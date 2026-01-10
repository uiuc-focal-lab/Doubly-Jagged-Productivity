import os, time, random, requests
from typing import Any, Dict, List, Optional

RETRYABLE_HTTP = {429, 500, 502, 503, 504}
RETRYABLE_EMPTY_FINISH = {"STOP"}  # treat as transient if content is empty
BLOCKISH_FINISH = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "OTHER"}

def _extract_text(candidate: Dict[str, Any]) -> str:
    content = (candidate.get("content") or {})
    parts = content.get("parts") or []
    # Only collect explicit text parts
    chunks = []
    for p in parts:
        if isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
            chunks.append(p["text"])
    return "".join(chunks)

def call_gemini_api(
    prompt: str,
    model: str = "gemini-2.5-pro",
    max_output_tokens: int = 8192,
    temperature: float = 0.0,
    top_p: float = 0.95,
    timeout_s: int = 120,
    max_retries: int = 8,
) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    model_path = model if model.startswith("models/") else f"models/{model}"
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent"

    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "thinkingConfig": {
                "thinkingBudget": 1024  # minimum for 2.5 Pro (can't be 0)
            },
            "maxOutputTokens": max_output_tokens,  # consider 8192/16384+ for long answers
            "temperature": temperature,
            "topP": top_p,
        },
    }

    backoff = 1.0
    last_debug: Optional[Dict[str, Any]] = None

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)

            if r.status_code in RETRYABLE_HTTP:
                time.sleep(backoff + random.random() * 0.25)
                backoff = min(backoff * 2, 30)
                continue

            r.raise_for_status()
            data = r.json()
            last_debug = data

            # If the prompt got blocked, candidates may be missing
            pf = data.get("promptFeedback") or {}
            print("prompt feedback", pf, data)
            if pf.get("blockReason"):
                raise RuntimeError(f"Gemini prompt blocked: {pf.get('blockReason')} | {pf}")

            cands: List[Dict[str, Any]] = data.get("candidates") or []
            if not cands:
                # treat as transient (some service-side cases present like this)
                time.sleep(backoff + random.random() * 0.25)
                backoff = min(backoff * 2, 30)
                continue

            cand0 = cands[0]
            finish = cand0.get("finishReason")
            text_raw = _extract_text(cand0)

            # If text is empty, decide whether it's a block vs transient
            if text_raw == "":
                # If finish indicates a block-like stop, surface it
                if finish in BLOCKISH_FINISH:
                    raise RuntimeError(
                        f"Gemini returned no text (finishReason={finish}). "
                        f"safetyRatings={cand0.get('safetyRatings')} promptFeedback={pf}"
                    )
                # Sometimes you get STOP with empty content; retry
                if finish in RETRYABLE_EMPTY_FINISH or finish is None:
                    time.sleep(backoff + random.random() * 0.25)
                    backoff = min(backoff * 2, 30)
                    continue

            # Return without stripping; let caller strip if they want
            return text_raw

        except (requests.RequestException, ValueError):
            time.sleep(backoff + random.random() * 0.25)
            backoff = min(backoff * 2, 30)
            continue

    # If we kept getting empties, show the last response metadata to debug
    raise RuntimeError(f"Gemini call failed after retries. Last response: {last_debug}")
