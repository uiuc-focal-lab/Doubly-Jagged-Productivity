from google import genai

def call_gemini_api(prompt: str, model: str, max_output_tokens: int = 2048,
               temperature: float = 0.1, top_p: float = 0.95,
               timeout_s: int = 120, max_retries: int = 8) -> str:
    """Gemini Developer API via REST generateContent."""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        raise ValueError("API_KEY not found in .env file")
                 
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set (add it to your .env)")

    model_path = model if model.startswith("models/") else f"models/{model}"
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent"
    params = {"key": GEMINI_API_KEY}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": temperature,
            "topP": top_p,
        },
    }

    backoff = 1.0
    for _ in range(max_retries):
        try:
            r = requests.post(url, params=params, json=payload, timeout=timeout_s)
            if r.status_code == 200:
                data = r.json()
                cands = data.get("candidates", [])
                if not cands:
                    return ""
                parts = cands[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                return text.strip()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue

    raise RuntimeError("Gemini call failed after retries")
