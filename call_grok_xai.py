import os
import requests
from typing import Optional


class GrokAPIError(RuntimeError):
    pass


def call_grok_api(
    prompt: str,
    model: str = "grok-4",
    *,
    max_output_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout_s: float = 180.0,
    base_url: Optional[str] = None,
) -> str:
    """
    Call xAI's Grok via the xAI API (OpenAI-compatible chat completions).

    Env vars:
        XAI_API_KEY   (required)
        XAI_BASE_URL  (optional; defaults to https://api.x.ai/v1)
    """
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise GrokAPIError("Missing XAI_API_KEY environment variable.")

    if base_url is None:
        base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/")

    url = f"{base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_output_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if resp.status_code >= 400:
        raise GrokAPIError(f"xAI API error {resp.status_code}: {resp.text[:2000]}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise GrokAPIError(f"Unexpected xAI response format: {data}") from e
