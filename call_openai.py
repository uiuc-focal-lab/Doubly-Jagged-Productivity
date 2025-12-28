import os
from openai import OpenAI

def call_openai_gpt(prompt: str, model: str, max_output_tokens: int = 2048,
                temperature: float = 0.1, max_retries: int = 8) -> str:
    """OpenAI Responses API (GPT-5 etc.). model is a model ID like 'gpt-5'."""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
                            
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set (add it to your .env or export it)")
    openai_client = OpenAI()
    backoff = 1.0
    for _ in range(max_retries):
        try:
            resp = openai_client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_output_tokens,
            )
            # SDK provides convenience property
            return (resp.output_text or "").strip()
        except Exception as e:
            msg = str(e)
            # basic retry on common transient errors (rate limits / 5xx)
            if any(s in msg for s in ["Rate limit", "429", "502", "503", "504", "temporarily", "timeout"]):
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            raise

    raise RuntimeError("OpenAI call failed after retries")

