import os
import time
from typing import Optional
from openai import OpenAI


def is_transient_openai_error(e: Exception) -> bool:
    msg = str(e)
    return any(s in msg for s in ["429", "Rate limit", "500", "502", "503", "504", "timeout", "temporarily"])


def extract_text_from_response(resp) -> str:
    txt = (getattr(resp, "output_text", None) or "").strip()
    if txt:
        return txt

    parts = []
    output = getattr(resp, "output", None) or []
    for item in output:
        if not isinstance(item, dict):
            item = getattr(item, "__dict__", {}) or {}
        if item.get("type") != "message":
            continue
        for c in item.get("content", []) or []:
            if isinstance(c, dict) and c.get("type") == "output_text":
                t = c.get("text") or ""
                if t:
                    parts.append(t)
    return "".join(parts).strip()


def wait_until_done(client: OpenAI, resp, poll_interval_s: float = 0.5):
    while getattr(resp, "status", None) in ("queued", "in_progress"):
        time.sleep(poll_interval_s)
        resp = client.responses.get(resp.id)
    return resp


def create_response_with_retries(
    client: OpenAI,
    *,
    model: str,
    input_text: str,
    max_output_tokens: int,
    previous_response_id: Optional[str] = None,
    max_retries: int = 6,
    backoff_start: float = 1.0,
):
    backoff = backoff_start
    for _ in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                input=input_text,
                max_output_tokens=max_output_tokens,
                text={"format": {"type": "text"}},  # plain text
                tools=[],
                tool_choice="none",
                # IMPORTANT: do NOT set store=False if you want previous_response_id to work
            )
            if previous_response_id:
                kwargs["previous_response_id"] = previous_response_id

            return client.responses.create(**kwargs)

        except Exception as e:
            if is_transient_openai_error(e):
                time.sleep(backoff)
                backoff = min(backoff * 2, 20)
                continue
            raise

    raise RuntimeError("OpenAI call failed after retries")


def call_openai_gpt(
    prompt: str,
    model: str = "gpt-5",
    max_output_tokens: int = 4096,
    max_retries: int = 6,
    max_continuations: int = 6,
    poll_interval_s: float = 0.5,
) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set")

    client = OpenAI()

    resp = create_response_with_retries(
        client,
        model=model,
        input_text=prompt,
        max_output_tokens=max_output_tokens,
        max_retries=max_retries,
    )
    resp = wait_until_done(client, resp, poll_interval_s=poll_interval_s)

    if getattr(resp, "error", None):
        raise RuntimeError(f"OpenAI error: {resp.error}")

    text = extract_text_from_response(resp)

    for _ in range(max_continuations):
        if not getattr(resp, "incomplete_details", None):
            break

        resp = create_response_with_retries(
            client,
            model=model,
            input_text="Continue exactly where you left off. Do not repeat earlier text.",
            previous_response_id=resp.id,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
        )
        resp = wait_until_done(client, resp, poll_interval_s=poll_interval_s)

        if getattr(resp, "error", None):
            raise RuntimeError(f"OpenAI error: {resp.error}")

        more = extract_text_from_response(resp)
        if more:
            text = (text + "\n" + more).strip() if text else more.strip()

    if not text:
        raise RuntimeError(
            "No text produced. "
            f"status={getattr(resp,'status',None)} "
            f"incomplete={getattr(resp,'incomplete_details',None)}"
        )

    return text.strip()

