"""
Mistral on Amazon Bedrock (via Bedrock Runtime Converse API).

This wrapper keeps the same "prompt -> string" interface as your other call_* modules.

Works with:
- raw Bedrock model IDs (e.g., "mistral.mistral-large-3-675b-instruct", "mistral.mistral-large-2407-v1:0"), or
- inference profile IDs (e.g., "us.mistral.pixtral-large-2502-v1:0"), or
- ARNs for models / inference profiles.
"""

from __future__ import annotations

from typing import Optional, List

import boto3
from botocore.exceptions import ClientError


def _extract_text_from_converse_response(resp: dict) -> str:
    """
    Standard Converse response shape:
      resp["output"]["message"]["content"] = [{"text": "..."} , ...]
    Some models may return multiple content blocks; join them.
    """
    content = (resp.get("output", {}) or {}).get("message", {}).get("content", [])
    if isinstance(content, list):
        texts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                texts.append(block["text"])
        return "\n".join(texts).strip()
    return ""


def call_mistral_bedrock(
    prompt: str,
    *,
    model_id: str = "mistral.mistral-large-3-675b-instruct",
    region_name: str = "us-east-1",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 0.9,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Call a Mistral model hosted on Amazon Bedrock using the Converse API.

    Returns the assistant's final text (no reasoning traces).
    """
    client = boto3.client("bedrock-runtime", region_name=region_name)

    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}],
        }
    ]

    inference_cfg = {
        "maxTokens": int(max_tokens),
        "temperature": float(temperature),
        "topP": float(top_p),
    }

    kwargs = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": inference_cfg,
    }
    if system_prompt:
        kwargs["system"] = [{"text": system_prompt}]

    try:
        resp = client.converse(**kwargs)
    except ClientError as e:
        raise RuntimeError(
            f"Bedrock Mistral call failed (modelId={model_id}, region={region_name}): {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Bedrock Mistral call failed (modelId={model_id}, region={region_name}): {e}"
        ) from e

    text = _extract_text_from_converse_response(resp)
    if not text:
        # As a last resort, try alternative keys (should rarely happen)
        return str(resp)
    return text
