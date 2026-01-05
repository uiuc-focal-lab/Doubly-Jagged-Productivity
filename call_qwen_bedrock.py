"""
Qwen on Amazon Bedrock (via Bedrock Runtime Converse API).

This wrapper keeps the same "prompt -> string" interface as your other call_* modules.
It works with either:
- a raw Bedrock model ID (e.g., "qwen.qwen3-next-80b-a3b"), or
- an inference profile ID (e.g., "us.qwen.qwen3-next-80b-a3b"), or
- an ARN for a model / inference profile.
"""

from __future__ import annotations

from typing import Optional, List

import boto3
from botocore.exceptions import ClientError


def _extract_text_from_converse_response(resp: dict) -> str:
    # Standard Converse shape for text: output.message.content[*].text
    content = (resp or {}).get("output", {}).get("message", {}).get("content", [])
    if isinstance(content, list):
        texts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                texts.append(block["text"])
        return "\n".join(texts).strip()
    return str(content).strip()


def call_qwen_bedrock(
    prompt: str,
    model_id: str,
    region_name: str = "us-east-1",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 0.9,
    stop_sequences: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Call a Qwen model on Bedrock using the Converse API and return the generated text.
    """
    client = boto3.client("bedrock-runtime", region_name=region_name)

    messages = [{"role": "user", "content": [{"text": prompt}]}]

    inference_cfg = {
        "maxTokens": int(max_tokens),
        "temperature": float(temperature),
        "topP": float(top_p),
    }
    if stop_sequences:
        inference_cfg["stopSequences"] = list(stop_sequences)

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
        raise RuntimeError(f"Bedrock Qwen call failed (modelId={model_id}, region={region_name}): {e}") from e
    except Exception as e:
        raise RuntimeError(f"Bedrock Qwen call failed (modelId={model_id}, region={region_name}): {e}") from e

    return _extract_text_from_converse_response(resp)
