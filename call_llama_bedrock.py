"""
Bedrock Meta Llama caller (Converse API).

This wrapper keeps the same "prompt -> string" interface as your other call_* modules.
"""

from __future__ import annotations

import os
from typing import Optional, List

import boto3
from botocore.exceptions import ClientError


def call_llama_bedrock(
    prompt: str,
    *,
    model_id: str = "meta.llama3-1-70b-instruct-v1:0",
    region_name: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 0.9,
    stop_sequences: Optional[List[str]] = None,
) -> str:
    """
    Invoke a Meta Llama model hosted on Amazon Bedrock using the Converse API.

    Args:
        prompt: User prompt (single-turn).
        model_id: Bedrock model ID, e.g. "meta.llama3-1-70b-instruct-v1:0".
        region_name: AWS region (e.g. "us-east-1"). Falls back to env vars.
        max_tokens: Max output tokens.
        temperature: Sampling temperature.
        top_p: Nucleus sampling p.
        stop_sequences: Optional list of stop strings (if you want to hard-stop).

    Returns:
        The model's generated text (first content block).
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return ""

    region = (
        region_name
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )

    client = boto3.client("bedrock-runtime", region_name=region)

    # Bedrock Converse expects a messages list with content blocks.
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    inference_cfg = {"maxTokens": int(max_tokens), "temperature": float(temperature), "topP": float(top_p)}
    if stop_sequences:
        inference_cfg["stopSequences"] = list(stop_sequences)

    try:
        resp = client.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig=inference_cfg,
        )
        # Standard Converse shape for text: output.message.content[0].text
        content = resp.get("output", {}).get("message", {}).get("content", [])
        if content and isinstance(content, list) and "text" in content[0]:
            return content[0]["text"]
        # Fallback: try to join any text blocks if present
        texts = []
        for block in content or []:
            t = block.get("text")
            if t:
                texts.append(t)
        return "\n".join(texts).strip()
    except (ClientError, Exception) as e:
        # Keep behavior consistent with other callers: raise so caller can retry/log.
        raise RuntimeError(f"Bedrock Llama invoke failed for model_id={model_id} in region={region}: {e}") from e
