import boto3
from botocore.exceptions import BotoCoreError, ClientError


def call_deepseek_bedrock(
    prompt: str,
    model_id: str,
    region_name: str = "us-east-1",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> str:
    """Call a DeepSeek model through Amazon Bedrock Converse."""

    client = boto3.client(
        "bedrock-runtime",
        region_name=region_name,
    )

    try:
        response = client.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            },
        )
    except (ClientError, BotoCoreError) as exc:
        raise RuntimeError(
            f"Failed to invoke DeepSeek model '{model_id}' through Bedrock"
        ) from exc

    content = response.get("output", {}).get("message", {}).get("content", [])

    text_blocks = []
    for block in content:
        if not isinstance(block, dict):
            continue

        if block.get("text"):
            text_blocks.append(block["text"])

        # Some DeepSeek responses may include a separate reasoning block.
        reasoning = block.get("reasoningContent", {})
        reasoning_text = reasoning.get("reasoningText", {})
        if isinstance(reasoning_text, dict) and reasoning_text.get("text"):
            # Ignore reasoning by default; only final text is needed for parsing.
            pass

    if not text_blocks:
        raise RuntimeError(
            f"DeepSeek returned no final text content. Response: {response}"
        )

    return "".join(text_blocks).strip()