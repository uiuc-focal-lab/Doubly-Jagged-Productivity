from botocore.exceptions import ClientError
import json
import time
import boto3

def call_claude_bedrock(prompt: str, model_id: str, max_tokens: int = 2048,
                       temperature: float = 0.1,
                       max_retries: int = 8) -> str:
    """
    Calls Claude on Bedrock using the Anthropic Messages API format.
    model_id can be the inference profile ID (e.g. us.anthropic...) or its ARN.
    """
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    claude_opus = "us.anthropic.claude-opus-4-5-20251101-v1:0"
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ],
    }

    backoff = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            data = json.loads(resp["body"].read())

            # Claude returns content blocks; concatenate any text blocks
            blocks = data.get("content", [])
            text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
            return text.strip()

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            # common transient errors: throttling / timeouts / service busy
            if code in {"ThrottlingException", "ModelTimeoutException", "ServiceUnavailableException"}:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            raise  # non-transient → bubble up

    raise RuntimeError("Bedrock Claude call failed after retries")
