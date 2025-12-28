import json
import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "us.anthropic.claude-opus-4-5-20251101-v1:0"
body = { "anthropic_version": "bedrock-2023-05-31", "max_tokens": 4000,"system": "You are a helpful assistant.","messages": [ {"role": "user", "content": "Write a 3-bullet summary of abstract interpretation."}],}
resp = client.invoke_model(modelId=model_id,body=json.dumps(body),contentType="application/json",accept="application/json",)
data = json.loads(resp["body"].read())
print(data["content"][0]["text"])
