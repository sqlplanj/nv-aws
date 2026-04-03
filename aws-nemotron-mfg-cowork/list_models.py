"""
Run this script to list all NVIDIA model IDs available in your AWS Bedrock account.
Usage:
    export AWS_BEARER_TOKEN_BEDROCK=<your-key>   # or set AWS credentials
    python list_models.py
"""

import os
import boto3

# ── Auth ──────────────────────────────────────────────────────────────────────
bearer = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

if bearer:
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = bearer

client = boto3.client("bedrock", region_name=region)

# ── List models ───────────────────────────────────────────────────────────────
resp = client.list_foundation_models(byProvider="nvidia")
models = resp.get("modelSummaries", [])

print(f"\nNVIDIA models available in {region}:\n")
print(f"{'modelId':<60} {'modelName'}")
print("-" * 100)
for m in sorted(models, key=lambda x: x["modelId"]):
    print(f"{m['modelId']:<60} {m.get('modelName', '')}")

print(f"\nTotal: {len(models)} models")
