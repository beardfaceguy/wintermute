#!/usr/bin/env python3
"""
Deploy huihui-ai/Qwen3-8B-abliterated on SageMaker.

Uses LMI V20 (vLLM 0.15.1, async mode) on ml.g5.2xlarge (24GB VRAM).

Usage:
    # Deploy
    python3 deploy_qwen3.py --profile experimental

    # Deploy with explicit HF token
    python3 deploy_qwen3.py --profile experimental --hf-token hf_xxx

    # Delete endpoint when done
    python3 deploy_qwen3.py --profile experimental --delete qwen3-8b-XXXXXX

Prerequisites:
    pip install sagemaker boto3
    aws sso login --profile experimental
    # Accept model terms at: https://huggingface.co/huihui-ai/Qwen3-8B-abliterated
"""
import argparse
import json
import os
import sys
import time

# Remove this script's own directory from sys.path so the 'sagemaker' package
# (installed in site-packages) isn't shadowed by the infra/sagemaker/ directory.
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _this_dir]

import boto3
import sagemaker
from sagemaker.model import Model

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "huihui-ai/Qwen3-8B-abliterated"
INSTANCE_TYPE = "ml.g5.2xlarge"   # 1x A10G, 24 GB VRAM — fits 8B bf16 (~16 GB weights)
REGION = "us-east-1"
ROLE_NAME = "SageMakerExecutionRole"

# LMI V20 container — vLLM 0.15.1, Qwen3 native support, async mode default
CONTAINER_URI = (
    f"763104351884.dkr.ecr.{REGION}.amazonaws.com"
    "/djl-inference:0.36.0-lmi20.0.0-cu128-v1.0"
)


def get_or_create_role(iam, account_id):
    role_arn = f"arn:aws:iam::{account_id}:role/{ROLE_NAME}"
    try:
        iam.get_role(RoleName=ROLE_NAME)
        print(f"IAM role exists: {role_arn}")
        return role_arn
    except iam.exceptions.NoSuchEntityException:
        pass

    print(f"Creating IAM role {ROLE_NAME}...")
    iam.create_role(
        RoleName=ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }],
        }),
        Description="SageMaker execution role for wintermute model endpoints",
    )
    for policy in [
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
    ]:
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=policy)
        print(f"  attached {policy.split('/')[-1]}")

    print("Waiting 15s for IAM propagation...")
    time.sleep(15)

    print(f"Role created: {role_arn}")
    return role_arn


def get_hf_token(explicit):
    if explicit:
        return explicit
    cache_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(cache_path):
        return open(cache_path).read().strip()
    return os.environ.get("HF_TOKEN")


def deploy(args):
    boto_session = boto3.session.Session(
        profile_name=args.profile,
        region_name=REGION,
    )
    iam = boto_session.client("iam")
    sts = boto_session.client("sts")

    account_id = sts.get_caller_identity()["Account"]
    print(f"Account: {account_id}  Region: {REGION}")

    role_arn = get_or_create_role(iam, account_id)
    hf_token = get_hf_token(args.hf_token)

    env = {
        "HF_MODEL_ID": MODEL_ID,
        "TENSOR_PARALLEL_DEGREE": "max",
        "OPTION_DTYPE": "bf16",
        "OPTION_MAX_MODEL_LEN": "8192",
        "OPTION_MAX_ROLLING_BATCH_SIZE": "8",
        "OPTION_GPU_MEMORY_UTILIZATION": "0.90",
    }
    if hf_token:
        env["HF_TOKEN"] = hf_token
        print("HF token: loaded")
    else:
        print("WARNING: no HF token found — model download may fail if repo is gated")

    sess = sagemaker.session.Session(boto_session=boto_session)
    endpoint_name = sagemaker.utils.name_from_base("qwen3-8b")

    print(f"\nDeploying {MODEL_ID}")
    print(f"  endpoint:  {endpoint_name}")
    print(f"  instance:  {INSTANCE_TYPE}")
    print(f"  container: {CONTAINER_URI}")
    print("  (model downloads from HF Hub at startup — allow 10-15 min)\n")

    model = Model(
        image_uri=CONTAINER_URI,
        role=role_arn,
        env=env,
        sagemaker_session=sess,
    )

    predictor = model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        container_startup_health_check_timeout=900,
    )

    print(f"\nEndpoint ready: {endpoint_name}")

    print("\nRunning smoke test...")
    response = predictor.predict({
        "inputs": "Hello, who are you?",
        "parameters": {"max_new_tokens": 128, "temperature": 0.7},
    })
    print("Response:", json.dumps(response, indent=2))

    out = {"endpoint_name": endpoint_name, "model_id": MODEL_ID, "instance": INSTANCE_TYPE}
    with open("endpoint.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nEndpoint config saved to endpoint.json")


def delete(args):
    boto_session = boto3.session.Session(profile_name=args.profile, region_name=REGION)
    sm = boto_session.client("sagemaker")
    print(f"Deleting endpoint: {args.delete}")
    sm.delete_endpoint(EndpointName=args.delete)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Deploy/delete Qwen3-8B-abliterated on SageMaker")
    parser.add_argument("--profile", default="experimental", help="AWS profile name")
    parser.add_argument("--hf-token", help="HuggingFace token (falls back to ~/.cache/huggingface/token)")
    parser.add_argument("--delete", metavar="ENDPOINT_NAME", help="Delete an existing endpoint")
    args = parser.parse_args()

    if args.delete:
        delete(args)
    else:
        deploy(args)


if __name__ == "__main__":
    main()
