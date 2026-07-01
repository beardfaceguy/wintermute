"""SageMaker serving backend — deploy a model to an LMI real-time endpoint.

Replaces the old infra/sagemaker/deploy_qwen3.py: deploys a model (HF Hub id or
S3 artifact) on the LMI v20 container and returns a ServingHandle carrying the
endpoint name. SageMaker endpoints are invoked via boto3 (not an OpenAI-compatible
URL), so the handle has no base_url.

Library use:
    from serving.sagemaker import SageMakerServeBackend
    h = SageMakerServeBackend(profile="experimental").deploy("huihui-ai/Qwen3-8B-abliterated")

CLI (equivalent to the old deploy_qwen3.py):
    python -m serving.sagemaker --profile experimental --model huihui-ai/Qwen3-8B-abliterated
    python -m serving.sagemaker --profile experimental --delete <endpoint-name>

The env builder / token loader are pure-tested; role resolution (create-if-
missing), model deployment, and endpoint deletion are seams (_resolve_role,
_deploy_model, _delete_endpoint) that do their boto3/sagemaker imports lazily
and are monkeypatched in tests.
"""

from __future__ import annotations

# Allow running as a bare script (`python serving/sagemaker.py`) in addition to
# `python -m serving.sagemaker` — the package imports below need the repo root.
if __package__ in (None, ""):
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import os  # noqa: E402
from typing import Any  # noqa: E402

from serving.base import ServeBackend, ServingHandle  # noqa: E402

REGION = "us-east-1"
ROLE_NAME = "SageMakerExecutionRole"
DEFAULT_INSTANCE = "ml.g5.2xlarge"

# LMI V20 container — vLLM 0.15.1 (same image used by infra/sagemaker/).
CONTAINER_URI = (
    f"763104351884.dkr.ecr.{REGION}.amazonaws.com"
    "/djl-inference:0.36.0-lmi20.0.0-cu128-v1.0"
)

_BASE_ENV = {
    "TENSOR_PARALLEL_DEGREE": "max",
    "OPTION_DTYPE": "bf16",
    "OPTION_MAX_MODEL_LEN": "8192",
    "OPTION_MAX_ROLLING_BATCH_SIZE": "8",
    "OPTION_GPU_MEMORY_UTILIZATION": "0.90",
}


def get_hf_token(explicit: str | None = None) -> str | None:
    """Resolve a HuggingFace token: explicit arg → ~/.cache/huggingface/token → HF_TOKEN."""
    if explicit:
        return explicit
    cache_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return f.read().strip()
    return os.environ.get("HF_TOKEN")


def build_serving_env(
    model_ref: str,
    env_extra: dict[str, str] | None = None,
    hf_token: str | None = None,
) -> dict[str, str]:
    """Build the LMI container env for a model.

    A HF Hub id is set as HF_MODEL_ID (downloaded at startup); an S3 artifact
    (s3://…) is mounted by the container, so no HF_MODEL_ID is set. A HF token is
    injected when provided (required for gated models).
    """
    env = dict(_BASE_ENV)
    if not model_ref.startswith("s3://"):
        env["HF_MODEL_ID"] = model_ref
    # TODO (S3 path, not yet exercised): an S3 artifact is mounted at
    # /opt/ml/model but LMI still needs to be told to load from there. A merged
    # HF tarball must include serving.properties (or set HF_MODEL_ID=/opt/ml/model)
    # or the container won't know what to serve. Validate when the S3 serving
    # path is first used.
    if hf_token:
        env["HF_TOKEN"] = hf_token
    if env_extra:
        env.update(env_extra)
    return env


# ── seams (lazy imports; monkeypatched in tests) ───────────────────────────────


def _resolve_role(profile: str) -> str:
    """Return the execution role ARN, creating the role if it doesn't exist."""
    import json
    import time

    import boto3

    session = boto3.session.Session(profile_name=profile, region_name=REGION)
    account_id = session.client("sts").get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{account_id}:role/{ROLE_NAME}"

    iam = session.client("iam")
    try:
        iam.get_role(RoleName=ROLE_NAME)
        return role_arn
    except iam.exceptions.NoSuchEntityException:
        pass

    iam.create_role(
        RoleName=ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "sagemaker.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }
        ),
        Description="SageMaker execution role for wintermute model endpoints",
    )
    for policy in [
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
    ]:
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=policy)
    time.sleep(15)  # IAM propagation
    return role_arn


def _deploy_model(
    model_ref: str,
    env: dict[str, str],
    role_arn: str,
    instance_type: str,
    endpoint_name: str,
    profile: str,
) -> str:
    # `import sagemaker` here resolves to the installed AWS SDK (absolute import),
    # not this module — safe as long as serving/ isn't placed directly on sys.path
    # (it isn't; the package is imported as serving.sagemaker from the repo root).
    import boto3
    import sagemaker
    from sagemaker.model import Model

    boto_session = boto3.session.Session(profile_name=profile, region_name=REGION)
    sess = sagemaker.session.Session(boto_session=boto_session)

    model_kwargs: dict[str, Any] = {
        "image_uri": CONTAINER_URI,
        "role": role_arn,
        "env": env,
        "sagemaker_session": sess,
    }
    if model_ref.startswith("s3://"):
        model_kwargs["model_data"] = model_ref

    model = Model(**model_kwargs)
    model.deploy(
        instance_type=instance_type,
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        container_startup_health_check_timeout=900,
    )
    return endpoint_name


def _delete_endpoint(profile: str, endpoint_name: str) -> None:
    import boto3

    boto3.session.Session(profile_name=profile, region_name=REGION).client(
        "sagemaker"
    ).delete_endpoint(EndpointName=endpoint_name)


class SageMakerServeBackend(ServeBackend):
    def __init__(self, profile: str = "experimental"):
        self.profile = profile

    def deploy(
        self,
        model_ref: str,
        *,
        instance_type: str = DEFAULT_INSTANCE,
        endpoint_name: str | None = None,
        served_model_name: str | None = None,
        env_extra: dict[str, str] | None = None,
        hf_token: str | None = None,
        **kwargs,
    ) -> ServingHandle:
        name = served_model_name or model_ref.rsplit("/", 1)[-1]
        ep_name = endpoint_name or _default_endpoint_name(name)
        token = get_hf_token(hf_token)
        env = build_serving_env(model_ref, env_extra=env_extra, hf_token=token)
        role_arn = _resolve_role(self.profile)
        ep = _deploy_model(model_ref, env, role_arn, instance_type, ep_name, self.profile)
        return ServingHandle(
            backend="sagemaker",
            model_name=name,
            endpoint_name=ep,
            extra={"instance_type": instance_type},
        )

    def delete(self, handle: ServingHandle) -> None:
        if not handle.endpoint_name:
            raise ValueError("handle has no endpoint_name to delete")
        _delete_endpoint(self.profile, handle.endpoint_name)


def _default_endpoint_name(model_name: str) -> str:
    import re

    base = re.sub(r"[^a-zA-Z0-9]+", "-", f"sft-{model_name}").strip("-").lower()
    return base[:63]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Deploy/delete a model on a SageMaker LMI endpoint"
    )
    parser.add_argument("--profile", default="experimental", help="AWS profile name")
    parser.add_argument("--model", help="HF model id or s3:// artifact to deploy")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE)
    parser.add_argument("--endpoint-name", help="Override the generated endpoint name")
    parser.add_argument("--hf-token", help="HF token (else ~/.cache/huggingface/token or HF_TOKEN)")
    parser.add_argument("--delete", metavar="ENDPOINT_NAME", help="Delete an existing endpoint")
    args = parser.parse_args(argv)

    backend = SageMakerServeBackend(profile=args.profile)
    if args.delete:
        backend.delete(ServingHandle(backend="sagemaker", model_name="", endpoint_name=args.delete))
        print(f"Deleted endpoint: {args.delete}")
        return
    if not args.model:
        parser.error("--model is required unless --delete is given")
    handle = backend.deploy(
        args.model,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
        hf_token=args.hf_token,
    )
    print(f"Endpoint deployed: {handle.endpoint_name}")


if __name__ == "__main__":
    main()
