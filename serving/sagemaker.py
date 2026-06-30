"""SageMaker serving backend — deploy a model to an LMI real-time endpoint.

The ServeBackend equivalent of infra/sagemaker/deploy_qwen3.py: it deploys a
model (HF Hub id or S3 model artifact) on the LMI v20 container and returns a
ServingHandle carrying the endpoint name. SageMaker endpoints are invoked via
boto3 (not an OpenAI-compatible URL), so the handle has no base_url.

The env builder is pure-tested; role resolution, model deployment, and endpoint
deletion are seams (_resolve_role, _deploy_model, _delete_endpoint) that do their
boto3/sagemaker imports lazily and are monkeypatched in tests.

NOTE: this supersedes infra/sagemaker/deploy_qwen3.py. Retiring that script and
updating its README + Vikunja #889 is tracked as Phase 2 housekeeping.
"""

from __future__ import annotations

from typing import Any

from serving.base import ServeBackend, ServingHandle

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


def build_serving_env(model_ref: str, env_extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build the LMI container env for a model.

    A HF Hub id is set as HF_MODEL_ID (downloaded at startup); an S3 artifact
    (s3://…) is mounted by the container, so no HF_MODEL_ID is set.
    """
    env = dict(_BASE_ENV)
    if not model_ref.startswith("s3://"):
        env["HF_MODEL_ID"] = model_ref
    # TODO (S3 path, not yet exercised): an S3 artifact is mounted at
    # /opt/ml/model but LMI still needs to be told to load from there. A merged
    # HF tarball must include serving.properties (or set HF_MODEL_ID=/opt/ml/model)
    # or the container won't know what to serve. Validate when the S3 serving
    # path is first used.
    if env_extra:
        env.update(env_extra)
    return env


# ── seams (lazy imports; monkeypatched in tests) ───────────────────────────────


def _resolve_role(profile: str) -> str:
    import boto3

    session = boto3.session.Session(profile_name=profile, region_name=REGION)
    account_id = session.client("sts").get_caller_identity()["Account"]
    return f"arn:aws:iam::{account_id}:role/{ROLE_NAME}"


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
        **kwargs,
    ) -> ServingHandle:
        name = served_model_name or model_ref.rsplit("/", 1)[-1]
        ep_name = endpoint_name or _default_endpoint_name(name)
        env = build_serving_env(model_ref, env_extra=env_extra)
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
