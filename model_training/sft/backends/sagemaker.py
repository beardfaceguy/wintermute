"""SageMakerBackend — run the SFT job as a SageMaker Training Job.

Submits a HuggingFace estimator whose entry point is model_training/sft/train.py.
The pure kwargs builder (`build_estimator_kwargs`) carries the mapping logic and
is unit-tested; the role resolution and estimator construction are isolated
behind seam functions (`_resolve_role`, `_make_estimator`) that do their
boto3/sagemaker imports lazily and are monkeypatched in tests.

Reuses the `experimental` account conventions from infra/sagemaker/ (profile,
region, execution role). Real runs are validated in phase C (Vikunja #911).

Data: cfg.data.train_path should be the S3 URI of the training JSONL; it is
passed to estimator.fit() as the "train" channel (mounted at
/opt/ml/input/data/train on the instance). The YAML config's train_path should
then point at that mounted path for the on-instance run.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

from model_training.sft.backends.base import TrainBackend
from model_training.sft.config import SFTConfig

DEFAULT_INSTANCE = "ml.g5.12xlarge"  # 4x A10G (96 GB) — comfortable for 8B LoRA
REGION = "us-east-1"
ROLE_NAME = "SageMakerExecutionRole"

# HuggingFace DLC framework versions (used when no explicit image_uri is given).
DEFAULT_TRANSFORMERS_VERSION = "4.49.0"
DEFAULT_PYTORCH_VERSION = "2.5.1"
DEFAULT_PY_VERSION = "py311"

ENTRY_POINT = "model_training/sft/train.py"


def _repo_root() -> str:
    # backends/sagemaker.py → model_training/sft/backends → ... → repo root
    return str(pathlib.Path(__file__).resolve().parents[3])


def _job_name(base_model: str) -> str:
    """DNS-safe SageMaker base job name derived from the model id."""
    name = re.sub(r"[^a-zA-Z0-9]+", "-", f"sft-{base_model}").strip("-").lower()
    return name[:63]


def build_estimator_kwargs(
    cfg: SFTConfig,
    *,
    role_arn: str,
    source_dir: str,
    config_path: str,
    instance_type: str = DEFAULT_INSTANCE,
    instance_count: int = 1,
    image_uri: str | None = None,
) -> dict[str, Any]:
    """Build the HuggingFace estimator constructor kwargs from an SFTConfig."""
    kwargs: dict[str, Any] = {
        "entry_point": ENTRY_POINT,
        "source_dir": source_dir,
        "role": role_arn,
        "instance_type": instance_type,
        "instance_count": instance_count,
        "base_job_name": _job_name(cfg.model.base_model),
        "hyperparameters": {"config": config_path},
    }
    if image_uri:
        kwargs["image_uri"] = image_uri
    else:
        kwargs["transformers_version"] = DEFAULT_TRANSFORMERS_VERSION
        kwargs["pytorch_version"] = DEFAULT_PYTORCH_VERSION
        kwargs["py_version"] = DEFAULT_PY_VERSION
    return kwargs


# ── seams (lazy imports; monkeypatched in tests) ───────────────────────────────


def _resolve_role(profile: str) -> str:
    import boto3

    session = boto3.session.Session(profile_name=profile, region_name=REGION)
    account_id = session.client("sts").get_caller_identity()["Account"]
    return f"arn:aws:iam::{account_id}:role/{ROLE_NAME}"


def _make_estimator(kwargs: dict[str, Any], profile: str):
    import boto3
    import sagemaker
    from sagemaker.huggingface import HuggingFace

    boto_session = boto3.session.Session(profile_name=profile, region_name=REGION)
    sess = sagemaker.session.Session(boto_session=boto_session)
    return HuggingFace(sagemaker_session=sess, **kwargs)


class SageMakerBackend(TrainBackend):
    def __init__(self, profile: str = "experimental"):
        self.profile = profile

    def run(
        self,
        cfg: SFTConfig,
        *,
        source_dir: str | None = None,
        config_path: str | None = None,
        instance_type: str | None = None,
        image_uri: str | None = None,
        wait: bool = True,
    ) -> str:
        cfg.validate()
        if config_path is None:
            raise ValueError("config_path (YAML config path within source_dir) is required")

        role_arn = _resolve_role(self.profile)
        kwargs = build_estimator_kwargs(
            cfg,
            role_arn=role_arn,
            source_dir=source_dir or _repo_root(),
            config_path=config_path,
            instance_type=instance_type or DEFAULT_INSTANCE,
            image_uri=image_uri,
        )
        estimator = _make_estimator(kwargs, self.profile)
        estimator.fit({"train": cfg.data.train_path}, wait=wait)
        return estimator.model_data
