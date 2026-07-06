"""
Launch a Titan pretraining job on SageMaker (script mode, PyTorch DLC).

Training is architecture-agnostic on SageMaker: the DLC provides torch+CUDA,
requirements.txt adds titans-trainer (fork), and train_entry.py runs the loop.
No BYOC needed for training (serving a Titan is the part that needs BYOC/EC2).

Prereqs (experimental account, us-east-1):
  - role: arn:aws:iam::491794274773:role/SageMakerExecutionRole
  - input staged at s3://<default-bucket>/titan-poc/.../ containing
    corpus.txt + vocab.json + merges.txt

Usage:
  python launch.py --dry-run --s3-input s3://<bucket>/titan-poc/dry-run-input/
"""
import argparse
import os

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROLE = "arn:aws:iam::491794274773:role/SageMakerExecutionRole"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="experimental")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--role", default=DEFAULT_ROLE)
    ap.add_argument("--instance-type", default="ml.g5.2xlarge")  # 1x A10G 24GB (quota 1)
    ap.add_argument("--s3-input", required=True, help="s3:// prefix with corpus.txt + tokenizer")
    ap.add_argument("--framework-version", default="2.3.0")
    ap.add_argument("--py-version", default="py311")
    ap.add_argument("--dry-run", action="store_true", help="tiny model + capped windows")
    ap.add_argument("--wait", action="store_true", help="stream logs until the job finishes")
    args = ap.parse_args()

    boto_sess = boto3.Session(profile_name=args.profile, region_name=args.region)
    sm_sess = sagemaker.Session(boto_session=boto_sess)

    if args.dry_run:
        hp = {"epochs": 1, "d-model": 256, "n-layers": 4, "n-heads": 4,
              "seq-len": 128, "batch-size": 8, "vocab-size": 8000, "max-windows": 200}
        max_run, base = 1800, "titan-dryrun"
    else:
        # ~50M-class dress rehearsal at cloud scale; tune up toward ~170M for the real run.
        hp = {"epochs": 1, "d-model": 512, "n-layers": 8, "n-heads": 8,
              "seq-len": 512, "batch-size": 16, "vocab-size": 8000, "max-windows": 0}
        max_run, base = 6 * 3600, "titan-pretrain"

    bucket = sm_sess.default_bucket()
    est = PyTorch(
        entry_point="train_entry.py",
        source_dir=HERE,  # ships requirements.txt too
        role=args.role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version=args.framework_version,
        py_version=args.py_version,
        hyperparameters=hp,
        sagemaker_session=sm_sess,
        output_path=f"s3://{bucket}/titan-poc/output/",
        max_run=max_run,
        base_job_name=base,
        disable_profiler=True,
    )
    print(f"launching {base} on {args.instance_type} | input={args.s3_input} | "
          f"output=s3://{bucket}/titan-poc/output/ | hp={hp}", flush=True)
    est.fit({"training": args.s3_input}, wait=args.wait)
    print(f"submitted job: {est.latest_training_job.name}", flush=True)


if __name__ == "__main__":
    main()
