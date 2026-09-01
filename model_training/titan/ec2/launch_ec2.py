"""
Launch (or dry-run/wire-up) the self-terminating Titan recall scale run on EC2 (#952).

Default mode does everything EXCEPT launch: creates the IAM instance profile,
uploads the run scripts to S3, and validates run_instances with DryRun=True.
Pass --go to actually launch. The instance is SELF-TERMINATING (see bootstrap.sh:
hard-cap shutdown + EXIT-trap shutdown + InstanceInitiatedShutdownBehavior=terminate),
so an unsupervised run cannot become runaway billing.

Prereqs: AWS profile `experimental` (us-east-1), AdministratorAccess.

  python launch_ec2.py                 # wire up + validate (no launch)
  python launch_ec2.py --go            # actually launch
  python launch_ec2.py --go --hard-cap-min 180
"""
import argparse
import json
import os
import time

import boto3
from botocore.exceptions import ClientError

HERE = os.path.dirname(os.path.abspath(__file__))
TITAN_DIR = os.path.dirname(HERE)
BASE_GPU_AMI_SSM = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
ROLE_NAME = "titan-ec2-runner"
PROFILE_NAME = "titan-ec2-runner"        # instance-profile name (== role name)
SCRIPTS = ["run_sweep.sh", "vram_probe.py"]  # from ec2/
TITAN_SCRIPTS = ["recall_lucidrains.py", "text_recall_lucidrains.py", "text_recall_adjacent.py"]  # from titan/
TOKENIZER_FILES = ["vocab.json", "merges.txt"]  # titan/tokenizer/ -> scripts/tokenizer/

TRUST = {"Version": "2012-10-17", "Statement": [
    {"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"},
     "Action": "sts:AssumeRole"}]}


def ensure_instance_profile(iam, bucket):
    """Idempotently create role + instance profile granting ONLY S3 to the
    titan-ec2 prefix (self-termination uses `shutdown`, not an IAM API)."""
    s3_policy = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
         "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/titan-ec2/*"]}]}
    try:
        iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(TRUST),
                        Description="Titan EC2 recall scale run (S3 only)")
        print(f"created role {ROLE_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
        print(f"role {ROLE_NAME} exists")
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="titan-s3",
                        PolicyDocument=json.dumps(s3_policy))
    try:
        iam.create_instance_profile(InstanceProfileName=PROFILE_NAME)
        print(f"created instance profile {PROFILE_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
        print(f"instance profile {PROFILE_NAME} exists")
    try:
        iam.add_role_to_instance_profile(InstanceProfileName=PROFILE_NAME, RoleName=ROLE_NAME)
    except ClientError as e:
        if e.response["Error"]["Code"] != "LimitExceeded":  # already attached
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="experimental")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--instance-type", default="g5.2xlarge")  # 1x A10G 24GB
    ap.add_argument("--bucket", default="sagemaker-us-east-1-491794274773")
    ap.add_argument("--tag", default="recall-scale")
    ap.add_argument("--ami", default="", help="override AMI (else latest DL base GPU)")
    ap.add_argument("--hard-cap-min", type=int, default=240, help="absolute wall-clock kill")
    ap.add_argument("--go", action="store_true", help="ACTUALLY launch (else dry-run validate)")
    args = ap.parse_args()

    sess = boto3.Session(profile_name=args.profile, region_name=args.region)
    ec2, iam, s3, ssm = sess.client("ec2"), sess.client("iam"), sess.client("s3"), sess.client("ssm")

    ami = args.ami or ssm.get_parameter(Name=BASE_GPU_AMI_SSM)["Parameter"]["Value"]
    s3_prefix = f"s3://{args.bucket}/titan-ec2/{args.tag}"

    # 1. IAM instance profile (S3 only)
    ensure_instance_profile(iam, args.bucket)

    # 2. upload run scripts to S3
    for f in SCRIPTS:
        s3.upload_file(os.path.join(HERE, f), args.bucket, f"titan-ec2/{args.tag}/scripts/{f}")
    for f in TITAN_SCRIPTS:
        s3.upload_file(os.path.join(TITAN_DIR, f), args.bucket, f"titan-ec2/{args.tag}/scripts/{f}")
    for f in TOKENIZER_FILES:
        s3.upload_file(os.path.join(TITAN_DIR, "tokenizer", f), args.bucket,
                       f"titan-ec2/{args.tag}/scripts/tokenizer/{f}")
    print(f"uploaded {SCRIPTS + TITAN_SCRIPTS} + tokenizer to {s3_prefix}/scripts/")

    # 3. build user-data from bootstrap.sh (inject S3 prefix + hard cap)
    with open(os.path.join(HERE, "bootstrap.sh")) as fh:
        boot = fh.read()
    user_data = (f"#!/bin/bash\nexport S3_PREFIX='{s3_prefix}' HARD_CAP_MIN='{args.hard_cap_min}'\n"
                 + boot.split("\n", 1)[1])  # drop bootstrap's own shebang line

    run_args = dict(
        ImageId=ami, InstanceType=args.instance_type, MinCount=1, MaxCount=1,
        UserData=user_data,
        IamInstanceProfile={"Name": PROFILE_NAME},
        InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                              "Ebs": {"VolumeSize": 100, "VolumeType": "gp3", "DeleteOnTermination": True}}],
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [{"Key": "Project", "Value": "titan-ec2"},
                                     {"Key": "Name", "Value": f"titan-{args.tag}"}]}],
    )
    print(f"AMI={ami} type={args.instance_type} hard_cap={args.hard_cap_min}m "
          f"shutdown_behavior=terminate results={s3_prefix}/titan_run.log")

    if not args.go:
        try:
            ec2.run_instances(DryRun=True, **run_args)
        except ClientError as e:
            if e.response["Error"]["Code"] == "DryRunOperation":
                print("\nDRY-RUN OK — wired up and validated. Launch with:  python launch_ec2.py --go")
                return
            raise
        return

    # --go: launch (retry briefly for IAM instance-profile propagation)
    for attempt in range(6):
        try:
            r = ec2.run_instances(**run_args)
            break
        except ClientError as e:
            if e.response["Error"]["Code"] in ("InvalidParameterValue", "InvalidIamInstanceProfile") and attempt < 5:
                print(f"IAM profile propagating... retry {attempt+1}/6")
                time.sleep(10)
                continue
            raise
    iid = r["Instances"][0]["InstanceId"]
    print(f"\nLAUNCHED {iid} ({args.instance_type}). Self-terminates on completion or at "
          f"{args.hard_cap_min}m hard cap.")
    print(f"watch:   aws s3 cp {s3_prefix}/titan_run.log - --profile {args.profile} --region {args.region}")
    print(f"status:  aws ec2 describe-instances --instance-ids {iid} "
          f"--query 'Reservations[0].Instances[0].State.Name' --profile {args.profile} --region {args.region} --output text")
    print(f"kill now: aws ec2 terminate-instances --instance-ids {iid} --profile {args.profile} --region {args.region}")


if __name__ == "__main__":
    main()
