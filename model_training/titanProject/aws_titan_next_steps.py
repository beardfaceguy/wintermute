#!/usr/bin/env python3
"""
AWS helper for Titans training environment.

This script keeps AWS setup idempotent and explicit:
- audit current environment state
- create the training security group if it is missing
- launch spot-first or on-demand EC2 runners with documented defaults
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_BUCKET = "alix-ai-ml-staging-data"
DEFAULT_PREFIX = "titan/"
DEFAULT_ROLE = "alix-llm-training-role"
DEFAULT_INLINE_POLICY = "alix-llm-training-s3"
DEFAULT_INSTANCE_PROFILE = "alix-llm-training-profile"
DEFAULT_SG_NAME = "alix-pc-llm-model-training"
DOC_SG_ID = "sg-0bec109715d614af7"
DEFAULT_KEY_NAME = "alix-pc-llm-training-key"
DEFAULT_INSTANCE_TYPE = "g6.2xlarge"
DEFAULT_AMI = "ami-0ad8dd83d01a01d3a"
DEFAULT_SSH_CIDR = "23.93.208.154/32"
DEFAULT_AZ = "us-east-1d"
DEFAULT_AZ_FALLBACKS = ["us-east-1d", "us-east-1c", "us-east-1b", "us-east-1f", "us-east-1a"]
DEFAULT_VOLUME_SIZE = 500

DEFAULT_TAGS: List[Dict[str, str]] = [
    {"Key": "Owner", "Value": "patrick.clawson"},
    {"Key": "Project", "Value": "Titan-LLM"},
    {"Key": "Env", "Value": "staging"},
    {"Key": "CostCenter", "Value": "ai-ml-training"},
    {"Key": "Purpose", "Value": "titan-training"},
    {"Key": "Name", "Value": "titan-train-staging-g6-2xlarge"},
]


@dataclass
class AwsContext:
    profile: str
    region: str


def _cmd_display(cmd: List[str]) -> str:
    return " ".join(cmd)


def run_aws(
    ctx: AwsContext,
    args: List[str],
    expect_json: bool = True,
    allow_failure: bool = False,
) -> Tuple[int, Optional[Dict[str, Any]], str, str]:
    base_args = ["--no-cli-pager", "--profile", ctx.profile, "--region", ctx.region] + args
    aws_bins: List[str] = []
    if os.environ.get("AWS_CLI_BIN"):
        aws_bins.append(os.environ["AWS_CLI_BIN"])
    aws_bins.extend(
        [
            "aws",
            "aws.exe",
            "/mnt/c/Program Files/Amazon/AWSCLIV2/aws.exe",
        ]
    )

    cmd: List[str] = []
    proc: Optional[subprocess.CompletedProcess[str]] = None
    for aws_bin in aws_bins:
        cmd = [aws_bin] + base_args
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            break
        except FileNotFoundError:
            continue

    if proc is None:
        raise RuntimeError(
            "AWS CLI executable was not found (tried `aws`, `aws.exe`, and default Windows AWS CLI path). "
            "Install AWS CLI v2 in WSL, or set `AWS_CLI_BIN` to a valid aws executable path."
        )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()

    payload: Optional[Dict[str, Any]] = None
    if expect_json and stdout:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            if not allow_failure:
                raise RuntimeError(
                    f"Failed to parse JSON output for `{_cmd_display(cmd)}`: {exc}\n"
                    f"stdout={stdout}\nstderr={stderr}"
                ) from exc

    if proc.returncode != 0 and not allow_failure:
        raise RuntimeError(
            f"`{_cmd_display(cmd)}` failed with exit code {proc.returncode}\n"
            f"stdout={stdout}\nstderr={stderr}"
        )

    return proc.returncode, payload, stdout, stderr


def flatten_instances(data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not data:
        return out
    for reservation in data.get("Reservations", []):
        out.extend(reservation.get("Instances", []))
    return out


def describe_sg_by_name(ctx: AwsContext, sg_name: str) -> List[Dict[str, Any]]:
    _, data, _, _ = run_aws(
        ctx,
        ["ec2", "describe-security-groups", "--filters", f"Name=group-name,Values={sg_name}"],
        expect_json=True,
        allow_failure=True,
    )
    if not data:
        return []
    return data.get("SecurityGroups", [])


def describe_sg_by_id(ctx: AwsContext, sg_id: str) -> List[Dict[str, Any]]:
    _, data, _, _ = run_aws(
        ctx,
        ["ec2", "describe-security-groups", "--group-ids", sg_id],
        expect_json=True,
        allow_failure=True,
    )
    if not data:
        return []
    return data.get("SecurityGroups", [])


def get_default_vpc_id(ctx: AwsContext) -> Optional[str]:
    _, data, _, _ = run_aws(
        ctx,
        ["ec2", "describe-vpcs", "--filters", "Name=isDefault,Values=true"],
        expect_json=True,
        allow_failure=True,
    )
    if not data:
        return None
    vpcs = data.get("Vpcs", [])
    if not vpcs:
        return None
    return vpcs[0].get("VpcId")


def ensure_sg(ctx: AwsContext, sg_name: str, ssh_cidr: str, vpc_id: Optional[str]) -> str:
    existing = describe_sg_by_name(ctx, sg_name)
    if existing:
        return existing[0]["GroupId"]

    resolved_vpc = vpc_id or get_default_vpc_id(ctx)
    if not resolved_vpc:
        raise RuntimeError(
            "No default VPC found. Re-run with --vpc-id to create the security group explicitly."
        )

    _, data, _, _ = run_aws(
        ctx,
        [
            "ec2",
            "create-security-group",
            "--group-name",
            sg_name,
            "--description",
            "Titan model training SSH access",
            "--vpc-id",
            resolved_vpc,
        ],
        expect_json=True,
    )
    if not data or "GroupId" not in data:
        raise RuntimeError("Security group creation did not return GroupId.")
    group_id = data["GroupId"]

    ip_permissions = json.dumps(
        [
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": ssh_cidr, "Description": "SSH for Titan training"}],
            }
        ]
    )
    run_aws(
        ctx,
        ["ec2", "authorize-security-group-ingress", "--group-id", group_id, "--ip-permissions", ip_permissions],
        expect_json=True,
        allow_failure=True,
    )

    tag_args = []
    for tag in DEFAULT_TAGS:
        tag_args.append(f"Key={tag['Key']},Value={tag['Value']}")
    run_aws(
        ctx,
        ["ec2", "create-tags", "--resources", group_id, "--tags"] + tag_args,
        expect_json=True,
        allow_failure=True,
    )
    return group_id


def resolve_sg_id(ctx: AwsContext, sg_id: Optional[str], sg_name: str) -> str:
    if sg_id:
        return sg_id
    by_name = describe_sg_by_name(ctx, sg_name)
    if not by_name:
        raise RuntimeError(
            f"Security group `{sg_name}` not found. Run `ensure-sg` first or pass --sg-id."
        )
    return by_name[0]["GroupId"]


def key_pair_exists(ctx: AwsContext, key_name: str) -> bool:
    rc, _, _, _ = run_aws(
        ctx,
        ["ec2", "describe-key-pairs", "--key-names", key_name],
        expect_json=True,
        allow_failure=True,
    )
    return rc == 0


def launch_instance(
    ctx: AwsContext,
    spot: bool,
    az_candidates: List[str],
    sg_id: str,
    key_name: str,
    ami: str,
    instance_type: str,
    instance_profile: str,
    volume_size: int,
) -> Dict[str, Any]:
    if not key_pair_exists(ctx, key_name):
        raise RuntimeError(
            f"Key pair `{key_name}` is not available in {ctx.region} for profile {ctx.profile}."
        )

    run_errors: List[str] = []
    tag_spec = json.dumps([{"ResourceType": "instance", "Tags": DEFAULT_TAGS}])
    bdm = json.dumps(
        [
            {
                "DeviceName": "/dev/sdf",
                "Ebs": {"VolumeSize": volume_size, "VolumeType": "gp3"},
            }
        ]
    )

    for az in az_candidates:
        base_args = [
            "ec2",
            "run-instances",
            "--image-id",
            ami,
            "--instance-type",
            instance_type,
            "--placement",
            f"AvailabilityZone={az}",
            "--iam-instance-profile",
            f"Name={instance_profile}",
            "--key-name",
            key_name,
            "--security-group-ids",
            sg_id,
            "--block-device-mappings",
            bdm,
            "--tag-specifications",
            tag_spec,
            "--count",
            "1",
        ]

        if spot:
            market = json.dumps(
                {
                    "MarketType": "spot",
                    "SpotOptions": {
                        "SpotInstanceType": "one-time",
                        "InstanceInterruptionBehavior": "terminate",
                    },
                }
            )
            base_args += ["--instance-market-options", market]

        rc, data, stdout, stderr = run_aws(
            ctx,
            base_args,
            expect_json=True,
            allow_failure=True,
        )
        if rc == 0 and data:
            instances = data.get("Instances", [])
            if not instances:
                raise RuntimeError("Launch call succeeded but returned no instance records.")
            first = instances[0]
            return {
                "InstanceId": first.get("InstanceId"),
                "AvailabilityZone": first.get("Placement", {}).get("AvailabilityZone", az),
                "State": first.get("State", {}).get("Name", "unknown"),
                "Spot": spot,
            }

        details = stderr or stdout or "unknown error"
        run_errors.append(f"{az}: {details}")

    joined = "\n".join(run_errors)
    raise RuntimeError(f"Failed to launch instance in all AZ candidates:\n{joined}")


def check_audit(ctx: AwsContext) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    identity_rc, ident, identity_stdout, identity_stderr = run_aws(
        ctx, ["sts", "get-caller-identity"], expect_json=True, allow_failure=True
    )
    out["identity"] = ident or {}
    out["identity_ok"] = identity_rc == 0
    out["identity_error"] = (identity_stderr or identity_stdout).strip() if identity_rc != 0 else ""

    bucket_ok_rc, _, _, _ = run_aws(
        ctx,
        ["s3api", "head-bucket", "--bucket", DEFAULT_BUCKET],
        expect_json=False,
        allow_failure=True,
    )
    out["bucket_exists"] = bucket_ok_rc == 0

    _, titan_listing, _, _ = run_aws(
        ctx,
        [
            "s3api",
            "list-objects-v2",
            "--bucket",
            DEFAULT_BUCKET,
            "--prefix",
            DEFAULT_PREFIX,
            "--delimiter",
            "/",
            "--max-keys",
            "1000",
        ],
        expect_json=True,
        allow_failure=True,
    )
    titan_prefixes = []
    if titan_listing:
        titan_prefixes = [entry.get("Prefix", "") for entry in titan_listing.get("CommonPrefixes", [])]
    out["titan_prefixes"] = titan_prefixes

    _, ckpt_listing, _, _ = run_aws(
        ctx,
        [
            "s3api",
            "list-objects-v2",
            "--bucket",
            DEFAULT_BUCKET,
            "--prefix",
            "titan/checkpoints/",
            "--max-keys",
            "1000",
        ],
        expect_json=True,
        allow_failure=True,
    )
    ckpt_keys = []
    if ckpt_listing:
        ckpt_keys = [entry.get("Key", "") for entry in ckpt_listing.get("Contents", [])]
    out["checkpoint_keys"] = ckpt_keys

    role_rc, role_payload, _, _ = run_aws(
        ctx,
        ["iam", "get-role", "--role-name", DEFAULT_ROLE],
        expect_json=True,
        allow_failure=True,
    )
    out["role_exists"] = role_rc == 0
    out["role_arn"] = (role_payload or {}).get("Role", {}).get("Arn")

    _, attached_payload, _, _ = run_aws(
        ctx,
        ["iam", "list-attached-role-policies", "--role-name", DEFAULT_ROLE],
        expect_json=True,
        allow_failure=True,
    )
    attached = (attached_payload or {}).get("AttachedPolicies", [])
    out["ssm_attached"] = any(
        p.get("PolicyArn") == "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore" for p in attached
    )

    inline_rc, inline_payload, _, _ = run_aws(
        ctx,
        [
            "iam",
            "get-role-policy",
            "--role-name",
            DEFAULT_ROLE,
            "--policy-name",
            DEFAULT_INLINE_POLICY,
        ],
        expect_json=True,
        allow_failure=True,
    )
    out["inline_policy_exists"] = inline_rc == 0
    out["inline_policy"] = (inline_payload or {}).get("PolicyDocument", {})

    profile_rc, profile_payload, _, _ = run_aws(
        ctx,
        ["iam", "get-instance-profile", "--instance-profile-name", DEFAULT_INSTANCE_PROFILE],
        expect_json=True,
        allow_failure=True,
    )
    out["instance_profile_exists"] = profile_rc == 0
    profile_roles = (profile_payload or {}).get("InstanceProfile", {}).get("Roles", [])
    out["instance_profile_role_names"] = [r.get("RoleName", "") for r in profile_roles]

    by_name = describe_sg_by_name(ctx, DEFAULT_SG_NAME)
    out["sg_by_name_count"] = len(by_name)
    out["sg_by_name_ids"] = [sg.get("GroupId", "") for sg in by_name]

    by_id = describe_sg_by_id(ctx, DOC_SG_ID)
    out["doc_sg_id_exists"] = len(by_id) > 0

    out["key_pair_exists"] = key_pair_exists(ctx, DEFAULT_KEY_NAME)

    _, project_instances_payload, _, _ = run_aws(
        ctx,
        [
            "ec2",
            "describe-instances",
            "--filters",
            "Name=tag:Project,Values=Titan-LLM",
            "Name=instance-state-name,Values=pending,running,stopping,stopped",
        ],
        expect_json=True,
        allow_failure=True,
    )
    project_instances = flatten_instances(project_instances_payload)
    out["project_instances"] = [
        {
            "InstanceId": i.get("InstanceId", ""),
            "State": i.get("State", {}).get("Name", ""),
            "Type": i.get("InstanceType", ""),
            "AZ": i.get("Placement", {}).get("AvailabilityZone", ""),
        }
        for i in project_instances
    ]

    _, g5_payload, _, _ = run_aws(
        ctx,
        [
            "ec2",
            "describe-instances",
            "--filters",
            "Name=instance-type,Values=g6.2xlarge,g6.xlarge,g5.2xlarge,g5.xlarge",
            "Name=instance-state-name,Values=pending,running,stopping,stopped",
        ],
        expect_json=True,
        allow_failure=True,
    )
    gpu_runner_instances = flatten_instances(g5_payload)
    out["gpu_runner_instances"] = [
        {
            "InstanceId": i.get("InstanceId", ""),
            "State": i.get("State", {}).get("Name", ""),
            "Type": i.get("InstanceType", ""),
            "AZ": i.get("Placement", {}).get("AvailabilityZone", ""),
            "NameTag": next((t.get("Value") for t in i.get("Tags", []) if t.get("Key") == "Name"), ""),
        }
        for i in gpu_runner_instances
    ]

    _, spot_payload, _, _ = run_aws(
        ctx,
        ["ec2", "describe-spot-instance-requests"],
        expect_json=True,
        allow_failure=True,
    )
    spot_requests = (spot_payload or {}).get("SpotInstanceRequests", [])
    out["spot_requests"] = [
        {
            "SpotInstanceRequestId": s.get("SpotInstanceRequestId", ""),
            "State": s.get("State", ""),
            "StatusCode": s.get("Status", {}).get("Code", ""),
            "InstanceId": s.get("InstanceId", ""),
        }
        for s in spot_requests
    ]

    return out


def checklist_from_audit(audit: Dict[str, Any]) -> List[Dict[str, str]]:
    prefixes = set(audit.get("titan_prefixes", []))
    code_present = "titan/code/" in prefixes
    data_present = "titan/data/" in prefixes
    checkpoints = audit.get("checkpoint_keys", [])
    baseline_ckpt = any("baseline" in key.lower() for key in checkpoints)
    project_instances = audit.get("project_instances", [])
    running_instances = [i for i in project_instances if i.get("State") in ("pending", "running")]

    launch_status = "done" if running_instances else "pending"
    bootstrap_status = "done" if code_present and data_present else "pending"
    if baseline_ckpt:
        baseline_status = "done"
        baseline_detail = "Baseline checkpoint objects found in s3://alix-ai-ml-staging-data/titan/checkpoints/."
    elif checkpoints:
        baseline_status = "pending"
        baseline_detail = "Checkpoint objects exist, but none appear baseline-labeled."
    else:
        baseline_status = "pending"
        baseline_detail = "No checkpoint objects found under titan/checkpoints/."

    return [
        {
            "item": "Launch training runner (spot-first g6.2xlarge with fallback)",
            "status": launch_status,
            "detail": (
                f"{len(running_instances)} runner(s) currently pending/running."
                if running_instances
                else "No pending/running Titan-tagged EC2 instances detected."
            ),
        },
        {
            "item": "Bootstrap instance (/mnt/data mount + code/data sync)",
            "status": bootstrap_status,
            "detail": (
                "S3 prefixes titan/code/ and titan/data/ detected."
                if bootstrap_status == "done"
                else "Missing titan/code/ and/or titan/data/ prefix objects in S3."
            ),
        },
        {
            "item": "Run baseline training and sync checkpoints to S3",
            "status": baseline_status,
            "detail": baseline_detail,
        },
    ]


def print_audit(audit: Dict[str, Any], ctx: AwsContext) -> None:
    identity = audit.get("identity", {})
    account = identity.get("Account", "unknown")
    arn = identity.get("Arn", "unknown")
    identity_error = str(audit.get("identity_error", "") or "").strip()
    print("== Environment Summary ==")
    print(f"Profile: {ctx.profile}")
    print(f"Region: {ctx.region}")
    print(f"Account: {account}")
    print(f"CallerArn: {arn}")
    if identity_error:
        error_lines = [line.strip() for line in identity_error.splitlines() if line.strip()]
        print(f"AuthError: {error_lines[-1] if error_lines else identity_error}")
    print()

    print("== Resource State ==")
    print(f"S3 bucket `{DEFAULT_BUCKET}` exists: {audit.get('bucket_exists')}")
    print(f"S3 titan prefixes: {audit.get('titan_prefixes', [])}")
    print(f"Checkpoint object count: {len(audit.get('checkpoint_keys', []))}")
    print(f"IAM role `{DEFAULT_ROLE}` exists: {audit.get('role_exists')}")
    print(f"SSM attached on role: {audit.get('ssm_attached')}")
    print(f"Inline policy `{DEFAULT_INLINE_POLICY}` exists: {audit.get('inline_policy_exists')}")
    print(f"Instance profile `{DEFAULT_INSTANCE_PROFILE}` exists: {audit.get('instance_profile_exists')}")
    print(f"Instance profile roles: {audit.get('instance_profile_role_names', [])}")
    print(f"Security group by name `{DEFAULT_SG_NAME}` count: {audit.get('sg_by_name_count')}")
    print(f"Security group IDs by name: {audit.get('sg_by_name_ids', [])}")
    print(f"Documented SG ID `{DOC_SG_ID}` exists: {audit.get('doc_sg_id_exists')}")
    print(f"Key pair `{DEFAULT_KEY_NAME}` exists: {audit.get('key_pair_exists')}")
    print(f"Titan-tagged instances: {audit.get('project_instances', [])}")
    print(f"Known Titan GPU runner instances: {audit.get('gpu_runner_instances', [])}")
    print(f"Spot requests: {audit.get('spot_requests', [])}")
    print()

    mismatches: List[str] = []
    if identity_error:
        mismatches.append(
            "AWS authentication/profile resolution failed. Resource checks may be incomplete until credentials are fixed."
        )
    else:
        if not audit.get("doc_sg_id_exists") and audit.get("sg_by_name_count", 0) > 0:
            mismatches.append(
                f"Doc SG ID `{DOC_SG_ID}` is absent, but SG name `{DEFAULT_SG_NAME}` exists with different ID(s)."
            )
        if audit.get("sg_by_name_count", 0) == 0:
            mismatches.append(f"SG `{DEFAULT_SG_NAME}` does not exist in this account/region.")
        if audit.get("bucket_exists") and len(audit.get("titan_prefixes", [])) == 0:
            mismatches.append("S3 bucket exists but no `titan/` prefixes found yet.")
        if not audit.get("instance_profile_exists") and audit.get("role_exists"):
            mismatches.append("Role exists but instance profile is missing.")

    print("== Open Items Checklist ==")
    for row in checklist_from_audit(audit):
        print(f"[{row['status']}] {row['item']}")
        print(f"  - {row['detail']}")
    print()

    print("== Mismatches ==")
    if mismatches:
        for mismatch in mismatches:
            print(f"- {mismatch}")
    else:
        print("- No major doc-vs-env mismatches detected from the current audit.")
    print()

    print("== Recommended Next Commands ==")
    if identity_error:
        print(
            "1) Verify profile exists:\n"
            "   /home/zombi/.local/bin/aws configure list-profiles"
        )
        print(
            "2) Authenticate/profile setup in WSL:\n"
            "   /home/zombi/.local/bin/aws configure sso --profile experimental-admin"
        )
        print(
            "3) Re-run audit after credentials are fixed:\n"
            "   python model_training/titanProject/aws_titan_next_steps.py audit"
        )
    else:
        print(
            "1) Ensure SG exists:\n"
            "   python model_training/titanProject/aws_titan_next_steps.py ensure-sg"
        )
        print(
            "2) Launch spot-first runner:\n"
            "   python model_training/titanProject/aws_titan_next_steps.py launch-spot"
        )
        print(
            "3) Re-audit after launch:\n"
            "   python model_training/titanProject/aws_titan_next_steps.py audit"
        )


def build_ctx(args: argparse.Namespace) -> AwsContext:
    profile = args.profile or os.environ.get("AWS_PROFILE", "experimental-admin")
    region = args.region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return AwsContext(profile=profile, region=region)


def handle_audit(args: argparse.Namespace) -> int:
    ctx = build_ctx(args)
    audit = check_audit(ctx)
    if args.json:
        print(json.dumps(audit, indent=2, sort_keys=True))
    else:
        print_audit(audit, ctx)
    return 0


def handle_ensure_sg(args: argparse.Namespace) -> int:
    ctx = build_ctx(args)
    group_id = ensure_sg(ctx, args.sg_name, args.ssh_cidr, args.vpc_id)
    print(f"Security group ready: {group_id}")
    return 0


def handle_launch(args: argparse.Namespace, spot: bool) -> int:
    ctx = build_ctx(args)
    sg_id = resolve_sg_id(ctx, args.sg_id, args.sg_name)
    azs = [az.strip() for az in args.az_order.split(",") if az.strip()] if spot else [args.az]
    launched = launch_instance(
        ctx=ctx,
        spot=spot,
        az_candidates=azs,
        sg_id=sg_id,
        key_name=args.key_name,
        ami=args.ami,
        instance_type=args.instance_type,
        instance_profile=args.instance_profile,
        volume_size=args.volume_size,
    )

    print("Instance launch requested.")
    print(json.dumps(launched, indent=2, sort_keys=True))
    print()
    print("Bootstrap/run reminder:")
    print("ssh ubuntu@<public-ip>  # or use your SSH config host")
    print("sudo mkdir -p /mnt/data")
    print('ROOT_DISK="/dev/$(lsblk -no PKNAME \\"$(findmnt -n -o SOURCE /)\\")"')
    print('DATA_DEV="$(lsblk -dpno NAME,TYPE | awk \'$2==\\"disk\\"{print $1}\' | grep -v \\"^${ROOT_DISK}$\\" | head -1)"')
    print('sudo mkfs -t xfs "$DATA_DEV"')
    print('UUID="$(sudo blkid -s UUID -o value "$DATA_DEV")"')
    print('echo "UUID=$UUID /mnt/data xfs defaults,nofail 0 2" | sudo tee -a /etc/fstab')
    print("sudo mount -a")
    print("cd /mnt/data/code/wintermute/model_training/titanProject")
    print("python train.py --config configs/config_baseline_nomem.yaml --device cuda --log-every 100")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AWS Titans hosting helper")
    parser.add_argument("--profile", default=None, help="AWS profile (default: env AWS_PROFILE or experimental-admin)")
    parser.add_argument("--region", default=None, help="AWS region (default: env AWS_DEFAULT_REGION or us-east-1)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Audit AWS env vs Titans open items")
    audit_parser.add_argument("--json", action="store_true", help="Emit raw JSON audit payload")

    sg_parser = subparsers.add_parser("ensure-sg", help="Create SG if missing (idempotent)")
    sg_parser.add_argument("--sg-name", default=DEFAULT_SG_NAME, help="Security group name")
    sg_parser.add_argument("--ssh-cidr", default=DEFAULT_SSH_CIDR, help="Allowed SSH CIDR for port 22")
    sg_parser.add_argument("--vpc-id", default=None, help="VPC ID override; default is account default VPC")

    spot_parser = subparsers.add_parser("launch-spot", help="Launch spot runner with AZ fallback")
    spot_parser.add_argument("--sg-id", default=None, help="Security group ID override")
    spot_parser.add_argument("--sg-name", default=DEFAULT_SG_NAME, help="Security group name lookup")
    spot_parser.add_argument("--key-name", default=DEFAULT_KEY_NAME, help="EC2 key pair name")
    spot_parser.add_argument("--ami", default=DEFAULT_AMI, help="AMI ID")
    spot_parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE, help="EC2 instance type")
    spot_parser.add_argument("--instance-profile", default=DEFAULT_INSTANCE_PROFILE, help="IAM instance profile name")
    spot_parser.add_argument("--volume-size", type=int, default=DEFAULT_VOLUME_SIZE, help="Root volume size GiB")
    spot_parser.add_argument(
        "--az-order",
        default=",".join(DEFAULT_AZ_FALLBACKS),
        help="Comma-separated AZ priority for spot launch fallback",
    )

    ondemand_parser = subparsers.add_parser("launch-ondemand", help="Launch on-demand runner in a single AZ")
    ondemand_parser.add_argument("--sg-id", default=None, help="Security group ID override")
    ondemand_parser.add_argument("--sg-name", default=DEFAULT_SG_NAME, help="Security group name lookup")
    ondemand_parser.add_argument("--key-name", default=DEFAULT_KEY_NAME, help="EC2 key pair name")
    ondemand_parser.add_argument("--ami", default=DEFAULT_AMI, help="AMI ID")
    ondemand_parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE, help="EC2 instance type")
    ondemand_parser.add_argument("--instance-profile", default=DEFAULT_INSTANCE_PROFILE, help="IAM instance profile name")
    ondemand_parser.add_argument("--volume-size", type=int, default=DEFAULT_VOLUME_SIZE, help="Root volume size GiB")
    ondemand_parser.add_argument("--az", default=DEFAULT_AZ, help="Single AZ for on-demand launch")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "audit":
            return handle_audit(args)
        if args.command == "ensure-sg":
            return handle_ensure_sg(args)
        if args.command == "launch-spot":
            return handle_launch(args, spot=True)
        if args.command == "launch-ondemand":
            return handle_launch(args, spot=False)
    except RuntimeError as err:
        print(str(err), file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
