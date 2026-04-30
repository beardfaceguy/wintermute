#!/usr/bin/env bash
# Multi-GPU GPT-medium pretrain (config_gpt_medium.yaml, ~407M params, DDP via torchrun) via SSM.
# Stdout/stderr -> CloudWatch (/aws/ssm/titan-llm-training) and S3 (ssm-logs/...).
# Default mode detaches the actual training process from SSM after bootstrap so long runs
# survive the SSM command timeout window. Final checkpoint sync and optional self-stop then
# happen on the instance itself.
#
# Requires a multi-GPU instance (g5.12xlarge = 4x A10G, g5.24xlarge = 4x A10G, etc.).
# NPROC_PER_NODE defaults to all available GPUs.
#
# One-time:
#   - create log group + attach scripts/aws_commands/iam/ssm_cloudwatch_logs_inline_policy.json
#   - if STOP_INSTANCE_ON_EXIT=1, also attach scripts/aws_commands/iam/ssm_long_run_self_stop_inline_policy.json
#   to alix-llm-training-role (see policy file headers).
#   - instance must have tag Purpose=titan-training (auto-applied on self-stop, but pre-tagging is safer).
# Set INSTANCE_ID before running.

set -euo pipefail

INSTANCE_ID="${INSTANCE_ID:-i-REPLACE_ME}"
REGION="${REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-experimental-admin}"
SSM_EXEC_TIMEOUT_SECONDS="${SSM_EXEC_TIMEOUT_SECONDS:-43200}"
SSM_DELIVERY_TIMEOUT_SECONDS="${SSM_DELIVERY_TIMEOUT_SECONDS:-43200}"
CW_LOG_GROUP="${CW_LOG_GROUP:-/aws/ssm/titan-llm-training}"
S3_BUCKET="${S3_BUCKET:-alix-ai-ml-staging-data}"
DETACH_TRAINING="${DETACH_TRAINING:-1}"
STOP_INSTANCE_ON_EXIT="${STOP_INSTANCE_ON_EXIT:-1}"
SYNC_FINAL_LOG_TO_S3="${SYNC_FINAL_LOG_TO_S3:-1}"
# REMOTE_RUN_ROOT is set dynamically on-instance after DATA_ROOT detection
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
LOG_PREFIX="ssm-logs/gpt-medium-pretrain-multigpu/$(date +%Y%m%d%H%M%S)"

if [[ "${INSTANCE_ID}" == "i-REPLACE_ME" ]]; then
  echo "Set INSTANCE_ID to your Titan multi-GPU training instance id." >&2
  echo "" >&2
  echo "IMPORTANT: If STOP_INSTANCE_ON_EXIT=1 (default), the instance MUST have:" >&2
  echo "  1. The IAM policy from iam/ssm_long_run_self_stop_inline_policy.json attached to its role" >&2
  echo "  2. Tag: Purpose=titan-training" >&2
  echo "" >&2
  echo "To tag an existing instance:" >&2
  echo "  aws ec2 create-tags --resources <instance-id> --tags Key=Purpose,Value=titan-training" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to build SSM --parameters JSON (brew install jq)." >&2
  exit 1
fi

REMOTE_SCRIPT="$(mktemp)"
PARAMS_JSON="$(mktemp)"
trap 'rm -f "${REMOTE_SCRIPT}" "${PARAMS_JSON}"' EXIT

{
printf '#!/bin/bash\n'
printf 'set -euo pipefail\n\n'
printf 'MAX_STEPS=%q\n' "${MAX_STEPS:-}"
printf 'LOG_EVERY=%q\n' "${LOG_EVERY:-100}"
printf 'SAVE_EVERY=%q\n' "${SAVE_EVERY:-}"
printf 'REGION=%q\n' "${REGION}"
printf 'RESUME_CKPT_S3_URI=%q\n' "${RESUME_CKPT_S3_URI:-}"
printf 'TRAIN_MAX_TOKENS_OVERRIDE=%q\n' "${TRAIN_MAX_TOKENS_OVERRIDE:-}"
printf 'VAL_MAX_TOKENS_OVERRIDE=%q\n' "${VAL_MAX_TOKENS_OVERRIDE:-}"
printf 'DETACH_TRAINING=%q\n' "${DETACH_TRAINING}"
printf 'STOP_INSTANCE_ON_EXIT=%q\n' "${STOP_INSTANCE_ON_EXIT}"
printf 'SYNC_FINAL_LOG_TO_S3=%q\n' "${SYNC_FINAL_LOG_TO_S3}"
printf 'NPROC_PER_NODE=%q\n\n' "${NPROC_PER_NODE}"
cat <<'EOF'
START_ISO=$(date -Iseconds)
echo "[meta] run_start_iso=${START_ISO}"

# ── Data root detection ──────────────────────────────────────────────
# AWS DL AMI instances (g5.*) use LVM for NVMe ephemeral at /opt/dlami/nvme.
# Other instances (g6.*) may have a separate EBS at /mnt/data.
# Fall back to creating /mnt/data on root volume if neither exists.
resolve_data_root() {
  if mountpoint -q /opt/dlami/nvme 2>/dev/null; then
    echo "/opt/dlami/nvme"
    return
  fi
  if mountpoint -q /mnt/data 2>/dev/null; then
    echo "/mnt/data"
    return
  fi
  if command -v lvm >/dev/null 2>&1; then
    vgchange -ay 2>/dev/null || true
    local lv_path
    lv_path=$(lvs --noheadings -o lv_path 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "${lv_path}" ]]; then
      mkdir -p /opt/dlami/nvme
      mount "${lv_path}" /opt/dlami/nvme 2>/dev/null || true
      if mountpoint -q /opt/dlami/nvme 2>/dev/null; then
        echo "/opt/dlami/nvme"
        return
      fi
    fi
  fi
  mkdir -p /mnt/data
  echo "/mnt/data"
}

DATA_ROOT="$(resolve_data_root)"
export DATA_ROOT
echo "[data-root] DATA_ROOT=${DATA_ROOT}"

# Auto-detect GPU count if NPROC_PER_NODE=auto
if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
  NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
  if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
    echo "[error] no GPUs detected and NPROC_PER_NODE=auto" >&2
    exit 1
  fi
  echo "[meta] auto-detected NPROC_PER_NODE=${NPROC_PER_NODE}"
fi

RUN_ID="gpt_medium_pretrain_multigpu_$(date +%Y%m%d%H%M%S)"
CHECKPOINT_DIR="${DATA_ROOT}/checkpoints/${RUN_ID}"
S3_PREFIX="s3://alix-ai-ml-staging-data/titan/checkpoints/${RUN_ID}/"
CODE_DIR="/home/ubuntu/wintermute"
CODE_BUNDLE_URI="${CODE_BUNDLE_URI:-s3://alix-ai-ml-staging-data/titan/code_bundles/titanProject_bundle.tar.gz}"
CODE_SYNC_DEST="${CODE_DIR}/model_training/titanProject"
CODE_BUNDLE_LOCAL="/tmp/titanProject_bundle.tar.gz"
DATA_DIR="${DATA_ROOT}/datasets"
TRAIN_LOCAL="${DATA_DIR}/train.txt"
VAL_LOCAL="${DATA_DIR}/val.txt"
CFG_LOCAL="${CODE_DIR}/model_training/titanProject/configs/config_gpt_medium.local.yaml"
MAX_STEPS="${MAX_STEPS:-}"
LOG_EVERY="${LOG_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-}"
RESUME_CKPT_S3_URI="${RESUME_CKPT_S3_URI:-}"
TRAIN_MAX_TOKENS_OVERRIDE="${TRAIN_MAX_TOKENS_OVERRIDE:-}"
VAL_MAX_TOKENS_OVERRIDE="${VAL_MAX_TOKENS_OVERRIDE:-}"
DETACH_TRAINING="${DETACH_TRAINING:-1}"
STOP_INSTANCE_ON_EXIT="${STOP_INSTANCE_ON_EXIT:-1}"
SYNC_FINAL_LOG_TO_S3="${SYNC_FINAL_LOG_TO_S3:-1}"
REMOTE_RUN_ROOT="${DATA_ROOT}/ssm_runs"
RUN_WORK_DIR="${REMOTE_RUN_ROOT}/${RUN_ID}"
TRAIN_LOG="${RUN_WORK_DIR}/train.log"
RUN_STATUS_JSON="${RUN_WORK_DIR}/run_status.json"
RUNNER_PID_FILE="${RUN_WORK_DIR}/runner.pid"
RUNNER_SCRIPT="${RUN_WORK_DIR}/run_training.sh"
DETACHED_LAUNCH_LOG="${RUN_WORK_DIR}/launcher.log"
S3_RUN_ARTIFACT_PREFIX="${S3_PREFIX%/}/run_artifacts"
export TRAIN_MAX_TOKENS_OVERRIDE VAL_MAX_TOKENS_OVERRIDE
export MAX_STEPS LOG_EVERY SAVE_EVERY NPROC_PER_NODE
export RESUME_CKPT_S3_URI
export RUN_ID CHECKPOINT_DIR S3_PREFIX CODE_DIR CFG_LOCAL DETACH_TRAINING
export STOP_INSTANCE_ON_EXIT SYNC_FINAL_LOG_TO_S3 RUN_WORK_DIR TRAIN_LOG
export RUN_STATUS_JSON REGION S3_RUN_ARTIFACT_PREFIX

echo "=== host ==="
date -Iseconds
uname -a || true
echo "=== disk ==="
df -h / "${DATA_ROOT}" 2>/dev/null || df -h || true
echo "=== memory ==="
free -h 2>/dev/null || true
echo "=== gpu ==="
nvidia-smi 2>/dev/null || echo "[warn] nvidia-smi unavailable"
echo "=== paths ==="
echo "DATA_ROOT=${DATA_ROOT}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "S3_PREFIX=${S3_PREFIX}"
echo "CODE_BUNDLE_URI=${CODE_BUNDLE_URI}"
echo "CODE_SYNC_DEST=${CODE_SYNC_DEST}"
echo "MAX_STEPS=${MAX_STEPS:-<config default>}"
echo "LOG_EVERY=${LOG_EVERY}"
echo "SAVE_EVERY=${SAVE_EVERY:-<config default>}"
echo "RESUME_CKPT_S3_URI=${RESUME_CKPT_S3_URI:-<unset>}"
echo "TRAIN_MAX_TOKENS_OVERRIDE=${TRAIN_MAX_TOKENS_OVERRIDE:-<unset>}"
echo "VAL_MAX_TOKENS_OVERRIDE=${VAL_MAX_TOKENS_OVERRIDE:-<unset>}"

mkdir -p "${CODE_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${DATA_DIR}"
mkdir -p "$(dirname "${CODE_SYNC_DEST}")"
mkdir -p "${RUN_WORK_DIR}"

echo "=== code bundle download start ==="
date -Iseconds
rm -rf "${CODE_SYNC_DEST}"
aws s3 cp "${CODE_BUNDLE_URI}" "${CODE_BUNDLE_LOCAL}" --only-show-errors
echo "=== code bundle download done ==="
date -Iseconds
echo "=== code bundle extract start ==="
tar -xzf "${CODE_BUNDLE_LOCAL}" -C "${CODE_DIR}"
rm -f "${CODE_BUNDLE_LOCAL}"
echo "=== code bundle extract done ==="
date -Iseconds
du -sh "${CODE_SYNC_DEST}" || true

cd "${CODE_DIR}"

echo "=== pip install ==="
date -Iseconds
python3 -m pip install -q --no-cache-dir numpy==1.26.4
python3 -m pip install -q --upgrade --no-cache-dir \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
python3 -m pip install -q --no-cache-dir sentencepiece boto3 pyarrow
python3 -m pip install -q --no-cache-dir titans-pytorch==0.5.3 --no-deps
python3 -m pip install -q --no-cache-dir \
  einops==0.8.2 einx==0.4.2 hyper-connections==0.4.9 axial-positional-embedding==0.3.12 \
  assoc-scan==0.0.4 ema-pytorch==0.7.9 tqdm fire loguru orjson tensordict==0.11.0 \
  x-transformers==2.17.7 rotary-embedding-torch==0.8.9 ninja pyvers cloudpickle frozendict --no-deps
echo "=== pip install done ==="

echo "=== torch cuda (after install) ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available());
gpus = torch.cuda.device_count();
print(f'gpu_count={gpus}');
[print(f'  gpu{i}: {torch.cuda.get_device_name(i)}') for i in range(gpus)]"

echo "=== dataset preload to local disk ==="
date -Iseconds
for i in 1 2 3; do
  aws s3 cp s3://alix-ai-ml-staging-data/titan/data/processed/train.txt "${TRAIN_LOCAL}" --only-show-errors && break
  echo "[warn] train.txt download failed (attempt ${i}); retrying..."
  sleep 5
done
for i in 1 2 3; do
  aws s3 cp s3://alix-ai-ml-staging-data/titan/data/processed/val.txt "${VAL_LOCAL}" --only-show-errors && break
  echo "[warn] val.txt download failed (attempt ${i}); retrying..."
  sleep 5
done
ls -lh "${TRAIN_LOCAL}" "${VAL_LOCAL}"

echo "=== token cache preload from S3 ==="
date -Iseconds
TOKEN_CACHE_S3="${TOKEN_CACHE_S3_URI:-s3://alix-ai-ml-staging-data/titan/token_cache/}"
TOKEN_CACHE_LOCAL="${DATA_DIR}/.titan_token_cache"
mkdir -p "${TOKEN_CACHE_LOCAL}"
if aws s3 ls "${TOKEN_CACHE_S3}" --no-cli-pager >/dev/null 2>&1; then
  aws s3 sync "${TOKEN_CACHE_S3}" "${TOKEN_CACHE_LOCAL}" --only-show-errors && \
    echo "[cache] synced pre-built token cache from ${TOKEN_CACHE_S3}" || \
    echo "[cache] token cache sync failed; will tokenize from scratch"
else
  echo "[cache] no pre-built token cache at ${TOKEN_CACHE_S3}; will tokenize from scratch"
fi
export TITAN_TOKEN_CACHE_TRUST_EXISTING=1

python3 - <<'PY'
import os
import yaml
from pathlib import Path

data_root = os.environ["DATA_ROOT"]
src = Path("model_training/titanProject/configs/config_gpt_medium.yaml")
dst = Path("model_training/titanProject/configs/config_gpt_medium.local.yaml")

cfg = yaml.safe_load(src.read_text())
cfg["data"]["train_path"] = f"{data_root}/datasets/train.txt"
cfg["data"]["val_path"] = f"{data_root}/datasets/val.txt"
train_override = os.environ.get("TRAIN_MAX_TOKENS_OVERRIDE")
val_override = os.environ.get("VAL_MAX_TOKENS_OVERRIDE")
if train_override:
    cfg["data"]["max_tokens"] = int(train_override)
if val_override:
    cfg["data"]["max_tokens_val"] = int(val_override)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"[meta] wrote local config: {dst}")
print(
    "[meta] local config caps:"
    f" train_max_tokens={cfg['data'].get('max_tokens')}"
    f" val_max_tokens={cfg['data'].get('max_tokens_val')}"
)

mcfg = cfg["model"]
params_est = (mcfg["vocab_size"] * mcfg["dim"] + mcfg.get("max_seq_len", 2048) * mcfg["dim"]
    + mcfg["depth"] * (4 * mcfg["dim"]**2 + 2 * mcfg["dim"] * mcfg["dim"] * mcfg["ff_mult"] + 2 * mcfg["dim"])
    + mcfg["dim"] + mcfg["vocab_size"] * mcfg["dim"])
tcfg = cfg["train"]
nproc = int(os.environ["NPROC_PER_NODE"])
tok_per_step = tcfg["batch_size"] * tcfg["seq_len"] * tcfg.get("grad_accum_steps", 1) * nproc
total_tok = tok_per_step * tcfg["max_steps"]
print(f"[meta] model_params_est={params_est:,} ({params_est/1e6:.0f}M)")
print(f"[meta] tokens_per_step={tok_per_step:,} (x{nproc} GPUs)")
print(f"[meta] total_tokens_est={total_tok:,} ({total_tok/1e9:.1f}B)")
PY

if [[ "${DETACH_TRAINING}" == "1" ]]; then
  cat >"${RUNNER_SCRIPT}" <<'RUNNER_EOF'
#!/bin/bash
set -euo pipefail

get_instance_id() {
  local token
  token="$(curl -fsS -X PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' 2>/dev/null || true)"
  if [[ -n "${token}" ]]; then
    curl -fsS -H "X-aws-ec2-metadata-token: ${token}" \
      http://169.254.169.254/latest/meta-data/instance-id
  else
    curl -fsS http://169.254.169.254/latest/meta-data/instance-id
  fi
}

sync_final_artifacts() {
  aws s3 sync "${CHECKPOINT_DIR}" "${S3_PREFIX}" --exclude "*" --include "ckpt_step_*.pt" --only-show-errors || \
    echo "[warn] final checkpoint sync failed"
  if [[ "${SYNC_FINAL_LOG_TO_S3}" == "1" && -f "${TRAIN_LOG}" ]]; then
    aws s3 cp "${TRAIN_LOG}" "${S3_RUN_ARTIFACT_PREFIX}/train.log" --only-show-errors || \
      echo "[warn] final train.log upload failed"
  fi
  if [[ -f "${RUN_STATUS_JSON}" ]]; then
    aws s3 cp "${RUN_STATUS_JSON}" "${S3_RUN_ARTIFACT_PREFIX}/run_status.json" --only-show-errors || \
      echo "[warn] final run_status upload failed"
  fi
}

finalize() {
  local exit_code=$?
  local end_iso
  end_iso="$(date -Iseconds)"
  export RUN_EXIT_CODE="${exit_code}"
  export END_ISO="${end_iso}"
  python3 - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "run_id": os.environ["RUN_ID"],
    "checkpoint_dir": os.environ["CHECKPOINT_DIR"],
    "s3_prefix": os.environ["S3_PREFIX"],
    "train_log": os.environ["TRAIN_LOG"],
    "nproc_per_node": int(os.environ["NPROC_PER_NODE"]),
    "ended_at": os.environ["END_ISO"],
    "exit_code": int(os.environ["RUN_EXIT_CODE"]),
}
Path(os.environ["RUN_STATUS_JSON"]).write_text(json.dumps(payload, indent=2) + "\n")
print(f"[meta] wrote run status: {os.environ['RUN_STATUS_JSON']}")
PY
  sync_final_artifacts
  if [[ "${STOP_INSTANCE_ON_EXIT}" == "1" ]]; then
    local instance_id=""
    instance_id="$(get_instance_id 2>/dev/null || true)"
    if [[ -n "${instance_id}" ]]; then
      echo "[self-stop] attempting to stop instance ${instance_id} in region ${REGION}"
      aws ec2 create-tags --resources "${instance_id}" --region "${REGION}" \
        --tags Key=Purpose,Value=titan-training 2>/dev/null || \
        echo "[self-stop] could not ensure Purpose tag (non-fatal)"
      if aws ec2 stop-instances --instance-ids "${instance_id}" --region "${REGION}" --output json 2>&1; then
        echo "[self-stop] stop-instances succeeded"
      else
        echo "[self-stop] FAILED — instance will keep running and incurring charges!"
        echo "[self-stop] ensure the instance role has ec2:StopInstances permission"
        echo "[self-stop] and the instance has tag Purpose=titan-training"
        echo "[self-stop] see: scripts/aws_commands/iam/ssm_long_run_self_stop_inline_policy.json"
      fi
    else
      echo "[self-stop] FAILED — could not determine instance id from metadata service"
    fi
  fi
  trap - EXIT
  exit "${exit_code}"
}

trap finalize EXIT

cd "${CODE_DIR}"
RESUME_ARGS=()
if [[ -n "${RESUME_CKPT_S3_URI}" ]]; then
  RESUME_CKPT_LOCAL="${RUN_WORK_DIR}/$(basename "${RESUME_CKPT_S3_URI}")"
  echo "[meta] resume checkpoint uri=${RESUME_CKPT_S3_URI}" | tee -a "${TRAIN_LOG}"
  aws s3 cp "${RESUME_CKPT_S3_URI}" "${RESUME_CKPT_LOCAL}" --only-show-errors
  RESUME_ARGS=(--resume "${RESUME_CKPT_LOCAL}")
fi
echo "[meta] detached runner start $(date -Iseconds) nproc=${NPROC_PER_NODE}" | tee -a "${TRAIN_LOG}"
PYTHONUNBUFFERED=1 torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port=29500 \
  model_training/titanProject/train.py \
  --config "${CFG_LOCAL}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --s3-checkpoint-uri "${S3_PREFIX}" \
  "${RESUME_ARGS[@]}" \
  ${MAX_STEPS:+--max-steps "${MAX_STEPS}"} \
  --log-every "${LOG_EVERY}" \
  ${SAVE_EVERY:+--save-every "${SAVE_EVERY}"} \
  --data-log-every-lines 50000 \
  --min-free-gb 20 \
  --aws-bin aws 2>&1 | tee -a "${TRAIN_LOG}"
RUNNER_EOF

  chmod +x "${RUNNER_SCRIPT}"
  nohup bash "${RUNNER_SCRIPT}" >"${DETACHED_LAUNCH_LOG}" 2>&1 </dev/null &
  RUNNER_PID=$!
  echo "${RUNNER_PID}" >"${RUNNER_PID_FILE}"
  sleep 5
  if ! kill -0 "${RUNNER_PID}" 2>/dev/null; then
    echo "[error] detached runner exited immediately"
    test -f "${DETACHED_LAUNCH_LOG}" && sed -n '1,160p' "${DETACHED_LAUNCH_LOG}"
    exit 1
  fi
  echo "[meta] detached runner pid=${RUNNER_PID}"
  echo "[meta] detached train log=${TRAIN_LOG}"
  echo "[meta] detached launcher log=${DETACHED_LAUNCH_LOG}"
  echo "[meta] detached status json=${RUN_STATUS_JSON}"
  echo "[meta] final artifacts prefix=${S3_RUN_ARTIFACT_PREFIX}"
  echo "[meta] nproc_per_node=${NPROC_PER_NODE}"
  echo "[meta] stop-on-exit=${STOP_INSTANCE_ON_EXIT} (requires ec2:StopInstances on the instance role)"
  echo "[meta] bootstrap complete; multi-GPU training now continues independently of this SSM command"
else
  RESUME_ARGS=()
  if [[ -n "${RESUME_CKPT_S3_URI}" ]]; then
    RESUME_CKPT_LOCAL="${RUN_WORK_DIR}/$(basename "${RESUME_CKPT_S3_URI}")"
    echo "[meta] resume checkpoint uri=${RESUME_CKPT_S3_URI}"
    aws s3 cp "${RESUME_CKPT_S3_URI}" "${RESUME_CKPT_LOCAL}" --only-show-errors
    RESUME_ARGS=(--resume "${RESUME_CKPT_LOCAL}")
  fi
  PYTHONUNBUFFERED=1 torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port=29500 \
    model_training/titanProject/train.py \
    --config "${CFG_LOCAL}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --s3-checkpoint-uri "${S3_PREFIX}" \
    "${RESUME_ARGS[@]}" \
    --log-every 100 \
    --data-log-every-lines 50000 \
    --min-free-gb 20 \
    --aws-bin aws

  END_ISO=$(date -Iseconds)
  echo "[meta] run_end_iso=${END_ISO}"
  echo "Run complete; instance left running for SSM to report success."
fi
EOF
} >"${REMOTE_SCRIPT}"

jq -n --rawfile script "${REMOTE_SCRIPT}" \
  --arg et "${SSM_EXEC_TIMEOUT_SECONDS}" \
  '{commands: [$script], executionTimeout: [$et]}' >"${PARAMS_JSON}"

CMD_ID=$(AWS_PROFILE="${AWS_PROFILE}" aws ssm send-command \
  --region "${REGION}" \
  --document-name "AWS-RunShellScript" \
  --comment "gpt-medium multi-GPU pretrain (~407M params, DDP) + CloudWatch + S3 logs" \
  --timeout-seconds "${SSM_DELIVERY_TIMEOUT_SECONDS}" \
  --instance-ids "${INSTANCE_ID}" \
  --cloud-watch-output-config "CloudWatchLogGroupName=${CW_LOG_GROUP},CloudWatchOutputEnabled=true" \
  --parameters "file://${PARAMS_JSON}" \
  --output-s3-bucket-name "${S3_BUCKET}" \
  --output-s3-key-prefix "${LOG_PREFIX}" \
  --query "Command.CommandId" --output text)

echo "Command ID: ${CMD_ID}"
echo "CloudWatch log group: ${CW_LOG_GROUP}"
echo "  Stream: ${CMD_ID}/${INSTANCE_ID}/aws-runShellScript/stdout (and stderr)"
echo "  Tail: AWS_PROFILE=${AWS_PROFILE} aws logs tail ${CW_LOG_GROUP} --follow --region ${REGION}"
echo "S3 SSM output (when flushed): s3://${S3_BUCKET}/${LOG_PREFIX}/${INSTANCE_ID}/awsrunShellScript/"
if [[ "${DETACH_TRAINING}" == "1" ]]; then
  echo "Detached mode: SSM only bootstraps and launches the remote multi-GPU runner."
  echo "The remote runner keeps training after SSM exits, uploads final artifacts, and can self-stop if the instance role allows it."
  echo "Check detached status: RUN_ID=<printed run id> INSTANCE_ID=${INSTANCE_ID} CMD_ID=${CMD_ID} bash scripts/aws_commands/check_detached_titan_status.sh"
else
  echo "Legacy foreground status: CMD_ID=${CMD_ID} INSTANCE_ID=${INSTANCE_ID} bash scripts/aws_commands/legacy/check_ssm_status.sh"
fi
