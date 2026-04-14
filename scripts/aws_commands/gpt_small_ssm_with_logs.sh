#!/usr/bin/env bash
# GPT-small SSM run with CloudWatch + S3 logging for stdout/stderr.
# Set INSTANCE_ID. Log group: /aws/ssm/titan-llm-training (IAM: scripts/aws_commands/iam/).

set -euo pipefail

INSTANCE_ID="${INSTANCE_ID:-i-REPLACE_ME}"
REGION="${REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-experimental-admin}"
SSM_EXEC_TIMEOUT_SECONDS="${SSM_EXEC_TIMEOUT_SECONDS:-43200}"
SSM_DELIVERY_TIMEOUT_SECONDS="${SSM_DELIVERY_TIMEOUT_SECONDS:-43200}"
CW_LOG_GROUP="${CW_LOG_GROUP:-/aws/ssm/titan-llm-training}"
S3_BUCKET="${S3_BUCKET:-alix-ai-ml-staging-data}"
LOG_PREFIX="ssm-logs/gpt-small/$(date +%Y%m%d%H%M%S)"

if [[ "${INSTANCE_ID}" == "i-REPLACE_ME" ]]; then
  echo "Set INSTANCE_ID to your instance id." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to build SSM --parameters JSON (brew install jq)." >&2
  exit 1
fi

REMOTE_SCRIPT="$(mktemp)"
PARAMS_JSON="$(mktemp)"
trap 'rm -f "${REMOTE_SCRIPT}" "${PARAMS_JSON}"' EXIT

cat >"${REMOTE_SCRIPT}" <<'EOF'
#!/bin/bash
set -euo pipefail

echo "=== host / disk / gpu ==="
date -Iseconds
uname -a || true
df -h / /mnt/data 2>/dev/null || df -h || true
nvidia-smi 2>/dev/null || true

RUN_ID="gpt_small_run_$(date +%Y%m%d%H%M%S)"
CHECKPOINT_DIR="/mnt/data/checkpoints/${RUN_ID}"
S3_PREFIX="s3://alix-ai-ml-staging-data/titan/checkpoints/${RUN_ID}/"
CODE_DIR="/home/ubuntu/wintermute"
CODE_BUNDLE_URI="${CODE_BUNDLE_URI:-s3://alix-ai-ml-staging-data/titan/code_bundles/titanProject_bundle.tar.gz}"
CODE_SYNC_DEST="${CODE_DIR}/model_training/titanProject"
CODE_BUNDLE_LOCAL="/tmp/titanProject_bundle.tar.gz"
DATA_DIR="/mnt/data/datasets"
TRAIN_LOCAL="${DATA_DIR}/train.txt"
VAL_LOCAL="${DATA_DIR}/val.txt"
CFG_LOCAL="${CODE_DIR}/model_training/titanProject/configs/config_gpt_small.local.yaml"
MAX_STEPS=20000  # adjust if needed

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$(dirname "$CODE_SYNC_DEST")"

echo "=== code bundle download start ==="
date -Iseconds
echo "CODE_BUNDLE_URI=${CODE_BUNDLE_URI}"
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

cd "$CODE_DIR"

echo "=== pip install: numpy ==="
date -Iseconds
python3 -m pip install --no-cache-dir numpy==1.26.4
echo "=== pip install: torch stack ==="
date -Iseconds
python3 -m pip install --upgrade --no-cache-dir \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
echo "=== pip install: tokenizer/io deps ==="
date -Iseconds
python3 -m pip install --no-cache-dir sentencepiece boto3 pyarrow
echo "=== pip install: titans-pytorch ==="
date -Iseconds
python3 -m pip install --no-cache-dir titans-pytorch==0.5.3 --no-deps
echo "=== pip install: remaining deps ==="
date -Iseconds
python3 -m pip install --no-cache-dir \
  einops==0.8.2 einx==0.4.2 hyper-connections==0.4.9 axial-positional-embedding==0.3.12 \
  assoc-scan==0.0.4 ema-pytorch==0.7.9 tqdm fire loguru orjson tensordict==0.11.0 \
  x-transformers==2.17.7 rotary-embedding-torch==0.8.9 ninja pyvers cloudpickle frozendict --no-deps

echo "=== torch cuda (after install) ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available());
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

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

python3 - <<'PY'
import yaml
from pathlib import Path

src = Path("model_training/titanProject/configs/config_gpt_small.yaml")
dst = Path("model_training/titanProject/configs/config_gpt_small.local.yaml")

cfg = yaml.safe_load(src.read_text())
cfg["data"]["train_path"] = "/mnt/data/datasets/train.txt"
cfg["data"]["val_path"] = "/mnt/data/datasets/val.txt"
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"[meta] wrote local config: {dst}")
PY

PYTHONUNBUFFERED=1 python3 model_training/titanProject/train.py \
  --config "${CFG_LOCAL}" \
  --device cuda \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --s3-checkpoint-uri "${S3_PREFIX}" \
  --save-every 1000 \
  --log-every 100 \
  --max-steps "${MAX_STEPS}" \
  --data-log-every-lines 50000 \
  --min-free-gb 20 \
  --aws-bin aws

echo "Run complete; instance left running for SSM to report success."
EOF

jq -n --rawfile script "${REMOTE_SCRIPT}" \
  --arg et "${SSM_EXEC_TIMEOUT_SECONDS}" \
  '{commands: [$script], executionTimeout: [$et]}' >"${PARAMS_JSON}"

CMD_ID=$(AWS_PROFILE="${AWS_PROFILE}" aws ssm send-command \
  --region "${REGION}" \
  --document-name "AWS-RunShellScript" \
  --comment "gpt-small long run with CloudWatch + S3 logs" \
  --timeout-seconds "${SSM_DELIVERY_TIMEOUT_SECONDS}" \
  --instance-ids "${INSTANCE_ID}" \
  --cloud-watch-output-config "CloudWatchLogGroupName=${CW_LOG_GROUP},CloudWatchOutputEnabled=true" \
  --parameters "file://${PARAMS_JSON}" \
  --output-s3-bucket-name "${S3_BUCKET}" \
  --output-s3-key-prefix "${LOG_PREFIX}" \
  --query "Command.CommandId" --output text)

echo "Command ID: ${CMD_ID}"
echo "CloudWatch: ${CW_LOG_GROUP} — stream ${CMD_ID}/${INSTANCE_ID}/aws-runShellScript/stdout"
echo "Tail: AWS_PROFILE=${AWS_PROFILE} aws logs tail ${CW_LOG_GROUP} --follow --region ${REGION}"
echo "S3: s3://${S3_BUCKET}/${LOG_PREFIX}/${INSTANCE_ID}/awsrunShellScript/"
echo "Use get-command-invocation; stop/terminate only after SSM reports Success."
