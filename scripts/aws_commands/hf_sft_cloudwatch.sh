#!/usr/bin/env bash
# Detached HuggingFace model SFT launcher via SSM.
# Downloads an HF model (Llama 3, Mistral, Qwen, etc.) and fine-tunes it
# with QLoRA (default) or full fine-tuning on an EC2 GPU instance.
# Set INSTANCE_ID and HF_MODEL_ID before running.

set -euo pipefail

INSTANCE_ID="${INSTANCE_ID:-i-REPLACE_ME}"
REGION="${REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-experimental-admin}"
SSM_EXEC_TIMEOUT_SECONDS="${SSM_EXEC_TIMEOUT_SECONDS:-43200}"
SSM_DELIVERY_TIMEOUT_SECONDS="${SSM_DELIVERY_TIMEOUT_SECONDS:-43200}"
CW_LOG_GROUP="${CW_LOG_GROUP:-/aws/ssm/titan-llm-training}"
S3_BUCKET="${S3_BUCKET:-alix-ai-ml-staging-data}"
STOP_INSTANCE_ON_EXIT="${STOP_INSTANCE_ON_EXIT:-1}"
SYNC_FINAL_LOG_TO_S3="${SYNC_FINAL_LOG_TO_S3:-1}"

# --- HF model config ---
HF_MODEL_ID="${HF_MODEL_ID:-REPLACE_WITH_HF_MODEL_ID}"
FINETUNE_MODE="${FINETUNE_MODE:-qlora}"   # qlora | lora | full
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# --- SFT data ---
# Set to "GENERATE" to run prepare_sft_mix.py on the instance (default).
# Set to an S3 URI or local path if data is already prepared.
SFT_TRAIN_PATH="${SFT_TRAIN_PATH:-GENERATE}"
SFT_VAL_PATH="${SFT_VAL_PATH:-GENERATE}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"

# --- Data generation params (only used when SFT_TRAIN_PATH=GENERATE) ---
OASST_TRAIN_PAIRS="${OASST_TRAIN_PAIRS:-24000}"
OASST_VAL_PAIRS="${OASST_VAL_PAIRS:-2000}"
OPENHERMES_TRAIN_PAIRS="${OPENHERMES_TRAIN_PAIRS:-20000}"
OPENHERMES_VAL_PAIRS="${OPENHERMES_VAL_PAIRS:-2000}"
SLIMORCA_TRAIN_PAIRS="${SLIMORCA_TRAIN_PAIRS:-10000}"
SLIMORCA_VAL_PAIRS="${SLIMORCA_VAL_PAIRS:-1000}"
LOGIC_TRAIN_PAIRS="${LOGIC_TRAIN_PAIRS:-1000}"
LOGIC_VAL_PAIRS="${LOGIC_VAL_PAIRS:-200}"

# --- Training hyperparams ---
RUN_ID_PREFIX="${RUN_ID_PREFIX:-hf_sft}"
STEPS="${STEPS:-3000}"
LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-250}"
EVAL_BATCHES="${EVAL_BATCHES:-20}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LR="${LR:-2.0e-04}"
LR_MIN="${LR_MIN:-2.0e-05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
SEQ_LEN="${SEQ_LEN:-2048}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"

# --- Multi-GPU (only for full fine-tuning) ---
NUM_GPUS="${NUM_GPUS:-1}"

LOG_PREFIX="ssm-logs/hf-sft/$(date +%Y%m%d%H%M%S)"

if [[ "${INSTANCE_ID}" == "i-REPLACE_ME" ]]; then
  echo "Set INSTANCE_ID to your GPU instance id." >&2
  exit 1
fi

if [[ "${HF_MODEL_ID}" == "REPLACE_WITH_HF_MODEL_ID" ]]; then
  echo "Set HF_MODEL_ID (e.g. meta-llama/Meta-Llama-3-8B)" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required (brew install jq)." >&2
  exit 1
fi

REMOTE_SCRIPT="$(mktemp)"
PARAMS_JSON="$(mktemp)"
trap 'rm -f "${REMOTE_SCRIPT}" "${PARAMS_JSON}"' EXIT

{
printf '#!/bin/bash\n'
printf 'set -euo pipefail\n\n'
printf 'REGION=%q\n' "${REGION}"
printf 'STOP_INSTANCE_ON_EXIT=%q\n' "${STOP_INSTANCE_ON_EXIT}"
printf 'SYNC_FINAL_LOG_TO_S3=%q\n' "${SYNC_FINAL_LOG_TO_S3}"
printf 'RUN_ID_PREFIX=%q\n' "${RUN_ID_PREFIX}"
printf 'HF_MODEL_ID=%q\n' "${HF_MODEL_ID}"
printf 'FINETUNE_MODE=%q\n' "${FINETUNE_MODE}"
printf 'LORA_RANK=%q\n' "${LORA_RANK}"
printf 'LORA_ALPHA=%q\n' "${LORA_ALPHA}"
printf 'SFT_TRAIN_PATH=%q\n' "${SFT_TRAIN_PATH}"
printf 'SFT_VAL_PATH=%q\n' "${SFT_VAL_PATH}"
printf 'USE_CHAT_TEMPLATE=%q\n' "${USE_CHAT_TEMPLATE}"
printf 'OASST_TRAIN_PAIRS=%q\n' "${OASST_TRAIN_PAIRS}"
printf 'OASST_VAL_PAIRS=%q\n' "${OASST_VAL_PAIRS}"
printf 'OPENHERMES_TRAIN_PAIRS=%q\n' "${OPENHERMES_TRAIN_PAIRS}"
printf 'OPENHERMES_VAL_PAIRS=%q\n' "${OPENHERMES_VAL_PAIRS}"
printf 'SLIMORCA_TRAIN_PAIRS=%q\n' "${SLIMORCA_TRAIN_PAIRS}"
printf 'SLIMORCA_VAL_PAIRS=%q\n' "${SLIMORCA_VAL_PAIRS}"
printf 'LOGIC_TRAIN_PAIRS=%q\n' "${LOGIC_TRAIN_PAIRS}"
printf 'LOGIC_VAL_PAIRS=%q\n' "${LOGIC_VAL_PAIRS}"
printf 'STEPS=%q\n' "${STEPS}"
printf 'LOG_EVERY=%q\n' "${LOG_EVERY}"
printf 'EVAL_EVERY=%q\n' "${EVAL_EVERY}"
printf 'EVAL_BATCHES=%q\n' "${EVAL_BATCHES}"
printf 'SAVE_EVERY=%q\n' "${SAVE_EVERY}"
printf 'LR=%q\n' "${LR}"
printf 'LR_MIN=%q\n' "${LR_MIN}"
printf 'WEIGHT_DECAY=%q\n' "${WEIGHT_DECAY}"
printf 'WARMUP_STEPS=%q\n' "${WARMUP_STEPS}"
printf 'BATCH_SIZE=%q\n' "${BATCH_SIZE}"
printf 'GRAD_ACCUM_STEPS=%q\n' "${GRAD_ACCUM_STEPS}"
printf 'SEQ_LEN=%q\n' "${SEQ_LEN}"
printf 'MIN_FREE_GB=%q\n' "${MIN_FREE_GB}"
printf 'NUM_GPUS=%q\n\n' "${NUM_GPUS}"
cat <<'EOF'
echo "[meta] run_start_iso=$(date -Iseconds)"

# ── Data root detection ──────────────────────────────────────────────
resolve_data_root() {
  if mountpoint -q /opt/dlami/nvme 2>/dev/null; then echo "/opt/dlami/nvme"; return; fi
  if mountpoint -q /mnt/data 2>/dev/null; then echo "/mnt/data"; return; fi
  if command -v lvm >/dev/null 2>&1; then
    vgchange -ay 2>/dev/null || true
    local lv_path
    lv_path=$(lvs --noheadings -o lv_path 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "${lv_path}" ]]; then
      mkdir -p /opt/dlami/nvme
      mount "${lv_path}" /opt/dlami/nvme 2>/dev/null || true
      if mountpoint -q /opt/dlami/nvme 2>/dev/null; then echo "/opt/dlami/nvme"; return; fi
    fi
  fi
  mkdir -p /mnt/data; echo "/mnt/data"
}

DATA_ROOT="$(resolve_data_root)"
export DATA_ROOT
echo "[data-root] DATA_ROOT=${DATA_ROOT}"

RUN_ID="${RUN_ID_PREFIX}_$(date +%Y%m%d%H%M%S)"
CHECKPOINT_DIR="${DATA_ROOT}/checkpoints/${RUN_ID}"
S3_PREFIX="s3://alix-ai-ml-staging-data/titan/checkpoints/${RUN_ID}/"
CODE_DIR="/home/ubuntu/wintermute"
CODE_BUNDLE_URI="${CODE_BUNDLE_URI:-s3://alix-ai-ml-staging-data/titan/code_bundles/titanProject_bundle.tar.gz}"
CODE_SYNC_DEST="${CODE_DIR}/model_training/titanProject"
CODE_BUNDLE_LOCAL="/tmp/titanProject_bundle.tar.gz"
HF_CACHE_DIR="${DATA_ROOT}/cache/huggingface"
REMOTE_RUN_ROOT="${DATA_ROOT}/ssm_runs"
RUN_WORK_DIR="${REMOTE_RUN_ROOT}/${RUN_ID}"
TRAIN_LOG="${RUN_WORK_DIR}/train.log"
RUN_STATUS_JSON="${RUN_WORK_DIR}/run_status.json"
RUNNER_SCRIPT="${RUN_WORK_DIR}/run_hf_sft.sh"
DETACHED_LAUNCH_LOG="${RUN_WORK_DIR}/launcher.log"
S3_RUN_ARTIFACT_PREFIX="${S3_PREFIX%/}/run_artifacts"
CFG_LOCAL="${CODE_DIR}/model_training/titanProject/configs/config_hf_sft.local.yaml"

export HF_HOME="${HF_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}/hub"
export HF_HUB_CACHE="${HF_CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"
TMPDIR="${HF_CACHE_DIR}/tmp"
export TMPDIR
mkdir -p "${TMPDIR}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"
mkdir -p "${CODE_DIR}" "${CHECKPOINT_DIR}" "${RUN_WORK_DIR}"

echo "=== host ==="
uname -a || true
echo "=== gpu ==="
nvidia-smi 2>/dev/null || echo "[warn] nvidia-smi unavailable"
echo "=== paths ==="
echo "HF_MODEL_ID=${HF_MODEL_ID}"
echo "FINETUNE_MODE=${FINETUNE_MODE}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "S3_PREFIX=${S3_PREFIX}"
echo "STEPS=${STEPS} LR=${LR} SEQ_LEN=${SEQ_LEN} BATCH_SIZE=${BATCH_SIZE}"

echo "=== code bundle ==="
rm -rf "${CODE_SYNC_DEST}"
aws s3 cp "${CODE_BUNDLE_URI}" "${CODE_BUNDLE_LOCAL}" --only-show-errors
tar -xzf "${CODE_BUNDLE_LOCAL}" -C "${CODE_DIR}"
rm -f "${CODE_BUNDLE_LOCAL}"

cd "${CODE_DIR}"

if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}}"
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}"

echo "=== pip install ==="
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
python3 -m pip install -q --no-cache-dir \
  "transformers>=4.40" "peft>=0.11" "bitsandbytes>=0.43" "accelerate>=0.30" \
  "jinja2>=3.1.0" datasets huggingface_hub
echo "=== pip install done ==="

python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ── SFT data preparation ─────────────────────────────────────────────
SFT_OUTPUT_DIR="${DATA_ROOT}/data/hf_sft_mix"
mkdir -p "${SFT_OUTPUT_DIR}"

if [[ "${SFT_TRAIN_PATH}" == "GENERATE" ]]; then
  echo "=== prepare SFT data (generate on instance) ==="
  date -Iseconds
  PYTHONUNBUFFERED=1 python3 model_training/titanProject/prepare_sft_mix.py \
    --output-dir "${SFT_OUTPUT_DIR}" \
    --output-format chat_text \
    --seed 42 \
    --oasst-train-pairs "${OASST_TRAIN_PAIRS}" \
    --oasst-val-pairs "${OASST_VAL_PAIRS}" \
    --openhermes-train-pairs "${OPENHERMES_TRAIN_PAIRS}" \
    --openhermes-val-pairs "${OPENHERMES_VAL_PAIRS}" \
    --slimorca-train-pairs "${SLIMORCA_TRAIN_PAIRS}" \
    --slimorca-val-pairs "${SLIMORCA_VAL_PAIRS}" \
    --logic-train-pairs "${LOGIC_TRAIN_PAIRS}" \
    --logic-val-pairs "${LOGIC_VAL_PAIRS}"
  SFT_TRAIN_PATH="${SFT_OUTPUT_DIR}/train_sft_mix.txt"
  SFT_VAL_PATH="${SFT_OUTPUT_DIR}/val_sft_mix.txt"
  echo "=== SFT data ready ==="
  wc -l "${SFT_TRAIN_PATH}" "${SFT_VAL_PATH}"
elif [[ "${SFT_TRAIN_PATH}" == s3://* ]]; then
  echo "=== download SFT data from S3 ==="
  SFT_TRAIN_LOCAL="${SFT_OUTPUT_DIR}/$(basename "${SFT_TRAIN_PATH}")"
  SFT_VAL_LOCAL="${SFT_OUTPUT_DIR}/$(basename "${SFT_VAL_PATH}")"
  aws s3 cp "${SFT_TRAIN_PATH}" "${SFT_TRAIN_LOCAL}" --only-show-errors
  aws s3 cp "${SFT_VAL_PATH}" "${SFT_VAL_LOCAL}" --only-show-errors
  SFT_TRAIN_PATH="${SFT_TRAIN_LOCAL}"
  SFT_VAL_PATH="${SFT_VAL_LOCAL}"
  wc -l "${SFT_TRAIN_PATH}" "${SFT_VAL_PATH}"
fi
export SFT_TRAIN_PATH SFT_VAL_PATH
export FINETUNE_MODE LORA_RANK LORA_ALPHA USE_CHAT_TEMPLATE
export SEQ_LEN BATCH_SIZE GRAD_ACCUM_STEPS LR LR_MIN WEIGHT_DECAY WARMUP_STEPS STEPS
export SAVE_EVERY EVAL_EVERY
export CFG_LOCAL
echo "SFT_TRAIN_PATH=${SFT_TRAIN_PATH}"
echo "SFT_VAL_PATH=${SFT_VAL_PATH}"

echo "=== write HF SFT config ==="
python3 - <<'PY'
import os, yaml
from pathlib import Path

cfg = {
    "model": {
        "variant": "gpt",
        "vocab_size": 50000,
        "dim": 1024,
        "depth": 24,
        "heads": 16,
        "ff_mult": 4,
        "max_seq_len": 2048,
    },
    "lora": {
        "enabled": os.environ["FINETUNE_MODE"] != "full",
        "rank": int(os.environ["LORA_RANK"]),
        "alpha": int(os.environ["LORA_ALPHA"]),
        "dropout": 0.05,
    },
    "quantization": {
        "load_in_4bit": os.environ["FINETUNE_MODE"] == "qlora",
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
    },
    "data": {
        "train_path": os.environ["SFT_TRAIN_PATH"],
        "val_path": os.environ["SFT_VAL_PATH"],
        "tokenizer_path": "unused",
        "use_chat_template": os.environ["USE_CHAT_TEMPLATE"] == "1",
    },
    "train": {
        "seq_len": int(os.environ["SEQ_LEN"]),
        "batch_size": int(os.environ["BATCH_SIZE"]),
        "grad_accum_steps": int(os.environ["GRAD_ACCUM_STEPS"]),
        "lr": float(os.environ["LR"]),
        "lr_min": float(os.environ["LR_MIN"]),
        "weight_decay": float(os.environ["WEIGHT_DECAY"]),
        "warmup_steps": int(os.environ["WARMUP_STEPS"]),
        "max_steps": int(os.environ["STEPS"]),
        "grad_clip": 1.0,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "cosine_decay": True,
        "save_every": int(os.environ["SAVE_EVERY"]),
        "eval_every": int(os.environ["EVAL_EVERY"]),
    },
}
dst = Path(os.environ["CFG_LOCAL"])
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"[meta] config written to {dst}")
PY

# Build finetune args
FINETUNE_ARGS=(
  --config "${CFG_LOCAL}"
  --hf-model "${HF_MODEL_ID}"
  --steps "${STEPS}"
  --log-every "${LOG_EVERY}"
  --eval-every "${EVAL_EVERY}"
  --eval-batches "${EVAL_BATCHES}"
  --save-every "${SAVE_EVERY}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --s3-checkpoint-uri "${S3_PREFIX}"
  --min-free-gb "${MIN_FREE_GB}"
  --aws-bin aws
  --lr "${LR}"
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
)

case "${FINETUNE_MODE}" in
  qlora) FINETUNE_ARGS+=(--qlora) ;;
  lora)  FINETUNE_ARGS+=(--lora) ;;
  full)  FINETUNE_ARGS+=(--no-lora) ;;
  *)     echo "[error] unknown FINETUNE_MODE=${FINETUNE_MODE}" >&2; exit 1 ;;
esac

if [[ "${USE_CHAT_TEMPLATE}" == "1" ]]; then
  FINETUNE_ARGS+=(--chat-template)
fi

cat >"${RUNNER_SCRIPT}" <<RUNNER_EOF
#!/bin/bash
set -euo pipefail

get_instance_id() {
  local token
  token="\$(curl -fsS -X PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' 2>/dev/null || true)"
  if [[ -n "\${token}" ]]; then
    curl -fsS -H "X-aws-ec2-metadata-token: \${token}" \
      http://169.254.169.254/latest/meta-data/instance-id
  else
    curl -fsS http://169.254.169.254/latest/meta-data/instance-id
  fi
}

finalize() {
  local exit_code=\$?
  echo "[meta] run ended at \$(date -Iseconds) exit_code=\${exit_code}"
  aws s3 sync "${CHECKPOINT_DIR}" "${S3_PREFIX}" --exclude "*" --include "step_*" --only-show-errors || true
  if [[ "${SYNC_FINAL_LOG_TO_S3}" == "1" && -f "${TRAIN_LOG}" ]]; then
    aws s3 cp "${TRAIN_LOG}" "${S3_RUN_ARTIFACT_PREFIX}/train.log" --only-show-errors || true
  fi
  if [[ "${STOP_INSTANCE_ON_EXIT}" == "1" ]]; then
    local iid=""
    iid="\$(get_instance_id 2>/dev/null || true)"
    if [[ -n "\${iid}" ]]; then
      echo "[self-stop] stopping \${iid}"
      aws ec2 create-tags --resources "\${iid}" --region "${REGION}" \
        --tags Key=Purpose,Value=titan-training 2>/dev/null || true
      aws ec2 stop-instances --instance-ids "\${iid}" --region "${REGION}" --output json 2>&1 || \
        echo "[self-stop] FAILED — instance will keep running!"
    fi
  fi
  trap - EXIT
  exit "\${exit_code}"
}
trap finalize EXIT

cd "${CODE_DIR}"
echo "[meta] HF SFT start \$(date -Iseconds)" | tee -a "${TRAIN_LOG}"

LAUNCHER="python3"
if [[ ${NUM_GPUS} -gt 1 ]]; then
  LAUNCHER="torchrun --nproc_per_node=${NUM_GPUS}"
fi

PYTHONUNBUFFERED=1 \${LAUNCHER} model_training/titanProject/finetune_sft.py ${FINETUNE_ARGS[@]} 2>&1 | tee -a "${TRAIN_LOG}"
RUNNER_EOF

chmod +x "${RUNNER_SCRIPT}"
nohup bash "${RUNNER_SCRIPT}" >"${DETACHED_LAUNCH_LOG}" 2>&1 </dev/null &
RUNNER_PID=$!
sleep 5
if ! kill -0 "${RUNNER_PID}" 2>/dev/null; then
  echo "[error] runner exited immediately"
  test -f "${DETACHED_LAUNCH_LOG}" && head -200 "${DETACHED_LAUNCH_LOG}"
  exit 1
fi
echo "[meta] detached runner pid=${RUNNER_PID}"
echo "[meta] train log=${TRAIN_LOG}"
echo "[meta] HF model=${HF_MODEL_ID} mode=${FINETUNE_MODE}"
echo "[meta] bootstrap complete; HF SFT running independently"
EOF
} >"${REMOTE_SCRIPT}"

jq -n --rawfile script "${REMOTE_SCRIPT}" \
  --arg et "${SSM_EXEC_TIMEOUT_SECONDS}" \
  '{commands: [$script], executionTimeout: [$et]}' >"${PARAMS_JSON}"

CMD_ID=$(AWS_PROFILE="${AWS_PROFILE}" aws ssm send-command \
  --region "${REGION}" \
  --document-name "AWS-RunShellScript" \
  --comment "HF model SFT (${HF_MODEL_ID}) ${FINETUNE_MODE} + CloudWatch" \
  --timeout-seconds "${SSM_DELIVERY_TIMEOUT_SECONDS}" \
  --instance-ids "${INSTANCE_ID}" \
  --cloud-watch-output-config "CloudWatchLogGroupName=${CW_LOG_GROUP},CloudWatchOutputEnabled=true" \
  --parameters "file://${PARAMS_JSON}" \
  --output-s3-bucket-name "${S3_BUCKET}" \
  --output-s3-key-prefix "${LOG_PREFIX}" \
  --query "Command.CommandId" --output text)

echo "Command ID: ${CMD_ID}"
echo "CloudWatch: ${CW_LOG_GROUP}"
echo "  Tail: AWS_PROFILE=${AWS_PROFILE} aws logs tail ${CW_LOG_GROUP} --follow --region ${REGION}"
echo "Model: ${HF_MODEL_ID} | Mode: ${FINETUNE_MODE} | GPUs: ${NUM_GPUS}"
