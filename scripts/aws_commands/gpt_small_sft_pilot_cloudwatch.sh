#!/usr/bin/env bash
# Detached GPT-small SFT pilot launcher via SSM.
# Prepares the current instruction mix, downloads the canonical base checkpoint + tokenizer,
# writes a GPT-small SFT config derived from the pretrain config, and runs finetune_sft.py.
# Stdout/stderr go to CloudWatch (/aws/ssm/titan-llm-training) and S3 (ssm-logs/...).
# Default mode detaches the actual SFT process from SSM after bootstrap so monitoring,
# final artifact sync, and optional instance self-stop happen on the instance itself.
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
STOP_INSTANCE_ON_EXIT="${STOP_INSTANCE_ON_EXIT:-0}"
SYNC_FINAL_LOG_TO_S3="${SYNC_FINAL_LOG_TO_S3:-1}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/mnt/data/ssm_runs}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-gpt_small_sft_pilot}"
BASE_CKPT_S3_URI="${BASE_CKPT_S3_URI:-s3://alix-ai-ml-staging-data/titan/checkpoints/gpt_small_pretrain_20260411004641/ckpt_step_40000.pt}"
TOKENIZER_S3_URI="${TOKENIZER_S3_URI:-s3://alix-ai-ml-staging-data/titan/tokenizers/new_bpe_50k/bpe_50k_fw_stack.model}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-/mnt/data/data/sft_mix}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/mnt/data/cache/huggingface}"
CLEAN_STALE_ROOT_HF_CACHE="${CLEAN_STALE_ROOT_HF_CACHE:-1}"
STEPS="${STEPS:-200}"
LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_BATCHES="${EVAL_BATCHES:-20}"
SAVE_EVERY="${SAVE_EVERY:-100}"
LR="${LR:-3.0e-05}"
LR_MIN="${LR_MIN:-1.0e-05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-25}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LEN="${SEQ_LEN:-1024}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-100000}"
OASST_TRAIN_PAIRS="${OASST_TRAIN_PAIRS:-24000}"
OASST_VAL_PAIRS="${OASST_VAL_PAIRS:-2000}"
OPENHERMES_TRAIN_PAIRS="${OPENHERMES_TRAIN_PAIRS:-20000}"
OPENHERMES_VAL_PAIRS="${OPENHERMES_VAL_PAIRS:-2000}"
SLIMORCA_TRAIN_PAIRS="${SLIMORCA_TRAIN_PAIRS:-10000}"
SLIMORCA_VAL_PAIRS="${SLIMORCA_VAL_PAIRS:-1000}"
LOGIC_TRAIN_PAIRS="${LOGIC_TRAIN_PAIRS:-1000}"
LOGIC_VAL_PAIRS="${LOGIC_VAL_PAIRS:-200}"
MAX_USER_CHARS="${MAX_USER_CHARS:-512}"
MAX_ASSISTANT_CHARS="${MAX_ASSISTANT_CHARS:-512}"
MAX_DIGIT_RATIO="${MAX_DIGIT_RATIO:-0.25}"
PREP_ALLOW_HTTP="${PREP_ALLOW_HTTP:-0}"
PREP_REJECT_ROLE_MARKERS="${PREP_REJECT_ROLE_MARKERS:-0}"
OASST_BEST_ONLY="${OASST_BEST_ONLY:-0}"
OASST_MIN_QUALITY="${OASST_MIN_QUALITY:-}"
OASST_MIN_HELPFULNESS="${OASST_MIN_HELPFULNESS:-}"
OASST_MAX_FAILS_TASK="${OASST_MAX_FAILS_TASK:-}"
OASST_MAX_SPAM="${OASST_MAX_SPAM:-}"
LOG_PREFIX="ssm-logs/gpt-small-sft/$(date +%Y%m%d%H%M%S)"

if [[ "${INSTANCE_ID}" == "i-REPLACE_ME" ]]; then
  echo "Set INSTANCE_ID to your Titan training instance id." >&2
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
printf 'REGION=%q\n' "${REGION}"
printf 'DETACH_TRAINING=%q\n' "${DETACH_TRAINING}"
printf 'STOP_INSTANCE_ON_EXIT=%q\n' "${STOP_INSTANCE_ON_EXIT}"
printf 'SYNC_FINAL_LOG_TO_S3=%q\n' "${SYNC_FINAL_LOG_TO_S3}"
printf 'REMOTE_RUN_ROOT=%q\n' "${REMOTE_RUN_ROOT}"
printf 'RUN_ID_PREFIX=%q\n' "${RUN_ID_PREFIX}"
printf 'BASE_CKPT_S3_URI=%q\n' "${BASE_CKPT_S3_URI}"
printf 'TOKENIZER_S3_URI=%q\n' "${TOKENIZER_S3_URI}"
printf 'SFT_OUTPUT_DIR=%q\n' "${SFT_OUTPUT_DIR}"
printf 'HF_CACHE_DIR=%q\n' "${HF_CACHE_DIR}"
printf 'CLEAN_STALE_ROOT_HF_CACHE=%q\n' "${CLEAN_STALE_ROOT_HF_CACHE}"
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
printf 'SEQ_LEN=%q\n' "${SEQ_LEN}"
printf 'MIN_FREE_GB=%q\n' "${MIN_FREE_GB}"
printf 'SHUFFLE_BUFFER=%q\n' "${SHUFFLE_BUFFER}"
printf 'OASST_TRAIN_PAIRS=%q\n' "${OASST_TRAIN_PAIRS}"
printf 'OASST_VAL_PAIRS=%q\n' "${OASST_VAL_PAIRS}"
printf 'OPENHERMES_TRAIN_PAIRS=%q\n' "${OPENHERMES_TRAIN_PAIRS}"
printf 'OPENHERMES_VAL_PAIRS=%q\n' "${OPENHERMES_VAL_PAIRS}"
printf 'SLIMORCA_TRAIN_PAIRS=%q\n' "${SLIMORCA_TRAIN_PAIRS}"
printf 'SLIMORCA_VAL_PAIRS=%q\n' "${SLIMORCA_VAL_PAIRS}"
printf 'LOGIC_TRAIN_PAIRS=%q\n' "${LOGIC_TRAIN_PAIRS}"
printf 'LOGIC_VAL_PAIRS=%q\n' "${LOGIC_VAL_PAIRS}"
printf 'MAX_USER_CHARS=%q\n' "${MAX_USER_CHARS}"
printf 'MAX_ASSISTANT_CHARS=%q\n' "${MAX_ASSISTANT_CHARS}"
printf 'MAX_DIGIT_RATIO=%q\n' "${MAX_DIGIT_RATIO}"
printf 'PREP_ALLOW_HTTP=%q\n' "${PREP_ALLOW_HTTP}"
printf 'PREP_REJECT_ROLE_MARKERS=%q\n' "${PREP_REJECT_ROLE_MARKERS}"
printf 'OASST_BEST_ONLY=%q\n' "${OASST_BEST_ONLY}"
printf 'OASST_MIN_QUALITY=%q\n' "${OASST_MIN_QUALITY}"
printf 'OASST_MIN_HELPFULNESS=%q\n' "${OASST_MIN_HELPFULNESS}"
printf 'OASST_MAX_FAILS_TASK=%q\n' "${OASST_MAX_FAILS_TASK}"
printf 'OASST_MAX_SPAM=%q\n\n' "${OASST_MAX_SPAM}"
cat <<'EOF'
START_ISO=$(date -Iseconds)
echo "[meta] run_start_iso=${START_ISO}"

RUN_ID="${RUN_ID_PREFIX}_$(date +%Y%m%d%H%M%S)"
CHECKPOINT_DIR="/mnt/data/checkpoints/${RUN_ID}"
S3_PREFIX="s3://alix-ai-ml-staging-data/titan/checkpoints/${RUN_ID}/"
CODE_DIR="/home/ubuntu/wintermute"
CODE_BUNDLE_URI="${CODE_BUNDLE_URI:-s3://alix-ai-ml-staging-data/titan/code_bundles/titanProject_bundle.tar.gz}"
CODE_SYNC_DEST="${CODE_DIR}/model_training/titanProject"
CODE_BUNDLE_LOCAL="/tmp/titanProject_bundle.tar.gz"
TOKENIZER_DIR="/mnt/data/tokenizers"
TOKENIZER_LOCAL="${TOKENIZER_DIR}/$(basename "${TOKENIZER_S3_URI}")"
BASE_CKPT_LOCAL="${CHECKPOINT_DIR}/$(basename "${BASE_CKPT_S3_URI}")"
CFG_LOCAL="${CODE_DIR}/model_training/titanProject/configs/config_gpt_small.sft.local.yaml"
RUN_WORK_DIR="${REMOTE_RUN_ROOT}/${RUN_ID}"
TRAIN_LOG="${RUN_WORK_DIR}/train.log"
RUN_STATUS_JSON="${RUN_WORK_DIR}/run_status.json"
RUNNER_PID_FILE="${RUN_WORK_DIR}/runner.pid"
RUNNER_SCRIPT="${RUN_WORK_DIR}/run_sft.sh"
DETACHED_LAUNCH_LOG="${RUN_WORK_DIR}/launcher.log"
S3_RUN_ARTIFACT_PREFIX="${S3_PREFIX%/}/run_artifacts"
SFT_META_JSON="${SFT_OUTPUT_DIR}/meta.json"
SFT_TRAIN_TXT="${SFT_OUTPUT_DIR}/train_sft_mix.txt"
SFT_VAL_TXT="${SFT_OUTPUT_DIR}/val_sft_mix.txt"
HF_HOME="${HF_CACHE_DIR}"
HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}/hub"
HF_HUB_CACHE="${HF_CACHE_DIR}/hub"
TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"

export RUN_ID CHECKPOINT_DIR S3_PREFIX CODE_DIR CFG_LOCAL RUN_WORK_DIR TRAIN_LOG
export RUN_STATUS_JSON REGION S3_RUN_ARTIFACT_PREFIX BASE_CKPT_S3_URI TOKENIZER_S3_URI
export BASE_CKPT_LOCAL TOKENIZER_LOCAL SFT_OUTPUT_DIR SFT_META_JSON SFT_TRAIN_TXT SFT_VAL_TXT
export STEPS LOG_EVERY EVAL_EVERY EVAL_BATCHES SAVE_EVERY MIN_FREE_GB
export LR LR_MIN WEIGHT_DECAY WARMUP_STEPS BATCH_SIZE SEQ_LEN SHUFFLE_BUFFER
export STOP_INSTANCE_ON_EXIT SYNC_FINAL_LOG_TO_S3
export HF_CACHE_DIR HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE

echo "=== host ==="
date -Iseconds
uname -a || true
echo "=== disk ==="
df -h / /mnt/data 2>/dev/null || df -h || true
echo "=== memory ==="
free -h 2>/dev/null || true
echo "=== gpu ==="
nvidia-smi 2>/dev/null || echo "[warn] nvidia-smi unavailable"
echo "=== paths ==="
echo "RUN_ID=${RUN_ID}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "S3_PREFIX=${S3_PREFIX}"
echo "BASE_CKPT_S3_URI=${BASE_CKPT_S3_URI}"
echo "TOKENIZER_S3_URI=${TOKENIZER_S3_URI}"
echo "SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR}"
echo "HF_CACHE_DIR=${HF_CACHE_DIR}"
echo "STEPS=${STEPS}"
echo "LOG_EVERY=${LOG_EVERY}"
echo "EVAL_EVERY=${EVAL_EVERY}"
echo "EVAL_BATCHES=${EVAL_BATCHES}"
echo "SAVE_EVERY=${SAVE_EVERY}"
echo "LR=${LR}"
echo "SEQ_LEN=${SEQ_LEN}"
echo "BATCH_SIZE=${BATCH_SIZE}"

mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${TOKENIZER_DIR}"
mkdir -p "${SFT_OUTPUT_DIR}"
mkdir -p "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"
mkdir -p "$(dirname "${CODE_SYNC_DEST}")"
mkdir -p "${RUN_WORK_DIR}"

if [[ "${CLEAN_STALE_ROOT_HF_CACHE}" == "1" ]]; then
  echo "=== clear stale root caches ==="
  du -sh /root/.cache 2>/dev/null || true
  rm -rf /root/.cache/huggingface /root/.cache/pip
  du -sh /root/.cache 2>/dev/null || true
  df -h / /mnt/data 2>/dev/null || df -h || true
fi

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

if [[ -f .env ]]; then
  set -a
  . ./.env
  set +a
fi
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}}"
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}"
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}"

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
echo "=== pip install: model deps ==="
date -Iseconds
python3 -m pip install --no-cache-dir titans-pytorch==0.5.3 --no-deps
python3 -m pip install --no-cache-dir \
  einops==0.8.2 einx==0.4.2 hyper-connections==0.4.9 axial-positional-embedding==0.3.12 \
  assoc-scan==0.0.4 ema-pytorch==0.7.9 tqdm fire loguru orjson tensordict==0.11.0 \
  x-transformers==2.17.7 rotary-embedding-torch==0.8.9 ninja pyvers cloudpickle frozendict --no-deps
echo "=== pip install: datasets ==="
date -Iseconds
python3 -m pip install --no-cache-dir datasets huggingface_hub

echo "=== torch cuda (after install) ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available());
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

echo "=== base checkpoint + tokenizer download ==="
date -Iseconds
for i in 1 2 3; do
  aws s3 cp "${BASE_CKPT_S3_URI}" "${BASE_CKPT_LOCAL}" --only-show-errors && break
  echo "[warn] base checkpoint download failed (attempt ${i}); retrying..."
  sleep 5
done
for i in 1 2 3; do
  aws s3 cp "${TOKENIZER_S3_URI}" "${TOKENIZER_LOCAL}" --only-show-errors && break
  echo "[warn] tokenizer download failed (attempt ${i}); retrying..."
  sleep 5
done
ls -lh "${BASE_CKPT_LOCAL}" "${TOKENIZER_LOCAL}"

echo "=== prepare SFT mix ==="
date -Iseconds
PREP_ARGS=(
  --output-dir "${SFT_OUTPUT_DIR}"
  --seed 42
  --oasst-train-pairs "${OASST_TRAIN_PAIRS}"
  --oasst-val-pairs "${OASST_VAL_PAIRS}"
  --openhermes-train-pairs "${OPENHERMES_TRAIN_PAIRS}"
  --openhermes-val-pairs "${OPENHERMES_VAL_PAIRS}"
  --slimorca-train-pairs "${SLIMORCA_TRAIN_PAIRS}"
  --slimorca-val-pairs "${SLIMORCA_VAL_PAIRS}"
  --logic-train-pairs "${LOGIC_TRAIN_PAIRS}"
  --logic-val-pairs "${LOGIC_VAL_PAIRS}"
  --max-user-chars "${MAX_USER_CHARS}"
  --max-assistant-chars "${MAX_ASSISTANT_CHARS}"
  --max-digit-ratio "${MAX_DIGIT_RATIO}"
)
if [[ "${PREP_ALLOW_HTTP}" == "1" ]]; then
  PREP_ARGS+=(--allow-http)
fi
if [[ "${PREP_REJECT_ROLE_MARKERS}" == "1" ]]; then
  PREP_ARGS+=(--reject-role-markers)
fi
if [[ "${OASST_BEST_ONLY}" == "1" ]]; then
  PREP_ARGS+=(--oasst-best-only)
fi
if [[ -n "${OASST_MIN_QUALITY}" ]]; then
  PREP_ARGS+=(--oasst-min-quality "${OASST_MIN_QUALITY}")
fi
if [[ -n "${OASST_MIN_HELPFULNESS}" ]]; then
  PREP_ARGS+=(--oasst-min-helpfulness "${OASST_MIN_HELPFULNESS}")
fi
if [[ -n "${OASST_MAX_FAILS_TASK}" ]]; then
  PREP_ARGS+=(--oasst-max-fails-task "${OASST_MAX_FAILS_TASK}")
fi
if [[ -n "${OASST_MAX_SPAM}" ]]; then
  PREP_ARGS+=(--oasst-max-spam "${OASST_MAX_SPAM}")
fi
PYTHONUNBUFFERED=1 python3 model_training/titanProject/prepare_sft_mix.py "${PREP_ARGS[@]}"
python3 - <<'PY'
import json
import os
from pathlib import Path

meta_path = Path(os.environ["SFT_META_JSON"])
if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    print("[meta] sft_mix_counts=", json.dumps(meta.get("counts", {}), sort_keys=True))
    print("[meta] sft_mix_paths=", json.dumps(meta.get("paths", {}), sort_keys=True))
else:
    print(f"[warn] expected SFT meta missing: {meta_path}")
PY

echo "=== write local SFT config ==="
python3 - <<'PY'
import os
import yaml
from pathlib import Path

src = Path("model_training/titanProject/configs/config_gpt_small.yaml")
dst = Path("model_training/titanProject/configs/config_gpt_small.sft.local.yaml")
cfg = yaml.safe_load(src.read_text())

cfg["data"]["train_path"] = os.environ["SFT_TRAIN_TXT"]
cfg["data"]["val_path"] = os.environ["SFT_VAL_TXT"]
cfg["data"]["tokenizer_path"] = os.environ["TOKENIZER_LOCAL"]
cfg["data"]["shuffle_buffer"] = int(os.environ["SHUFFLE_BUFFER"])

cfg["train"]["seq_len"] = int(os.environ["SEQ_LEN"])
cfg["train"]["batch_size"] = int(os.environ["BATCH_SIZE"])
cfg["train"]["lr"] = float(os.environ["LR"])
cfg["train"]["lr_min"] = float(os.environ["LR_MIN"])
cfg["train"]["weight_decay"] = float(os.environ["WEIGHT_DECAY"])
cfg["train"]["warmup_steps"] = int(os.environ["WARMUP_STEPS"])
cfg["train"]["max_steps"] = int(os.environ["STEPS"])
cfg["train"]["save_every"] = int(os.environ["SAVE_EVERY"])
cfg["train"]["eval_every"] = int(os.environ["EVAL_EVERY"])
cfg["train"]["cosine_decay"] = False
cfg["train"].pop("grad_accum_steps", None)
cfg["train"].pop("target_tokens", None)
cfg["data"].pop("max_tokens", None)
cfg["data"].pop("max_tokens_val", None)

dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"[meta] wrote local config: {dst}")
print(
    "[meta] local SFT config:"
    f" seq_len={cfg['train']['seq_len']}"
    f" batch={cfg['train']['batch_size']}"
    f" lr={cfg['train']['lr']}"
    f" train_path={cfg['data']['train_path']}"
    f" val_path={cfg['data']['val_path']}"
)
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
  aws s3 sync "${CHECKPOINT_DIR}" "${S3_PREFIX}" --exclude "*" --include "ckpt_sft_step_*.pt" --only-show-errors || \
    echo "[warn] final checkpoint sync failed"
  if [[ "${SYNC_FINAL_LOG_TO_S3}" == "1" && -f "${TRAIN_LOG}" ]]; then
    aws s3 cp "${TRAIN_LOG}" "${S3_RUN_ARTIFACT_PREFIX}/train.log" --only-show-errors || \
      echo "[warn] final train.log upload failed"
  fi
  if [[ -f "${RUN_STATUS_JSON}" ]]; then
    aws s3 cp "${RUN_STATUS_JSON}" "${S3_RUN_ARTIFACT_PREFIX}/run_status.json" --only-show-errors || \
      echo "[warn] final run_status upload failed"
  fi
  if [[ -f "${CFG_LOCAL}" ]]; then
    aws s3 cp "${CFG_LOCAL}" "${S3_RUN_ARTIFACT_PREFIX}/config_gpt_small.sft.local.yaml" --only-show-errors || \
      echo "[warn] final config upload failed"
  fi
  if [[ -f "${SFT_META_JSON}" ]]; then
    aws s3 cp "${SFT_META_JSON}" "${S3_RUN_ARTIFACT_PREFIX}/sft_meta.json" --only-show-errors || \
      echo "[warn] final sft meta upload failed"
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
    "config_path": os.environ["CFG_LOCAL"],
    "sft_meta_path": os.environ["SFT_META_JSON"],
    "base_ckpt_s3_uri": os.environ["BASE_CKPT_S3_URI"],
    "tokenizer_s3_uri": os.environ["TOKENIZER_S3_URI"],
    "steps": int(os.environ["STEPS"]),
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
      aws ec2 stop-instances --instance-ids "${instance_id}" --region "${REGION}" --output json >/dev/null || \
        echo "[warn] instance self-stop failed; attach the self-stop IAM policy if this should be automatic"
    else
      echo "[warn] could not determine instance id for self-stop"
    fi
  fi
  trap - EXIT
  exit "${exit_code}"
}

trap finalize EXIT

cd "${CODE_DIR}"
echo "[meta] detached runner start $(date -Iseconds)" | tee -a "${TRAIN_LOG}"
PYTHONUNBUFFERED=1 python3 model_training/titanProject/finetune_sft.py \
  --config "${CFG_LOCAL}" \
  --ckpt "${BASE_CKPT_LOCAL}" \
  --device cuda \
  --steps "${STEPS}" \
  --log-every "${LOG_EVERY}" \
  --eval-every "${EVAL_EVERY}" \
  --eval-batches "${EVAL_BATCHES}" \
  --save-every "${SAVE_EVERY}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --s3-checkpoint-uri "${S3_PREFIX}" \
  --min-free-gb "${MIN_FREE_GB}" \
  --aws-bin aws 2>&1 | tee -a "${TRAIN_LOG}"
RUNNER_EOF

  chmod +x "${RUNNER_SCRIPT}"
  nohup bash "${RUNNER_SCRIPT}" >"${DETACHED_LAUNCH_LOG}" 2>&1 </dev/null &
  RUNNER_PID=$!
  echo "${RUNNER_PID}" >"${RUNNER_PID_FILE}"
  sleep 5
  if ! kill -0 "${RUNNER_PID}" 2>/dev/null; then
    echo "[error] detached runner exited immediately"
    test -f "${DETACHED_LAUNCH_LOG}" && sed -n '1,200p' "${DETACHED_LAUNCH_LOG}"
    exit 1
  fi
  echo "[meta] detached runner pid=${RUNNER_PID}"
  echo "[meta] detached train log=${TRAIN_LOG}"
  echo "[meta] detached launcher log=${DETACHED_LAUNCH_LOG}"
  echo "[meta] detached status json=${RUN_STATUS_JSON}"
  echo "[meta] final artifacts prefix=${S3_RUN_ARTIFACT_PREFIX}"
  echo "[meta] stop-on-exit=${STOP_INSTANCE_ON_EXIT} (requires ec2:StopInstances on the instance role)"
  echo "[meta] bootstrap complete; SFT now continues independently of this SSM command"
else
  PYTHONUNBUFFERED=1 python3 model_training/titanProject/finetune_sft.py \
    --config "${CFG_LOCAL}" \
    --ckpt "${BASE_CKPT_LOCAL}" \
    --device cuda \
    --steps "${STEPS}" \
    --log-every "${LOG_EVERY}" \
    --eval-every "${EVAL_EVERY}" \
    --eval-batches "${EVAL_BATCHES}" \
    --save-every "${SAVE_EVERY}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --s3-checkpoint-uri "${S3_PREFIX}" \
    --min-free-gb "${MIN_FREE_GB}" \
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
  --comment "gpt-small SFT pilot + CloudWatch + S3 logs" \
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
  echo "Detached mode: SSM only bootstraps and launches the remote SFT runner."
  echo "Check detached status: RUN_ID=<printed run id> INSTANCE_ID=${INSTANCE_ID} CMD_ID=${CMD_ID} bash scripts/aws_commands/check_detached_titan_status.sh"
else
  echo "Legacy foreground status: CMD_ID=${CMD_ID} INSTANCE_ID=${INSTANCE_ID} bash scripts/aws_commands/legacy/check_ssm_status.sh"
fi
