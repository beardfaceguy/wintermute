#!/bin/bash
cd "$(dirname "$0")" || exit 1

PYTHON_VERSION="${VLLM_PYTHON_VERSION:-3.11.9}"
PYENV_PYTHON="$HOME/.pyenv/versions/$PYTHON_VERSION/bin/python"
if [ ! -x "$PYENV_PYTHON" ]; then
  echo "❌ Python $PYTHON_VERSION not found at expected location: $PYENV_PYTHON"
  exit 1
fi

VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-$HOME/models/wizard-vicuna-awq}"
VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-wizard-vicuna-7b-awq}"
VLLM_QUANT="${VLLM_QUANT:-awq}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_memory_usage_threshold=0.95
export RAY_object_store_memory=32g
export RAY_preallocate_plasma=1
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
export RAY_disable_usage_stats=1

python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --model "$VLLM_MODEL_PATH" \
  --served-model-name "$VLLM_SERVED_NAME" \
  --quantization "$VLLM_QUANT" \
  --dtype auto \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --block-size 32 \
  --gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
  --tokenizer-pool-type ray \
  --tokenizer-pool-size 4 \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --swap-space 8 \
  --disable-log-requests \
  --enforce-eager \
  --port "$VLLM_PORT" &

VLLM_PID=$!


# Function to handle shutdown
cleanup() {
  echo "Received termination signal. Shutting down vLLM (PID $VLLM_PID)..."
  kill -SIGTERM "$VLLM_PID"
  wait "$VLLM_PID"
  echo "vLLM shutdown complete."
}

# Trap Ctrl+C (SIGINT) and termination (SIGTERM)
trap cleanup SIGINT SIGTERM

# Wait for vLLM to exit
wait "$VLLM_PID"
