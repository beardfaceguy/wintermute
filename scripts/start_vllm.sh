#!/bin/bash
cd "$(dirname "$0")" || exit 1

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_memory_usage_threshold=0.95
export RAY_object_store_memory=32g
export RAY_preallocate_plasma=1
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
export RAY_disable_usage_stats=1

# Start vLLM in background
python -m vllm.entrypoints.openai.api_server \
  --model "$HOME/models/Nous-Hermes-2-Mistral-7B-DPO-AWQ" \
  --served-model-name mistral-7b-instruct-awq \
  --quantization awq \
  --dtype auto \
  --max-model-len 2048 \
  --block-size 32 \
  --gpu-memory-utilization 0.90 \
  --tokenizer-pool-type ray \
  --tokenizer-pool-size 4 \
  --max-num-seqs 4 \
  --swap-space 8 \
  --disable-log-requests \
  --enforce-eager \
  --port 8001 &

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
