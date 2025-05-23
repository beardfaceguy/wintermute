#!/bin/bash
cd "$(dirname "$0")" || exit 1

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
RAY_memory_usage_threshold=0.9 RAY_object_store_memory=1g \
python -m vllm.entrypoints.openai.api_server \
  --model "$HOME/models/mistral-7b-instruct-awq" \
  --served-model-name mistral-7b-instruct-awq \
  --quantization awq \
  --dtype auto \
  --max-model-len 2048 \
  --block-size 32 \
  --gpu-memory-utilization 0.88 \
  --tokenizer-pool-type ray \
  --tokenizer-pool-size 1 \
  --max-num-seqs 4 \
  --swap-space 8 \
  --disable-log-requests \
  --enforce-eager \
  --port 8001

