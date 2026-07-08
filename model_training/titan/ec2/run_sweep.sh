#!/bin/bash
# VRAM / sequence-length probe (Vikunja #952): find the seq ceiling of the proven
# small MAC (d256/L2/1-mem, sliding-window attn) on the A10G with the current
# no-accelerated-scan neural memory. Tells us whether long-context TEXT recall can
# run at a useful seq length as-is, or needs the accelerated-scan CUDA kernel /
# gradient checkpointing first. Each cell is a FRESH process so an OOM can't poison
# the next measurement.
set -uo pipefail
PY="${PY:?}"; SCRIPTS="${SCRIPTS:?}"
cd "$SCRIPTS" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==================== VRAM/SEQ PROBE (small MAC d256/L2/1-mem, sliding, window 128) ===================="
for seq in 256 512 1024 2048; do
  for batch in 1 4 8; do
    timeout 8m "$PY" vram_probe.py --seq "$seq" --batch "$batch" --segment-len 128 --nmem-seg 16 \
      || echo "PROBE seq=$seq batch=$batch RESULT=TIMEOUT_OR_ERR"
  done
done

echo "-------------------- coarser memory granularity (cheaper scan) at long seq --------------------"
for seq in 1024 2048; do
  timeout 8m "$PY" vram_probe.py --seq "$seq" --batch 4 --segment-len 128 --nmem-seg 64 \
    || echo "PROBE seq=$seq batch=4 nmem_seg=64 RESULT=TIMEOUT_OR_ERR"
done

echo "==================== PROBE COMPLETE ===================="
