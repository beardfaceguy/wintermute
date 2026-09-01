#!/bin/bash
# Option (b) / #952: does MODEL SCALE crack needle-in-natural-text recall?
# The small MAC (6.8M, d256/L2/1-mem) learned clean MQAR but failed EVERY
# text/needle-in-filler variant (settled at the value marginal). Here we scale the
# model (d512/L6, 2 neural-memory layers, ~20M+) on the SAME tasks:
#   adj128_big  - adjacent-key MQAR-in-text, in-window: can scale do needle-copy at all?
#   adj512_big  - adjacent-key, seq512 sliding: does memory carry it through long filler?
#   tpl512_big  - the natural "value is" template, seq512: the true product task.
# far-recall (depth<=0.25) high => scale cracks it. Still marginal => real wall.
set -uo pipefail
PY="${PY:?}"; SCRIPTS="${SCRIPTS:?}"
cd "$SCRIPTS" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run () {  # run <name> <timeout> <script> <args...>
  local name="$1" tmo="$2" script="$3"; shift 3
  echo "==================== SWEEP $name ($script, timeout $tmo) ===================="
  timeout "$tmo" "$PY" -u "$script" "$@" 2>&1 | sed "s/^/[$name] /"
  echo "[$name] exit_status=${PIPESTATUS[0]}"
}

# 1. adjacent-key, in-window (seq128) — easiest needle-copy; can a bigger model do it?
run adj128_big 45m text_recall_adjacent.py --seq 128 --segment-len 128 --nmem-seg 16 \
    --dim 512 --layers 6 --heads 8 --neural-mem-layers 2 4 \
    --batch-size 4 --steps 15000 --eval-every 1000 --early-stop-far 0.8

# 2. adjacent-key, seq512 sliding (window 128) — memory-forced needle through long filler
run adj512_big 75m text_recall_adjacent.py --seq 512 --segment-len 128 --nmem-seg 32 \
    --dim 512 --layers 6 --heads 8 --neural-mem-layers 2 4 \
    --batch-size 2 --steps 20000 --eval-every 1000 --early-stop-far 0.8

# 3. natural "value is" template, seq512 — the actual product task at scale
run tpl512_big 75m text_recall_lucidrains.py --seq 512 --segment-len 128 --nmem-seg 32 \
    --dim 512 --layers 6 --heads 8 --neural-mem-layers 2 4 \
    --batch-size 2 --steps 20000 --eval-every 1000 --early-stop-far 0.8

echo "==================== SWEEP COMPLETE ===================="
