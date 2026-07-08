#!/bin/bash
# Scaled MEMORY-FORCED associative-recall sweep for the Stage-A cloud run (#952).
# Called by bootstrap.sh with PY (venv python) and SCRIPTS (dir with
# recall_lucidrains.py) exported. Extends the local result (4-pair memory-forced
# acc 0.93 on an 8GB card) to a bigger MAC + multiple neural-memory layers + more
# key->value pairs on the A10G (24GB).
#
# All runs use --sliding --segment-len 4 => the queried key sits far beyond the
# local attention window, so recall MUST route through the neural memory (not
# attention). Per-run `timeout` keeps the total well under the instance hard cap.
set -uo pipefail
PY="${PY:?}"; SCRIPTS="${SCRIPTS:?}"
cd "$SCRIPTS" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run () {  # run <name> <timeout> <args...>
  local name="$1" tmo="$2"; shift 2
  echo "==================== SWEEP $name (timeout $tmo) ===================="
  timeout "$tmo" "$PY" -u recall_lucidrains.py "$@" 2>&1 | sed "s/^/[$name] /"
  echo "[$name] exit_status=${PIPESTATUS[0]}"
}

# 0. env / VRAM sanity — the proven local config; confirms A10G + deps work fast.
run sanity 20m --n-pairs 4  --sliding --segment-len 4 --nmem-seg 16 \
    --dim 256 --depth 2 --steps 3000 --batch-size 16 --early-stop-acc 0.9

# 1. bigger MAC + TWO neural-memory layers, 8 pairs memory-forced.
run p8 50m  --n-pairs 8  --sliding --segment-len 4 --nmem-seg 16 \
    --dim 512 --depth 4 --heads 8 --neural-mem-layers 2 4 \
    --steps 15000 --batch-size 16 --early-stop-acc 0.9

# 2. 16 pairs — the real capacity test.
run p16 60m --n-pairs 16 --sliding --segment-len 4 --nmem-seg 16 \
    --dim 512 --depth 4 --heads 8 --neural-mem-layers 2 4 \
    --steps 20000 --batch-size 16 --early-stop-acc 0.9

# 3. 32 pairs — stretch; deeper model + three memory layers.
run p32 60m --n-pairs 32 --sliding --segment-len 4 --nmem-seg 16 \
    --dim 512 --depth 6 --heads 8 --neural-mem-layers 2 4 6 \
    --steps 25000 --batch-size 12 --early-stop-acc 0.9

echo "==================== SWEEP COMPLETE ===================="
