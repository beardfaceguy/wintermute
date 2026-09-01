#!/bin/bash
# EC2 user-data: self-contained, SELF-TERMINATING Titan recall scale run (#952).
#
# THREE independent termination layers so an unsupervised run can NEVER become
# runaway billing:
#   1. `shutdown -h +HARD_CAP_MIN` scheduled as the very first action — absolute
#      wall-clock kill even if everything below hangs.
#   2. `trap ... EXIT` -> `shutdown -h now` — fires on normal completion OR any
#      failure (no `set -e`, so we always reach it).
#   3. the instance is launched with InstanceInitiatedShutdownBehavior=terminate,
#      so any OS shutdown TERMINATES the instance (no lingering stopped-but-billing).
# The log is synced to S3 every 5 min, so partial results survive a hard-cap kill.
# Self-termination uses only `shutdown` (no IAM terminate perm needed); the
# instance profile grants S3 only.
set -uo pipefail

HARD_CAP_MIN="${HARD_CAP_MIN:-240}"     # absolute cap (minutes); injected by launcher
S3="${S3_PREFIX:?}"                      # s3://<bucket>/titan-ec2/<tag>
LOG=/var/log/titan_run.log

# ── SAFETY LAYER 1: schedule the hard cap before doing anything that could hang ─
shutdown -h "+${HARD_CAP_MIN}" "titan hard cap ${HARD_CAP_MIN}m" || true

# ── SAFETY LAYER 2: always upload the log and shut down on exit ─────────────────
finish () { aws s3 cp "$LOG" "$S3/titan_run.log" || true; shutdown -h now || true; }
trap finish EXIT

exec > >(tee -a "$LOG") 2>&1
echo "==== titan ec2 bootstrap start $(date -u) (hard cap ${HARD_CAP_MIN}m) ===="
nvidia-smi || true

# periodic log sync so a hard-cap kill still leaves partial results in S3
( while true; do sleep 300; aws s3 cp "$LOG" "$S3/titan_run.log" >/dev/null 2>&1 || true; done ) &

# ── env: fresh venv + torch + titans-pytorch (pip resolves a consistent set) ────
# The base DL GPU AMI's system python3 lacks ensurepip, so install python3-venv first.
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y python3-venv
cd /home/ubuntu || exit 1
python3 -m venv tvenv
# shellcheck disable=SC1091
. tvenv/bin/activate
pip install -q --upgrade pip
pip install -q torch                       # default CUDA build; runs on the AMI's CUDA 13.2 driver
pip install -q titans-pytorch              # pulls tensordict/x-transformers/etc. for this torch
pip install -q datasets tokenizers         # text-recall filler streaming + BPE tokenizer
tvenv/bin/python -c "import torch, titans_pytorch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || { echo 'ENV BUILD FAILED — aborting (instance will self-terminate)'; exit 1; }

# ── fetch scripts + run the sweep ──────────────────────────────────────────────
mkdir -p /home/ubuntu/scripts
aws s3 cp "$S3/scripts/" /home/ubuntu/scripts/ --recursive
export PY=/home/ubuntu/tvenv/bin/python SCRIPTS=/home/ubuntu/scripts
bash /home/ubuntu/scripts/run_sweep.sh

echo "==== titan ec2 bootstrap done $(date -u) ===="
# trap EXIT uploads the log and shuts down -> instance terminates
