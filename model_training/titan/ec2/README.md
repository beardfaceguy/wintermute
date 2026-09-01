# EC2 Titan recall scale run (Stage-A / #952)

Runs the **memory-forced** associative-recall sweep on an A10G (24 GB) GPU to prove
the lucidrains MAC neural memory scales past the 8 GB local ceiling. Extends the
local result (4-pair memory-forced, acc 0.93) to a bigger MAC + multiple
neural-memory layers + more key→value pairs.

## Why EC2 (not SageMaker)
lucidrains/titans-pytorch has a finicky dependency tree already proven in the
gaming-pc `titan-venv`. EC2 reproduces that env directly (fresh venv, `pip install
torch` + `titans-pytorch`) with no fixed-DLC torch-version constraints to fight.

## Safety — self-terminating, cannot run away
Three independent termination layers (see `bootstrap.sh`):
1. `shutdown -h +HARD_CAP_MIN` scheduled first thing (default **240 min** absolute cap).
2. `trap ... EXIT` → `shutdown -h now` — fires on success **or** any failure.
3. Instance launched with `InstanceInitiatedShutdownBehavior=terminate` → any OS
   shutdown **terminates** the instance (no lingering stopped-but-billing).

The log syncs to S3 every 5 min, so partial results survive a hard-cap kill.
Self-termination uses only `shutdown` (no IAM terminate permission); the instance
profile grants S3 only. A failed bootstrap therefore self-destructs within minutes,
so even the first launch is safe to run unattended.

## What it runs
`bootstrap.sh` (EC2 user-data) builds the venv, then runs `run_sweep.sh`:
| step | pairs | model | mem layers | note |
|---|---|---|---|---|
| sanity | 4  | d256/L2 | 1 | env/VRAM check (proven local config) |
| p8  | 8  | d512/L4 | 2,4   | bigger MAC, memory-forced |
| p16 | 16 | d512/L4 | 2,4   | capacity test |
| p32 | 32 | d512/L6 | 2,4,6 | stretch |

All use `--sliding --segment-len 4` → the queried key is far beyond the local
attention window, so recall must route through the neural memory. Per-run
`timeout`s sum under the hard cap; each early-stops at query-acc 0.9.

## Usage
```bash
export AWS_PROFILE=experimental   # us-east-1, AdministratorAccess

# wire up + validate WITHOUT launching (creates IAM profile, uploads scripts,
# DryRun-validates run_instances):
python launch_ec2.py

# actually launch (self-terminating):
python launch_ec2.py --go
# options: --hard-cap-min 180  --instance-type g5.2xlarge  --tag recall-scale
```

On `--go` it prints the instance id + commands to watch the S3 log, check state,
and force-kill early. Results land at
`s3://sagemaker-us-east-1-491794274773/titan-ec2/<tag>/titan_run.log`.

## Cost
g5.2xlarge on-demand ≈ $1.2/hr. The full sweep is bounded well under the 4 h hard
cap; expect a few dollars. Instance + its gp3 EBS are deleted on termination.
