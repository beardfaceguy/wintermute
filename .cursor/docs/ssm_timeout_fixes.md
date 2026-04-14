# SSM Run Command timeouts and flakiness (research + project fixes)

This note collects what others and AWS document about `ExecutionTimedOut`, `DeliveryTimedOut`, stuck `InProgress`, and `Undeliverable`, and what we changed in this repo.

## Symptoms we saw

- Long training via SSM stayed `InProgress` while logs stopped; concurrent probe commands returned `Undeliverable`.
- Document-style / IPC failures on some command IDs.
- One run hit `ExecutionTimedOut` after about **one hour** while mapping a dataset (GPT-2 smoke).
- Successful long runs sometimes never showed `Success` because the instance shut down before the agent reported completion (separate issue; see `.cursor/rules/ssm-command-completion.md`).

## How AWS models timeouts (official behavior)

From [Understanding command statuses](https://docs.aws.amazon.com/systems-manager/latest/userguide/monitor-commands.html):

1. **`--timeout-seconds` on `send-command`** maps to the console **Timeout (seconds)** field. It participates in the **total** timeout budget together with the document’s execution timeout.
2. **`AWS-RunShellScript`** exposes **`executionTimeout`** (default **3600** seconds in the document). That is the **execution** limit: if the shell script runs longer than this, the invocation can end as **`ExecutionTimedOut`**.
3. **Total timeout** is described as combining the delivery/outer timeout and the document execution timeout; if the command is not completed within the combined envelope, you can see **`DeliveryTimedOut`** depending on state.
4. **Maximum** execution timeout for Run Command is **172800 seconds (48 hours)** (commonly cited; confirm in your account/document version).
5. **`Undeliverable`**: the service could not deliver the command to the node (instance down, agent not responding, etc.). It is **not** the same as “no stdout for a long time.”
6. **`InProgress`** until the agent reports a terminal state; if the agent never reports (crash, forced power-off, broken agent), the invocation can remain stuck until timeouts.

**Critical repo finding:** Several long-running scripts used `--timeout-seconds 43200` but built parameters as `{ "commands": [...] }` **only**, with **no `executionTimeout`**. That leaves the document default **3600 s** execution limit. That matches an approximately **one-hour** `ExecutionTimedOut` independent of the larger `--timeout-seconds`. **Fix:** pass **`executionTimeout`** in the same JSON as `commands`, set consistently with expected runtime (up to 48h cap).

References:

- AWS: [Understanding command statuses](https://docs.aws.amazon.com/systems-manager/latest/userguide/monitor-commands.html) (timeout composition, `ExecutionTimedOut` vs `DeliveryTimedOut`, `Undeliverable`).
- Stack Overflow and re:Post threads discuss confusion between **`--timeout-seconds`** and **plugin/document execution timeout**; the document parameters win for shell runtime.

## Other fixes reported by operators

| Issue | Mitigation |
| --- | --- |
| Default **3600 s** execution cap on `AWS-RunShellScript` | Set **`executionTimeout`** in parameters (strings in a JSON array, e.g. `["43200"]`). |
| Delivery vs execution mix-up | Raise **both** `--timeout-seconds` and **`executionTimeout`** so the combined envelope covers the full job. |
| SSM Agent bugs / upgrades | GitHub [aws/amazon-ssm-agent#430](https://github.com/aws/amazon-ssm-agent/issues/430): commands could be lost around agent updates; keep agent updated. |
| Too many concurrent commands | Reports of ~**5** concurrent `InProgress` commands per instance; extra commands may queue or misbehave—avoid probe spam during long runs. |
| No logs | Use **`--output-s3-bucket-name`** / **`--output-s3-key-prefix`** and/or **`--cloud-watch-output-config`** so output survives stuck UI state. |
| Instance shutdown before agent ACK | Do not **`shutdown`** in the same document as training; stop instance **after** SSM shows **Success** (project rule). |

## Does `sleep` with no output cause timeout?

**No**—not as a separate “idle” timeout for Run Command. The agent tracks the running process; **`sleep`** keeps the process alive. **`ExecutionTimedOut`** is based on **elapsed time** vs **`executionTimeout`**, not on bytes to stdout. Lack of stdout does **not** explain **`Undeliverable`** (that points to delivery/agent/instance). For **psychological** debugging, the test script can **`echo`** every minute; that does not change the execution-timeout semantics.

## Code changes in this repo

- **`scripts/aws_commands/gpt_small_pretrain_long_cloudwatch.sh`**, **`gpt_small_ssm_with_logs.sh`**, **`gpt_small_fresh_10k_with_logs.sh`**: parameters now include **`executionTimeout`** aligned with **`SSM_EXEC_TIMEOUT_SECONDS`** (default `43200`), alongside existing **`--timeout-seconds`**.

## Test script

- **`scripts/aws_commands/ssm_timeout_sleep_test.sh`** — modes:
  - **`SSM_QUICK_TEST=1`** — ~2 min, `ExecutionTimedOut` (sanity check that `executionTimeout` is enforced).
  - **`SSM_LONG_VERIFY=1`** — default **4200 s (~70 min)** sleep, **`executionTimeout=7200`**, **`--timeout-seconds 43200`** — use this to prove jobs **over one hour** finish with **`Success`** (not the legacy 3600 s cap).
  - Otherwise: **`SLEEP_SEC`** default 4000 (~1h07), or set explicitly; **`INCLUDE_EXEC_TIMEOUT=false`** reproduces the old hourly timeout.

- **`scripts/aws_commands/ssm_timeout_wait_for_command.sh`** — polls until the invocation leaves `InProgress` (use after starting a test).

### How to run the test (after `aws sso login`)

```bash
export AWS_PROFILE=experimental-admin
export INSTANCE_ID=i-xxxxxxxxxxxxxxxxx

# A) Quick (~2m): expect ExecutionTimedOut
SSM_QUICK_TEST=1 ./scripts/aws_commands/ssm_timeout_sleep_test.sh

# B) Long over-1h verification (~70m): expect Success (leave instance running)
SSM_LONG_VERIFY=1 ./scripts/aws_commands/ssm_timeout_sleep_test.sh
# Optional: SLEEP_SEC=4500 SSM_LONG_VERIFY=1  # longer sleep, same safety margins

# C) Reproduce legacy failure (~1h07 to fail): no executionTimeout in parameters
INCLUDE_EXEC_TIMEOUT=false SLEEP_SEC=4000 ./scripts/aws_commands/ssm_timeout_sleep_test.sh
```

Poll / wait:

```bash
source /tmp/ssm_timeout_sleep_test_last.env   # CMD_ID / INSTANCE_ID / REGION / AWS_PROFILE
./scripts/aws_commands/legacy/check_ssm_status.sh
./scripts/aws_commands/ssm_timeout_wait_for_command.sh   # blocks until terminal status
```

## Iteration log (live verification)

| Attempt | Change | Result | Notes |
| --- | --- | --- | --- |
| 1 | Add **`executionTimeout`** to long-run shell parameter JSON (`gpt_small_*.sh`) | **Done (code)** | `jq` now emits `executionTimeout` alongside `commands`; delivery timeout via `SSM_DELIVERY_TIMEOUT_SECONDS`. |
| 2 | `ssm_timeout_sleep_test.sh` + `SSM_QUICK_TEST=1` on `i-095a84b978335b4f9` | **`TimedOut` / `ExecutionTimedOut`** (~2m) | Confirms the parameter is enforced; not “idle” detection. |
| 3 | Short success run (`SLEEP_SEC=200`, `executionTimeout=400`) | **Not fully polled** in session | Optional spot-check. |
| 4 | **`SSM_LONG_VERIFY=1`** (4200s sleep, exec 7200, delivery 43200) | **`Success`** | `CMD_ID=fc720351-058d-4b73-9348-0a2c4a34924f`, `i-095a84b978335b4f9`: after **~70m** wall time, **`get-command-invocation`** → **`Status=Success`**, **`ResponseCode=0`**. Confirms jobs **>1h** complete when **`executionTimeout`** is set (not the legacy 3600s default alone). |

The execution-timeout hypothesis is **confirmed** for this environment: quick test showed **`ExecutionTimedOut`** when cap was 120s; long verify showed **`Success`** for 4200s sleep with **`executionTimeout=7200`**.
