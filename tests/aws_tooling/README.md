# aws_tooling test suite

Unit tests for the AWS lifecycle / monitoring tooling tracked in the
[Vikunja AWS Tooling project](#) (project id `68`).

## What's covered

These tests guard two pieces of safety-critical infrastructure:

| Component | What we assert | Why it matters |
|---|---|---|
| `~/.local/bin/aws-instance-watcher` | state-machine fires the right ntfy / exits the right rc for every observable transition (running, terminated, post-prune `None`, auth-fail, transient flake, threshold tunable) | Two production runs failed silently when this script took the wrong silent-loop branch. We never want to "wonder if it's still running". |
| `model_training/titanProject/scripts/lib/aws_lifecycle.sh` | pre-terminate hooks complete BEFORE `terminate-instances` is called, AND `terminate-instances` always either succeeds or falls back to `shutdown -h now`. Even when hooks fail, even when IMDSv2 is unreachable, even when the AWS API rejects the call. | This is the property that prevents (a) torching checkpoints/logs by terminating too early, and (b) stranding a billing instance idle, both of which we've actually hit. |

## What's NOT covered (deliberately)

* Real AWS lifecycle (launching, watching, terminating actual EC2 spot
  instances). That's a separate **integration** test tracked as a
  followup task in the AWS Tooling Vikunja project — it costs ~$0.001/run
  and tests the same surface end-to-end with real cloud APIs.
* `aws_titan_next_steps.py` (the launcher helper). That's a separate
  followup; it'll get its own tests when it's promoted to a generic
  Wintermute launcher.
* The `aws s3 sync` code-package upload pattern. Tracked as a separate
  Vikunja task ("codify S3 sync helper").

## Running locally

```bash
# All tests:
bash tests/aws_tooling/run_all.sh

# A single suite:
bash tests/aws_tooling/test_aws_instance_watcher.sh
bash tests/aws_tooling/test_aws_lifecycle.sh
```

Total wall time on the dev box: <1 second. No network calls, no AWS
credentials needed.

## How the mocking works

`test_helpers.sh` provides:

* `setup_mocks` / `teardown_mocks` — creates a temp dir, prepends a
  `mock_bin/` to `PATH`, opens a `mock_calls.log` file. The system under
  test calls `aws`, `curl`, `shutdown`, etc. through `PATH` — so it gets
  our mocks instead of the real binaries.
* `write_mock <name> <body>` — drops an executable shell stub at
  `mock_bin/<name>` whose body has access to `MOCK_LOG`. Every
  invocation appends an `argv` line to the log so tests can assert call
  count and ordering.
* Assertion helpers: `assert_eq`, `assert_contains`, `assert_file_contains`,
  `assert_rc_eq`, `assert_calls_in_order`. The last one is the most
  important — it walks `MOCK_LOG` and verifies the regex sequence appears
  in order, which is how we encode the safety property "hook before
  terminate".

## Why shell tests instead of Python/pytest

The tools under test are bash scripts. Testing them via shell mocks
keeps the test harness close to the production runtime — same
`PATH`-based dispatch, same shell semantics around `||`, `2>/dev/null`,
trap propagation, etc. The bugs that bit us in production were exactly
these shell-semantics edge cases (e.g. `aws ... 2>/dev/null || echo
"lookup-failed"` swallowing a non-zero exit code into a normal-looking
case branch). A Python harness would mock those away.

## Adding tests

1. Add a `test_<thing>` function to one of the existing files. Inside:
   call `setup_mocks`, register any specialized mocks via `write_mock`,
   run the system under test, assert via the helpers, call
   `teardown_mocks`.
2. Append the function name to the `run_tests` line at the bottom of
   the file.
3. If you're adding a new tool, create a new `test_<tool>.sh` and add it
   to the `TESTS` array in `run_all.sh`.

Tests should each finish in <1s and require zero external state. If a
test is hard to write in this style, that's usually a signal the
production code needs a small refactor for testability (the way
`run_titan_arms_ssm.sh` got a `lib/aws_lifecycle.sh` extracted for
exactly this reason).

## Out-of-band manual verification

A handful of behaviors are too coarse for unit tests but worth a
periodic smoke check:

```bash
# Three real watcher invocations against ntfy.sh — proves the
# round-trip end-to-end. Uses disposable test topics.
TOPIC="watcher-smoke-$(date +%s)"

POLL_INTERVAL_SEC=2 AUTH_FAIL_THRESHOLD=2 \
  ~/.local/bin/aws-instance-watcher i-090fcbf322a58970e "smoke pruned" "$TOPIC" || true
AWS_PROFILE=does-not-exist POLL_INTERVAL_SEC=2 AUTH_FAIL_THRESHOLD=2 \
  ~/.local/bin/aws-instance-watcher i-090fcbf322a58970e "smoke auth-fail" "$TOPIC" || true

sleep 3
curl -fsS "https://ntfy.sh/${TOPIC}/json?since=5m&poll=1"
# expect: 2 messages with title "* WATCHER BROKEN", priority max
```

This doesn't cost AWS billing (the second run uses a bogus profile, the
first runs against an already-terminated instance). Use it after any
material change to the watcher.
