# Legacy AWS Command Helpers

This directory keeps obsolete helpers that are still useful for older, foreground SSM workflows.

Current status:
- Use `../check_detached_titan_status.sh` for the active detached Titan long-run flow.
- Use `check_ssm_status.sh` only for older one-shot or timeout-debug paths where the training job is still tied directly to the original SSM command.

Why this exists:
- The current Titan long-run launcher defaults to detached execution, so the old foreground-only polling helper is no longer the main status path.
- Keeping the legacy helper avoids breaking older notes and ad hoc debug workflows while making the preferred path explicit.
