# SSM Command Completion Rule

## Rule: Do Not Shutdown Inside SSM Training Commands

**MANDATORY**: Let SSM-run training/eval scripts exit cleanly. Do not call `shutdown` in the same SSM command; stop/terminate the instance separately after the command completes.

### When This Rule Applies:
- Any long-running SSM `AWS-RunShellScript` used for training/eval, data prep, or exports.
- Scripts that sync checkpoints/logs to S3 and then stop the instance.

### Required Actions:
1. **No in-script shutdown**: Avoid `shutdown`/`poweroff` within the SSM command. Issue a separate stop/terminate after SSM reports `Success`.
2. **If shutdown is unavoidable**: Add a brief `sleep` after training/sync so the agent can report completion before power-off.
3. **No backgrounding**: Do not background the training process; ensure the script waits for training and sync to finish, then exits 0.
4. **Logs**: Prefer both S3 (`--output-s3-bucket-name` / `--output-s3-key-prefix`) and CloudWatch on long runs: `--cloud-watch-output-config "CloudWatchLogGroupName=/aws/ssm/...,CloudWatchOutputEnabled=true"`. The instance role needs `logs:PutLogEvents`, `CreateLogStream`, `DescribeLogStreams`, `CreateLogGroup` on that log group **and** `logs:DescribeLogGroups` on `*` (SSM agent calls it before writing). See `scripts/aws_commands/iam/ssm_cloudwatch_logs_inline_policy.json`.

### Purpose:
- Ensures SSM can update status to `Success`/`Failed` and avoids stuck `InProgress` commands.
- Preserves logs and reduces confusion about run completion.

---

**This rule keeps SSM jobs observable and prevents hidden shutdown-related failures.**
