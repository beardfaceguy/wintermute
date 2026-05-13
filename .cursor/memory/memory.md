# Memory - Alix Estate Manager Development Insights

## 2026-03-21 AWS / vLLM Hosting Notes
- `vllm/vllm-openai-cpu:v0.17.1` works for HF GPT-2 export; allow ~30–35s warmup before probing `/v1/models` or requests will reset.
- Serve command used: `docker run -d --name vllm-test -p 127.0.0.1:8010:8000 -v /mnt/data/hf_runs/gpt2_nomem_small:/model vllm/vllm-openai-cpu:v0.17.1 --model /model --dtype float32 --port 8000`.
- ECR push of `vllm/vllm-openai:latest` is large and may show “Unavailable” for layers while uploading; retrying push after partial upload reuses cached layers.
- ECR pushes from EC2 require `ecr:GetAuthorizationToken`, `BatchCheckLayerAvailability`, `GetDownloadUrlForLayer`, `BatchGetImage`, `InitiateLayerUpload`, `UploadLayerPart`, `CompleteLayerUpload`, `PutImage` on the instance role. The first push from an M-series (arm64) host produced an arm64 image; rebuilding/pushing from amd64 is needed for GPU hosts.
- `vllm/vllm-openai:latest` pulled with `--platform linux/amd64` runs on g5 (A10G) with HF GPT-2 export: `docker run -d --gpus all -p 127.0.0.1:8010:8000 -v /mnt/data/hf_runs/gpt2_nomem_small:/model 491794274773.dkr.ecr.us-east-1.amazonaws.com/vllm-openai:latest --model /model --dtype float16 --port 8000`. Warmup/compile ~80s; `/v1/models` OK.
- Stable run settings from latest test: add `--gpu-memory-utilization 0.85`; expect ~90s warmup; `/v1/completions` succeeded on g5.xlarge.

## 2026-03-27 Titan GPT-small SSM + CloudWatch
- `send-command` with `--cloud-watch-output-config CloudWatchLogGroupName=/aws/ssm/titan-llm-training,CloudWatchOutputEnabled=true` streams stdout/stderr to CloudWatch during the run (unlike S3 plugin output, which may appear mainly after completion).
- Instance role must allow `logs:DescribeLogGroups` on `*` as well as `CreateLogGroup` / `CreateLogStream` / `DescribeLogStreams` / `PutLogEvents` on the target log group; otherwise SSM agent logs `AccessDeniedException` in `/var/log/amazon/ssm/errors.log` and no log streams appear.
- Launch scripts: `scripts/aws_commands/gpt_small_pretrain_long_cloudwatch.sh` (long pretrain), `gpt_small_ssm_with_logs.sh`, `gpt_small_fresh_10k_with_logs.sh`; IAM sample: `scripts/aws_commands/iam/ssm_cloudwatch_logs_inline_policy.json`.

## Architecture Patterns Discovered

### Field Implementation Patterns
- **Estate Email Pattern**: Field added directly to Estate model with @unique constraint, available for both creation and updates
- **taxId Pattern**: Field added via migration, nullable initially, unique constraint added later
- **scanBoxId Pattern**: Auto-generated on backend during estate creation, display-only in frontend
- **Deceased Fields Pattern**: Added to Deceased table via migration, handled in writeCoreEstateInformation mutation

### GraphQL Resolver Priority
- **Generated resolvers** are registered FIRST in resolver array
- **Custom resolvers** are registered LAST
- **Frontend mutations** may use generated resolvers instead of custom ones
- **Solution**: Add logic to BOTH generated and custom resolvers

### UI Conditional Rendering Patterns
- **Estate Email**: Always display with fallback "-" when empty
- **Test Account**: Boolean field with conditional styling
- **Status Fields**: Color-coded based on status values
- **Boolean Fields**: Checkbox or toggle display
- **Complex Conditional Logic**: Multiple conditions for field visibility
- **Fallback Display**: Show "-" or placeholder when field is null/empty
- **scanBoxId Pattern**: Conditional rendering - only display when value exists (no fallback)
- **Copy Functionality Pattern**: Use `navigator.clipboard.writeText()` with `useNotify` for user feedback

### Database Migration Strategy
- **Use postgres superuser** for schema changes when regular user lacks privileges
- **Follow established patterns** (taxId, Estate Email) for new field implementations
- **Apply pending migrations** before making schema changes
- **Create database backups** after migrations for quick restart capability

### Node Version Requirements
- **Critical**: Project requires Node 23+ to match deployment environment
- **Issue**: Node 20.18.1 causes generator failures and prevents client generation
- **Solution**: Default Node version set to 23 via `nvm alias default 23`
- **Status**: ✅ UPDATED - Default Node version now 23.11.1, matches QA3 deployment environment
- **Policy**: Local environment must match deployment servers (Node 23), not the other way around

### Prisma Version Requirements
- **Critical**: Project requires Prisma version `^5.18.0` (as specified in package.json)
- **Issue**: `typegraphql-prisma` only works with specific Prisma versions
- **Check**: Run `yarn why prisma` to verify installed version
- **Fix**: If version mismatch, reinstall correct Prisma version with `yarn add prisma@^5.18.0`

## Environment Gotchas

### Backend Startup Procedure
- **ALWAYS switch to Node 23 first**: `export NVM_DIR="$HOME/.config/nvm" && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && nvm use 23`
- **Then run in background**: `yarn start` (NEVER foreground - user loses control)
- **CRITICAL**: Wait 1-2 minutes for full startup - server does Prisma generation, service initialization, and database connections
- **Success indicator**: Look for "🚀 Server ready at http://localhost:8080/graphql" message
- **Frontend commands**: Use `yarn dev` (NOT `yarn start` - that command doesn't exist in frontend)

### Dependency Resolution
- **Backend (alix-api)**: **ALWAYS use yarn** - never use npm
- **Frontend (alix-estate-manager)**: **ALWAYS use yarn** - never use npm  
- **Package manager consistency**: Mixed npm/yarn causes CI/CD failures and deployment issues
- **MUI Lab conflicts**: React 19 vs MUI Lab compatibility resolved with legacy peer deps
- **Backend requires Node 23+**: Use nvm to switch versions

### Service Management
- **NEVER run services in foreground**: Always use `is_background: true` or user will need to kill them manually
- **Multiple instances**: Old ts-node-dev processes weren't properly killed, causing conflicts
- **Clean startup**: Kill all existing processes before restarting services

### AWS Training Environment Audit
- **Use repo helper for AWS checks**: `python model_training/titanProject/aws_titan_next_steps.py audit` provides a done/pending checklist for S3/IAM/SG/key/instances/spot/checkpoints.
- **Idempotent continuation flow**: Run `ensure-sg`, then `launch-spot` (or `launch-ondemand`) from the same helper script for repeatable setup.
- **Local Titan runs with S3-backed inputs**: activate `.venv`, export `AWS_PROFILE=experimental-admin` and `AWS_SDK_LOAD_CONFIG=1`, then use `python -u ... | tee ...` so `boto3` sees SSO credentials and the log file is not stalled by stdout buffering.
- **Local Titan interpreter gotcha**: the shell's default `python3` may still point at the system interpreter even when the venv exists. For reproducible local Titan runs, prefer the explicit interpreter path `.venv/bin/python`.
- **Titan workstation handoff note**: the checked-in Titan workflow is not tied to a specific laptop, but local virtualenv paths are host-specific. On a new x86 Linux laptop, recreate `.venv` from `requirements.txt` there instead of copying it; the canonical long-run execution path remains the AWS detached Ubuntu runner. (2026-05-13 cleanup: consolidated `.venv_docs` + empty `.venv_titan` into a single `.venv`.)
- **WSL AWS CLI fallback**: helper now tries `aws`, `aws.exe`, and `/mnt/c/Program Files/Amazon/AWSCLIV2/aws.exe`; set `AWS_CLI_BIN` if your executable is elsewhere.
- **WSL install gotcha**: when `awscli` apt package is unavailable, install AWS CLI v2 from zip into `~/.local/bin` and export `AWS_CLI_BIN=/home/zombi/.local/bin/aws` for script runs.
- **Zip extraction gotcha**: Python `zipfile` extraction may drop executable bits for AWS installer files; restore execute perms (`/tmp/aws/install`, `/tmp/aws/dist/aws`) before install.
- **Auth clarity improvement**: `aws_titan_next_steps.py audit` now prints explicit `AuthError` when the selected profile is missing, instead of silently showing all resources as `False`.
- **WSL context gotcha**: running `wsl -d ...` inside an existing WSL shell can invoke the Linux `wsl` (Wsman Shell), not Windows WSL launcher; in WSL use AWS commands directly.
- **SSM fallback pattern**: when SSH key is missing locally but IAM role includes `AmazonSSMManagedInstanceCore`, use SSM `send-command`/`get-command-invocation` for bootstrap and diagnostics.
- **EC2 bootstrap path**: using SSM, restore only the needed Titan subtree from `s3://alix-ai-ml-staging-data/titan/code/wintermute/model_training/titanProject` into `/home/ubuntu/wintermute/model_training/titanProject` for smoke/pretrain bring-up, and emit explicit phase markers around code sync, pip installs, and dataset preload so CloudWatch shows where time is being spent.
- **Titan code-restore optimization**: the mirrored `model_training/titanProject` tree in S3 is not safe to restore verbatim during AWS smoke/pretrain bring-up because it contains large generated artifacts under `logs/`, especially `logs/LLM/data`, that can exhaust the root disk before training starts. Current launcher pattern: restore from a small code-only tarball bundle at `s3://alix-ai-ml-staging-data/titan/code_bundles/titanProject_bundle.tar.gz`, excluding `model_training/titanProject/logs`, `*.pt`, `*.pyc`, and `__pycache__`, then extract into `/home/ubuntu/wintermute`.
- **Titan launcher env gotcha**: if the remote launcher’s inline Python reads `TRAIN_MAX_TOKENS_OVERRIDE` / `VAL_MAX_TOKENS_OVERRIDE` via `os.environ`, those shell variables must be explicitly `export`ed in the remote script. Plain shell assignment is not visible to the Python subprocess and silently falls back to the baseline token caps.
- **Titan validated AWS smoke recipe**: the current known-good AWS smoke path is `scripts/aws_commands/gpt_small_fresh_10k_with_logs.sh` on `g6.xlarge`, using the bundle restore path plus `MAX_STEPS=10`, `LOG_EVERY=1`, `TRAIN_MAX_TOKENS_OVERRIDE=1000000`, and `VAL_MAX_TOKENS_OVERRIDE=50000`. That configuration reached visible CUDA optimizer steps through `step=10` and completed with SSM `Success`, so future bring-up should start from that exact pattern before promoting to longer runs.
- **Titan long-run timeout lesson**: the first full `config_gpt_small.yaml` AWS run proved that a `40000`-step single-GPU pretrain can easily exceed the old `SSM_EXEC_TIMEOUT_SECONDS=43200` / `SSM_DELIVERY_TIMEOUT_SECONDS=43200` (`12h`) defaults in `gpt_small_pretrain_long_cloudwatch.sh`. The job itself stayed healthy and synced checkpoints through `16000`, but SSM timed out long before training completed. Do not assume the SSM envelope is long enough just because bootstrap succeeded.
- **Titan local-watchdog failure mode**: a workstation-side `caffeinate` monitor is not a reliable sole cleanup path for multi-hour AWS runs when it depends on local AWS SSO refresh. In the timeout incident, the monitor crashed hours before the training command timed out because `aws` could no longer refresh the local SSO token. Cleanup for long runs must not depend exclusively on a local long-lived SSO session.
- **Titan durable long-run pattern**: `scripts/aws_commands/gpt_small_pretrain_long_cloudwatch.sh` now defaults to `DETACH_TRAINING=1`, meaning SSM handles bootstrap only and then launches a remote wrapper on the instance for the real training process. That wrapper performs a final checkpoint sync on exit, can upload `train.log` and a run-status JSON to S3, and can optionally stop the instance with `STOP_INSTANCE_ON_EXIT=1`. To make self-stop work, attach `scripts/aws_commands/iam/ssm_long_run_self_stop_inline_policy.json` to the training role in addition to the existing CloudWatch logs policy.
- **Titan detached-run status helper**: use `scripts/aws_commands/check_detached_titan_status.sh` with `RUN_ID` and `INSTANCE_ID` after launching detached long runs. It checks EC2 state, optionally shows the original bootstrap SSM command status, runs a short SSM probe against `/mnt/data/ssm_runs/<RUN_ID>`, tails the detached `train.log`, and lists synced checkpoints from S3. This is the preferred monitoring path once bootstrap has already exited.
- **Titan legacy foreground status helper**: `scripts/aws_commands/check_ssm_status.sh` is obsolete for the current detached long-run flow and now lives at `scripts/aws_commands/legacy/check_ssm_status.sh`. Keep it only for older foreground SSM timeout-debug paths.
- **Titan legacy helper directory**: `scripts/aws_commands/legacy/README.md` explains why obsolete foreground-only helpers are preserved and points operators to `scripts/aws_commands/check_detached_titan_status.sh` as the default status path.
- **Titan detached self-stop bug**: the detached runner finalizer needs `REGION` assigned inside the generated remote script before `set -u` cleanup runs. Otherwise the run can finish, write `run_status.json`, and sync artifacts, but automatic `stop-instances` will fail with `REGION: unbound variable`.
- **Titan detached smoke overrides**: `scripts/aws_commands/gpt_small_pretrain_long_cloudwatch.sh` now accepts `MAX_STEPS`, `LOG_EVERY`, and `SAVE_EVERY` env overrides so the detached path itself can be validated with tiny runs before committing to a full 40k-step job.
- **Titan detached resume support**: `scripts/aws_commands/gpt_small_pretrain_long_cloudwatch.sh` now accepts `RESUME_CKPT_S3_URI` and downloads that checkpoint onto the instance before launching `train.py --resume ...`. This makes it possible to continue older Titan detached runs from an S3-synced checkpoint instead of always starting from step 0.
- **Titan checkpoint lineage rule**: treat `gpt_small_pretrain_20260411004641/ckpt_step_40000.pt` as the current canonical base-model checkpoint and branch later work from it cleanly. Start SFT from that base, but keep SFT checkpoints in a separate lineage from any future corpus-training extensions. If more base-model training is needed, resume from a pretrain checkpoint and then rerun SFT from the newer base rather than trying to continue pretraining from an SFT checkpoint.
- **Titan current SFT launcher rule**: use `scripts/aws_commands/gpt_small_sft_pilot_cloudwatch.sh` for the first GPT-small SFT smoke/pilot on AWS. It downloads the canonical base checkpoint and tokenizer, rebuilds the current instruction mix on-instance, writes a local SFT config derived from `config_gpt_small.yaml`, and can run detached with final artifact upload and optional self-stop.
- **Titan SFT cache placement rule**: on reused EC2 runners, Hugging Face dataset and hub caches can silently fill `/root/.cache` and leave the root disk near-fatal before SFT even starts. Keep SFT launcher HF caches under `/mnt/data` (for example `HF_HOME=/mnt/data/cache/huggingface` plus dataset/hub subdirs) and, when deliberately recovering a dirty runner, clear stale `/root/.cache/huggingface` before relaunch.
- **Titan code-bundle freshness gotcha**: the detached AWS SFT launcher restores `model_training/titanProject` from `s3://alix-ai-ml-staging-data/titan/code_bundles/titanProject_bundle.tar.gz`, so new repo-side prep/loader changes are not present remotely until that tarball is rebuilt and uploaded. A launch can fail with seemingly impossible argument mismatches if the local code and the S3 bundle drift.
- **Titan SFT config/history gotcha**: `model_training/titanProject/configs/config_sft_pilot_oasst1_dolly.yaml` is historical and should not be treated as the current source of truth for GPT-small SFT from `ckpt_step_40000.pt`. The present `prepare_sft_mix.py` sources are `OASST1`, `OpenHermes`, `SlimOrca`, and optional GSM8K-style logic pairs, so current launcher/config generation should derive model shape from the active GPT-small pretrain config rather than the older MAC-era pilot config.
- **Titan first SFT smoke lesson**: the first detached GPT-small SFT smoke from `ckpt_sft_step_200.pt` validated the infrastructure path but not the model quality. Qualitative prompts still showed repetition, malformed task execution, and `User:` / `Assistant:` leakage in completions, so treat that checkpoint as path validation only.
- **Titan tightened SFT recipe knob**: `prepare_sft_mix.py` now supports `--reject-role-markers` to drop samples whose user or assistant text already contains literal `User:` / `Assistant:` markers. Use this for the next tighter SFT pilot together with a narrower early mix (favor `OASST1` + a smaller `OpenHermes` slice, drop `SlimOrca` and logic boosters initially, tighten char caps) before scaling steps back up.
- **Titan tightened 600-step SFT lesson**: the tighter `600`-step pilot improved the infra path and the eval curve, but the checkpoint is still not a deployable assistant. On the fixed five-prompt qualitative suite, `ckpt_sft_step_600.pt` stayed more on-topic than `ckpt_sft_step_200.pt` in places, yet still leaked `User:` / `Assistant:` turns into completions, answered arithmetic incorrectly, and drifted off-task on story/debugging prompts. Treat `ckpt_sft_step_600.pt` as a better experiment result, not a promotion candidate.
- **Titan decode-sweep lesson**: a stricter inference sweep across the tightened-run checkpoints (`200` / `400` / `600`) with `top_k=1,temp=1.0` and `top_k=5,temp=0.3` did not rescue behavior. Lower-entropy decoding reduced some randomness, but the same core defects remained: repetition loops, explicit `User:` / `Assistant:` leakage in later checkpoints, and incorrect arithmetic/task completion. That means the next gain likely has to come from training-data/format changes rather than inference settings alone.
- **Titan OASST1-only pilot lesson**: the narrower `OASST1`-only `600`-step pilot (`gpt_small_sft_oasst1_v1_20260411192919`) also failed the fixed qualitative suite. Compared with the mixed tightened run, it sometimes sounded more like a polite chat assistant at the first sentence, but it still quickly collapsed into multi-turn leakage, incorrect factual/math answers, and off-task dialogue. Narrowing the mix alone was not enough.
- **Titan masked-SFT objective rule**: `finetune_sft.py` no longer should be treated as a plain next-token trainer over the full flattened `User: ... Assistant: ...` line. The new SFT path builds sample-local masked labels so prompt tokens are ignored (`-100`) and loss starts at the first assistant answer token, with an explicit end-of-sample token appended when available. Keep pretraining on the old continuous-token loader, but treat SFT as answer-only supervision from here forward.
- **Titan OASST quality-gate rule**: `prepare_sft_mix.py` now supports opt-in `OASST1` quality gates (`--oasst-best-only`, `--oasst-min-quality`, `--oasst-min-helpfulness`, `--oasst-max-fails-task`, `--oasst-max-spam`). Use these when you want a smaller, cleaner `OASST1` slice; the change was motivated by direct samples showing low-rank, spammy, `fails_task`-flagged `OASST1` answers still entering the SFT target text.
- **Titan zero-count source-selection rule**: in `prepare_sft_mix.py`, a requested pair count of `0` now means "disable this source" rather than "use all available". The old behavior silently polluted supposed single-source experiments, and sources with both train/val counts at `0` should now be skipped entirely during load to avoid wasted bootstrap time.
- **Titan cleaned OASST1 pilot lesson**: the cleaned, quality-gated masked `OASST1` pilot (`gpt_small_sft_clean_oasst1_v1_20260411200834`) improved loss on a much smaller, cleaner corpus, but the fixed qualitative suite still showed repetition, wrong arithmetic, and malformed task completions. Practical implication: do not spend a longer run on that recipe family without a stronger base.
- **Titan OpenHermes-only pilot lesson**: the corrected `OpenHermes`-only masked pilot (`gpt_small_sft_openhermes_v2_20260411201927`) reached much better eval perplexity than the cleaned `OASST1` pilot, yet the final checkpoint still failed qualitatively with repetition, wrong arithmetic, and bad task execution. Practical implication: better-looking SFT loss on this base checkpoint is still not a reliable proxy for assistant behavior.
- **Titan branch-split rule**: keep the Titan GPT-small effort split into two explicit branches. Branch A is a GPT-2 bootstrap validation branch used to prove the workflow can produce a usable assistant. Branch B is the controlled from-scratch branch used for the actual long-term behavior-control goal. Do not let branch-level experiment noise accumulate in the umbrella Titan project; keep umbrella issues for cross-branch decisions and summaries.
- **Titan GPT-2 bootstrap rationale**: GPT-2 small/medium are acceptable as workflow-proof bases because they are pretrained next-token models rather than modern RLHF chat models, but they still carry OpenAI/WebText priors and should not be treated as the final "fully controlled" base lineage. Use them to validate the recipe, then port the proven recipe onto the controlled scratch branch.
- **Titan GPT-2 bootstrap convention**: use `hf://gpt2` as the Branch A source identifier for both tokenizer and model weights. `train.py` should use `--init-from hf://gpt2` for continued-training style workflows, while `finetune_sft.py`, `generate.py`, `inference_smoke.py`, `chat_repl.py`, and `chat_http.py` can use `--ckpt hf://gpt2` directly. Keep GPT-2-specific model dimensions and `vocab_size: 50257` in `model_training/titanProject/configs/config_gpt2_small.yaml`.
- **Titan GPT-2 bootstrap implementation rule**: prefer the exact `hf_gpt2` wrapper path over trying to map GPT-2 weights into the local `x-transformers` `GPTLM`. The mapping path can load tensors, but it did not preserve native GPT-2 qualitative behavior well enough for Branch A validation. The reliable Branch A bootstrap path is `variant: hf_gpt2` plus `hf://gpt2`.
- **Titan instruction-format SFT rule**: for Branch A instruction tuning, prefer `prepare_sft_mix.py --output-format instruction_jsonl` over flattening everything into one-line `User:` / `Assistant:` text. The JSONL path preserves internal newlines and gives `finetune_sft.py` enough structure to rebuild the Raschka-style prompt (`### Instruction`, optional `### Input`, `### Response`) at load time while keeping one sample per record.
- **Titan local GPT-2 instruction smoke lesson**: the first tiny local Branch A smoke (`48` train / `12` val OASST1 rank-0 instruction JSONL samples, `20` CPU SFT steps from `hf://gpt2`) proved the workflow path end to end but did not produce a clean assistant checkpoint. The smoke checkpoint shifted tone toward an explicit assistant persona, yet still showed incorrect arithmetic, `Assistant:` leakage on chat-style prompts, and response-header echo/repetition on instruction-style prompts. Treat this as workflow validation plus bug discovery, not a promotion candidate.
- **Titan local smoke prompt-family rule**: for instruction JSONL smoke configs, local qualitative checks should default to instruction-style prompts rather than legacy chat-style prompts. `inference_smoke.py` now supports `--prompt-family auto` and infers instruction prompts from instruction JSONL configs, while `prompt_formats.py` trims repeated `### Response:` / role markers from reported completions so prompt-boundary artifacts do not get mistaken for model-content behavior.
- **Titan local recipe-iteration lesson**: on the tiny Branch A GPT-2 smoke, adding GSM8K logic samples made arithmetic completions more answer-shaped but still wrong. A tiny curated direct-answer booster file (`model_training/titanProject/data/smoke_instruction_boosters.jsonl`) was the first local tweak that produced a directionally correct arithmetic answer on the instruction-mode smoke suite, but the result was still unstable across separate generations and often ran past the short answer into unrelated text. Treat the curated booster path as the best local candidate so far, but not yet as a stable proof recipe.
- **Titan narrative-booster lesson**: after the slightly longer low-LR local Branch A rerun (`cla98_longlr_smoke_20260411231842`), the remaining qualitative gap is no longer best described as a general "needs more steps" problem. The current curated direct-answer-heavy mix can preserve formatting and arithmetic well enough to pass the local rubric overall, yet still fail short story-style prompts semantically. The next local Branch A follow-up should therefore target narrative/composition examples specifically before trying broader or longer reruns of the same mix.
- **Titan minimal story-booster result**: a small checked-in narrative booster slice plus a story-weighted tiny merged smoke dataset (`sft_smoke_instruction_story_curated`) improved the Branch A local story prompt from generic factual prose about cats to an on-topic short story-like answer, while preserving the arithmetic win (`2 plus 2 is 4.`). However, the resulting story is still shallow/generic, so this path should be treated as directionally correct but not yet a sufficient proof of good narrative quality.
- **Titan tiny-narrative cleanup caveat**: on the local Branch A GPT-2 smoke, tightening the first story-weighted booster slice into a smaller cat-and-kite-only set with no duplicated rows regressed the result instead of improving it. The rerun lost the arithmetic win and still failed the story prompt, so tiny-recipe quality here is sensitive to weight/diversity tradeoffs rather than monotonically improving with "cleaner" supervision.
- **Titan story-rubric tightening rule**: the instruction smoke story prompt should not count as a pass merely because `cat` and `kite` appear. `inference_smoke.py` now expects story-like action/structure for prompt 2, so historical keyword-only story passes are no longer reliable evidence of narrative quality.
- **Titan cla99 decode-sweep lesson**: a tiny decode sweep on the current best local narrative checkpoint (`checkpoints_sft_smoke/cla99_story_curated/ckpt_sft_step_80.pt`) did not rescue story quality. Near-greedy decoding can preserve arithmetic, and longer outputs can change the failure mode, but neither produces a genuine short-story pass under the stronger story rubric. Treat the remaining Branch A local gap as recipe/data-limited rather than primarily decode-limited.
- **Titan prompt-family-balanced booster lesson**: after the decode sweep confirmed the remaining CLA-99 gap was recipe-limited, a broader story slice plus tiny arithmetic stabilizer still was not enough by itself. The strongest local result came from a stricter hand-built booster set matched directly to the three smoke prompt families: one-sentence self-intro, blue-cat/red-kite microstory, and arithmetic. That balanced exact-prompt-family variant (`cla99_micro_balanced`) is now the best local Branch A candidate under the stronger story rubric.
- **Titan deterministic-local-recipe goal**: for the current Branch A local tightening loop, the target is no longer just one better smoke checkpoint. The recipe should be expressible as checked-in booster inputs plus a deterministic dataset-build step so the same bounded local run can be reproduced with minimal agent intervention. The next local pass after `cla99_micro_balanced` should focus on prompt-1 self-introduction quality while preserving the existing story + arithmetic wins and improving the reproducibility of generating the candidate dataset itself.
- **Titan deterministic-builder lesson**: the new local smoke-variant builder (`scripts/build_smoke_instruction_variant.py`) successfully converts checked-in booster inputs into a reproducible instruction-smoke dataset, which is the right direction for the minimal-intervention goal. But the first builder-driven prompt-1-focused recipe (`cla99_micro_balanced_v2`) regressed on both story and arithmetic, so reproducibility improved before quality did. Future local tightening should continue from the scripted path, but treat `cla99_micro_balanced` as the current quality baseline until a script-built variant matches or beats it.
- **Titan local decode-stop lesson**: for the Branch A GPT-2 instruction smoke, adding EOS-aware stopping plus prompt-family stop strings materially cleaned up outputs without retraining. With stop handling enabled, the curated checkpoint could emit short clean answers like `I'm a programmer.` and `2 plus 2 is 4.` This means part of the prior failure pattern was decode overshoot, not only weak model content. Also, the default `inference_smoke.py` threshold (`min_completion_chars=20`) is too blunt for short-answer instruction prompts and can mark correct short completions as failures.
- **Titan local checkpoint-save guard gotcha**: `finetune_sft.py` defaults to `--min-free-gb 20`, which can silently skip checkpoint writes during intentionally tiny local smoke runs on a workstation with less free space. For small local validation runs where the checkpoint directory is known and bounded, explicitly lower `--min-free-gb` so the smoke can actually save the checkpoint you intend to inspect.
- **Ubuntu Python packaging gotcha**: some WSL images have `python3` without `pip` and without `ensurepip`; install pip via `get-pip.py --user --break-system-packages` before installing CLI tools.
- **Local HF CLI path**: `hf` installs to `/home/<user>/.local/bin/hf`; use absolute path or add `~/.local/bin` to PATH.
- **HF token sanity check**: before scripted `hf auth login --token`, verify token file has non-zero stripped length to avoid non-obvious `Option '--token' requires an argument` failures.
- **HF CLI command variant**: with `huggingface_hub` 1.6.0, use `hf auth whoami` for identity check (plain `hf whoami` is not a valid command).
- **HF auth on remote via SSM**: safest pattern is upload token file as short-lived encrypted S3 object, run `hf auth login` on EC2 via SSM using that object, then delete the object in script cleanup.
- **Remote AWS CLI compatibility**: some EC2 images still have AWS CLI variant that rejects `--no-cli-pager`; for SSM-run `aws s3 sync/cp` commands, avoid that flag unless version is confirmed.
- **Titans checkpoint persistence**: run eval/checkpoint hooks at step-level (inside the batch loop), not epoch-end, or long datasets delay checkpointing too much; `train.py` now supports `--save-every`, `--checkpoint-dir`, and `--s3-checkpoint-uri` for periodic durable sync.
- **Sync resilience**: when wiring periodic S3 sync into training loops, treat sync as best-effort (log warning on missing `aws`/sync failure) so model training itself continues.
- **Titan AWS hardware recommendation after CLA-42**: prefer `g6.2xlarge` for main GPT-small pretraining and `g6.xlarge` only for short smoke runs. `g6.2xlarge` keeps a single `24 GiB` GPU but doubles host RAM to `32 GiB`, which is materially safer for the current preload/tokenization path and still fits the documented `$50/mo` guarded burst model better than `g5.xlarge`.
- **Titan DLAMI disk-selection gotcha**: on the current GPU AMI, a non-root NVMe device may already be consumed by the DLAMI ephemeral LVM stack and mounted at `/opt/dlami/nvme`. Do not bootstrap `/mnt/data` by selecting the "first non-root disk"; explicitly target the blank attached EBS data volume or verify with `lsblk -f` / `findmnt` first.
- **Titan full-corpus memory trap**: the current `data.py` path still accumulates token IDs in a Python list before converting to a tensor. Combined with `config_gpt_small.yaml` using `max_tokens=1_000_000_000`, this is enough to overwhelm `16 GiB` hosts during tokenization and can manifest as EC2 instance reachability failure, SSM `ConnectionLost`, ghost `InProgress`, and `Undeliverable` follow-up probes. Short-term workaround: use `TRAIN_MAX_TOKENS_OVERRIDE` / `VAL_MAX_TOKENS_OVERRIDE` in `gpt_small_pretrain_long_cloudwatch.sh` for smaller bring-up passes. Long-term fix: move to disk-backed token storage (shards or memmap) with lazy window slicing instead of whole-corpus materialization.
- **Titan disk-backed token cache (CLA-48)**: `model_training/titanProject/data.py` now uses a sharded token-cache format: one cache directory per `(path, tokenizer fingerprint, max_tokens)` key, a `manifest.json`, and multiple `uint32` shard files reopened lazily via memmap-backed readers. `train.py` / `finetune_sft.py` now derive the cache key from a SHA-256 fingerprint of tokenizer contents, not just the tokenizer path, so tokenizer updates at the same path invalidate cleanly. Keep `TITAN_TOKEN_CACHE_SHARD_SIZE_TOKENS` available as the controlled override for validation and future shard-size benchmarks.
- **Titan cache-build concurrency gotcha**: the disk-backed cache writer must always use a unique temp build directory or filename per process/build attempt before the final atomic replace. Shared temp names cause same-key concurrent builders to clobber each other. The current implementation fixes this with a PID/UUID-scoped temp directory; next follow-up is shard-size benchmarking on the real corpus, not more cache-identity work.
- **Titan shard-size benchmark harness**: the real Titan S3 tokenizer/corpus path now validates successfully with forced tiny shards (`TITAN_TOKEN_CACHE_SHARD_SIZE_TOKENS=256`, `--max-tokens 2048`), including same-process and fresh-process cache reuse. That means future shard-size experiments can focus on performance trade-offs rather than correctness debugging.
- **Titan shard-size identity and first range**: shard size itself must be part of the cache key, not just manifest metadata, or benchmark/production runs with different shard settings will silently reuse the wrong cache layout. The first real benchmark sweep on an `8M`-token Titan slice showed the practical sweet spot is not the extremes: `512K` had the best reuse time, `2M` had the best multi-shard build time, and the single-shard `8M` case only barely won raw build time while giving up the operational benefits of sharding. Current recommended next sweep range: `512K` to `2M`.
- **Titan current shard-size default**: the tighter follow-up sweep across `512K`, `768K`, `1M`, `1.5M`, and `2M` on the same `8M`-token slice reinforced `2M` as the best current default. In that narrowed range it gave the fastest build while keeping a still-sharded layout (`4` shards). `1M` is the main conservative fallback if future operational concerns favor more shards over a modest build-time cost.
- **Titan fine-grained shard result**: a final local sweep around `2M` using literal token counts (`1,750,000`, `2,000,000`, `2,150,000`, `2,500,000`) did not support moving higher. `1,750,000` was the fastest build point in that band and avoided the reuse regression seen at `2,000,000` and `2,150,000`. Current best measured setting on this harness is `1,750,000`, with `2,000,000` still acceptable as the cleaner round-number fallback.
- **Titan shard benchmark noise is real**: the drift-check rerun across `1M` through `2.1M` showed nontrivial run-to-run instability that is unlikely to be explained by shard size alone. Concrete examples: `1.0M` had a normal build but a one-off `4.14s` reuse spike, and `1.9M` had a build slowdown immediately followed by a strong `2.0M` result. The stable conclusion is not a single magic shard size; it is that the useful region is still roughly `1.5M` to `2.1M`, and future benchmarking should control for external variance with repeated/interleaved runs or explicit cache-warmness controls.
- **Titan controlled shard benchmark outcome**: once the benchmark was upgraded to `3` repeats, interleaved order, forced cold builds, paired fresh-process reuses, and per-phase load capture, the earlier dramatic outliers mostly disappeared. On that controlled harness, `1,750,000` had the best mean build time (`13.89s`) with low variance (`0.31s`) and solid reuse behavior, while `1,800,000` was the closest practical neighbor. The control lesson is as important as the winner: repeated/interleaved benchmarking is necessary here because single-pass sweeps overstate noise.
- **Titan controlled rerun drift result**: repeating the same controlled benchmark with the same seed still produced meaningful drift for some shard sizes, so even the improved harness does not eliminate all external variance. The important part is which sizes stayed good anyway: `1,750,000` and `1,800,000` remained the strongest stable region across both controlled runs, while `1,500,000` and `1,900,000` showed much higher build variance and should not be treated as robust defaults. Current practical guidance: prefer `1,750,000` for best repeated build performance or `1,800,000` for the slightly rounder stable alternative.
- **Titan active shard-size default**: `model_training/titanProject/data.py` now sets `DEFAULT_SHARD_SIZE_TOKENS = 1_750_000`. The env var `TITAN_TOKEN_CACHE_SHARD_SIZE_TOKENS` still overrides it for experiments, but the no-override path now matches the best repeated controlled benchmark result.
- **Inference regression check**: keep a single-command smoke harness (`inference_smoke.py`) that loads checkpoint once and runs a fixed prompt suite with JSON pass/fail output; this is the fastest end-to-end health check after training/infrastructure changes.
- **Qualitative interface check**: keep a minimal interactive REPL (`chat_repl.py`) for fast human sanity checks after a successful smoke test; include `/reset` and context trimming so multi-turn prompts stay within seq_len budget.
- **No-dependency serving option**: use a stdlib HTTP server (`chat_http.py`) for quick model integration tests without adding web dependencies; provide `/health`, `/chat`, `/reset`, keep per-session in-memory history, and guard generation with a lock for thread safety.
- **Built-in browser tester**: serving a simple HTML page at `GET /` (same `chat_http.py`) is enough for manual chat checks without Gradio; keep JSON API endpoints unchanged for scripts.
- **Safe exposure pattern for testing**: when publishing model HTTP endpoint, open SG app port only to operator `/32` (not `0.0.0.0/0` on ingress) and verify with `curl` health check before sharing the link.
- **Path proxy gotcha (`/titan`)**: when placing the model UI behind an nginx path prefix, frontend calls must be path-relative (derive API base from `window.location.pathname`) or requests will incorrectly hit root paths (`/health`, `/chat`) and fail.
- **Domain bring-up checklist**: for custom domain + path (`wint3rmute.com/titan`), you need three aligned layers: DNS A record -> EC2 public IP, SG ingress on `80/443`, and nginx reverse proxy location to backend (`/titan/` -> `127.0.0.1:8000/`).
- **SFT pilot recipe**: a practical first chat-improvement pass is `OASST1 + Dolly` converted to one-line `User: ... Assistant: ...` examples and trained for a short 600-step supervised run from the base checkpoint; this quickly shifts behavior toward assistant-style responses without a full retrain.



## Key Lessons Learned

### Implementation Methodology
- **Jira-First Analysis**: MUST check Jira task requirements first before analyzing UI patterns
- **Complete Field Analysis**: MUST analyze ALL existing fields before implementing new ones to understand patterns
- **Backend-First Verification**: Always check if functionality already exists in backend services
- **Complete Data Flow Tracing**: Database → Backend Service → GraphQL → Frontend mutation → UI display
- **CRITICAL: Requirement Clarification**: NEVER assume requirements or add "common practices" without explicit user approval
- **MAJOR LESSON: Unauthorized Character Exclusions**: Added character exclusions (I,O,0,1) to scanBoxId without user approval - major overstep that violated requirement clarification rule
- **CRITICAL: Minimal Changes Principle**: ALWAYS GO FOR THE MINIMUM CHANGES NEEDED TO COMPLETE THE TASK - Never make architectural changes that affect many components without explicit approval
- **RECENT EXAMPLE**: After successfully implementing backOff for scanBoxId retry logic, agent started investigating other retry patterns in the file - this was scope creep beyond the specific task requirement
- **CRITICAL: Deprecated Code Alert**: ALWAYS ALERT WHEN WORKING WITH DEPRECATED CODE - Stop and ask for team discussion or explicit approval before modifying deprecated code

## ALIX Coding Strategies (From PR Reviews)

### Database-First Philosophy
- **Prefer database-native solutions** over application-level complexity
- **Use PostgreSQL sequences** for auto-incrementing values when appropriate
- **Leverage database functions** like `lpad()` and `to_hex()` for formatting
- **Use `DEFAULT` values** to handle automatic generation at the database level
- **Prefer declarative solutions** over imperative application code
- **Consider the database as part of the solution** rather than just storage

### Simplicity Over Complexity
- **Keep implementations simple** - if the database can handle it, let it
- **Avoid over-engineering** when database features can solve the problem elegantly
- **Prefer simpler implementations** over complex custom solutions
- **Avoid creating services for simple operations** that can be handled inline
- **Consider if a service adds value** or just adds unnecessary complexity
- **Use built-in JavaScript methods** instead of manual character iteration
- **Avoid unnecessary complexity** - if a one-liner can achieve the same result, prefer it

### Prisma ORM Standards
- **Use Prisma ORM methods** instead of raw SQL for simple database operations
- **Prefer `tx.modelName.find()`** over `tx.$queryRaw` when the operation is straightforward
- **Use Prisma's fluent API** instead of raw SQL for database operations
- **Leverage Prisma's query optimization** capabilities
- **Use Prisma's type-safe query builders** for better maintainability
- **Reserve raw SQL** only for complex operations that cannot be expressed with Prisma ORM
- **Maintain consistency** with the codebase's ORM usage patterns

### Database Schema Standards
- **All models primary/foreign keys must be in UUID format** (`id String @id @default(uuid())`)
- **Use appropriate database data types** to enforce constraints rather than relying on CHECK constraints
- **Avoid creating separate counter tables** when the unique constraint on the target field already provides the necessary uniqueness guarantee
- **Follow established naming conventions** for database tables (not snake_case)
- **Use database transactions** for operations that must be atomic
- **Include ID generation in the same transaction** as the main operation

### Code Quality and Maintenance
- **Remove unused files** when implementation changes make them obsolete
- **Keep codebase clean** by eliminating dead code
- **Avoid obvious comments** that simply restate what the code is doing
- **Comments should add value** by explaining "why" not "what"
- **Remove unnecessary error handlers** that handle edge cases that shouldn't occur in production
- **Use database seeding** instead of runtime initialization methods for initial data
- **Don't use error handling as migration reminders** - handle migrations properly during deployment

### Service Architecture Patterns
- **Use instance-based services** instead of static methods when following service pattern
- **Initialize dependencies in constructor** for better organization and maintainability
- **Follow dependency injection pattern** for better testability and modularity
- **Reserve services for complex business logic** that requires multiple steps or external dependencies
- **Consider multiple approaches** before settling on a complex solution
- **Evaluate external libraries** like CUID2 for ID generation when appropriate

### Error Handling and Business Logic
- **Consider business impact** of error conditions and limits
- **Plan for edge cases** that could block core business functionality
- **Ensure graceful degradation** or alternative approaches when limits are reached
- **Database schema issues** should be handled through proper migrations during deployment, not through runtime error handling
- **The application should assume the database schema is correct and up-to-date**

### Code Analysis Requirements
- **MUST examine git history** for each field type to see actual implementation approach
- **MUST check backend services** not just GraphQL schema - business logic lives in services
- **MUST verify frontend mutations** request all needed fields in response
- **MUST understand conditional rendering patterns** for UI behavior decisions



## Critical Commands

### Backend Startup
```bash
# Switch to Node 23
export NVM_DIR="$HOME/.config/nvm" && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && nvm use 23

# Start backend in background
yarn start
```

### Frontend Startup
```bash
# Install dependencies
yarn install

# Start development server
yarn dev
```

### Database Operations
```bash
# Check migration status
npx prisma migrate status

# Apply pending migrations
npx prisma migrate deploy

# Regenerate Prisma client
npx prisma generate

# Create database backup
pg_dump -h localhost -p 5432 -U patrickclawson -d alix > alix_backup.sql
```

### Service Management
```bash
# Kill all backend processes
pkill -f "ts-node-dev"
pkill -f "yarn start"

# Check running services
ps aux | grep -E "(yarn|ts-node|npm)"
```

---

## 2026-04-08 Linear-first technical companion pattern

- For new project-management work in this repo, Linear is the execution source of truth and local markdown should act only as a repository-specific technical companion.
- The approved Linear integration path is the `plugin-linear-linear` MCP server.
- A good default structure is: Linear project -> program issue -> phase parent issues -> actionable child issues, with active progress recorded in issue comments.
- Local technical docs should follow the `IMPLEMENTATION_DOC_*` companion style and avoid duplicating live status, ownership, or checklist tracking from Linear.

*Last updated: 2026-04-08*
*This file contains cross-cutting insights useful across multiple implementations and sessions.*