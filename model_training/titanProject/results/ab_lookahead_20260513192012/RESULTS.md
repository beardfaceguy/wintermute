# MAC lookahead vs baseline — 25k-step A/B (Vikunja #186)

**Run ID:** `20260513192012`
**Hardware:** g6.2xlarge spot (1× L4, 24 GiB), us-east-1c
**Wall clock:** Run A ~1h 59m, Run B ~1h 57m (single instance, sequential)
**Code state:** branch `main` at the commit that introduced
`config_mac_{lookahead,baseline}_25k.yaml`
(`model_training/titanProject/scripts/run_ab_lookahead_ssm.sh`).
**Logs:** S3 `s3://alix-ai-ml-staging-data/titan/checkpoints/mac_{lookahead,baseline}_25k_20260513192012/`
also mirrored at `model_training/titanProject/logs/ab_lookahead_20260513192012/`
(gitignored, local reference).

## Question

For our standard MAC config (dim=384, depth=5, heads=6, segment_len=256,
seq_len=512, batch=4, 50M-token train budget, 2M-token val budget), does
turning on the `titans-pytorch` 0.5.3 lookahead-family flags
(`store_with_lookahead_value`, `neural_memory_add_value_residual`,
`neural_mem_gate_attn_output`) lower validation loss vs the same
arch with all three off? Same hyperparams everywhere else.

## Result

**No — baseline wins**, and the gap grows with training.

| step  | A: lookahead | B: baseline | Δ (B − A) | winner |
|------:|-------------:|------------:|----------:|--------|
|  2500 | 5.7532       | 5.8884      | +0.135    | A      |
|  5000 | 5.4219       | 5.4634      | +0.041    | A      |
|  7500 | 5.2509       | 5.2483      | −0.003    | tie    |
| 10000 | 5.1259       | 5.1047      | −0.021    | B      |
| 12500 | 5.0255       | 4.9897      | −0.036    | B      |
| 15000 | 4.9450       | 4.8968      | −0.048    | B      |
| 17500 | 4.8814       | 4.8275      | −0.054    | B      |
| 20000 | 4.8334       | 4.7768      | −0.057    | B      |
| 22500 | 4.8004       | 4.7423      | −0.058    | B      |
| **25000** | **4.7843** | **4.7244** | **−0.060** | **B** |

Final perplexity: **A=119.62, B=112.66** → baseline is **~7 ppl better** at the
end of training.

Throughput was identical: **~7,400 tok/s** sustained on both arms (so the
flags aren't paying for themselves in either quality or speed).

## Reading

* Lookahead is *slightly* better in the warmup phase (first ~5k steps), the
  curves cross around step 7.5k, and from 10k onward baseline pulls ahead
  monotonically — the spread is widening at 25k.
* The three flags are bundled together in this run, so we don't yet know
  which one is responsible. The ablation in
  `results/ablation_lookahead_*/` (next experiment) splits them out.
* Possible alternative explanations worth ruling out before declaring the
  features bad in general:
  * Insufficient steps/data — the lookahead pathway adds capacity that may
    only pay off at much higher token counts than 50M.
  * LR schedule isn't tuned for the modified gradient flow (flags change
    the neural-memory update path).
  * One of the three flags is harmful in isolation and it's dragging the
    other two down.

## Action items

* [done] Commit configs + runner (commit `91d7dc7`).
* [next] 3-arm ablation (one flag at a time) to identify the culprit —
  see `scripts/run_titan_arms_ssm.sh` and the
  `config_mac_{lookahead,residual,gate}_only_25k.yaml` configs.
* [bug] Self-terminate trap silently no-op'd at end of this run
  (instance idled ~2hr after `rc=0`). IMDSv2 token fetch returned empty;
  no fallback. Fix landed in `run_titan_arms_ssm.sh` before next launch
  (retries IMDSv2, falls back to `/var/lib/cloud/data/instance-id`,
  final fallback to `shutdown -h now`).
