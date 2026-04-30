# IMPLEMENTATION_DOC_TITAN_BRANCH_EXECUTION - Branch Split Board
- Role: Repository-local execution board for the Titan Branch A / Branch B split
- Execution source of truth: Linear project and issues, not this file
- Last updated: 2026-04-12
- Owner / editors: Cursor Agent, Project Team

## Purpose
- This file captures the agreed branch split for the Titan GPT-small effort after the current scratch base failed multiple SFT-quality gates.
- It exists to keep the umbrella Titan project readable while still giving agents one local place to understand the two-branch execution model.
- Linear remains canonical for live status, ownership, due dates, blockers, and comments.

## Branch Model
- Branch A: `Titan Branch A - GPT-2 Bootstrap Validation`
  - Goal: prove the training, SFT, prompting, and evaluation workflow can produce a working small assistant when the base model is already competent.
  - Non-goal: become the final model lineage if full behavioral control is still a hard requirement.
- Branch B: `Titan Branch B - Controlled GPT-Small From Scratch`
  - Goal: build the final model from a base whose behavior comes from operator-chosen pretraining data and objectives.
  - Non-goal: keep re-polishing the old scratch checkpoint lineage without changing the base-model plan.

## Relationship Between Branches
- Branch A is the workflow-proof branch.
- Branch B is the product-intent branch.
- Required ordering:
  - complete Branch A first
  - freeze one known-good Branch A workflow
  - apply that exact workflow to Branch B after the scratch base passes its own pre-SFT gates
- Shared evaluation rule:
  - use the same prompt suite, decode settings, and qualitative gate across both branches
  - do not move the goalposts between Branch A and Branch B

## Linear Structure
- Umbrella project: `Titan GPT-small Pretraining Stabilization`
- Program issue: `CLA-33`
- Branch projects:
  - `Titan Branch A - GPT-2 Bootstrap Validation`
  - `Titan Branch B - Controlled GPT-Small From Scratch`
- Project hygiene rule:
  - umbrella project gets summaries and cross-branch decisions
  - branch projects get execution noise, experiments, blockers, and iteration details

## Branch A Phase Board
### Phase A1 - GPT-2 base-model integration
- `A1.1` Add GPT-2 small weight-loading path to `titanProject`
- `A1.2` Add GPT-2 medium weight-loading path to `titanProject`
- `A1.3` Add GPT-2-compatible configs and checkpoint lineage naming
- `A1.4` Verify tokenizer/model compatibility for GPT-2 bootstrap

### Phase A2 - Homogeneous instruction-format SFT pipeline
- `A2.1` Add Raschka-style instruction-template prep path
- `A2.2` Preserve formatting and newlines in instruction SFT data
- `A2.3` Add branch-specific clean train/val split for instruction tuning
- `A2.4` Wire masked SFT loader to the new instruction format

### Phase A3 - Local workflow proof on GPT-2 small
- `A3.1` Run local GPT-2 small SFT smoke on a tiny instruction shard
- `A3.2` Run the fixed qualitative prompt suite on the local smoke checkpoint
- `A3.3` Fix workflow or data-format bugs surfaced by the local smoke

### Phase A4 - AWS bounded GPT-2 bootstrap pilots
- `A4.1` Launch bounded AWS GPT-2 small SFT pilot
- `A4.2` Evaluate GPT-2 small pilot on the fixed prompt suite
- `A4.3` Launch bounded AWS GPT-2 medium fallback pilot if needed
- `A4.4` Compare GPT-2 small vs GPT-2 medium bootstrap quality

### Phase A5 - Freeze Branch A reference recipe
- `A5.1` Freeze Branch A reference config and prompt format
- `A5.2` Record Branch A promotion gate and expected qualitative outputs
- `A5.3` Document Branch A as the reference SFT workflow for Branch B

## Branch A Success Gate
- At least one GPT-2 bootstrap path produces a mostly coherent assistant on the fixed prompt suite.
- Simple arithmetic, list-format prompts, and basic factual prompts are no longer degenerating into loops.
- Prompt formatting is stable and there is no obvious turn leakage or malformed continuation collapse.

## Branch B Phase Board
### Phase B1 - Scratch restart baseline definition
- `B1.1` Define a fresh scratch lineage and retire the current base from the main path
- `B1.2` Freeze the plain GPT decoder architecture for the scratch restart
- `B1.3` Define tokenizer policy for the controlled scratch branch
- `B1.4` Define restart configs and naming rules

### Phase B2 - Controlled pretraining corpus and boundaries
- `B2.1` Audit current pretraining corpus assumptions and contamination risks
- `B2.2` Add explicit document or EOS boundary handling to the pretraining stream
- `B2.3` Define a control-first pretraining corpus subset
- `B2.4` Add corpus metadata and provenance tracking for Branch B

### Phase B3 - Scratch pretraining quality gates
- `B3.1` Add base continuation evaluation harness before SFT
- `B3.2` Define scratch-base promotion thresholds
- `B3.3` Run local scratch pretraining smoke and continuation checks
- `B3.4` Run bounded AWS scratch pretraining validation

### Phase B4 - Apply Branch A reference SFT recipe to the scratch base
- `B4.1` Port the Branch A instruction-format prep path onto the scratch base
- `B4.2` Run bounded scratch-base SFT pilot with the Branch A recipe
- `B4.3` Evaluate the scratch-base SFT checkpoint on the fixed prompt suite
- `B4.4` Compare the scratch-base SFT result directly against the Branch A bootstrap baseline

### Phase B5 - Decide Branch B promotion or further pretraining iteration
- `B5.1` Decide whether Branch B is good enough to replace Branch A
- `B5.2` If not, choose the next lever explicitly: corpus, scale, or pretraining duration
- `B5.3` Record Branch B lessons and the next restart rules

## Branch B Success Gate
- The scratch base produces coherent raw continuations before any SFT begins.
- The frozen Branch A post-training recipe improves that base into a usable small assistant.
- The resulting behavior can be traced back to operator-chosen pretraining data and post-training data rather than inherited GPT-2 priors.

## First Active Issues
- Branch A:
  - `A1.1` Add GPT-2 small weight-loading path to `titanProject`
  - `A2.1` Add Raschka-style instruction-template prep path
  - `A3.1` Run local GPT-2 small SFT smoke on a tiny instruction shard
- Branch B:
  - open `B1.1` early so the lineage split is explicit, but keep it pending until Branch A freezes a reference workflow

## Operating Notes
- Do not continue spending major effort on the old `gpt_small_pretrain_20260411004641` lineage as the assistant base candidate.
- If Branch A fails on both `gpt2-small` and `gpt2-medium`, assume the workflow or data path is still broken before blaming the scratch base again.
- If Branch A succeeds and Branch B fails, assume the remaining problem is base-model quality rather than SFT formatting.
