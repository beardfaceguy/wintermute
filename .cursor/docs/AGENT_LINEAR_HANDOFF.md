# Linear-First Handoff For Another Agent

## Purpose
Use this instruction set if you are continuing work in this environment after the team switched from local markdown task tracking to Linear-first execution.

Linear is now the source of truth for project execution.
Local markdown docs are now only technical companions.

## Core Rule
- Read Linear first.
- Work from Linear issues.
- Update Linear as you make progress.
- Do not use local implementation docs as the primary tracker.

## Which Linear Setup To Use
- Use the Linear MCP plugin server: `plugin-linear-linear`
- Do not prefer any duplicate or legacy Linear MCP unless you are explicitly troubleshooting access
- Assume Linear MCP auth may already be cached in Cursor
- If browser automation is required for Linear settings, you may still need a live logged-in browser session even if MCP access works

## Required Working Order
1. Open the relevant Linear project.
2. Open the top-level program issue for that project.
3. Open the relevant phase/workstream parent issue.
4. Read the active child issue(s).
5. Read the most recent comments before making changes.
6. Only after that, read local repo docs for technical context.

## Status Accuracy Rules
- Open issues must not be written as if their outcomes are already achieved.
- Use `Done when` for completion criteria, not present-tense completion language.
- Use `Current status` or `Remaining work` for what is true now.
- If an issue is `In Progress`, `Planned`, `Blocked`, or `Backlog`, its description should still read as unfinished work.
- If you are translating an older markdown plan into Linear, capture historical completed work under completed phases/issues and describe only unresolved work under the active phase.

## Comment Scope Rules
- In a Linear comment, the `Done:` section must only include work that belongs to the current issue or phase.
- Do not restate accomplishments from earlier phases as if they are progress on the active phase.
- If earlier work matters for context, reference the earlier issue instead of counting it as new progress here.
- Keep current-phase comments focused on the current workstream's status, next steps, and blockers.

## What Belongs In Linear
- Status
- Assignee / ownership
- Due dates
- Blockers
- Progress updates
- Task breakdown
- Milestones / timeline

## What Belongs In Local Docs
- Repo-specific technical notes
- File paths and touchpoints
- Environment gotchas
- Commands worth reusing
- Architecture or implementation notes that would help a future agent

## What Must Not Happen
- Do not maintain a second checklist in markdown if the work already exists in Linear
- Do not treat `IMPLEMENTATION_*.md` files as the source of truth for progress
- Do not copy issue-by-issue status from Linear into local docs
- Do not leave work completed in code without updating the related Linear issue
- Do not write an open issue as if it is already complete
- Do not mix earlier-phase accomplishments into the active phase's status comment

## Preferred Statuses
- `Backlog`
- `Planned`
- `In Progress`
- `Blocked`
- `Done`

## Required Comment Format
When you make meaningful progress, leave a concise Linear comment like this:

```md
Status: In Progress

Done:
- Completed item

Next:
- Next item

Blockers:
- None

Links:
- PR/doc/issue references
```

## Verification Step After Linear Updates
- After updating a Linear issue, re-read:
  - the issue description
  - the latest comment
  - the active child issues
- Confirm they all tell the same story.
- Fix any wording that overstates completion, mixes phases, or contradicts the live child issue states.

## Current Linear Projects

### 1. Linear adoption project
- Project: `Cursor + Linear PM Adoption`
- Team: `Clawcorp` (`CLA`)
- Program issue: `CLA-5`
- Phase parents:
  - `CLA-15` Phase 1 workflow design and taxonomy
  - `CLA-14` Phase 2 Cursor + Linear integration validation
  - `CLA-13` Phase 3 pilot rollout and adoption
- Stage 2 pilot issue: `CLA-17`

Use this project if you are working on the workflow itself.

### 2. AE-2002 project
- Project: `AE-2002 - One-click Docker environment for Alix repos`
- Program issue: `CLA-19`
- Historical phases already captured:
  - `CLA-23` Phase 1 PM local setup automation (`Done`)
  - `CLA-22` Phase 2 quick-launch infra and app orchestration (`Done`)
  - `CLA-21` Phase 3 idempotency hardening (`Done`)
  - `CLA-20` Phase 4 runtime resilience hardening (`Done`)
- Active remaining phase:
  - `CLA-24` Phase 5 PM docs and account sync finalization (`In Progress`)

## AE-2002 Active Work
If you are working on AE-2002, start with `CLA-24` and its child issues:

- `CLA-30` Finalize PM operator runbook in README
- `CLA-25` Finalize PM staging test-account list and mapping requirements
- `CLA-29` Decide whether Cognito PM user sync should be automated or documented as a runbook
- `CLA-27` Implement or validate Cognito user-pool sync for PM accounts
- `CLA-28` Validate fallback database backup option end to end
- `CLA-31` Add explicit post-restore data integrity verification queries
- `CLA-26` Optional hardening: reuse dev-process sweep logic in quick-launch-up conflict path

## Known AE-2002 Reality
- AE-2002 has already been translated into Linear-first structure
- Historical completed work is already represented in Linear
- The remaining real blocker is around PM account list / mapping and Cognito-side sync decisions
- If you make progress on AE-2002, update `CLA-24` and the specific child issue you touched

## Local Files To Read Only After Linear
- `CURSOR_README.md`
- `.cursor/docs/IMPLEMENTATION_DOC_LINEAR.md`
- `.cursor/memory/memory.md`
- In AE-2002 work specifically: the relevant local implementation doc only as a technical companion, not as the execution tracker

## Decision Rule
If you are unsure whether an update belongs in Linear or markdown:
- Put execution tracking in Linear
- Put repo-specific technical context in markdown

## Handoff Standard
Before ending your session:
- Make sure the relevant Linear issue state is correct
- Leave a short Linear comment using the required format
- Update local docs only if you discovered technical context that future agents would otherwise miss

## Short Version
Read Linear first.
Use `plugin-linear-linear`.
Work from the issue hierarchy.
Comment progress in Linear.
Use local docs only for technical context.
