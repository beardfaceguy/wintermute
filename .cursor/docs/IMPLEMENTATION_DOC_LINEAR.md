# IMPLEMENTATION_DOC_LINEAR - Local Technical Companion
- Role: Repository-local technical companion for the Linear adoption effort
- Execution source of truth: Linear project and issues, not this file
- Last updated: 2026-04-30
- Owner / editors: Cursor Agent, Project Team

## Purpose
- This file exists to give agents repo-specific context that does not fit cleanly into Linear fields.
- Linear is the canonical tracker for status, ownership, due dates, and progress updates.
- This file should stay concise and technical. It should not become a second project tracker.

## Linear Source Of Truth
- Project: `Cursor + Linear PM Adoption`
- Team: `Clawcorp` (`CLA`)
- Program issue: `CLA-5` - Cursor + Linear PM adoption tracker
- Phase parents:
  - `CLA-15` - Phase 1 workflow design and taxonomy
  - `CLA-14` - Phase 2 Cursor + Linear integration validation
  - `CLA-13` - Phase 3 pilot rollout and adoption
- Timeline:
  - Original target: 2026-04-14 to 2026-04-24
  - Linear adoption is operationally active; verify current milestone status in Linear.
- Stage 2 pilot target:
  - `AE-2002 - One-click Docker environment for Alix repos`
  - Program issue: `CLA-19`

## How Agents Should Work
- Start in Linear first:
  - Read the project description.
  - Open the program issue and phase issue relevant to the task.
  - Read active child issues and recent comments.
- Use this file second:
  - Pull repo-specific context, file paths, and workflow notes from here.
  - Add only technical notes or handoff information that would help future agents in this repository.
- Update work in Linear, not here:
  - Move issue state in Linear.
  - Add comments in Linear for progress updates, blockers, and next steps.
  - Only update this file when repository-local context changes.

## Recommended Linear Taxonomy
- Project:
  - Holds goal, scope, target date, and high-level timeline.
- Parent issue:
  - Represents a phase or workstream.
- Child issue:
  - Represents a concrete actionable task with owner, priority, due date, and definition of done.
- Comment:
  - Represents agent or human progress updates.
- Preferred statuses:
  - `Backlog`
  - `Planned`
  - `In Progress`
  - `Blocked`
  - `Done`

## Agent Update Format
- Use short issue comments in Linear with this structure:

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

## Repository-Specific Context
- Approved Linear integration path is the plugin MCP server: `plugin-linear-linear`.
- The legacy duplicate Linear MCP server should stay disabled unless there is a specific troubleshooting reason to re-enable it.
- Linear auth may already be cached in Cursor, so enabling the plugin may not trigger a new auth prompt.
- MCP auth and browser-session auth are separate practical concerns:
  - MCP calls to `plugin-linear-linear` worked without an interactive auth prompt once access was available.
  - Browser automation against Linear settings required a live browser login session and manual user login before team workflow settings could be changed.
- This setup currently assumes a single human operator: Patrick.
- The local `linear/docs` files in this repository are useful background material, but they are not the operational source of truth for project status.

## Local Files Worth Checking
- `CURSOR_README.md`
  - Project-level Cursor rules and `.cursor` guidance.
- `.cursor/docs/IMPLEMENTATION_DOC_LINEAR.md`
  - This local technical companion.
- `.cursor/docs/implementation_doc_template.md`
  - Template for future local technical companion docs.
- `.cursor/memory/memory.md`
  - Cross-cutting reusable knowledge when relevant.

## Technical Notes
- No repository code changes are currently required for the Linear migration itself.
- The main implementation work is process and integration validation:
  - confirm workflow/status mapping
  - validate Cursor read/write flows against Linear
  - document single-user auth behavior and agent access assumptions
  - run the pilot and capture follow-up improvements
- Stage 2 pilot translation status:
  - AE-2002 has been translated from its implementation doc into a dedicated Linear project.
  - Historical completed phases were represented as Done parent issues.
  - Remaining work is concentrated under Phase 5 with active issues for PM docs, Cognito sync decisions, fallback backup validation, and post-restore integrity checks.

## Auth And Access Assumptions
- Human operator model:
  - Patrick is the only human expected to operate this Linear setup.
- Approved agent access path:
  - Use `plugin-linear-linear` for Linear MCP operations.
- Stable assumptions future agents can rely on:
  - Linear project and issue read/write flows are working through the plugin MCP.
  - Cached MCP auth may allow access without re-prompting in Cursor.
  - Browser-based settings changes may still require an active logged-in browser session.
- Out of scope:
  - Multi-user onboarding flows
  - Team-wide access provisioning documentation for additional human contributors

## Open Technical Questions
- Should agents always update Linear comments directly, or should some updates also be summarized in repository docs for auditability.
- Do we need any custom Linear labels for phases, or is the parent issue hierarchy sufficient.
- Do we want a dedicated repo rule that tells all agents to consult Linear before reading local implementation docs.

## Handoff Checklist
- Confirm the relevant Linear issue is linked to the correct parent phase.
- Leave a concise Linear comment with `Status`, `Done`, `Next`, and `Blockers`.
- Update this file only if you discovered technical context that future agents would otherwise miss.
- Do not duplicate issue-by-issue status here.

## Changelog
- [2026-04-06] Reframed this file from a task tracker into a repo-local technical companion with Linear as the execution source of truth.
- [2026-04-06] Added project hierarchy, agent workflow guidance, and a standard Linear comment format.
- [2026-04-06] Documented the verified single-user auth model, browser-vs-MCP auth behavior, and final preferred status taxonomy.
- [2026-04-06] Recorded AE-2002 as the selected Stage 2 pilot and noted its translated Linear project structure.
