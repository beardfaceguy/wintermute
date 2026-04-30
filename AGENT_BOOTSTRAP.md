# Agent Bootstrap

Start here. This file tells you what to read and where to find project state.

## Required Reading (in order)

1. **`CURSOR_README.md`** — Project overview, directory map, current status, prerequisites, and next steps.
2. **`.cursor/docs/Wintermute_Master_Spec.md`** — Architecture vision, component descriptions, and the full roadmap.
3. **`.cursor/docs/IMPLEMENTATION_MCP_CORE.md`** — What was built in the most recent session: MCP servers, agent runner, SQL agent, file inventory, design decisions, and what comes next.

## Linear (Project Tracker)

All task tracking lives in **Linear** under the **Clawcorp** workspace.

### How to connect

Linear is available via the built-in Cursor MCP server `plugin-linear-linear`. You do not need to install anything — just call the MCP tools directly.

### Key commands

| Action | MCP Tool | Example Arguments |
|--------|----------|-------------------|
| List active Wintermute issues | `list_issues` | `{"project": "Wintermute Platform", "state": "started"}` |
| List planned issues | `list_issues` | `{"project": "Wintermute Platform", "state": "unstarted"}` |
| Get full issue details | `get_issue` | `{"id": "CLA-140"}` |
| Update an issue | `save_issue` | `{"id": "CLA-140", "state": "In Progress"}` |
| List all projects | `list_projects` | `{}` |

### Active Wintermute issues

| ID | Title | Status |
|----|-------|--------|
| CLA-136 | [Program] Wintermute Core MCP Infrastructure | In Progress |
| CLA-143 | Train GPT-Medium (407M) — pretrain + SFT | Done (pretraining + SFT complete 2026-04-30) |

Check Linear for current active issues — the table above is a snapshot and may be stale.

### Other Linear projects (not active in this workspace)

- **Titan GPT-small Pretraining Stabilization** — completed/closed
- **bbhmm** — crypto trading bot (separate workspace)
- **Hosting Options** — GPU hosting research

## Optional Reading (load on demand)

These files are only needed if the user explicitly asks to resume the related work:

- **`.cursor/docs/IMPLEMENTATION_AWS_titan_llm_model_training.md`** — AWS training infrastructure history
- **`.cursor/docs/llm_training_project.md`** — LLM training project summary and results
- **`.cursor/docs/TRAINING_RUN_GPT_MEDIUM_20260419.md`** — GPT-Medium 407M run log
- **`.cursor/docs/SFT_PIPELINE_GUIDE.md`** — SFT data formats, config, and pipeline reference
- **`.cursor/docs/AGENT_LINEAR_HANDOFF.md`** — Linear-first agent working order
- **`.cursor/docs/archive/`** — Superseded historical docs (GPT-small details, early AWS plan, Tessier-Ashpool vision)
