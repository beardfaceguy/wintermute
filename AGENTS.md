# Agent Bootstrap

Start here. This file tells you what to read and where to find project state.

## Required Reading (in order)

1. **`CURSOR_README.md`** — Project overview, directory map, current status, prerequisites, and next steps.
2. **`.cursor/docs/Wintermute_Master_Spec.md`** — Architecture vision, component descriptions, and the full roadmap.
3. **`.cursor/docs/IMPLEMENTATION_MCP_CORE.md`** — What was built in the most recent session: MCP servers, agent runner, SQL agent, file inventory, design decisions, and what comes next.

## Vikunja (Project Tracker)

All task tracking lives in **Vikunja**, a self-hosted instance running at https://vikunja.wint3rmute.com.

### How to connect

Vikunja is available via the `vikunja` MCP server (tools prefixed `vikunja-vikunja_*`). The server is registered in Cursor's MCP config and points at the local instance using a `tk_…` API token for the `beardface` user.

### Key commands

| Action | MCP Tool | Example Arguments |
|--------|----------|-------------------|
| List projects | `vikunja_projects` | `{"subcommand": "list"}` |
| Get a project | `vikunja_projects` | `{"subcommand": "get", "id": <project_id>}` |
| List tasks in a project | `vikunja_tasks` | `{"subcommand": "list", "projectId": <project_id>}` |
| Get a task | `vikunja_tasks` | `{"subcommand": "get", "id": <task_id>}` |
| Update a task | `vikunja_tasks` | `{"subcommand": "update", "id": <task_id>, "done": true}` |
| Create a task | `vikunja_tasks` | `{"subcommand": "create", "projectId": <project_id>, "title": "..."}` |
| Add a comment | `vikunja_tasks` | `{"subcommand": "comment", "id": <task_id>, "comment": "..."}` |

### Workflow

Before starting work, check Vikunja for relevant tasks. If none exist, create one. Mark tasks in progress as you work and done when complete. The user should be able to open Vikunja at any time and see an accurate picture of what's done, what's in progress, and what's next.

Check Vikunja for current active tasks — do not rely on stale snapshots in this file.

### Vikunja MCP server quirks

The Vikunja MCP server ([democratize-technology/vikunja-mcp](https://github.com/democratize-technology/vikunja-mcp)) has a few sharp edges around partial updates. Treat `vikunja_projects.update` and `vikunja_tasks.bulk-update` as **full replace, not partial patch**.

- **`vikunja_projects.update` requires `title`.** Calling `update` with only `id` + `description` (or any other subset that omits `title`) returns `Invalid Data`. Always include the existing `title` even when you don't intend to change it.
- **`vikunja_projects.update` resets `parent_project_id` to `0` if you don't pass `parentProjectId`.** A child project will silently become a top-level project. Always pass `parentProjectId` when updating any field on a child project — fetch the current parent first if you don't already know it.
- **`vikunja_tasks.bulk-update` wipes other fields.** Calling `bulk-update` with `field: "done"`, `value: true` clears `description` and `priority` on every targeted task. Either:
  - Re-apply lost fields with per-task single `update` calls afterward (single `update` correctly preserves omitted fields, including `done`), or
  - Avoid `bulk-update` entirely and use parallel single `update` calls.

All three are the same root cause: the server sends a partial PATCH that Vikunja treats as a full object replace, so omitted fields get cleared. Upstream tracking:

- [#44](https://github.com/democratize-technology/vikunja-mcp/issues/44) — `vikunja_projects.update` requires `title`
- [#45](https://github.com/democratize-technology/vikunja-mcp/issues/45) — `vikunja_projects.update` resets `parent_project_id`
- [#46](https://github.com/democratize-technology/vikunja-mcp/issues/46) — `vikunja_tasks.bulk-update` wipes other fields
- [#37](https://github.com/democratize-technology/vikunja-mcp/issues/37) — same family on the **task-update** path (silent project moves; `labels` ignored on `create`)

## Optional Reading (load on demand)

These files are only needed if the user explicitly asks to resume the related work:

- **`.cursor/docs/IMPLEMENTATION_AWS_titan_llm_model_training.md`** — AWS training infrastructure history
- **`.cursor/docs/llm_training_project.md`** — LLM training project summary and results
- **`.cursor/docs/TRAINING_RUN_GPT_MEDIUM_20260419.md`** — GPT-Medium 407M run log
- **`.cursor/docs/SFT_PIPELINE_GUIDE.md`** — SFT data formats, config, and pipeline reference
- **`.cursor/docs/AGENT_LINEAR_HANDOFF.md`** — (legacy, Linear-era) agent working order — superseded by Vikunja workflow above
- **`.cursor/docs/archive/`** — Superseded historical docs (GPT-small details, early AWS plan, Tessier-Ashpool vision)
