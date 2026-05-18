# Wintermute

Wintermute takes its name from the AI in *Neuromancer*. The ambition is broader than a chatbot: a **modular AI that can accumulate skill from real workflows**, check its own reasoning where it matters, and **improve over time** without collapsing into brittle one-off hacks.

Full architecture, roadmap nuance, and implementation history live in the docs linked below—this file is the **north star**, not an inventory.

## Vision

The master spec summarizes the ultimate goal:

1. **Learn through interaction** with real-world systems (**MCP** tools: databases, memory, repos, shell where appropriate)—not only from static corpora.

2. **Verify reasoning and behavior** with **test-driven and structured checks**, so successes and failures are grounded in reproducible signals.

3. **Persist what works** in **long-term memory** (semantic / RAG-oriented storage): strategies, rationales, and dead ends worth avoiding—not just conversational logs.

4. **Evolve the system**: swap components, evaluate models, refine training and data (**wintermute-trainer-style** abstraction from thought paths)—so improvement is deliberate and observable.

Together, that describes **recursive refinement with guardrails**: the system repeatedly tries, checks, remembers, and (when warranted) retrains or reconfigures—under human-defined boundaries.

## Sanity, memory tiers, and failsafes

The spec layers **Freud**, **Jung**, and **Adler** as auditors (contradiction and quality, symbolic drift, goal alignment)—with a trajectory from exploratory **Live** memory to vetted **Cold** memory rather than trusting every embedding equally.

Separately, **Blade Runner** sketches **oversight and last-resort control** so autonomy does not quietly outrun intentional limits. These pieces are roadmap and design scaffolding; maturity varies by component.

## This repository

The codebase anchors on a **FastAPI / vLLM** backend, **`talkingHead`** web UI, **MCP** servers (**mcp-memory**, **mcp-postgres**, etc.), an **AgentRunner** for tool-using loops, a **Freud-side** auditing direction, and **`model_training/titanProject/`** (**Titan**, pretraining, SFT, and related harnesses). Exact layout and status change often.

**Labs:** The same checkout is deliberately used as a **sandbox for new AI mechanics**, not only the Wintermute-shaped spine above. Recent threads include **HRMs** (hierarchical reasoning models), **DAGs** (directed acyclic graphs) structuring plans and pipelines, and Titan-scale **training-from-scratch and fine-tuning** experiments beside the MCP stack. Some of that lands as exploratory code paths or notebooks with shorter half-lives—treat unclear corners as prototypes unless a doc ties them back to shipped behavior.

## Documentation

| Document | Purpose |
| -------- | ------- |
| [`CURSOR_README.md`](CURSOR_README.md) | Cursor agent onboarding, directory map, current status snapshot |
| [`.cursor/docs/Wintermute_Master_Spec.md`](.cursor/docs/Wintermute_Master_Spec.md) | Vision, modules, auditors, Blade Runner oversight, roadmap detail |
| [`AGENTS.md`](AGENTS.md) | Tracker (Vikunja) workflow and bootstrap reading order |

For day-to-day development, **start with `CURSOR_README.md`**, then drill into `.cursor/docs/` as needed.
