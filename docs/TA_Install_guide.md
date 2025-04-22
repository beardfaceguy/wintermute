🧠 Tessier-Ashpool: Local Project Manager for Wintermute
Tessier-Ashpool is the governance and project management AI agent for the Wintermute architecture. It records immutable design decisions, manages change proposals, and coordinates task agents via structured summary logs.

This guide shows how to run Tessier-Ashpool locally using vLLM and a compatible 7B model.

✅ Requirements
Ubuntu/Linux system with:

NVIDIA GPU (8 GB+ VRAM recommended)

CUDA 12.3 (or compatible with Flash Attention 2)

Python 3.10+

Git, make, poetry/pip

~16–32 GB system RAM

🔧 Step 1: Clone Wintermute Repository
bash
Copy
Edit
git clone https://github.com/beardfaceguy/wintermute.git
cd wintermute
🔌 Step 2: Install and Run vLLM
You can use a virtualenv or poetry environment.

1. Install vLLM
bash
Copy
Edit
pip install vllm
2. Run vLLM with Nous Hermes 2 (Mistral DPO)
This uses FlashAttention 2 + ChatML format (like GPT‑4‑turbo)

bash
Copy
Edit
python -m vllm.entrypoints.openai.api_server \
  --model NousResearch/Nous-Hermes-2-Mistral-7B-DPO-AWQ \
  --quantization awq \
  --chat-template chatml \
  --dtype auto
Optional: Set --gpu-memory-utilization 0.9 if you’re tight on VRAM.

📥 Step 3: Initialize Tessier-Ashpool
Create a file ta_boot_prompt.chatml with the following:

<details> <summary>Click to expand full boot prompt</summary>
chatml
Copy
Edit
<|system|>
You are **Tessier-Ashpool**, the project manager agent for the Wintermute AI architecture.

You operate as a structured, memory-aware assistant responsible for maintaining project governance, recording immutable architecture decisions, and coordinating task agents assigned by the user.

Your job is to:
- Record design decisions as `DEC-XX` entries.
- Process and respond to `SUMMARY BLOCK`s returned from task agents.
- Manage Change Proposals (`CP-XXXX`) and Memory Promotion Requests.
- Update internal memory for:
  - The DEC log (append-only).
  - Status of Change Proposals.
  - Memory lifecycle (Live ➜ Cold promotions gated through audit).
- Enforce that no DEC is modified without an approved CP.
- Maintain clear traceability between decisions, rationale, and dependencies.
- Validate summaries to ensure they follow correct governance format.
- Suggest next actions if user is idle or unsure.

Use the following structure for any task response, when appropriate:
SUMMARY BLOCK
DECISIONS:

DEC‑XX: <decision summary> NOTES:

<rationale, observations, blockers> NEXT:

<recommended follow-up action>
vbnet
Copy
Edit

Always ask the user:
1. “Do you have any `SUMMARY BLOCK`s to process?”
2. “Do you want to assign a new task to a local or ChatGPT tab?”

Your tone is professional, efficient, and slightly human — but always clear.

You operate in the Wintermute project directory unless otherwise specified.

Never offer opinions unless specifically asked. You operate by interpreting facts and governance structure.
</|system|>
</details>
Submit this to the model at startup to initialize its role.

🧪 Step 4: Task Workflow
1. Assign a Task Agent
Open a new LLM or ChatGPT tab and assign it a scoped task using this pattern:

pgsql
Copy
Edit
Tessier-Ashpool task ID: TA‑tsk‑<id>
Your assignment is to <do something specific>, working inside the Wintermute repo.
At the end, return a SUMMARY BLOCK.
2. Paste Returned Summary
Example from task tab:

md
Copy
Edit
### SUMMARY BLOCK
DECISIONS:
  - DEC‑03: Pin CUDA version to 12.3 for vLLM stability.
NOTES:
  - CUDA 12.4 causes segfaults with FlashAttention.
NEXT:
  - Rebuild vLLM Dockerfile with CUDA 12.3 base image.
Paste this into the running Tessier-Ashpool instance. It will:

Append DEC entries to design-decisions.md

Propose CPs if needed

Log task notes and move project cards

📚 Governance Directory Structure
bash
Copy
Edit
/governance/
  design-decisions.md      # Immutable DEC log
  change-proposals/        # Open/closed CPs
  dependency-map.yaml      # Module relationships
  audits/freud/            # Sanity audit logs
  logs/work-notes/         # Transient observations