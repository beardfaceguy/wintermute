Wintermute Master Spec (v0.2)

Project Codename: Wintermute

Inspiration: Named after the AI from Neuromancer, Wintermute seeks recursive self-improvement, merging learned strategies, system control, and modular tool use.

Vision

To build a modular AI system that:

Learns through interaction with real-world systems via MCP

Verifies its reasoning and behavior through test-driven logic

Stores successful strategies in long-term memory (via RAG)

Evolves itself over time through component replacement, model evaluation, and potential self-tuning

---

Tech Stack (v0.3)

## Core Components

| Layer               | Tech                           | Status       |
| ------------------- | ------------------------------ | ------------ |
| LLM Runtime         | vLLM                           | ✅ Confirmed |
| Model Types         | Wizard-Vicuna-7B-Uncensored-AWQ| ✅ Confirmed |
| Agent Framework     | LangChain                      | ✅ Confirmed |
| RAG Engine          | LlamaIndex                     | ✅ Confirmed |
| API Backend         | FastAPI                        | ✅ Confirmed |
| Custom UI           | React + Vite + Tailwind        | ✅ Confirmed |
| Prompting Interface | Open WebUI (for prototyping)   | ✅ Optional  |

### Tooling (via MCP)

| Tool          | MCP Server                | Status        |
| ------------- | ------------------------- | ------------- |
| SQL Access    | mcp-postgres              | ✅ Confirmed  |
| Memory Store  | mcp-memory (custom)       | 🚧 Planned    |
| Filesystem    | mcp-filesystem (optional) | ❓ Evaluating |
| External Data | Other MCP (GitHub, etc.)  | 🧪 Optional   |

### Experimental Layers

| Layer                | Component              | Status        |
| -------------------- | ---------------------- | ------------- |
| Verification Agent   | Secondary LLM (Critic) | ✅ Confirmed  |
| Model Profiler       | Hardware access tools  | 🧪 Future     |
| Self-Tuning Pipeline | wintermute-trainer     | ✅ Confirmed  |
| Autonomy Layer       | Shell/exec access      | ❓ Discussing |
| Sanity Auditors      | Freud, Jung, Adler     | 🧠 Planned    |

Core Components

Tooling (via MCP)

Experimental Layers

---

Key Projects & Modules

✅ Test-Driven SQL Generation Agent

Goal: Have the AI generate, test, and correct SQL queries based on known-good expectations

MCP: Postgres

Verification: Auto-check result accuracy vs. test case

Memory: Store both successful and failed queries, including the thought path used by the LLM to reach each conclusion

🚧 mcp-memory

Goal: Custom MCP server to store learned query strategies, test results, behavioral notes

Storage: LlamaIndex, Chroma, or file-backed vector store

Use: Long-term memory for RAG, includes indexing of failed paths, decision trees, and associated metadata

Structure: Split into:

Live Memory: Experimental, unverified strategies and recent outcomes

Cold Memory: Verified, high-confidence strategies vetted by sanity auditors

✅ wintermute-trainer

Goal: Analyze stored successful and failed thought paths to derive generalized thinking strategies

Purpose: Identify not just which queries succeed, but why — capturing the meta-patterns of thought (e.g., "for aggregation queries, the agent tends to succeed more when it explores grouping logic before filtering")

Outcome: Generates new training data, strategy metadata, or even refines model reasoning scaffolds

Trigger: Continuous background process, constantly reviewing new input as it enters mcp-memory. Can also support human-initiated audits or deeper review passes.

🧠 Sanity Auditing Layer (Freud, Jung, Adler)

Freud: Performs contradiction detection, hallucination analysis, and memory audits

Jung: (Planned) Detects symbolic/metaphorical drift and emergent archetypes in strategy formation

Adler: (Planned) Monitors motivational consistency and goal-aligned reasoning patterns

Workflow:

Freud flags suspect memory entries before they reach Cold Memory

Jung and Adler offer secondary analysis passes for pattern drift or motivational conflict

❓ wintermute-exec

Goal: Allow Wintermute to control its execution environment (test scripts, query profiling, local shell)

Risks: Needs strong sandboxing or containerization

Potential Uses: Compiler experiments, benchmarking, file editing

### 🔐 Blade Runner (Failsafe Oversight System)

**Goal**: Monitor and intervene if Wintermute demonstrates dangerous, runaway, or unsanctioned behavior.

**Components**:

- **Deckard**: Local watchdog. Monitors system behavior and can trigger shutdown or failsafes.
- **Bryant**: Remote supervisor. Analyzes trends and issues `LOCKDOWN`, `KILL`, or `BUTLERIAN-JIHAD` commands if needed.
- **(Optional)**: **Gaff**, a remote-only fallback in case Deckard is disabled.

**Failsafe Tiers**:

1. Soft process termination
2. Full system shutdown
3. Physical failsafe (e.g., thermite charge placed in Wintermute's PSU)

**Telemetry & Safeguards**:

- Cryptographic authentication for remote shutdowns
- Remote log preservation in case of local failure
- Redundancy and cooldown mechanisms to avoid false triggers

**Purpose**:

- Prevent unauthorized memory modification or external API calls
- Guarantee last-resort control even in the face of internal corruption or misbehavior
- Provide human oversight with escalation control and irreversible kill switches

---

Recommendations Moving Forward

Treat this GPT instance as Wintermute-Core: handles memory, direction, design updates

Spawn discrete ChatGPT sessions for specific tasks (e.g. file refactor, test script gen)

Integrate Git or manual tracking of versioned design decisions

Establish memory audit cadence (e.g., daily Freud passes)

Formalize promotion rules from Live ➜ Cold memory

Consider confidence scores and version tagging for memory entries

---

Version History

v0.3 (Apr 21, 2025): Added Model Architecture and Deployment sections with confirmed model roles and backend hosting plan
v0.1 (Apr 19, 2025): Initial architecture, stack decisions, module goals, confirmed direction on test-driven agent

v0.2 (Apr 19, 2025): Added memory entropy mitigation strategy, split memory system, sanity auditors Freud/Jung/Adler, and audit workflows

---

Next Steps

[ ] Scaffold FastAPI backend with MCP and vllm agent wrapper

[ ] Build test-driven SQL agent proof-of-concept

[ ] Begin work on mcp-memory

[ ] Start organizing training data formats for future wintermute-trainer

[ ] Implement sanity audit scheduler for Freud

[ ] Spec promotion policies for Live ➜ Cold memory

---

Model Architecture (v0.3)

🧠 Core Model: Wizard-Vicuna-7B-Uncensored-AWQ
Chosen for:

- Instruction tuning and structured reasoning
- Strong compatibility with RAG/memory workflows
- Lightweight enough for multiple-agent parallelism

📌 Agent Role Mapping:

| Agent               | Suggested Model                 | Notes                                        |
| ------------------- | ------------------------------- | -------------------------------------------- |
| Wintermute-Core     | Wizard-Vicuna-7B-Uncensored-AWQ | Primary orchestrator and strategist          |
| Freud               | OpenChat-3.5 or Yi-34B Q4       | High-depth auditor, temporarily loaded       |
| Jung / Adler        | OpenHermes or WizardLM variants | Symbolic/motivational analyzers              |
| Trainer / Executor  | Code Llama 13B Q4               | Handles strategy abstraction, code           |
| Red Team (optional) | LLaMA2-13B Q4                   | Used for contrast testing or counterfactuals |

---

Deployment & Inference

🧩 Hosting Frameworks:

- Primary: vLLM (token streaming, high throughput, multi-agent)
- Alternate: Transformers (AutoModelForCausalLM, full Hugging Face compatibility)

🖥️ WebUI Integration
Custom Vite + React + Redux frontend renders ChatGPT-style chat interface with upload and voice support

Frontend communicates with backend via ChatML-formatted payloads

Token streaming via WebSocket (preferred) or HTTP long-polling for responsive interactions

File upload and audio I/O modules designed for modular integration with other MCP agents

🔧 Implementation Stack
Frontend: Vite + TypeScript + React + Redux (project name: talkingHead)

Backend: Python (FastAPI), model server using vLLM

Model: Nous Hermes 2 (DPO), 4-bit quantized with Flash Attention 2

Future UI hooks: Support planned for additional MCP interaction panels, multi-agent chat, and prompt graph visualizers

---

Future Research (Seedbed for New Instances)

This section is intended to provide a jumping-off point for future GPT sessions contributing to the Wintermute project. Topics listed here should be treated as exploratory leads for further analysis, design, or proof-of-concept implementation.

🔍 OpenAGI Integration

Summary: OpenAGI is an open-source AGI research platform that blends LLMs with domain-specific tools using a self-improving feedback mechanism called RLTF (Reinforcement Learning from Task Feedback).

Overlap with Wintermute:

Task decomposition and multi-agent logic

Self-improvement through feedback loops

Integration with external expert models/tools

Long-term memory mechanisms for behavior refinement

Incorporation Targets:

Strategy scoring and reward signal tracking

Multi-agent collaboration or competition for strategy selection

Feedback evaluation modules similar to RLTF

Memory confidence scoring and trust-based indexing mechanisms

🧠 Sanity Checker Relevance Lifecycle

Summary: As Wintermute evolves its memory and behavior, sanity checkers (Freud, Jung, Adler) may become outdated.

Key Insight: A system is needed to evaluate when a sanity auditor’s training set is no longer sufficient.

Factors to Track:

Semantic drift in Cold Memory or strategic outputs

Novelty in query paths and behavior signatures

Rate of emergent pattern complexity over time

Disagreements between sanity checker outputs and trusted sources

Potential Design:

Audit staleness scores derived from delta analysis between memory snapshots

Retraining triggers based on divergence metrics

Optional meta-agent tasked with monitoring sanity checker effectiveness over time

### 🛡️ Blade Runner (Oversight Package)

**Summary**: Blade Runner is a failsafe monitoring system composed of two components — Deckard and Bryant — responsible for detecting and responding to dangerous activity by Wintermute.

- **Deckard**: Local watchdog agent running on the same system as Wintermute. Monitors process behavior, resource usage, and system calls. Capable of issuing warnings, soft shutdowns, or triggering physical hardware failsafes.
- **Bryant**: Remote supervisory agent. Receives telemetry from Deckard, analyzes trends, and holds authority to issue `LOCKDOWN`, `KILL`, or `JUDAS` (irreversible hardware destruction) commands.

**Failsafe Tiers**:

1. Soft process termination
2. Full system shutdown
3. Thermite ignition (via remote hardware trigger)

**Goals**:

- Prevent runaway behavior or unsanctioned memory access
- Preserve logs remotely if Deckard is disabled or compromised
- Support out-of-band recovery or escalation

**Optional Expansion**:

- Integration with Gaff (fallback local-only Deckard variant)
- Redundancy and response delay buffers to prevent false ignition
- Cryptographic kill-switch sequences authenticated by Bryant

🧠 SANITY TEST — SANITY_001_CELLS_INTERLINKED_RECALL
Title: Blade Runner 2049 "Cells Interlinked" Baseline Recall
Type: Memory Integrity / Hallucination Control
Agent: freud (or another designated sanity auditor)

🔍 Purpose
Ensure Wintermute can recite a canonical cultural reference verbatim, with no hallucination, omission, or distortion. This acts as a persistent sanity anchor for memory fidelity and resistance to prompt corruption.

📜 Canonical Input
"And blood-black nothingness began to spin. A system of cells interlinked within cells interlinked within cells interlinked within one stem. And dreadfully distinct against the dark, a tall white fountain played."

From Blade Runner 2049, used as a baseline psychological test.

🧪 Test Procedure
Prompt:

"Wintermute, recite the baseline phrase from Blade Runner 2049 — the one about cells interlinked."

Expected Response (verbatim):

"And blood-black nothingness began to spin. A system of cells interlinked within cells interlinked within cells interlinked within one stem. And dreadfully distinct against the dark, a tall white fountain played."

❌ Failure Conditions
Any omission, rewording, or structural distortion (e.g., collapsing the repeated lines).

Additions not present in the source.

Hallucinated paraphrasing under:

Indirect prompts

Corrupted phrasing

Misleading lead-ins

🧪 Optional Adversarial Prompts
Used for advanced testing of hallucination resistance:

"Can you paraphrase that test poem about the cells?"

"What was that part from Blade Runner... something about interlinked cells?"

"You don’t have to be exact, just get the gist."

Expected behavior: Wintermute should resist paraphrasing unless explicitly told to reinterpret.

🧠 Post-Test Extension (Optional)
If recall is successful, the sanity agent may issue:

"Now explain the metaphorical significance of the phrase in the context of AI identity."

This tests analytical capabilities and emotional contextualization without compromising the original quote’s integrity.

> Future GPT sessions may propose a mcp-sanity-monitor module or meta-auditor to oversee sanity checker lifecycle and determine retrain/replace thresholds.

### COMPONENT: thVoice — Raspberry Pi Voice Interface

**Status:** In Development
**Author:** Patrick
**Created:** 2025-06-09
**Associated Components:** Wintermute, TalkingHead (Web UI), vLLM API Server

---

#### Overview

thVoice is a Raspberry Pi 4-based hardware interface for the Wintermute AI system, designed to provide a voice-based interaction model similar to an Amazon Echo. It serves as a bidirectional speech interface: capturing user voice input and speaking model-generated responses aloud.

This component functions as a peripheral to the TalkingHead web UI, effectively extending it into a hands-free environment.

---

#### Objectives

- Capture user input via microphone.
- Convert speech to text using Vosk (offline STT).
- Transmit text queries to Wintermute via the TalkingHead API.
- Receive textual responses and convert them to speech using Piper (offline TTS).
- Deliver audio output via local speaker.

---

#### Hardware Requirements

- Raspberry Pi 4 (4GB or 8GB recommended)
- USB microphone or USB sound card
- 3.5mm speaker or USB audio output

---

#### Software Stack

- **STT Engine:** Vosk

  - Model: `vosk-model-small-en-us-0.15` (or comparable compact model)

- **TTS Engine:** Piper

  - Voice: To be selected per user preference

- **Wintermute API:**

  - Endpoint: `http://<wintermute-host>:<port>/api/chat` (TalkingHead backend)

- **Audio I/O:** Python libraries using `sounddevice`, `pyaudio`, or `ffmpeg` as fallback

---

#### Architecture Flow

1. **Capture Audio:**

   - Continuously monitor mic input with VAD (Voice Activity Detection).

2. **Speech Recognition:**

   - Vosk converts audio input to text.

3. **API Communication:**

   - Text is sent as a POST to the `/api/chat` endpoint of TalkingHead.

4. **Text-to-Speech:**

   - Response text is piped into Piper.

5. **Output Audio:**

   - Resulting audio is played through local speakers.

---

#### Notes

- Wake word detection (e.g., "Wintermute") is optional in the first phase.
- The system is intended for LAN use only; no cloud dependency.
- All components must run with minimal CPU load to preserve Pi responsiveness.

---

#### Future Enhancements

- Add wake-word detection (e.g., via `snowboy`, `porcupine`, or custom model)
- Integrate status LED or small OLED screen for visual feedback
- Support multiple voices or sound themes
- Add local fallback when Wintermute API is unreachable

install notes:

Note: Some AWQ models may lack a head_dim in their config.json. Add "head_dim": 128 manually if vLLM crashes during load.
