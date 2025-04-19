# Wintermute Master Spec (v0.2)

## Project Codename: Wintermute
**Inspiration**: Named after the AI from *Neuromancer*, Wintermute seeks recursive self-improvement, merging learned strategies, system control, and modular tool use.

## Vision
To build a modular AI system that:
- Learns through interaction with real-world systems via MCP
- Verifies its reasoning and behavior through test-driven logic
- Stores successful strategies in long-term memory (via RAG)
- Evolves itself over time through component replacement, model evaluation, and potential self-tuning

---

## Tech Stack (v0.2)

### Core Components
| Layer                | Tech                        | Status       |
|---------------------|-----------------------------|--------------|
| LLM Runtime         | vLLM                        | ✅ Confirmed  |
| Model Types         | Mistral 7B, LLaMA2, etc.    | ✅ Confirmed  |
| Agent Framework     | LangChain                   | ✅ Confirmed  |
| RAG Engine          | LlamaIndex                  | ✅ Confirmed  |
| API Backend         | FastAPI                     | ✅ Confirmed  |
| Custom UI           | React + Vite + Tailwind     | ✅ Confirmed  |
| Prompting Interface | Open WebUI (for prototyping)| ✅ Optional   |

### Tooling (via MCP)
| Tool              | MCP Server              | Status       |
|------------------|-------------------------|--------------|
| SQL Access       | mcp-postgres            | ✅ Confirmed  |
| Memory Store     | mcp-memory (custom)     | 🚧 Planned    |
| Filesystem       | mcp-filesystem (optional)| ❓ Evaluating |
| External Data    | Other MCP (GitHub, etc.)| 🧪 Optional   |

### Experimental Layers
| Layer                 | Component                  | Status       |
|----------------------|----------------------------|--------------|
| Verification Agent   | Secondary LLM (Critic)     | ✅ Confirmed  |
| Model Profiler       | Hardware access tools      | 🧪 Future     |
| Self-Tuning Pipeline | wintermute-trainer         | ✅ Confirmed  |
| Autonomy Layer       | Shell/exec access          | ❓ Discussing |
| Sanity Auditors      | Freud, Jung, Adler         | 🧠 Planned    |

---

## Key Projects & Modules

### ✅ Test-Driven SQL Generation Agent
- **Goal**: Have the AI generate, test, and correct SQL queries based on known-good expectations
- **MCP**: Postgres
- **Verification**: Auto-check result accuracy vs. test case
- **Memory**: Store both successful and failed queries, including the **thought path** used by the LLM to reach each conclusion

### 🚧 mcp-memory
- **Goal**: Custom MCP server to store learned query strategies, test results, behavioral notes
- **Storage**: LlamaIndex, Chroma, or file-backed vector store
- **Use**: Long-term memory for RAG, includes indexing of failed paths, decision trees, and associated metadata
- **Structure**: Split into:
  - **Live Memory**: Experimental, unverified strategies and recent outcomes
  - **Cold Memory**: Verified, high-confidence strategies vetted by sanity auditors

### ✅ wintermute-trainer
- **Goal**: Analyze stored successful and failed thought paths to derive **generalized thinking strategies**
- **Purpose**: Identify not just which queries succeed, but *why* — capturing the meta-patterns of thought (e.g., "for aggregation queries, the agent tends to succeed more when it explores grouping logic before filtering")
- **Outcome**: Generates new training data, strategy metadata, or even refines model reasoning scaffolds
- **Trigger**: Continuous background process, constantly reviewing new input as it enters `mcp-memory`. Can also support human-initiated audits or deeper review passes.

### 🧠 Sanity Auditing Layer (Freud, Jung, Adler)
- **Freud**: Performs contradiction detection, hallucination analysis, and memory audits
- **Jung**: (Planned) Detects symbolic/metaphorical drift and emergent archetypes in strategy formation
- **Adler**: (Planned) Monitors motivational consistency and goal-aligned reasoning patterns
- **Workflow**:
  - Freud flags suspect memory entries before they reach Cold Memory
  - Jung and Adler offer secondary analysis passes for pattern drift or motivational conflict

### ❓ wintermute-exec
- **Goal**: Allow Wintermute to control its execution environment (test scripts, query profiling, local shell)
- **Risks**: Needs strong sandboxing or containerization
- **Potential Uses**: Compiler experiments, benchmarking, file editing

---

## Recommendations Moving Forward
- Treat this GPT instance as **Wintermute-Core**: handles memory, direction, design updates
- Spawn discrete ChatGPT sessions for specific tasks (e.g. file refactor, test script gen)
- Integrate Git or manual tracking of versioned design decisions
- Establish memory audit cadence (e.g., daily Freud passes)
- Formalize promotion rules from Live ➜ Cold memory
- Consider confidence scores and version tagging for memory entries

---

## Version History
- **v0.1 (Apr 19, 2025)**: Initial architecture, stack decisions, module goals, confirmed direction on test-driven agent
- **v0.2 (Apr 19, 2025)**: Added memory entropy mitigation strategy, split memory system, sanity auditors Freud/Jung/Adler, and audit workflows

---

## Next Steps
- [ ] Scaffold FastAPI backend with MCP and LangChain agent wrapper
- [ ] Build test-driven SQL agent proof-of-concept
- [ ] Begin work on `mcp-memory`
- [ ] Start organizing training data formats for future `wintermute-trainer`
- [ ] Implement sanity audit scheduler for Freud
- [ ] Spec promotion policies for Live ➜ Cold memory

---

## Future Research (Seedbed for New Instances)
This section is intended to provide a jumping-off point for future GPT sessions contributing to the Wintermute project. Topics listed here should be treated as exploratory leads for further analysis, design, or proof-of-concept implementation.

### 🔍 OpenAGI Integration
- **Summary**: OpenAGI is an open-source AGI research platform that blends LLMs with domain-specific tools using a self-improving feedback mechanism called RLTF (Reinforcement Learning from Task Feedback).
- **Overlap with Wintermute**:
  - Task decomposition and multi-agent logic
  - Self-improvement through feedback loops
  - Integration with external expert models/tools
  - Long-term memory mechanisms for behavior refinement
- **Incorporation Targets**:
  - Strategy scoring and reward signal tracking
  - Multi-agent collaboration or competition for strategy selection
  - Feedback evaluation modules similar to RLTF
  - Memory confidence scoring and trust-based indexing mechanisms

> Future GPT sessions should consider reviewing the [OpenAGI GitHub repo](https://github.com/aiplanethub/openagi) and its design principles to evaluate specific components for integration or adaptation.

