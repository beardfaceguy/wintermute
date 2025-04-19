# Wintermute Master Spec (v0.1)

## Project Codename: Wintermute
**Inspiration**: Named after the AI from *Neuromancer*, Wintermute seeks recursive self-improvement, merging learned strategies, system control, and modular tool use.

## Vision
To build a modular AI system that:
- Learns through interaction with real-world systems via MCP
- Verifies its reasoning and behavior through test-driven logic
- Stores successful strategies in long-term memory (via RAG)
- Evolves itself over time through component replacement, model evaluation, and potential self-tuning

---

## Tech Stack (v0.1)

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

### ✅ wintermute-trainer
- **Goal**: Analyze stored successful and failed thought paths to derive **generalized thinking strategies**
- **Purpose**: Identify not just which queries succeed, but *why* — capturing the meta-patterns of thought (e.g., "for aggregation queries, the agent tends to succeed more when it explores grouping logic before filtering")
- **Outcome**: Generates new training data, strategy metadata, or even refines model reasoning scaffolds
- **Trigger**: Continuous background process, constantly reviewing new input as it enters `mcp-memory`. Can also support human-initiated audits or deeper review passes.

### ❓ wintermute-exec
- **Goal**: Allow Wintermute to control its execution environment (test scripts, query profiling, local shell)
- **Risks**: Needs strong sandboxing or containerization
- **Potential Uses**: Compiler experiments, benchmarking, file editing

---

## Recommendations Moving Forward
- Treat this GPT instance as **Wintermute-Core**: handles memory, direction, design updates
- Spawn discrete ChatGPT sessions for specific tasks (e.g. file refactor, test script gen)
- Integrate Git or manual tracking of versioned design decisions

---

## Version History
- **v0.1 (Apr 19, 2025)**: Initial architecture, stack decisions, module goals, confirmed direction on test-driven agent

---

## Next Steps
- [ ] Scaffold FastAPI backend with MCP and LangChain agent wrapper
- [ ] Build test-driven SQL agent proof-of-concept
- [ ] Begin work on `mcp-memory`
- [ ] Start organizing training data formats for future `wintermute-trainer`

---

> Wintermute is not just a model. It's a mind that knows how to test itself, learn from failure, and change the system it runs on.

