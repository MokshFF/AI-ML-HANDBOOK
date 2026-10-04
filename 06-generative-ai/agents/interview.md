# AI Agents Interview Questions & Answers

### Q1: What is the ReAct framework, and why does it outperform pure Action or pure Chain-of-Thought approaches?
**Answer:**
ReAct (Reasoning + Acting) alternates between internal verbal reasoning ("Thought"), external environmental actuation ("Action"), and absorbing feedback ("Observation").
- **Pure Chain-of-Thought (CoT)** operates in a closed loop without external grounding, leading to hallucinations when facts change or factual errors propagate.
- **Pure Action (Act-only)** lacks deliberative reasoning, failing to formulate multi-step strategies or self-correct when API outputs return unexpected formats.
- **ReAct** combines both: reasoning traces help the model formulate goals, track sub-goals, and synthesize observations; tool actions retrieve grounded facts that steer the reasoning process.

---

### Q2: What are the main limitations of unbounded while-loop agents (e.g., AutoGPT), and how do State Graphs solve them?
**Answer:**
1. **Unbounded While-Loop Failures**:
   - **Infinite loops**: Repeating identical tool calls when facing non-fatal API errors.
   - **Context overflow**: Growing execution trajectories quickly exhaust LLM context windows.
   - **Lack of determinism**: Hard to enforce compliance pipelines or business logic gates.
   - **State invisibility**: Internal state is trapped in unstructured text prompts.
2. **State Graph Solutions**:
   - Explicit nodes and conditional edges enforce structured control flow.
   - Guardrails such as max iteration counts per node terminate stuck cycles.
   - Typed state dictionaries maintain explicit, auditable memory.
   - Checkpointing allows pausing execution for Human-in-the-Loop review and deterministic replay.

---

### Q3: Explain the Model Context Protocol (MCP) and why the industry is converging on it.
**Answer:**
Prior to MCP, every AI agent framework (LangChain, LlamaIndex, AutoGen, CrewAI, proprietary SDKs) defined its own tool and context wrapper format. Developers had to rewrite integrations for every framework.
**MCP (Model Context Protocol)** is an open JSON-RPC 2.0 standard:
- Defines uniform endpoints: `tools/list`, `tools/call`, `resources/read`, and `prompts/list`.
- Standardizes transport layers (stdio for local child processes, SSE/HTTP for remote services).
- Decouples tool providers (databases, APIs, developer tools) from agent runtimes, enabling plug-and-play interoperability across any compliant AI platform.

---

### Q4: How do you design memory for an agent requiring both session awareness and long-term personalization?
**Answer:**
A robust architecture employs a multi-tiered memory hierarchy:
1. **Working / Short-Term Memory**:
   - Recent message turns maintained within the prompt window.
   - Summarization or sliding window truncation when approaching context limits.
2. **Semantic / Long-Term Memory**:
   - Vector database storing past user facts, preferences, and key entities.
   - Embed queries to retrieve relevant facts ($top\text{-}k$ cosine similarity) and inject them into system instructions.
3. **Episodic / Procedural Memory**:
   - Successful past trajectory logs (few-shot exemplars) demonstrating how previous complex tasks were solved.

---

### Q5: How do you safeguard agent systems against destructive or malicious tool calls?
**Answer:**
1. **Human-in-the-Loop (HITL)**: Annotate dangerous tools with `requires_approval=True` and block execution until confirmed by a human.
2. **Input Validation**: Strictly validate arguments against JSON schemas, forbidding shell injection or unsafe file paths.
3. **Least Privilege & Sandboxing**: Execute generated code inside ephemeral containerized sandboxes with restricted network access.
4. **Output Verification**: Validate tool response format and redact sensitive tokens (passwords, keys) before passing observations back to the LLM.
