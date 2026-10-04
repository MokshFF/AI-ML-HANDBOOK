# LLMOps Interview Questions & Answers

### Q1: How does LLMOps differ from classical MLOps?
**Answer:**
1. **Artifact of Record**: In classical MLOps, the primary artifact is a trained model binary (weights). In LLMOps, the core artifact is a composite application consisting of prompt templates, few-shot exemplars, retrieval indexes, and API configurations.
2. **Training vs Prompting**: Most LLM deployments utilize foundation models via APIs or open-source checkpoints. The primary development cycle is prompt engineering, retrieval tuning, and fine-tuning rather than full pre-training.
3. **Observability & Telemetry**: Classical MLOps tracks tabular distribution drift and classification metrics (AUC, F1). LLMOps tracks token counts, inference cost in dollars, Time to First Token (TTFT), hallucination rates, and LLM-as-a-judge scores.
4. **Latency Profiles**: Classical models infer in 5-20 ms; LLMs take 500 ms to 5 seconds, requiring streaming responses (Server-Sent Events) and specialized inference engines.

---

### Q2: Why is Prompt Versioning essential, and how should a Prompt Registry be designed?
**Answer:**
**The Need**:
Hardcoding prompts in application source code creates hidden dependencies. Changing a single word in a prompt can alter output formatting, break downstream JSON parsers, and spike token costs.
**Prompt Registry Architecture**:
1. **Declarative Metadata**: Each prompt template has a canonical name, input variables, targeted model family, temperature, and semantic version number (`v1`, `v2`, `...`).
2. **Environment Separation**: Versioned prompts are promoted across environments (`dev` -> `staging` -> `prod`). Application code queries `registry.get_active_prompt("support_triage", env="prod")`, allowing prompt updates without redeploying microservices.
3. **Automated CI Regression Gates**: Any prompt edit triggers an automated evaluation run against a test suite of representative inputs to verify that accuracy does not regress.

---

### Q3: How do you implement end-to-end tracing for LLM applications (the Langfuse / OpenTelemetry pattern)?
**Answer:**
1. **Trace ID Propagation**: Assign a unique `trace_id` to each incoming user interaction, propagating it through nested sub-calls (retrieval step, guardrail check, LLM call, tool call).
2. **Span Metadata**: Each span logs:
   - Input prompt and generated completion.
   - Exact model identifier (e.g., `gpt-4o-2024-08-06`).
   - Token consumption (prompt tokens, completion tokens, cached tokens).
   - Exact cost in USD calculated from token pricing tiers.
   - Latency (Time to First Token and total generation duration).
3. **User Feedback Association**: Link asynchronous user reactions (thumbs up/down, copy action, correction) to the specific `trace_id` for automated evaluation dataset curation.
