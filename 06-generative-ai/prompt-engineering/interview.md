# Prompt Engineering - Interview Questions

### 1. When would you pick few-shot over zero-shot, and what makes examples effective?
When the output format or label semantics are hard to describe. Effective examples are representative, format-consistent, label-balanced, and include a tricky case; order and recency can bias outputs, so evaluate permutations.

### 2. How do you reliably get JSON out of an LLM?
Specify a schema, use native structured-output/JSON-mode features where available, then validate in code, retry with the validation errors, cap retries, and fall back safely. Valid structure does not imply valid facts; add semantic checks.

### 3. Should you ask the model to "show its reasoning"?
It can help on multi-step tasks, but explanations may be unfaithful, cost tokens, and may be withheld by some systems. Ask for a short justification plus final answer, and verify the answer with tools or tests. Do not rely on, or attempt to extract, private reasoning.

### 4. Design a tool-calling prompt. Who executes the tool?
Declare tools with JSON schemas and a strict reply format; the **application** parses, validates arguments, enforces permissions (and human approval for risky actions), executes, and returns the result as data. The model only proposes calls.

### 5. How would you evaluate whether prompt B beats prompt A?
Same labelled test set, same model/settings, a defined metric, multiple samples if sampling is stochastic, report intervals or a paired test, check subgroup regressions. Avoid overfitting to the dev set.

### 6. What is prompt injection and how does it relate to templates?
Untrusted text inside a prompt can carry instructions. Delimiters and templates help but do not solve it; see [`../safety-alignment/`](../safety-alignment/).

### 7. Why might the same prompt behave differently after a model upgrade?
Different training/post-training, instruction following, formatting habits, and safety behaviour. Keep a regression suite.

### 8. Temperature for structured extraction vs. creative writing?
Low (often 0) for extraction/classification to reduce variance; higher for diversity. Nondeterminism can persist even at 0.

### Coding drill
Implement `validate_schema` for `required`, `enum`, numeric bounds and nested arrays, and write tests showing `True` is not accepted as an `integer`.
