# Prompt Engineering

> **Last reviewed:** 2026-10. Model behaviour differs across vendors and versions; always re-evaluate prompts when you change models. This module is deliberately provider-agnostic (`model_fn: str -> str`).

## Learning objectives
Write robust prompts and prompt templates, use zero-shot / few-shot / role prompting, obtain **validated structured outputs**, prompt for tool use safely, apply reasoning-style prompting without extracting private reasoning, and **evaluate** prompts quantitatively.

## 1. Zero-shot, few-shot and role prompting
- **Zero-shot**: instruction only. Cheapest; works when the task is well specified.
- **Few-shot** (Brown et al., 2020): add input/output demonstrations. Gains depend on example *format*, label balance, ordering, and coverage - not just count. Keep a consistent layout (`build_few_shot`).
- **Role / system prompting**: state audience, tone, scope and constraints. Roles shape style more than they add knowledge, and a role statement is **not** a security boundary ([`../safety-alignment/`](../safety-alignment/)).

## 2. Prompt templates
Treat prompts as versioned artifacts: parameterised templates (`PromptTemplate`) with required variables, stored in version control, and tested. Separate *instructions* from *untrusted data* with clear delimiters, and never splice untrusted text into the instruction part without treating it as data.

## 3. Structured outputs
Downstream code needs machine-readable output. Layers of defence, from weakest to strongest:
1. Describe the format in the prompt (and show the schema).
2. Use provider features such as JSON mode / schema-constrained decoding when available (names and guarantees vary; check official docs).
3. **Always validate** (`validate_schema`) and **repair or reject** (`structured_call` feeds validation errors back, bounded retries).
Valid JSON is not the same as *correct* content: validate semantics too (ranges, enums, referential integrity).

## 4. Reasoning-style prompting (without exposing private reasoning)
Prompting for intermediate steps ("let's think step by step", Kojima et al., 2022; chain-of-thought, Wei et al., 2022) can improve accuracy on multi-step problems in some models. Practical guidance:
- Request a **brief, user-facing justification** and a clearly separated **final answer** field, instead of asking for a raw transcript of internal reasoning. Verbalised reasoning is not guaranteed to be a faithful account of how the model got its answer, and many systems keep internal reasoning private.
- Prefer **verifiable** outputs: compute with tools, check with code, run tests, self-consistency voting (Wang et al., 2022).
- Newer "reasoning" models often do internal deliberation by themselves; verbose step-by-step instructions can be unnecessary or counterproductive. Measure, don't assume.

## 5. Tool-use prompting
Declare each tool (name, description, JSON-schema parameters) and a strict call format; parse and **validate** the call (`parse_tool_call`), check authorisation, then execute in application code and return results as data. Frameworks and provider "function calling" APIs implement the same idea. See [`../agents/`](../agents/).

## 6. Prompt evaluation
Prompt changes are experiments:
1. Build a labelled **test set** covering typical, edge, and adversarial cases; keep a held-out split.
2. Pick a **scorer** (exact match, schema validity, programmatic check, rubric/LLM-judge - see [`../evaluation/`](../evaluation/)).
3. Compare variants on identical cases and report **confidence intervals** (`evaluate_prompt_variants`). Four cases can't distinguish 75% from 100%.
4. Re-run on every model/version/prompt change (regression testing).

## Common mistakes
- Judging a prompt on 3 hand-picked examples.
- Parsing free text with regexes instead of validating structured output.
- Putting secrets or authorisation logic in the prompt.
- Assuming a prompt tuned for one model transfers unchanged to another.
- Requesting hidden reasoning verbatim instead of verifiable answers plus short justifications.

## Code and notebook
- [`code/prompt_tools.py`](code/prompt_tools.py): `PromptTemplate`, `build_few_shot`, `extract_json`, `validate_schema`, `structured_call`, tool-call helpers, `evaluate_prompt_variants`.
- [`code/test_prompt_tools.py`](code/test_prompt_tools.py): 7 tests.
- [`notebook.ipynb`](notebook.ipynb): offline walkthrough with mock models.
