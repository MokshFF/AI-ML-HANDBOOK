"""
Prompt engineering utilities that work with ANY text-in/text-out model.

`model_fn` is always a plain callable ``str -> str`` so examples are vendor-neutral
and testable offline with deterministic mock models.
"""

import json
import math
import re
import string
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ModelFn = Callable[[str], str]


# --------------------------------------------------------------------------- #
# Templates and few-shot construction
# --------------------------------------------------------------------------- #
class PromptTemplate:
    """`{field}` placeholders; literal braces are written as `{{` and `}}`."""

    def __init__(self, template: str):
        self.template = template
        self.fields = {f for _, f, _, _ in string.Formatter().parse(template) if f}

    def render(self, **values: Any) -> str:
        missing = self.fields - values.keys()
        extra = values.keys() - self.fields
        if missing:
            raise KeyError(f"missing template variables: {sorted(missing)}")
        if extra:
            raise KeyError(f"unexpected template variables: {sorted(extra)}")
        return self.template.format(**values)


def build_few_shot(instruction: str, examples: Sequence[Tuple[str, str]], query: str,
                   input_label: str = "Input", output_label: str = "Output") -> str:
    """Zero-shot when `examples` is empty, few-shot otherwise. Consistent format matters."""
    parts = [instruction.strip(), ""]
    for x, y in examples:
        parts += [f"{input_label}: {x}", f"{output_label}: {y}", ""]
    parts += [f"{input_label}: {query}", f"{output_label}:"]
    return "\n".join(parts)


def role_prompt(role: str, task: str, constraints: Sequence[str] = ()) -> str:
    lines = [f"You are {role}.", task.strip()]
    if constraints:
        lines.append("Constraints:")
        lines += [f"- {c}" for c in constraints]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Structured outputs: extract -> validate -> repair loop
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> Optional[Any]:
    """Return the first balanced JSON object/array found in `text` (handles code fences)."""
    text = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder()
    for m in re.finditer(r"[\{\[]", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
            return obj
        except json.JSONDecodeError:
            continue
    return None


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
          "array": list, "object": dict}


def validate_schema(obj: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    """Minimal JSON-Schema subset: type, enum, required, properties, items, minimum, maximum."""
    errors: List[str] = []
    t = schema.get("type")
    if t:
        py = _TYPES[t]
        ok = isinstance(obj, py) and not (t in ("integer", "number") and isinstance(obj, bool))
        if t == "integer" and isinstance(obj, float):
            ok = False
        if not ok:
            return [f"{path}: expected {t}, got {type(obj).__name__}"]
    if "enum" in schema and obj not in schema["enum"]:
        errors.append(f"{path}: {obj!r} not in {schema['enum']}")
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if "minimum" in schema and obj < schema["minimum"]:
            errors.append(f"{path}: {obj} < minimum {schema['minimum']}")
        if "maximum" in schema and obj > schema["maximum"]:
            errors.append(f"{path}: {obj} > maximum {schema['maximum']}")
    if isinstance(obj, dict):
        for k in schema.get("required", []):
            if k not in obj:
                errors.append(f"{path}: missing required '{k}'")
        for k, sub in schema.get("properties", {}).items():
            if k in obj:
                errors += validate_schema(obj[k], sub, f"{path}.{k}")
    if isinstance(obj, list) and "items" in schema:
        for i, item in enumerate(obj):
            errors += validate_schema(item, schema["items"], f"{path}[{i}]")
    return errors


def structured_call(model_fn: ModelFn, prompt: str, schema: Dict[str, Any],
                    max_retries: int = 2) -> Tuple[Optional[Any], int, List[str]]:
    """Call the model, parse + validate, and feed validation errors back on failure.

    Returns (parsed_or_None, attempts_used, last_errors).
    """
    full = f"{prompt}\n\nRespond with a single JSON object matching this schema:\n{json.dumps(schema)}"
    errors: List[str] = []
    for attempt in range(1, max_retries + 2):
        raw = model_fn(full)
        obj = extract_json(raw)
        errors = ["no JSON found"] if obj is None else validate_schema(obj, schema)
        if not errors:
            return obj, attempt, []
        full = (f"{prompt}\n\nYour previous reply was invalid:\n- " + "\n- ".join(errors) +
                f"\nReturn ONLY corrected JSON matching:\n{json.dumps(schema)}")
    return None, max_retries + 1, errors


# --------------------------------------------------------------------------- #
# Tool-use prompting
# --------------------------------------------------------------------------- #
def render_tool_prompt(tools: Sequence[Dict[str, Any]], user_request: str) -> str:
    """Describe tools and the exact reply format. The *application* executes tools, never the model."""
    spec = json.dumps(list(tools), indent=2)
    return (
        "You may call at most one tool. Available tools (JSON):\n" + spec + "\n\n"
        'To call a tool reply with ONLY: {"tool": "<name>", "arguments": {...}}\n'
        'If no tool is needed reply with ONLY: {"tool": null, "answer": "<text>"}\n\n'
        f"User request: {user_request}"
    )


def parse_tool_call(reply: str, tools: Sequence[Dict[str, Any]]) -> Tuple[Optional[str], Dict[str, Any], List[str]]:
    """Parse and validate a tool call against the declared parameter schema."""
    obj = extract_json(reply)
    if not isinstance(obj, dict) or "tool" not in obj:
        return None, {}, ["reply is not a tool-call object"]
    if obj["tool"] is None:
        return None, {"answer": obj.get("answer", "")}, []
    by_name = {t["name"]: t for t in tools}
    if obj["tool"] not in by_name:
        return None, {}, [f"unknown tool {obj['tool']!r}"]
    args = obj.get("arguments", {})
    errs = validate_schema(args, by_name[obj["tool"]]["parameters"], "arguments")
    return (obj["tool"], args, errs) if not errs else (None, {}, errs)


# --------------------------------------------------------------------------- #
# Prompt evaluation
# --------------------------------------------------------------------------- #
def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def evaluate_prompt_variants(variants: Dict[str, Callable[[str], str]], cases: Sequence[Tuple[str, str]],
                             model_fn: ModelFn, scorer: Callable[[str, str], bool]) -> Dict[str, Dict[str, Any]]:
    """`variants` maps name -> function(input)->prompt. Returns pass rate + 95% Wilson CI per variant."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, make in variants.items():
        passed = sum(bool(scorer(model_fn(make(x)), gold)) for x, gold in cases)
        lo, hi = wilson_interval(passed, len(cases))
        out[name] = {"pass_rate": passed / len(cases), "ci95": (lo, hi), "n": len(cases)}
    return out


def exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()
