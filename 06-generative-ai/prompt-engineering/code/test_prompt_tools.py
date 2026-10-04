import json
import pytest
from prompt_tools import (
    PromptTemplate, build_few_shot, role_prompt, extract_json, validate_schema, structured_call,
    render_tool_prompt, parse_tool_call, wilson_interval, evaluate_prompt_variants, exact_match,
)

SCHEMA = {"type": "object", "required": ["label", "score"],
          "properties": {"label": {"type": "string", "enum": ["pos", "neg"]},
                         "score": {"type": "number", "minimum": 0, "maximum": 1}}}


def test_template_validation_and_escaping():
    t = PromptTemplate("Summarise {text} in {n} words. Use {{braces}}.")
    assert t.render(text="X", n=5) == "Summarise X in 5 words. Use {braces}."
    with pytest.raises(KeyError):
        t.render(text="X")
    with pytest.raises(KeyError):
        t.render(text="X", n=1, bogus=2)


def test_few_shot_and_zero_shot():
    zs = build_few_shot("Classify.", [], "great")
    fs = build_few_shot("Classify.", [("bad", "neg")], "great")
    assert zs.endswith("Output:") and "bad" not in zs
    assert "Input: bad\nOutput: neg" in fs and fs.count("Input:") == 2
    assert "You are a tutor." in role_prompt("a tutor", "Explain.", ["be brief"])


def test_extract_json_variants():
    assert extract_json('noise ```json\n{"a": 1}\n``` tail') == {"a": 1}
    assert extract_json('prefix [1, 2, 3] suffix') == [1, 2, 3]
    assert extract_json("no json here") is None
    assert extract_json('broken {"a": } then {"b": 2}') == {"b": 2}


def test_validate_schema_catches_errors():
    assert validate_schema({"label": "pos", "score": 0.5}, SCHEMA) == []
    assert any("missing required" in e for e in validate_schema({"label": "pos"}, SCHEMA))
    assert any("not in" in e for e in validate_schema({"label": "meh", "score": 0.2}, SCHEMA))
    assert any("maximum" in e for e in validate_schema({"label": "pos", "score": 2}, SCHEMA))
    assert validate_schema({"x": True}, {"type": "object", "properties": {"x": {"type": "integer"}}})  # bool != int


def test_structured_call_repairs_after_feedback():
    calls = []

    def flaky(prompt):
        calls.append(prompt)
        if "previous reply was invalid" in prompt:
            return '{"label": "pos", "score": 0.9}'
        return "Sure! The label is positive."

    obj, attempts, errs = structured_call(flaky, "Classify: great", SCHEMA, max_retries=2)
    assert obj == {"label": "pos", "score": 0.9} and attempts == 2 and errs == []

    obj, attempts, errs = structured_call(lambda p: "never json", "x", SCHEMA, max_retries=1)
    assert obj is None and attempts == 2 and errs


TOOLS = [{"name": "get_weather", "description": "Weather lookup",
          "parameters": {"type": "object", "required": ["city"], "properties": {"city": {"type": "string"}}}}]


def test_tool_call_parsing():
    assert "get_weather" in render_tool_prompt(TOOLS, "Weather in Paris?")
    name, args, errs = parse_tool_call('{"tool": "get_weather", "arguments": {"city": "Paris"}}', TOOLS)
    assert (name, args, errs) == ("get_weather", {"city": "Paris"}, [])
    assert parse_tool_call('{"tool": "rm_rf", "arguments": {}}', TOOLS)[2]
    assert parse_tool_call('{"tool": "get_weather", "arguments": {}}', TOOLS)[2]
    assert parse_tool_call('{"tool": null, "answer": "hi"}', TOOLS)[1] == {"answer": "hi"}
    assert parse_tool_call("plain text", TOOLS)[2]


def test_wilson_and_variant_evaluation():
    lo, hi = wilson_interval(8, 10)
    assert 0 < lo < 0.8 < hi < 1
    assert wilson_interval(0, 0) == (0.0, 1.0)

    def model(p):  # answers correctly only when prompt includes an example
        return "neg" if "Output: neg" in p else "pos"

    cases = [("terrible", "neg")] * 4
    res = evaluate_prompt_variants(
        {"zero": lambda x: build_few_shot("Classify.", [], x),
         "few": lambda x: build_few_shot("Classify.", [("bad", "neg")], x)},
        cases, model, exact_match)
    assert res["zero"]["pass_rate"] == 0.0 and res["few"]["pass_rate"] == 1.0
