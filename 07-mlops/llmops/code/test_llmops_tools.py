import pytest
from llmops_tools import (
    PromptRegistry,
    LLMTelemetryTracer,
    EvalTestCase,
    run_prompt_eval_pipeline
)


def test_prompt_registry_versioning_and_promotion():
    reg = PromptRegistry()
    p1 = reg.register_prompt(
        name="sql_generator",
        template="Translate this natural language to SQL: {query}",
        input_variables=["query"],
        model_name="gpt-4o-mini"
    )
    assert p1.version == 1
    assert p1.environment == "dev"
    assert p1.render(query="Show users") == "Translate this natural language to SQL: Show users"

    # Promote p1 to prod
    reg.promote("sql_generator", version=1, environment="prod")
    assert reg.get_active_prompt("sql_generator", "prod").version == 1

    # Register p2 and promote
    p2 = reg.register_prompt(
        name="sql_generator",
        template="Given Postgres schema, write SQL for: {query}. Return only SQL.",
        input_variables=["query"],
        model_name="gpt-4o"
    )
    assert p2.version == 2
    reg.promote("sql_generator", version=2, environment="prod")
    assert reg.get_active_prompt("sql_generator", "prod").version == 2
    assert p1.environment == "archived"


def test_llm_telemetry_tracer():
    tracer = LLMTelemetryTracer()

    # Log 2 calls
    tracer.record_call(
        prompt_name="summarizer",
        prompt_version=1,
        model_name="gpt-4o-mini",
        prompt_tokens=1000,
        completion_tokens=200,
        latency_ms=250.0
    )
    tracer.record_call(
        prompt_name="summarizer",
        prompt_version=1,
        model_name="gpt-4o-mini",
        prompt_tokens=2000,
        completion_tokens=500,
        latency_ms=450.0
    )

    summary = tracer.get_summary()
    assert summary["total_calls"] == 2
    assert summary["total_tokens"] == 3700
    assert summary["total_cost_usd"] > 0.0
    assert summary["mean_latency_ms"] == 350.0


def test_automated_eval_pipeline():
    reg = PromptRegistry()
    p = reg.register_prompt(
        name="classifier",
        template="Classify sentiment: {text}",
        input_variables=["text"]
    )

    def mock_model(prompt):
        return "Positive" if "great" in prompt or "good" in prompt else "Negative"

    test_suite = [
        EvalTestCase(inputs={"text": "This is a great product"}, expected_substring="Positive"),
        EvalTestCase(inputs={"text": "A very good service"}, expected_substring="Positive"),
        EvalTestCase(inputs={"text": "Terrible and broke down"}, expected_substring="Negative")
    ]

    res = run_prompt_eval_pipeline(p, mock_model, test_suite)
    assert res["score"] == 1.0
    assert res["passed"] == 3
    assert res["is_ready_for_prod"] is True
