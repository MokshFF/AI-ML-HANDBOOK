"""
LLMOps Infrastructure:
1. Prompt Version Registry with semantic versioning and deployment environments.
2. Token Usage, Cost & Latency Telemetry Tracer.
3. Automated LLM Regression Evaluation Pipeline.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# 1. Prompt Version Registry
# --------------------------------------------------------------------------- #
@dataclass
class PromptVersion:
    name: str
    version: int
    template: str
    input_variables: List[str]
    model_name: str
    temperature: float = 0.0
    environment: str = "dev"  # "dev", "staging", "prod"
    created_at: float = field(default_factory=time.time)

    def render(self, **kwargs) -> str:
        for var in self.input_variables:
            if var not in kwargs:
                raise ValueError(f"Missing variable '{var}' in prompt template '{self.name}'")
        return self.template.format(**kwargs)


class PromptRegistry:
    """Manages prompt iterations, versions, and promotional environments."""
    def __init__(self):
        self.prompts: Dict[str, List[PromptVersion]] = {}

    def register_prompt(
        self,
        name: str,
        template: str,
        input_variables: List[str],
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0
    ) -> PromptVersion:
        if name not in self.prompts:
            self.prompts[name] = []
        version = len(self.prompts[name]) + 1
        pv = PromptVersion(
            name=name,
            version=version,
            template=template,
            input_variables=input_variables,
            model_name=model_name,
            temperature=temperature
        )
        self.prompts[name].append(pv)
        return pv

    def promote(self, name: str, version: int, environment: str = "prod") -> PromptVersion:
        if name not in self.prompts:
            raise KeyError(f"Prompt '{name}' not found")
        versions = self.prompts[name]
        target = next((v for v in versions if v.version == version), None)
        if not target:
            raise KeyError(f"Version {version} not found for prompt '{name}'")
        # Demote previous prod
        if environment == "prod":
            for v in versions:
                if v.environment == "prod":
                    v.environment = "archived"
        target.environment = environment
        return target

    def get_active_prompt(self, name: str, environment: str = "prod") -> Optional[PromptVersion]:
        if name not in self.prompts:
            return None
        for v in reversed(self.prompts[name]):
            if v.environment == environment:
                return v
        return None


# --------------------------------------------------------------------------- #
# 2. Token, Cost & Latency Telemetry Tracer
# --------------------------------------------------------------------------- #
MODEL_PRICING_PER_MILLION = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00}
}


@dataclass
class LLMTrace:
    trace_id: str
    prompt_name: str
    prompt_version: int
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    status: str = "SUCCESS"


class LLMTelemetryTracer:
    """Logs LLM calls and computes operational cost, token counts, and latency percentiles."""
    def __init__(self):
        self.traces: List[LLMTrace] = []

    def record_call(
        self,
        prompt_name: str,
        prompt_version: int,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        status: str = "SUCCESS"
    ) -> LLMTrace:
        pricing = MODEL_PRICING_PER_MILLION.get(model_name, {"input": 1.0, "output": 2.0})
        cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000.0
        trace = LLMTrace(
            trace_id=f"tr_{uuid.uuid4().hex[:8]}",
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=round(cost, 6),
            status=status
        )
        self.traces.append(trace)
        return trace

    def get_summary(self) -> Dict[str, Any]:
        if not self.traces:
            return {"total_calls": 0, "total_cost_usd": 0.0, "mean_latency_ms": 0.0}
        total_cost = sum(t.cost_usd for t in self.traces)
        total_tokens = sum(t.prompt_tokens + t.completion_tokens for t in self.traces)
        latencies = [t.latency_ms for t in self.traces]
        return {
            "total_calls": len(self.traces),
            "total_cost_usd": round(total_cost, 4),
            "total_tokens": total_tokens,
            "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
            "p95_latency_ms": round(float(sorted(latencies)[int(0.95 * len(latencies))]), 2)
        }


# --------------------------------------------------------------------------- #
# 3. Automated LLM Regression Evaluation Pipeline
# --------------------------------------------------------------------------- #
@dataclass
class EvalTestCase:
    inputs: Dict[str, Any]
    expected_substring: str


def run_prompt_eval_pipeline(
    prompt_version: PromptVersion,
    mock_llm_fn: Callable[[str], str],
    test_cases: Sequence[EvalTestCase]
) -> Dict[str, Any]:
    """Runs test suite against prompt version to catch quality regressions before deployment."""
    passed = 0
    failures = []
    for i, tc in enumerate(test_cases):
        rendered = prompt_version.render(**tc.inputs)
        output = mock_llm_fn(rendered)
        if tc.expected_substring.lower() in output.lower():
            passed += 1
        else:
            failures.append({
                "test_index": i,
                "rendered_prompt": rendered,
                "output": output,
                "expected": tc.expected_substring
            })
    score = passed / len(test_cases) if test_cases else 0.0
    return {
        "prompt_name": prompt_version.name,
        "version": prompt_version.version,
        "score": round(score, 4),
        "total_tests": len(test_cases),
        "passed": passed,
        "failed": len(failures),
        "is_ready_for_prod": score >= 0.90
    }
