"""
Agent building blocks without any LLM dependency: tools, a bounded agent loop,
memory, a LangGraph-style state graph, supervisor routing, an MCP-style tool
interface, and agent evaluation.

The "model" is always a plain Python callable (a policy) so behaviour is deterministic
and testable. In a real system the policy is an LLM call that returns the same
action dictionaries.
"""

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

END = "__end__"


# --------------------------------------------------------------------------- #
# Tools and registry (function calling)
# --------------------------------------------------------------------------- #
@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]          # JSON-schema-like: {"required": [...], "properties": {k: {"type": ...}}}
    fn: Callable[..., Any]
    risk: str = "low"                   # "high" tools need human approval


_PYTYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}


def validate_args(schema: Dict[str, Any], args: Dict[str, Any]) -> List[str]:
    errs = [f"missing required '{k}'" for k in schema.get("required", []) if k not in args]
    props = schema.get("properties", {})
    for k, v in args.items():
        if k not in props:
            errs.append(f"unexpected argument '{k}'")
            continue
        t = _PYTYPES.get(props[k].get("type", ""), object)
        if not isinstance(v, t) or (props[k].get("type") in ("integer", "number") and isinstance(v, bool)):
            errs.append(f"argument '{k}' should be {props[k].get('type')}")
    return errs


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def specs(self) -> List[Dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in self.tools.values()]

    def call(self, name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        """Validate then execute. Returns (ok, result_or_error_message). Never raises for model mistakes."""
        if name not in self.tools:
            return False, f"unknown tool '{name}'"
        errs = validate_args(self.tools[name].parameters, args)
        if errs:
            return False, "invalid arguments: " + "; ".join(errs)
        try:
            return True, self.tools[name].fn(**args)
        except Exception as e:  # tool failures become observations the agent can react to
            return False, f"tool error: {type(e).__name__}: {e}"


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod}


def safe_calc(expression: str) -> float:
    """Arithmetic only (no eval): parses the AST and allows a tiny whitelist of operators."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant) and abs(node.right.value) > 10:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expression, mode="eval"))


# --------------------------------------------------------------------------- #
# Bounded agent loop (ReAct-style: act -> observe -> repeat)
# --------------------------------------------------------------------------- #
@dataclass
class AgentResult:
    status: str                      # "done" | "max_steps" | "error"
    final: Optional[str]
    steps: int
    trace: List[Dict[str, Any]] = field(default_factory=list)


Policy = Callable[[str, List[Dict[str, Any]]], Dict[str, Any]]


def run_agent(policy: Policy, registry: ToolRegistry, task: str, max_steps: int = 6,
              approve: Optional[Callable[[str, Dict[str, Any]], bool]] = None) -> AgentResult:
    """policy(task, history) -> {"tool": name, "arguments": {...}} or {"final": text}.

    Safety rails: hard step limit, validated tool calls, human-approval hook for high-risk tools,
    tool errors fed back as observations.
    """
    trace: List[Dict[str, Any]] = []
    for step in range(1, max_steps + 1):
        action = policy(task, trace)
        if "final" in action:
            trace.append({"step": step, "action": "final"})
            return AgentResult("done", action["final"], step, trace)
        name, args = action.get("tool"), action.get("arguments", {})
        tool = registry.tools.get(name)
        if tool and tool.risk == "high" and not (approve and approve(name, args)):
            ok, obs = False, "denied: human approval required for high-risk tool"
        else:
            ok, obs = registry.call(name, args)
        trace.append({"step": step, "tool": name, "arguments": args, "ok": ok, "observation": obs})
    return AgentResult("max_steps", None, max_steps, trace)


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
class ShortTermMemory:
    """Rolling conversation window (what fits in the context)."""

    def __init__(self, max_items: int = 6):
        self.max_items, self.items = max_items, []

    def add(self, role: str, content: str) -> None:
        self.items.append((role, content))
        self.items = self.items[-self.max_items:]


class LongTermMemory:
    """Persistent facts retrieved by keyword overlap (swap for a vector store in production)."""

    def __init__(self):
        self.facts: List[Tuple[str, set]] = []

    @staticmethod
    def _toks(s: str) -> set:
        return set(re.findall(r"[a-z0-9]+", s.lower()))

    def add(self, fact: str) -> None:
        self.facts.append((fact, self._toks(fact)))

    def search(self, query: str, k: int = 3) -> List[str]:
        q = self._toks(query)
        scored = sorted(((len(q & t) / max(1, len(q | t)), f) for f, t in self.facts), reverse=True)
        return [f for s, f in scored[:k] if s > 0]


# --------------------------------------------------------------------------- #
# Workflow as an explicit state graph (LangGraph-style concept)
# --------------------------------------------------------------------------- #
class StateGraph:
    """Nodes are functions state -> partial update. Edges are static or routed by a function."""

    def __init__(self):
        self.nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.edges: Dict[str, str] = {}
        self.routers: Dict[str, Callable[[Dict[str, Any]], str]] = {}
        self.entry: Optional[str] = None

    def add_node(self, name: str, fn) -> "StateGraph":
        self.nodes[name] = fn
        return self

    def set_entry(self, name: str) -> "StateGraph":
        self.entry = name
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        self.edges[src] = dst
        return self

    def add_conditional_edges(self, src: str, router) -> "StateGraph":
        self.routers[src] = router
        return self

    def run(self, state: Dict[str, Any], max_steps: int = 20) -> Tuple[Dict[str, Any], List[Tuple[str, Dict[str, Any]]]]:
        """Returns (final_state, checkpoints). Checkpoints enable replay/debugging/resume."""
        if self.entry is None:
            raise ValueError("entry node not set")
        cur, checkpoints = self.entry, []
        state = dict(state)
        for _ in range(max_steps):
            if cur == END:
                return state, checkpoints
            state = {**state, **self.nodes[cur](dict(state))}
            checkpoints.append((cur, dict(state)))
            if cur in self.routers:
                cur = self.routers[cur](state)
            else:
                cur = self.edges.get(cur, END)
        raise RuntimeError("graph exceeded max_steps (possible infinite loop)")


# --------------------------------------------------------------------------- #
# Multi-agent: supervisor / router
# --------------------------------------------------------------------------- #
def supervisor_route(task: str, workers: Dict[str, Tuple[Sequence[str], Callable[[str], str]]]) -> Tuple[str, str]:
    """Pick the worker whose keywords best match the task; returns (worker_name, result)."""
    words = set(re.findall(r"[a-z0-9]+", task.lower()))
    scores = {n: len(words & set(kw)) for n, (kw, _) in workers.items()}
    best = max(scores, key=lambda n: (scores[n], n))
    if scores[best] == 0:
        return "none", "no suitable worker"
    return best, workers[best][1](task)


# --------------------------------------------------------------------------- #
# MCP-style tool interface (simplified illustration of the idea)
# --------------------------------------------------------------------------- #
def describe_tools_mcp_style(registry: ToolRegistry) -> List[Dict[str, Any]]:
    """Mirror the shape of an MCP `tools/list` entry: name, description, inputSchema."""
    return [{"name": t.name, "description": t.description,
             "inputSchema": {"type": "object", **t.parameters}} for t in registry.tools.values()]


def handle_jsonrpc(registry: ToolRegistry, request: Dict[str, Any]) -> Dict[str, Any]:
    """Toy JSON-RPC 2.0 handler for `tools/list` and `tools/call`.

    This only illustrates the request/response shape. The real Model Context Protocol
    also defines initialisation/capability negotiation, transports, resources, prompts,
    authorization, etc. - read the official specification.
    """
    rid, method, params = request.get("id"), request.get("method"), request.get("params", {})
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": describe_tools_mcp_style(registry)}}
    if method == "tools/call":
        ok, out = registry.call(params.get("name", ""), params.get("arguments", {}))
        return {"jsonrpc": "2.0", "id": rid,
                "result": {"content": [{"type": "text", "text": str(out)}], "isError": not ok}}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}}


# --------------------------------------------------------------------------- #
# Agent evaluation
# --------------------------------------------------------------------------- #
def evaluate_agent(cases: Sequence[Dict[str, Any]], run: Callable[[str], AgentResult]) -> Dict[str, float]:
    """Each case: {"task", "check": fn(final)->bool, optional "expected_tools": [...]}.

    Reports task success, mean steps, invalid-tool-call rate, and expected-tool recall.
    """
    ok = steps = calls = bad = hit = exp_total = 0
    for c in cases:
        res = run(c["task"])
        success = res.status == "done" and bool(c["check"](res.final))
        ok += success
        steps += res.steps
        used = [t for t in res.trace if "tool" in t]
        calls += len(used)
        bad += sum(not t["ok"] for t in used)
        if "expected_tools" in c:
            used_names = {t["tool"] for t in used}
            hit += len(used_names & set(c["expected_tools"]))
            exp_total += len(c["expected_tools"])
    n = len(cases)
    return {"success_rate": ok / n, "mean_steps": steps / n,
            "invalid_call_rate": bad / calls if calls else 0.0,
            "expected_tool_recall": hit / exp_total if exp_total else 1.0}
