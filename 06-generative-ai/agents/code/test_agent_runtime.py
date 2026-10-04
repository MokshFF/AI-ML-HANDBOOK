import pytest
from agent_runtime import *


def make_registry():
    r = ToolRegistry()
    r.register(Tool("calc", "Evaluate arithmetic", {"required": ["expression"], "properties": {"expression": {"type": "string"}}}, safe_calc))
    r.register(Tool("lookup", "Look up a fact", {"required": ["key"], "properties": {"key": {"type": "string"}}},
                    lambda key: {"capital_of_france": "Paris"}[key]))
    r.register(Tool("send_email", "Send an email", {"required": ["to"], "properties": {"to": {"type": "string"}}},
                    lambda to: f"sent to {to}", risk="high"))
    return r


def test_safe_calc():
    assert safe_calc("2 + 3 * 4") == 14
    assert safe_calc("-(2 ** 3)") == -8
    for bad in ("__import__('os').system('x')", "open('f')", "2 ** 999", "a + 1"):
        with pytest.raises(Exception):
            safe_calc(bad)


def test_registry_validates_and_handles_errors():
    r = make_registry()
    assert r.call("calc", {"expression": "1+1"}) == (True, 2)
    ok, msg = r.call("calc", {})
    assert not ok and "missing required" in msg
    ok, msg = r.call("calc", {"expression": 5})
    assert not ok and "should be string" in msg
    assert not r.call("nope", {})[0]
    ok, msg = r.call("lookup", {"key": "missing"})
    assert not ok and "tool error" in msg
    assert len(r.specs()) == 3


def test_agent_loop_completes_and_recovers_from_error():
    r = make_registry()

    def policy(task, hist):
        if not hist:
            return {"tool": "lookup", "arguments": {"key": "typo"}}            # fails
        if hist[-1]["ok"] is False:
            return {"tool": "lookup", "arguments": {"key": "capital_of_france"}}  # recovers
        return {"final": f"The capital is {hist[-1]['observation']}."}

    res = run_agent(policy, r, "capital of France?")
    assert res.status == "done" and res.final == "The capital is Paris." and res.steps == 3
    assert [t.get("ok") for t in res.trace[:2]] == [False, True]


def test_agent_step_limit_and_human_approval():
    r = make_registry()
    loop = lambda task, hist: {"tool": "calc", "arguments": {"expression": "1+1"}}
    res = run_agent(loop, r, "x", max_steps=3)
    assert res.status == "max_steps" and res.final is None and res.steps == 3

    mail = lambda task, hist: {"tool": "send_email", "arguments": {"to": "a@b.c"}} if not hist else {"final": "ok"}
    denied = run_agent(mail, r, "x")
    assert denied.trace[0]["ok"] is False and "denied" in denied.trace[0]["observation"]
    allowed = run_agent(mail, r, "x", approve=lambda n, a: True)
    assert allowed.trace[0]["ok"] is True


def test_memory():
    st = ShortTermMemory(2)
    for i in range(4):
        st.add("user", str(i))
    assert [c for _, c in st.items] == ["2", "3"]
    lt = LongTermMemory()
    lt.add("The user prefers metric units")
    lt.add("The project deadline is Friday")
    assert lt.search("what units does the user prefer")[0].startswith("The user prefers")
    assert lt.search("zzz qqq") == []


def test_state_graph_routing_loop_and_guard():
    g = StateGraph()
    g.add_node("draft", lambda s: {"text": s.get("text", "") + "x", "tries": s.get("tries", 0) + 1})
    g.add_node("check", lambda s: {"ok": len(s["text"]) >= 3})
    g.set_entry("draft").add_edge("draft", "check")
    g.add_conditional_edges("check", lambda s: END if s["ok"] else "draft")
    final, cps = g.run({})
    assert final["text"] == "xxx" and final["tries"] == 3
    assert [n for n, _ in cps] == ["draft", "check"] * 3
    bad = StateGraph().add_node("a", lambda s: {}).set_entry("a").add_edge("a", "a")
    with pytest.raises(RuntimeError):
        bad.run({}, max_steps=5)
    with pytest.raises(ValueError):
        StateGraph().run({})


def test_supervisor_routing():
    workers = {"math": (["calculate", "sum", "multiply"], lambda t: "math result"),
               "search": (["find", "lookup", "who"], lambda t: "search result")}
    assert supervisor_route("please calculate the sum", workers) == ("math", "math result")
    assert supervisor_route("who is the author", workers)[0] == "search"
    assert supervisor_route("sing a song", workers)[0] == "none"


def test_mcp_style_interface():
    r = make_registry()
    listing = handle_jsonrpc(r, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert {t["name"] for t in listing["result"]["tools"]} == {"calc", "lookup", "send_email"}
    assert listing["result"]["tools"][0]["inputSchema"]["type"] == "object"
    call = handle_jsonrpc(r, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "calc", "arguments": {"expression": "6*7"}}})
    assert call["result"] == {"content": [{"type": "text", "text": "42"}], "isError": False}
    err = handle_jsonrpc(r, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "calc", "arguments": {}}})
    assert err["result"]["isError"] is True
    assert handle_jsonrpc(r, {"id": 4, "method": "bogus"})["error"]["code"] == -32601


def test_agent_evaluation_metrics():
    r = make_registry()

    def policy(task, hist):
        if not hist:
            return {"tool": "calc", "arguments": {"expression": task}}
        return {"final": str(hist[-1]["observation"])}

    cases = [{"task": "2+2", "check": lambda f: f == "4", "expected_tools": ["calc"]},
             {"task": "3*3", "check": lambda f: f == "10", "expected_tools": ["calc"]},   # wrong expectation -> failure
             {"task": "bad(", "check": lambda f: True, "expected_tools": ["calc"]}]
    m = evaluate_agent(cases, lambda t: run_agent(policy, r, t))
    assert m["success_rate"] == pytest.approx(2 / 3)   # third case "succeeds" with an error string but check is lenient
    assert m["invalid_call_rate"] == pytest.approx(1 / 3)
    assert m["expected_tool_recall"] == 1.0 and m["mean_steps"] == 2.0
