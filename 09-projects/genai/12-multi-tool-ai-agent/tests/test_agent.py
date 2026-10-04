"""Tests for Autonomous Multi-Tool Agent."""
from tools import calculator, knowledge_lookup
from agent import AutonomousAgent

def test_tools():
    assert calculator("10 + 5 * 2") == "20"
    assert "Guido" in knowledge_lookup("python_release")

def test_agent_execution():
    agent = AutonomousAgent()
    res = agent.execute_plan("calculate (50 - 10) / 2")
    assert res["steps_taken"] == 1
    assert "20.0" in res["final_answer"]
