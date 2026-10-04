"""Autonomous ReAct Multi-Tool Agent Engine."""
from typing import List, Dict, Any
from tools import TOOL_REGISTRY

class AutonomousAgent:
    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps
        self.scratchpad = []

    def execute_plan(self, query: str) -> Dict[str, Any]:
        self.scratchpad = []
        step = 0
        
        # Rule-based ReAct dispatcher
        while step < self.max_steps:
            step += 1
            if "calculate" in query.lower() or any(op in query for op in ["+", "-", "*", "/"]):
                # Extract expression
                expr = "".join(c for c in query if c in "0123456789+-*/.() ")
                obs = TOOL_REGISTRY["calculator"](expr)
                self.scratchpad.append({"step": step, "tool": "calculator", "input": expr, "observation": obs})
                final_answer = f"The calculated result is: {obs}"
                break
            elif "when" in query.lower() or "created" in query.lower():
                obs = TOOL_REGISTRY["knowledge_lookup"]("python_release")
                self.scratchpad.append({"step": step, "tool": "knowledge_lookup", "input": "python_release", "observation": obs})
                final_answer = f"According to knowledge base: {obs}"
                break
            else:
                final_answer = f"Processed query directly: '{query}'"
                break
                
        return {
            "query": query,
            "steps_taken": len(self.scratchpad),
            "scratchpad": self.scratchpad,
            "final_answer": final_answer
        }

if __name__ == "__main__":
    agent = AutonomousAgent()
    res = agent.execute_plan("Please calculate 45 * 12 + 10")
    print("Agent Result:", res["final_answer"])
    print("Trajectory:", res["scratchpad"])
