"""Registered Tools for Autonomous Agent."""
from typing import Dict, Any

def calculator(expression: str) -> str:
    """Safely evaluates basic arithmetic expressions."""
    allowed = set("0123456789+-*/.() ")
    if not set(expression).issubset(allowed):
        return "Error: Invalid characters in math expression"
    try:
        # Safe eval of pure arithmetic
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Math Error: {e}"

def knowledge_lookup(topic: str) -> str:
    """Queries internal knowledge base for factual attributes."""
    kb = {
        "python_release": "Python was originally created by Guido van Rossum in 1991.",
        "transformer_paper": "Attention Is All You Need was published in 2017.",
        "resnet_depth": "The original ResNet architecture features 152 layers."
    }
    return kb.get(topic.strip().lower(), "Topic not found in knowledge base.")

TOOL_REGISTRY = {
    "calculator": calculator,
    "knowledge_lookup": knowledge_lookup
}
