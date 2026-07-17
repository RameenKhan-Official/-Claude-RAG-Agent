"""
Tool definitions for the Claude agent.

Each tool has:
  1. A JSON schema (for Claude's `tools` parameter)
  2. A Python function that implements it

`TOOL_REGISTRY` maps tool name -> callable, so the agent loop can
dispatch a `tool_use` block to the right implementation generically.
"""

from __future__ import annotations

import ast
import operator as op
from typing import Callable

from src.retrieval.retriever import Retriever

# --------------------------------------------------------------------------- #
# Tool schemas (sent to the Claude API)
# --------------------------------------------------------------------------- #

TOOL_SCHEMAS = [
    {
        "name": "search_documents",
        "description": (
            "Search the indexed knowledge base for text relevant to a query. "
            "Use this whenever the user asks a question that might be answered "
            "by the documents they've uploaded, before answering from general knowledge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, phrased as a natural-language question or topic.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to retrieve (default 4).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate a basic arithmetic expression (add, subtract, multiply, divide, "
            "exponentiation, parentheses). Use this for any numeric computation instead "
            "of computing it mentally."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A basic arithmetic expression, e.g. '(120000 * 0.2) / 12'.",
                }
            },
            "required": ["expression"],
        },
    },
]


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #

_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate a restricted arithmetic AST (no names, no calls)."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def calculate(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression and return the result as a string."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


def make_search_documents(retriever: Retriever) -> Callable[..., str]:
    """Bind a Retriever instance into a `search_documents(query, top_k=4)` tool function."""

    def search_documents(query: str, top_k: int = 4) -> str:
        return retriever.retrieve_as_context(query, top_k=top_k)

    return search_documents


def build_tool_registry(retriever: Retriever) -> dict[str, Callable[..., str]]:
    """Build the name -> function mapping the agent uses to execute tool calls."""
    return {
        "search_documents": make_search_documents(retriever),
        "calculate": calculate,
    }
