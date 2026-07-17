"""
ClaudeAgent: a thin orchestration layer around the Anthropic Messages API
that implements the standard tool-use loop:

  1. Send the conversation + tool schemas to Claude.
  2. If Claude responds with `tool_use` blocks, execute each one locally
     and send the results back as `tool_result` blocks.
  3. Repeat until Claude returns a plain text answer (or a turn limit
     is hit, as a safety valve against infinite tool loops).

This is intentionally provider-call-shaped rather than framework-shaped
(no LangChain/etc.) so it's easy to read end-to-end and easy to explain
in an interview.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import anthropic

from src.agent.tools import TOOL_SCHEMAS, build_tool_registry
from src.config import settings
from src.retrieval.retriever import Retriever

SYSTEM_PROMPT = """\
You are a helpful assistant with access to a document knowledge base and a \
calculator tool. When a user's question could be answered from indexed \
documents, call `search_documents` before answering. If nothing relevant \
is retrieved, say so plainly rather than guessing. Use `calculate` for any \
arithmetic instead of computing it yourself. Keep answers concise and cite \
the source names returned by search_documents when you use them.\
"""


@dataclass
class AgentResponse:
    text: str
    tool_calls: list[dict] = field(default_factory=list)


class ClaudeAgent:
    def __init__(
        self,
        retriever: Retriever,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_turns: int | None = None,
    ):
        self.client = anthropic.Anthropic(api_key=api_key or settings.anthropic_api_key)
        self.model = model or settings.claude_model
        self.max_tokens = max_tokens or settings.max_tokens
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_turns = max_turns or settings.max_agent_turns
        self.tool_registry: dict[str, Callable[..., str]] = build_tool_registry(retriever)

    def _call_claude(self, messages: list[dict]):
        return self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

    def _execute_tool(self, name: str, tool_input: dict) -> str:
        fn = self.tool_registry.get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'"
        try:
            return fn(**tool_input)
        except Exception as exc:
            return f"Error running tool '{name}': {exc}"

    def chat(self, user_message: str, history: list[dict] | None = None) -> AgentResponse:
        """
        Run one full user turn, including any tool-use round-trips.

        `history` is a list of prior {"role": ..., "content": ...} messages
        (excluding the new user_message), letting the caller maintain
        multi-turn conversation state.
        """
        messages = list(history or [])
        messages.append({"role": "user", "content": user_message})

        tool_calls_made: list[dict] = []

        for _ in range(self.max_turns):
            response = self._call_claude(messages)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                final_text = "".join(block.text for block in response.content if block.type == "text")
                return AgentResponse(text=final_text, tool_calls=tool_calls_made)

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result_text = self._execute_tool(block.name, block.input)
                tool_calls_made.append({"name": block.name, "input": block.input, "result": result_text})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        return AgentResponse(
            text="I wasn't able to finish reasoning about that within the allotted tool-use turns.",
            tool_calls=tool_calls_made,
        )
