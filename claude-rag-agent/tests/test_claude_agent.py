from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.agent.claude_agent import ClaudeAgent


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name, input_, block_id="tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=block_id)


def _make_response(content, stop_reason):
    return SimpleNamespace(content=content, stop_reason=stop_reason)


@patch("src.agent.claude_agent.anthropic.Anthropic")
def test_chat_returns_direct_text_without_tool_use(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_response(
        content=[_text_block("Hello there!")], stop_reason="end_turn"
    )
    mock_anthropic_cls.return_value = mock_client

    agent = ClaudeAgent(retriever=MagicMock(), api_key="fake-key")
    response = agent.chat("Hi")

    assert response.text == "Hello there!"
    assert response.tool_calls == []
    mock_client.messages.create.assert_called_once()


@patch("src.agent.claude_agent.anthropic.Anthropic")
def test_chat_executes_tool_then_returns_final_text(mock_anthropic_cls):
    mock_client = MagicMock()
    tool_call_response = _make_response(
        content=[_tool_use_block("calculate", {"expression": "2+2"})],
        stop_reason="tool_use",
    )
    final_response = _make_response(content=[_text_block("The answer is 4.")], stop_reason="end_turn")
    mock_client.messages.create.side_effect = [tool_call_response, final_response]
    mock_anthropic_cls.return_value = mock_client

    agent = ClaudeAgent(retriever=MagicMock(), api_key="fake-key")
    response = agent.chat("What is 2+2?")

    assert response.text == "The answer is 4."
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "calculate"
    assert response.tool_calls[0]["result"] == "4"
    assert mock_client.messages.create.call_count == 2


@patch("src.agent.claude_agent.anthropic.Anthropic")
def test_chat_stops_after_max_turns_if_tool_use_never_ends(mock_anthropic_cls):
    mock_client = MagicMock()
    looping_response = _make_response(
        content=[_tool_use_block("calculate", {"expression": "1+1"})],
        stop_reason="tool_use",
    )
    mock_client.messages.create.return_value = looping_response
    mock_anthropic_cls.return_value = mock_client

    agent = ClaudeAgent(retriever=MagicMock(), api_key="fake-key", max_turns=3)
    response = agent.chat("loop forever")

    assert "wasn't able to finish" in response.text
    assert mock_client.messages.create.call_count == 3


@patch("src.agent.claude_agent.anthropic.Anthropic")
def test_chat_handles_unknown_tool_gracefully(mock_anthropic_cls):
    mock_client = MagicMock()
    tool_call_response = _make_response(
        content=[_tool_use_block("unknown_tool", {})],
        stop_reason="tool_use",
    )
    final_response = _make_response(content=[_text_block("Fallback answer.")], stop_reason="end_turn")
    mock_client.messages.create.side_effect = [tool_call_response, final_response]
    mock_anthropic_cls.return_value = mock_client

    agent = ClaudeAgent(retriever=MagicMock(), api_key="fake-key")
    response = agent.chat("trigger unknown tool")

    assert response.tool_calls[0]["result"].startswith("Error: unknown tool")
    assert response.text == "Fallback answer."
