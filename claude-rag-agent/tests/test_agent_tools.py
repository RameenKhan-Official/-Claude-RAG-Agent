from unittest.mock import MagicMock

from src.agent.tools import build_tool_registry, calculate, make_search_documents


def test_calculate_basic_arithmetic():
    assert calculate("2 + 3 * 4") == "14"


def test_calculate_handles_parentheses_and_division():
    assert calculate("(10 + 2) / 4") == "3.0"


def test_calculate_rejects_unsafe_expression():
    result = calculate("__import__('os').system('echo hi')")
    assert result.startswith("Error")


def test_calculate_rejects_name_references():
    result = calculate("os.getcwd()")
    assert result.startswith("Error")


def test_make_search_documents_delegates_to_retriever():
    mock_retriever = MagicMock()
    mock_retriever.retrieve_as_context.return_value = "some context"

    search_fn = make_search_documents(mock_retriever)
    result = search_fn(query="test query", top_k=2)

    mock_retriever.retrieve_as_context.assert_called_once_with("test query", top_k=2)
    assert result == "some context"


def test_build_tool_registry_contains_expected_tools():
    registry = build_tool_registry(MagicMock())
    assert set(registry.keys()) == {"search_documents", "calculate"}
    assert callable(registry["calculate"])
