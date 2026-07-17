"""
Minimal terminal demo: index the sample documents and chat with the agent.

Usage:
    python scripts/cli_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.agent.claude_agent import ClaudeAgent
from src.config import settings
from src.embeddings.embedder import get_embedder
from src.retrieval.retriever import Retriever

SAMPLE_DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def main() -> None:
    if not settings.anthropic_api_key:
        print("ERROR: Set ANTHROPIC_API_KEY in your environment or .env file first.")
        sys.exit(1)

    print("Indexing sample documents...")
    embedder = get_embedder(settings.embedding_model_name)
    retriever = Retriever(embedder=embedder)
    paths = list(SAMPLE_DOCS_DIR.glob("*.txt"))
    count = retriever.index_files(paths, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    print(f"Indexed {count} chunks from {len(paths)} file(s).\n")

    agent = ClaudeAgent(retriever=retriever)
    history: list[dict] = []

    print("Ask a question about the sample employee handbook (Ctrl+C to quit).\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not user_input:
            continue

        response = agent.chat(user_input, history=history)
        print(f"\nAssistant: {response.text}\n")

        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"  [tool call] {tc['name']}({tc['input']})")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response.text})


if __name__ == "__main__":
    main()
