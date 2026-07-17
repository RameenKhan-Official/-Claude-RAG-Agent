"""
Streamlit front-end for the Claude RAG Agent.

Lets a user upload documents, builds a FAISS index over them, and chats
with a Claude-powered agent that can call `search_documents` and
`calculate` as tools while answering.

Run with:  streamlit run src/app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

# Allow `streamlit run src/app.py` to resolve `src.*` imports regardless of cwd.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.agent.claude_agent import ClaudeAgent
from src.config import settings
from src.embeddings.embedder import get_embedder
from src.retrieval.retriever import Retriever

st.set_page_config(page_title="Claude RAG Agent", page_icon="🤖", layout="wide")


@st.cache_resource(show_spinner=False)
def _get_retriever(embedding_model_name: str) -> Retriever:
    embedder = get_embedder(embedding_model_name)
    return Retriever(embedder=embedder)


def _init_session_state() -> None:
    if "retriever" not in st.session_state:
        st.session_state.retriever = _get_retriever(settings.embedding_model_name)
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []  # display history: [{"role", "content"}]
    if "claude_history" not in st.session_state:
        st.session_state.claude_history = []  # raw Anthropic-format history
    if "indexed_count" not in st.session_state:
        st.session_state.indexed_count = 0


def _ensure_agent() -> ClaudeAgent:
    if st.session_state.agent is None:
        st.session_state.agent = ClaudeAgent(retriever=st.session_state.retriever)
    return st.session_state.agent


def main() -> None:
    st.title("🤖 Claude RAG Agent")
    st.caption("Upload documents, then chat with a Claude-powered agent that retrieves relevant context on demand.")

    _init_session_state()

    if not settings.anthropic_api_key:
        st.warning("No ANTHROPIC_API_KEY found. Set it in a `.env` file (see `.env.example`) before chatting.")

    with st.sidebar:
        st.header("📄 Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload documents (.txt, .md, .pdf)",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input("Chunk size", 200, 3000, settings.chunk_size, step=100)
        with col2:
            chunk_overlap = st.number_input("Chunk overlap", 0, 500, settings.chunk_overlap, step=20)

        if st.button("Build / Update Index", disabled=not uploaded_files):
            with st.spinner("Indexing documents..."):
                paths = []
                for f in uploaded_files:
                    suffix = Path(f.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(f.getvalue())
                        paths.append(tmp.name)

                added = st.session_state.retriever.index_files(
                    paths, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap)
                )
                st.session_state.indexed_count += added

            st.success(f"Indexed {added} new chunks (total: {st.session_state.indexed_count}).")

        st.metric("Chunks indexed", st.session_state.indexed_count)

        st.divider()
        if st.button("Reset conversation"):
            st.session_state.messages = []
            st.session_state.claude_history = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                with st.expander("🔧 Tool calls made"):
                    for tc in msg["tool_calls"]:
                        st.markdown(f"**{tc['name']}**({tc['input']})")
                        st.code(tc["result"][:1000])

    user_input = st.chat_input("Ask something...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                agent = _ensure_agent()
                response = agent.chat(user_input, history=st.session_state.claude_history)
                st.markdown(response.text)

                if response.tool_calls:
                    with st.expander("🔧 Tool calls made"):
                        for tc in response.tool_calls:
                            st.markdown(f"**{tc['name']}**({tc['input']})")
                            st.code(tc["result"][:1000])

                st.session_state.claude_history.append({"role": "user", "content": user_input})
                st.session_state.claude_history.append({"role": "assistant", "content": response.text})
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.text, "tool_calls": response.tool_calls}
                )
            except Exception as exc:
                st.error(f"Error: {exc}")


if __name__ == "__main__":
    main()
