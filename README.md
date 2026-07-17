# 🤖 Claude RAG Agent

A document-grounded chatbot that combines **retrieval-augmented generation (RAG)**
with an **agentic tool-use loop**, built on the Claude API. Upload documents,
and the agent decides *for itself* when to search your knowledge base versus
when to reach for a calculator tool — instead of a fixed retrieve-then-generate
pipeline, Claude drives the loop.

## Why this project

Most RAG demos hard-code a single "retrieve, then generate" step. This one
gives Claude actual tools (`search_documents`, `calculate`) and lets the model
decide when to call them, how many times, and how to combine the results —
the same pattern used in production agentic systems. It's designed to be
readable end-to-end in one sitting, with no framework (LangChain, LlamaIndex)
in the way of understanding what's actually happening on each API call.

## Architecture

```
                    ┌─────────────────┐
   User message ──▶ │   ClaudeAgent    │
                    │  (tool-use loop) │
                    └───────┬─────────┘
                            │ tool_use: search_documents(query)
                            ▼
                    ┌─────────────────┐
                    │    Retriever     │
                    │ (embed + search) │
                    └───────┬─────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
          ┌─────────────┐     ┌─────────────────┐
          │  Embedder    │     │ FAISSVectorStore │
          │ (Sentence-   │     │  (similarity      │
          │ Transformers)│     │   search + meta)  │
          └─────────────┘     └─────────────────┘

   Document upload ──▶ document_loader (load + chunk) ──▶ Retriever.index_files
```

**Flow of a single turn:**
1. The user's message + conversation history + tool schemas are sent to Claude.
2. If Claude emits a `tool_use` block (e.g. `search_documents`), the agent runs
   the corresponding Python function locally and sends the result back as a
   `tool_result`.
3. This repeats — Claude can call `search_documents` and then `calculate` in
   the same turn — until Claude returns a plain-text answer or a max-turn
   safety limit is hit.

## Features

- 🔎 **Retrieval-augmented generation** — PDF/TXT/Markdown ingestion, chunking,
  Sentence-Transformer embeddings, and a FAISS similarity index.
- 🛠️ **Agentic tool use** — Claude decides when to call `search_documents` vs.
  `calculate` vs. answering directly, using the real Anthropic tool-use API.
- 🧩 **Modular, swappable design** — embedder, vector store, and LLM client are
  each behind a small interface, so any piece (e.g. FAISS → Pinecone) can be
  swapped without touching the others.
- ✅ **Real test suite** — 25+ pytest tests covering chunking edge cases, the
  FAISS wrapper, the safe expression evaluator, and the full tool-use loop
  (mocked, no API key needed to run tests).
- 🖥️ **Two front ends** — a Streamlit chat UI and a plain terminal CLI demo.
- ⚙️ **CI on every push** — GitHub Actions runs `ruff` and the full test suite
  on Python 3.10 and 3.11.

## Project structure

```
claude-rag-agent/
├── app.py                        # `streamlit run app.py` entry point
├── src/
│   ├── config.py                 # env-driven settings (model, chunk size, etc.)
│   ├── app.py                    # Streamlit UI
│   ├── ingestion/
│   │   └── document_loader.py    # load + chunk .txt/.md/.pdf
│   ├── embeddings/
│   │   └── embedder.py           # Sentence-Transformers wrapper
│   ├── vectorstore/
│   │   └── faiss_store.py        # FAISS index + metadata, save/load
│   ├── retrieval/
│   │   └── retriever.py          # embed + index + search, glued together
│   └── agent/
│       ├── tools.py              # tool schemas + safe implementations
│       └── claude_agent.py       # the tool-use loop around the Anthropic API
├── scripts/
│   └── cli_demo.py               # terminal chat demo using the sample docs
├── data/sample_docs/             # sample document for a quick first run
├── tests/                        # pytest suite (mocks the Anthropic client)
├── .github/workflows/tests.yml   # CI: lint + test on push/PR
├── requirements.txt
├── pyproject.toml                # ruff config
└── .env.example
```

## Getting started

### Prerequisites
- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
git clone https://github.com/<your-username>/claude-rag-agent.git
cd claude-rag-agent
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

### Run the Streamlit app

```bash
streamlit run app.py
```

Upload a `.txt`, `.md`, or `.pdf` file in the sidebar, click **Build/Update
Index**, then chat. Expand "🔧 Tool calls made" under any response to see
exactly which tools Claude invoked and what they returned.

### Run the terminal demo (no upload needed)

```bash
python scripts/cli_demo.py
```

This indexes `data/sample_docs/company_handbook.txt` automatically — try
asking *"How many PTO days do I get?"* or *"What's the reimbursement policy
for a $200 expense?"*

### Run the tests

```bash
pytest
```

The test suite mocks the Anthropic client and the embedding model, so it runs
fully offline — no API key required.

## Configuration

All settings are environment variables (see `.env.example`), read once at
startup in `src/config.py`:

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Claude API key |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Model used for generation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Document chunking parameters |
| `TOP_K` | `4` | Chunks retrieved per query |
| `MAX_AGENT_TURNS` | `6` | Safety limit on tool-use round-trips per turn |

## Design decisions worth calling out in an interview

- **No orchestration framework.** The tool-use loop in `claude_agent.py` is
  ~50 lines of plain API calls. That's a deliberate choice: it's easier to
  reason about, debug, and explain than a framework abstraction, and it
  mirrors exactly what the Anthropic docs show for tool use.
- **Character-based chunking, not token-based.** Keeps the ingestion path
  dependency-free (no tokenizer download) — a reasonable tradeoff for a
  project that should run anywhere with just `pip install`.
- **A safe, sandboxed calculator**, not `eval()`. `tools.py` parses
  expressions with Python's `ast` module and only permits a fixed whitelist of
  arithmetic operators — it will reject arbitrary code execution attempts.
- **Lazy imports for heavy dependencies** (`faiss`, `sentence-transformers`)
  keep module import fast and let most of the test suite run without them
  installed at all.

## Possible extensions

- Swap FAISS for a managed vector DB (Pinecone, Weaviate, pgvector) behind the
  same `FAISSVectorStore`-shaped interface.
- Add a `web_search` tool alongside `search_documents` and `calculate`.
- Stream responses token-by-token using the Anthropic streaming API.
- Add conversation persistence (SQLite) so chat history survives a restart.

## License

MIT — see [LICENSE](./LICENSE).

