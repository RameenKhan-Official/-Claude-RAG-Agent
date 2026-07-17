"""
Central configuration for the Claude RAG Agent.

All tunable parameters are read from environment variables (with sane
defaults) so the app can be configured for different environments
(local dev, CI, deployment) without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # --- Anthropic / Claude -------------------------------------------------
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    claude_model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "1024")))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.3")))
    max_agent_turns: int = field(default_factory=lambda: int(os.getenv("MAX_AGENT_TURNS", "6")))

    # --- Ingestion / chunking ------------------------------------------------
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "120")))

    # --- Embeddings -----------------------------------------------------------
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # --- Retrieval --------------------------------------------------------------
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))

    # --- Vector store persistence ------------------------------------------------
    index_dir: str = field(default_factory=lambda: os.getenv("INDEX_DIR", "storage/index"))


settings = Settings()
