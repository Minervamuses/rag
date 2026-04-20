"""Central configuration for the RAG workspace."""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Single collection for all knowledge chunks
KNOWLEDGE_COLLECTION = "knowledge"


def _default_persist_dir() -> str:
    """Resolve the store directory.

    The store is a rag-specific artifact (Chroma + JSON); it conceptually
    belongs to rag, not the host. Default to ``store/`` at the rag repo root,
    so running ingest from any host lands data in rag's own territory rather
    than polluting the host project. Order:
      1. ``KMS_STORE_DIR`` env var (explicit override, host may still pin
         it wherever it wants).
      2. ``<rag repo root>/store/``.
    """
    env = os.environ.get("KMS_STORE_DIR")
    if env:
        return env
    rag_repo_root = Path(__file__).resolve().parent.parent
    return str(rag_repo_root / "store")


@dataclass
class KMSConfig:
    """Central configuration for the RAG workspace."""

    # Storage
    persist_dir: str = field(default_factory=_default_persist_dir)

    # Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 100
    encoding_model: str = "o200k_base"

    # Embedding
    embed_model: str = "bge-m3"

    # LLM
    llm_model: str = "z-ai/glm-5"

    # Evaluation LLMs
    gen_llm_model: str = "google/gemini-3.1-pro-preview"
    judge_llm_model: str = "openai/gpt-5.2"
    filter_llm_model: str = "llama3.1:8b"

    # Retrieval
    default_k: int = 10

    # Agent context controls
    agent_max_messages: int = 20
    agent_max_tool_interactions: int = 4

    # Agent turn compaction
    agent_turns_per_compaction: int = 10
    agent_compaction_model: str | None = None
    agent_compaction_max_tokens: int = 800

    # Collection names
    raw_collection: str = "raw"

    def raw_json_path(self) -> str:
        """Path to the raw chunks JSON file."""
        return f"{self.persist_dir}/raw.json"

    def folder_meta_path(self) -> str:
        """Path to the folder metadata JSON file."""
        return f"{self.persist_dir}/folder_meta.json"
