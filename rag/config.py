"""Central configuration for the RAG library.

Only settings rag itself needs belong here. Hosts (e.g. the ``app`` agent
repo) layer their own config on top via subclassing.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

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
class RAGConfig:
    """Configuration for the rag library (storage, chunking, embedding, tagger)."""

    # Storage
    persist_dir: str = field(default_factory=_default_persist_dir)

    # Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 100
    encoding_model: str = "o200k_base"

    # Embedding
    embed_model: str = "bge-m3"

    # Retrieval
    default_k: int = 10

    # LLM used by rag's own tagger
    tagger_model: str = "z-ai/glm-5"

    # Collection names
    raw_collection: str = "raw"

    def raw_json_path(self) -> str:
        """Path to the raw chunks JSON file."""
        return f"{self.persist_dir}/raw.json"

    def folder_meta_path(self) -> str:
        """Path to the folder metadata JSON file."""
        return f"{self.persist_dir}/folder_meta.json"
