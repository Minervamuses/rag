"""Framework-neutral RAG library."""

from rag.api import explore, get_context, list_chunks, search
from rag.cli.ingest import ingest_repo, ingest_single
from rag.config import RAGConfig
from rag.sync import list_diff, prune_orphans
from rag.tools import TOOL_SCHEMAS, dispatch
from rag.types import ContextChunk, ContextWindow, FolderSummary, Hit, Inventory

__all__ = [
    "ContextChunk",
    "ContextWindow",
    "FolderSummary",
    "Hit",
    "Inventory",
    "RAGConfig",
    "TOOL_SCHEMAS",
    "dispatch",
    "explore",
    "get_context",
    "ingest_repo",
    "ingest_single",
    "list_chunks",
    "list_diff",
    "prune_orphans",
    "search",
]
