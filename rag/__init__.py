"""Framework-neutral RAG library."""

from rag.api import explore, get_context, list_chunks, search
from rag.config import RAGConfig
from rag.types import ContextChunk, ContextWindow, FolderSummary, Hit, Inventory

__all__ = [
    "ContextChunk",
    "ContextWindow",
    "FolderSummary",
    "Hit",
    "Inventory",
    "RAGConfig",
    "explore",
    "get_context",
    "list_chunks",
    "search",
]
