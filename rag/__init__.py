"""Framework-neutral RAG library."""

from rag.api import explore, get_context, list_chunks, search
from rag.config import RAGConfig
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
    "list_chunks",
    "search",
]
