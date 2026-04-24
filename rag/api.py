"""Public Python API for the KMS."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from rag.config import RAGConfig, KNOWLEDGE_COLLECTION
from rag.filters import build_where, date_to_int
from rag.retriever.vector import VectorRetriever
from rag.store.chroma_store import ChromaStore
from rag.store.json_store import JSONStore
from rag.types import ContextChunk, ContextWindow, FolderSummary, Hit, Inventory
from rag.utils.paths import extract_date

if TYPE_CHECKING:
    from langchain_core.documents import Document


_store_cache: dict[tuple[str, str], ChromaStore] = {}
_store_cache_lock = threading.Lock()


def _get_store(cfg: RAGConfig) -> ChromaStore:
    """Return a process-wide ChromaStore, one per (persist_dir, collection).

    Why: chromadb's SharedSystemClient caches a System per persist_dir and
    pops it on client release. Instantiating a new Chroma client per search
    race-conditions against that cache under LangGraph's ToolNode
    ThreadPoolExecutor (KeyError on `_identifier_to_system[identifier]`).
    """
    key = (cfg.persist_dir, KNOWLEDGE_COLLECTION)
    with _store_cache_lock:
        store = _store_cache.get(key)
        if store is None:
            store = ChromaStore(KNOWLEDGE_COLLECTION, cfg)
            _store_cache[key] = store
        return store


def _doc_to_hit(doc: "Document") -> Hit:
    """Convert a LangChain document into a framework-neutral Hit."""
    metadata = dict(doc.metadata or {})
    raw_tags = metadata.get("tags")
    if isinstance(raw_tags, str):
        try:
            tags = json.loads(raw_tags)
        except json.JSONDecodeError:
            tags = []
    elif isinstance(raw_tags, list):
        tags = raw_tags
    else:
        tags = []

    return Hit(
        pid=str(metadata.get("pid", "")),
        chunk_id=int(metadata.get("chunk_id", 0) or 0),
        text=doc.page_content,
        file_path=str(metadata.get("file_path", "")),
        category=metadata.get("category"),
        file_type=metadata.get("file_type"),
        folder=metadata.get("folder"),
        date=int(metadata.get("date", 0) or 0),
        tags=tags,
        metadata=metadata,
    )


def search(
    query: str,
    *,
    k: int = 5,
    folder_prefix: str | None = None,
    category: str | None = None,
    file_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    config: RAGConfig | None = None,
) -> list[Hit]:
    """Semantic search with optional metadata filters.

    `folder_prefix` is a strict ``str.startswith`` match on the chunk's
    folder field, matching `list_chunks` semantics. Because Chroma cannot
    express prefix matching in a metadata where-clause, this filter is
    applied in Python after vector retrieval; extra candidates are fetched
    up front so the returned list can still reach `k` entries in the common
    case.
    """
    cfg = config or RAGConfig()
    where = build_where(
        category=category,
        file_type=file_type,
        date_from=date_from,
        date_to=date_to,
    )
    store = _get_store(cfg)
    retriever = VectorRetriever(store)
    retrieval_k = k * 3 if folder_prefix else k
    docs = retriever.retrieve(query, k=retrieval_k, where=where)
    hits = [_doc_to_hit(doc) for doc in docs]
    if folder_prefix is not None:
        prefix = folder_prefix.rstrip("/")
        hits = [h for h in hits if (h.folder or "").startswith(prefix)]
    return hits[:k]


def explore(
    *,
    category: str | None = None,
    config: RAGConfig | None = None,
) -> Inventory:
    """Return a structured inventory of what's in the knowledge base."""
    cfg = config or RAGConfig()
    meta_path = Path(cfg.folder_meta_path())
    if not meta_path.exists():
        return Inventory(categories={}, tags=[], date_range=None, folders=[])

    with meta_path.open("r", encoding="utf-8") as file_obj:
        folder_meta: dict[str, dict] = json.load(file_obj)

    categories: dict[str, int] = {}
    all_tags: set[str] = set()
    dates: list[int] = []
    folders: list[FolderSummary] = []

    for folder_rel, meta in folder_meta.items():
        tags = meta.get("tags", [])
        summary = meta.get("summary", "")
        cat = tags[0] if tags else "unknown"

        categories[cat] = categories.get(cat, 0) + 1
        all_tags.update(tags)

        date = extract_date(folder_rel)
        if date:
            dates.append(date)

        if category and cat != category:
            continue

        folders.append(
            FolderSummary(
                folder=folder_rel,
                category=cat,
                tags=list(tags),
                summary=summary,
            )
        )

    date_range = (min(dates), max(dates)) if dates else None
    return Inventory(
        categories=categories,
        tags=sorted(all_tags),
        date_range=date_range,
        folders=folders,
    )


def list_chunks(
    *,
    folder_prefix: str | None = None,
    pid: str | None = None,
    category: str | None = None,
    file_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    config: RAGConfig | None = None,
) -> list[Hit]:
    """Enumerate every stored chunk, optionally filtered.

    Filter arguments mirror `search` so agents and eval harnesses can move
    between ranked and unranked retrieval without relearning the surface.
    All filters are applied in Python over the JSON backup (no embedding
    model or Chroma round-trip). `folder_prefix` is a strict
    ``str.startswith`` match; `date_from` / `date_to` take ``YYYY-MM-DD``
    strings and drop chunks whose `date == 0` (no YYYYMMDD folder on disk).
    """
    cfg = config or RAGConfig()
    json_store = JSONStore(cfg.raw_json_path())
    docs = json_store.get(pid=pid)
    hits = [_doc_to_hit(doc) for doc in docs]
    if folder_prefix is not None:
        prefix = folder_prefix.rstrip("/")
        hits = [h for h in hits if (h.folder or "").startswith(prefix)]
    if category is not None:
        hits = [h for h in hits if h.category == category]
    if file_type is not None:
        hits = [h for h in hits if h.file_type == file_type]
    if date_from is not None:
        lo = date_to_int(date_from)
        hits = [h for h in hits if h.date and h.date >= lo]
    if date_to is not None:
        hi = date_to_int(date_to)
        hits = [h for h in hits if h.date and h.date <= hi]
    return hits


def get_context(
    pid: str,
    chunk_id: int,
    *,
    window: int = 1,
    config: RAGConfig | None = None,
) -> ContextWindow | None:
    """Return the target chunk plus neighbours from the same document."""
    cfg = config or RAGConfig()
    json_store = JSONStore(cfg.raw_json_path())
    bounded_window = min(max(window, 0), 3)
    all_docs = json_store.get(pid=pid)
    if not all_docs:
        return None

    all_docs.sort(key=lambda doc: doc.metadata.get("chunk_id", 0))
    target_idx = next(
        (idx for idx, doc in enumerate(all_docs) if doc.metadata.get("chunk_id") == chunk_id),
        None,
    )
    if target_idx is None:
        available = [doc.metadata.get("chunk_id") for doc in all_docs]
        raise ValueError(
            f"chunk_id={chunk_id} not found in pid={pid!r}. Available: {available}"
        )

    start = max(0, target_idx - bounded_window)
    end = min(len(all_docs), target_idx + bounded_window + 1)
    chunks = [
        ContextChunk(
            chunk_id=int(doc.metadata.get("chunk_id", 0) or 0),
            text=doc.page_content,
            is_target=(doc.metadata.get("chunk_id") == chunk_id),
        )
        for doc in all_docs[start:end]
    ]
    return ContextWindow(
        pid=pid,
        target_chunk_id=chunk_id,
        chunks=chunks,
        total_chunks_in_doc=len(all_docs),
    )
