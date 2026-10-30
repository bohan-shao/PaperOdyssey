"""Manage retrieval components owned by a dedicated offline experiment process."""

from __future__ import annotations

from retrieval import dense, lookup, pipeline, rerank, sparse


def require_fresh_runtime() -> None:
    """Refuse components loaded with an earlier config instead of silently timing stale models."""
    if (pipeline._LOADED or lookup._STORE is not None or sparse._RETRIEVER is not None
            or dense._RETRIEVER is not None or dense._ENCODER is not None or rerank._RERANKER is not None):
        raise RuntimeError("run the experiment in a fresh dedicated process")


def clear_runtime(*, release_models: bool = False) -> None:
    """Release experiment-owned readers, optionally models; never call in an online server."""
    stores = {lookup._STORE}
    for retriever in (sparse._RETRIEVER, dense._RETRIEVER):
        if retriever is not None:
            stores.add(retriever._lookup)
    sparse._RETRIEVER = dense._RETRIEVER = None
    for store in stores:
        if store is not None:
            store.close()
    lookup._STORE = None
    pipeline._LOADED.clear()
    if release_models:
        dense._ENCODER = None
        rerank._RERANKER = None
