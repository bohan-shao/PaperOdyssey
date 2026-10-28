"""Unified retrieval entry point: `retrieve()` is the only retrieval interface for other modules.

One query flows through (every step is a config switch, so the ablation matrix is config-only):

    sparse: BM25 top 150 ──┐  RRF union -> paper/section selection (50) -> cross-encoder
                           ├─ diverse top_k -> optional bounded parent excerpt
    dense: BGE top 150 ────┘  (all budgets and innovations are configurable)

- `retrieval.sparse.enabled` / `retrieval.dense.enabled`: with only one on, its list goes straight
  to the selection / rerank / top_k step (no fusion); at least one must be on.
- `retrieval.parallel`: sparse runs on a worker thread while dense runs on the calling thread.
- `retrieval.filters.enabled`: off = the filters argument is ignored (the Q3 ablation).
- `retrieval.rerank.enabled`: off = the fused (or single) list is cut to top_k.
- `retrieval.parent_doc.enabled`: the final hits carry their parent section text (passage_id stays
  the child's); applied after reranking, so the cross-encoder always scores the child text.

Hit stages tell the configuration level: "bm25" | "dense" (one retriever, no rerank), "hybrid"
(fused, no rerank), "rerank". `latency_ms` has one entry per stage that ran (query_ms, sparse_ms,
dense_ms, fusion_ms, selection_ms, rerank_ms, parent_ms) plus total_ms; channel timings include
query encoding and hydration.

Thread-safe: orchestration calls retrieve() from several threads at once; each component
serializes its own model / JVM access. The enabled components (BM25 index, dense index + encoder,
reranker, lookup store) are loaded one at a time under one lock before any parallel work, once
per process: loading imports pyserini / sentence-transformers, which both import transformers,
and two threads importing transformers at once fail ("cannot import name 'AutoTokenizer'").
Any failure in an enabled stage (index not built, model missing) raises RuntimeError naming the
stage: nothing falls back silently, so an ablation run can never quietly turn into another
configuration.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from config import get_config

from . import rerank, sparse
from .fusion import reciprocal_rank_fusion
from .lookup import get_lookup_store
from .query import query_views
from .selection import select_hits
from .types import RetrieveResult, RetrievalFilters, ScoredPassage

_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="retrieval")  # sparse beside dense
_COMPONENTS = (("sparse", True), ("dense", True), ("rerank", True), ("parent_doc", False))  # (switch, default)
_LOADED: set[str] = set()
_LOAD_LOCK = threading.Lock()


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def _sparse_search(query: str, filters: RetrievalFilters | None,
                   top_k: int) -> tuple[list[ScoredPassage], int]:
    return sparse.search(query, filters, top_k)


def _dense_search(query: str, filters: RetrievalFilters | None,
                  top_k: int) -> tuple[list[ScoredPassage], int]:
    from . import dense  # numpy/faiss/torch: imported on first use, not with the package

    return dense.search(query, filters, top_k)


def _rerank(query: str, candidates: list[ScoredPassage], top_k: int) -> list[ScoredPassage]:
    return rerank.rerank(query, candidates, top_k)


def _run(stage: str, fn: Any, *args: Any) -> Any:
    """Call one stage; any failure becomes a RuntimeError that names the stage."""
    try:
        return fn(*args)
    except Exception as exc:
        raise RuntimeError(f"{stage} failed: {exc}") from exc


def _loaders() -> dict[str, Callable[[], object]]:
    def load_dense() -> object:
        from . import dense

        return dense.get_dense_retriever()

    return {"sparse": sparse.get_sparse_retriever, "dense": load_dense,
            "rerank": rerank.get_reranker, "parent_doc": get_lookup_store}


def _ensure_loaded(cfg: dict) -> None:
    """Load every enabled component, one at a time, the first time it is needed."""
    wanted = [name for name, default in _COMPONENTS if cfg.get(name, {}).get("enabled", default)]
    if all(name in _LOADED for name in wanted):
        return
    with _LOAD_LOCK:
        loaders = _loaders()
        for name in wanted:
            if name not in _LOADED:
                _run(f"loading {name}", loaders[name])
                _LOADED.add(name)


def _candidates(query: str, filters: RetrievalFilters | None, cfg: dict,
                latency: dict[str, int], sparse_query: str | None = None) -> list[ScoredPassage]:
    """Sparse and/or dense top lists, fused when both are on."""
    use_sparse, use_dense = cfg["sparse"].get("enabled", True), cfg["dense"].get("enabled", True)
    if not (use_sparse or use_dense):
        raise RuntimeError("retrieval.sparse and retrieval.dense are both disabled")
    sparse_args = (query if sparse_query is None else sparse_query, filters, cfg["sparse"].get("top_k", 50))
    dense_args = (query, filters, cfg["dense"].get("top_k", 50))

    sparse_hits: list[ScoredPassage] = []
    dense_hits: list[ScoredPassage] = []
    pending = None
    if use_sparse and use_dense and cfg.get("parallel", True):
        pending = _POOL.submit(_run, "sparse retrieval", _sparse_search, *sparse_args)
    elif use_sparse:
        sparse_hits, latency["sparse_ms"] = _run("sparse retrieval", _sparse_search, *sparse_args)
    try:
        if use_dense:
            dense_hits, latency["dense_ms"] = _run("dense retrieval", _dense_search, *dense_args)
    except BaseException:
        # Offline experiments may close readers/restore config as soon as retrieve returns.
        # Drain the sibling worker before propagating the original dense failure.
        if pending is not None:
            pending.cancel()
            try:
                pending.result()
            except BaseException:
                pass
        raise
    if pending is not None:
        sparse_hits, latency["sparse_ms"] = pending.result()

    if not (use_sparse and use_dense):
        return sparse_hits if use_sparse else dense_hits
    fusion = cfg.get("fusion", {})
    start = time.perf_counter()
    fused = reciprocal_rank_fusion(sparse_hits, dense_hits, k=fusion.get("rrf_k", 60),
                                   weights=fusion.get("weights", (1.0, 1.0)),
                                   top_k=None if cfg.get("diversity", {}).get("enabled", False)
                                   else fusion.get("top_k", 50))
    latency["fusion_ms"] = _ms(start)
    return fused


def retrieve(
    query: str,
    filters: RetrievalFilters | None = None,
    top_k: int | None = None,
) -> RetrieveResult:
    """Unified retrieval entry point.

    Args:
        query: the search query (a sub-query produced by the orchestration layer's
            query understanding).
        filters: metadata filtering (year/category); ignored when retrieval.filters.enabled
            is false.
        top_k: number of final results to return (default retrieval.rerank.top_k, 10).

    Returns:
        RetrieveResult (hits + per-stage latencies).

    Raises:
        RuntimeError: when an index has not been built, a model or backend is unavailable, or
            both retrievers are disabled.
    """
    start = time.perf_counter()
    cfg = get_config()["retrieval"]
    top_k = cfg["rerank"].get("top_k", 10) if top_k is None else top_k
    if not cfg.get("filters", {}).get("enabled", True):
        filters = None
    latency: dict[str, int] = {}

    if top_k <= 0 or not query.strip():
        return RetrieveResult(hits=[], total_candidates=0, latency_ms={"total_ms": _ms(start)})
    stage_start = time.perf_counter()
    query, sparse_query = query_views(query, cfg.get("query_processing", {}))
    if cfg.get("query_processing", {}).get("enabled", False):
        latency["query_ms"] = _ms(stage_start)

    _ensure_loaded(cfg)
    candidates = _candidates(query, filters, cfg, latency, sparse_query)
    diversity = cfg.get("diversity", {})
    if diversity.get("enabled", False):
        stage_start = time.perf_counter()
        candidates = select_hits(candidates, cfg.get("fusion", {}).get("top_k", 50), diversity)
        latency["selection_ms"] = _ms(stage_start)
    else:
        # Recall depth is independent of the scoring budget, including single-channel runs.
        candidates = candidates[:max(0, cfg.get("fusion", {}).get("top_k", 50))]
    if cfg["rerank"].get("enabled", True):
        stage_start = time.perf_counter()
        hits = _run("rerank", _rerank, query, candidates,
                    len(candidates) if diversity.get("enabled", False) else top_k)
        latency["rerank_ms"] = _ms(stage_start)
    else:
        hits = candidates if diversity.get("enabled", False) else candidates[:top_k]
    if diversity.get("enabled", False):
        stage_start = time.perf_counter()
        hits = select_hits(hits, top_k, diversity, final=True)
        latency["selection_ms"] += _ms(stage_start)
    if cfg.get("parent_doc", {}).get("enabled", False):
        stage_start = time.perf_counter()
        hits = _run("parent-document lookup", get_lookup_store().with_parent_text, hits)
        latency["parent_ms"] = _ms(stage_start)
    latency["total_ms"] = _ms(start)
    return RetrieveResult(hits=hits, total_candidates=len(candidates), latency_ms=latency)


def warm_up() -> None:
    """Load every enabled index and model now, so the first real query is not the slow one."""
    retrieve("warm up")
