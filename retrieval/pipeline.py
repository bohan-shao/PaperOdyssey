"""Unified retrieval entry point (interface discipline: the orchestration layer may only call here). TODO.

Implementation requirements (see retrieval/README.md for details):
- `retrieve(query, filters, top_k) -> RetrieveResult`:
  1) run sparse/dense in parallel (ThreadPoolExecutor or asyncio, config `retrieval.parallel`)
  2) each top-50 → fusion (RRF)
  3) optional rerank (`retrieval.rerank.enabled`) → top-10
  4) aggregate per-stage latencies (sparse_ms/dense_ms/fusion_ms/rerank_ms/total_ms)
- Toggles map to the ablation matrix (EVALUATION.md §2): the four configurations only
  change config; this function has no hardcoded branches.
- Result fields per SCHEMA.md §8 (stage marks the current configuration level).

Constraints: this file is the system's only retrieval entry point; verify with small
corpus runs first, then real indices.
"""

from __future__ import annotations

from .types import RetrieveResult, RetrievalFilters


def retrieve(
    query: str,
    filters: RetrievalFilters | None = None,
    top_k: int = 10,
) -> RetrieveResult:
    """Unified retrieval entry point.

    Args:
        query: the search query (a sub-query produced by the orchestration layer's
            query understanding).
        filters: metadata filtering (year/category).
        top_k: number of final results to return (default 10).

    Returns:
        RetrieveResult (hits + per-stage latencies).

    Raises:
        RuntimeError: when the index has not been built or all retrieval backends are
            unavailable.
    """
    raise NotImplementedError  # TODO: implement; see retrieval/README.md
