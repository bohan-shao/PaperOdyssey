"""Pyserini (Lucene) BM25 sparse retrieval. TODO.

Implementation requirements (see retrieval/README.md for details):
- Build index: `build_index(passages_path, index_dir)` converts passages.jsonl into a
  Lucene index with Pyserini; text is the search field, year/categories are filter fields.
- Search: `search(query, filters, top_k) -> list[ScoredPassage]`;
  filters constrain via Lucene filters (year/category) and do not participate in scoring.
- Scoring: native BM25 scores; ScoredPassage.stage="bm25".
- Latency: record the search time (milliseconds) and return it with the results (pipeline aggregates).
- Fallback: when Java/PyLucene is unavailable, implement the same interface with rank-bm25
  (in-module branch, config `retrieval.sparse.backend: pyserini|rankbm25`); note differences
  in the report.

Constraints: SQL LIKE is forbidden; index and queries must consistently use the same backend.
"""

from __future__ import annotations

from pathlib import Path

from .types import RetrievalFilters, ScoredPassage


def build_index(passages_path: str | Path, index_dir: str | Path) -> None:
    """Build the BM25 index from passages.jsonl.

    Args:
        passages_path: data/processed/passages.jsonl.
        index_dir: data/indices/bm25.

    Raises:
        RuntimeError: when index construction fails (e.g. backend unavailable).
    """
    raise NotImplementedError  # TODO: implement; see retrieval/README.md


def search(
    query: str,
    filters: RetrievalFilters | None,
    top_k: int,
) -> tuple[list[ScoredPassage], int]:
    """BM25 search.

    Args:
        query: the search query (a keyword string or natural language, provided by
            the orchestration layer).
        filters: metadata filtering (None = no filtering).
        top_k: number of results to return.

    Returns:
        (hits, latency_ms).
    """
    raise NotImplementedError  # TODO: implement; see retrieval/README.md
