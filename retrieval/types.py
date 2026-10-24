"""Retrieval type contracts (kept in sync with docs/SCHEMA.md).

Any change to this file must be synced with: docs/SCHEMA.md + callers + contract
tests, via a cross-group PR.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievalFilters:
    """Metadata filtering conditions (Q3 innovation: timeline/category retrieval).

    year_min/year_max and categories are all optional; None means no filtering.
    """

    year_min: int | None = None
    year_max: int | None = None
    categories: list[str] | None = None


@dataclass(frozen=True)
class ScoredPassage:
    """A retrieved passage hit (a hits element in SCHEMA.md §8).

    Attributes:
        passage_id: globally unique passage ID (SCHEMA.md §2).
        paper_id: ID of the containing paper.
        text: child text, or a bounded parent excerpt in parent-document mode; a failed
            anchor or oversized child preserves the child (SCHEMA.md section 9).
        score: the score at the current stage (BM25 score / cosine similarity / rerank score).
        rank: the rank at the current stage (1-based).
        stage: the stage that produced this record ("bm25"|"dense"|"hybrid"|"rerank").
        title/year/categories: paper-level metadata (for UI display and review).
        parent_section_id: the parent unit ID in parent-document mode (may be None).
    """

    passage_id: str
    paper_id: str
    text: str
    score: float
    rank: int
    stage: str
    title: str = ""
    year: int | None = None
    categories: list[str] = field(default_factory=list)
    parent_section_id: str | None = None


@dataclass(frozen=True)
class RetrieveResult:
    """The complete return of a single retrieve() call.

    Attributes:
        hits: the sorted top-k passages (rank 1..top_k).
        total_candidates: selected pool size before reranking/final truncation, bounded
            by fusion.top_k for hybrid and single-channel retrieval, even without reranking.
        latency_ms: per-stage latencies (for Q2 timing / report performance).
    """

    hits: list[ScoredPassage]
    total_candidates: int
    latency_ms: dict[str, int] = field(default_factory=dict)
