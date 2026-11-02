"""Orchestration type contracts (synced with docs/SCHEMA.md §4–§6).

Changing this file requires syncing docs/SCHEMA.md + all callers + contract tests, via a reviewed PR.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SubQuery:
    """A single retrieval sub-query.

    Attributes:
        query: Natural-language search statement (used by dense/rerank).
        keywords: BM25 keyword string (may be empty; falls back to query text).
        filters: Metadata filters (year/category), None means no restriction.
    """

    query: str
    keywords: str = ""
    filters: dict[str, object] | None = None


@dataclass(frozen=True)
class Criterion:
    """A single review criterion.

    Attributes:
        criterion: Criterion text (e.g. "must be published after 2023").
        source: "fixed" (fixed rubric item) | "extracted" (from the user input).
    """

    criterion: str
    source: str = "extracted"


@dataclass(frozen=True)
class QueryPlan:
    """Query-understanding output (SCHEMA.md §4)."""

    input_text: str
    sub_queries: list[SubQuery] = field(default_factory=list)
    criteria: list[Criterion] = field(default_factory=list)


@dataclass(frozen=True)
class ReviewRecord:
    """Review result for one paper (SCHEMA.md §5).

    Attributes:
        paper_id: Paper ID.
        verdict: "pass" | "reject".
        score: Score from 1–5 (5 is best); 0 = not reviewed (the plain RAG baseline, D55).
        reasons: Review reasons (for UI display and report case studies).
        covered: Aspects of the input requirements covered by this paper.
        missing: Aspects not covered by this paper that other papers still need to fill in.
        round: The round in which the review happened (starting at 1). For a paper reviewed again
            (D53), the round in which its kept verdict was first reached.
        evidence: The review basis (title/abstract snippets/top passage IDs), may be empty. A
            re-reviewed paper adds `review_rounds`, every round in which it was reviewed (D53).
    """

    paper_id: str
    verdict: str
    score: int
    reasons: str = ""
    covered: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    round: int = 1
    evidence: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalResult:
    """The complete output of one Paper Search run_query call (SCHEMA.md §6).

    This is the search result only — it contains no generated answer. The downstream
    task (downstream/) consumes this and produces its own task output (D24).
    `papers` holds passing records in `passing_order_key` order; `reviews` retains one pass/reject
    record per reviewed paper (a re-reviewed paper's reviews merged, D53); `retrieval_ranks` maps every reviewed paper to its best retrieval rank
    in this run (D47). Legacy cached results have an unavailable trace (`reviews=[]`) and no ranks.
    """

    query_id: str
    input_text: str
    query_plan: QueryPlan
    papers: list[ReviewRecord]
    coverage_report: dict[str, object]
    latency_by_stage: dict[str, int] = field(default_factory=dict)
    reviews: list[ReviewRecord] = field(default_factory=list)
    retrieval_ranks: dict[str, int] = field(default_factory=dict)


Progress = Callable[[str, Mapping[str, object]], None]
"""Optional step callback of `run_query` and `downstream.run` (SCHEMA.md §6.1, D58).

Called with an event name and its details from the thread running the search; the UI shows each
step as it happens. It never changes the result.
"""


def passing_order_key(paper_id: str, score: int, retrieval_ranks: Mapping[str, int]) -> tuple[int, float, str]:
    """Sort key of `FinalResult.papers` (SCHEMA.md §6, D47).

    Descending review score, then the paper's best retrieval rank (1 = the top passage of some
    sub-query's retrieve() result; a paper without a rank, as in legacy results, sorts last), then
    paper ID. Review scores are 1-5 integers (0 for every paper of the plain RAG baseline, D55),
    so ties are common: the rank keeps the retriever's order among them instead of arXiv-ID order.
    """
    rank = retrieval_ranks.get(paper_id)
    return -score, float(rank) if rank is not None else float("inf"), paper_id
