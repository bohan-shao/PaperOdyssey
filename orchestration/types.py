"""Orchestration type contracts (synced with docs/SCHEMA.md §4–§6).

Changing this file requires syncing docs/SCHEMA.md + all callers + contract tests, via a reviewed PR.
"""

from __future__ import annotations

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
        score: Score from 1–5 (5 is best).
        reasons: Review reasons (for UI display and report case studies).
        covered: Aspects of the input requirements covered by this paper.
        missing: Aspects not covered by this paper that other papers still need to fill in.
        round: The round in which the review happened (starting at 1).
        evidence: The review basis (title/abstract snippets/top passage IDs), may be empty.
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
    """

    query_id: str
    input_text: str
    query_plan: QueryPlan
    papers: list[ReviewRecord]
    coverage_report: dict[str, object]
    latency_by_stage: dict[str, int] = field(default_factory=dict)
