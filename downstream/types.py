"""Cited paper-QA contracts; see docs/SCHEMA.md section 10."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from orchestration.types import FinalResult


@dataclass(frozen=True)
class EvidenceSource:
    """A numbered passage supplied to the answer model (text is the exact excerpt sent)."""

    number: int
    paper_id: str
    passage_id: str
    title: str
    year: int | None
    section: str
    url: str | None
    text: str


@dataclass(frozen=True)
class DownstreamResult:
    query_id: str
    input_text: str
    answer: str
    status: Literal["answered", "insufficient_evidence"]
    sources: list[EvidenceSource]
    cited_source_numbers: list[int]
    search_result: FinalResult
    latency_by_stage: dict[str, int] = field(default_factory=dict)
    prompt_version: str = "1.0.0"
