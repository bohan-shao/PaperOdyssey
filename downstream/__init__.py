"""Simple question answering grounded in retrieved paper passages."""

from __future__ import annotations

from .qa import DownstreamError, answer_from_search, run
from .types import DownstreamResult, EvidenceSource

__all__ = ["run", "answer_from_search", "DownstreamError", "DownstreamResult", "EvidenceSource"]
