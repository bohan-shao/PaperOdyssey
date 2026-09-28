"""Simple question answering grounded in retrieved paper passages."""

from __future__ import annotations

from .qa import DownstreamError, run
from .types import DownstreamResult, EvidenceSource

__all__ = ["run", "DownstreamError", "DownstreamResult", "EvidenceSource"]
