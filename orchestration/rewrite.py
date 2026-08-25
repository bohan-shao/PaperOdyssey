"""Feedback-based rewriting: generate supplementary queries from the missing aspects found in review. TODO.

Implementation requirements (see orchestration/README.md for details):
- Input: the aggregated list of missing aspects + the list of existing sub-queries (to avoid duplicates).
- Output: 1–2 supplementary SubQuery items (LLM-generated; the prompt requires semantics distinct from existing queries).
- Calls llm.client.chat_json(); failure fallback: no supplementary queries (skip rewriting for this round).
- Record the rewrite-stage latency.

Constraints: supplementary queries are added to the next round of retrieval; this module neither retrieves nor reviews.
"""

from __future__ import annotations

from .types import SubQuery


def rewrite(missing_aspects: list[str], existing_queries: list[str]) -> list[SubQuery]:
    """Generate supplementary retrieval queries from missing aspects.

    Args:
        missing_aspects: Uncovered aspects aggregated from review (deduplicated).
        existing_queries: The list of existing sub-query texts (to avoid duplication).

    Returns:
        A list of supplementary SubQuery items (0–2; 0 means no rewriting is needed).
    """
    raise NotImplementedError  # TODO: implement, see orchestration/README.md
