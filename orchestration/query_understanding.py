"""Query understanding: the LLM splits a paragraph of text into sub-queries + keywords + requirement criteria. TODO.

Implementation requirements (see orchestration/README.md for details):
- Input: the user's input_text (a paragraph of text).
- Output: QueryPlan (1–config.num_sub_queries sub-queries + a criteria list).
- Calls llm.client.chat_json() (prompt templates in llm/prompt.py),
  requiring the LLM to strictly output JSON ({sub_queries, criteria}); retry once on parse failure,
  and if it still fails, fall back to a single sub-query (query=the original text).
- Sub-query fields: query (natural language), keywords (BM25 keyword string),
  filters (extracted from the input, e.g. "after 2023" → year_min, category words → categories).
- Criteria: source marks fixed (hits a fixed rubric item) or extracted (input-specific).
- Record the understanding-stage latency (milliseconds) for latency_by_stage.

Constraints: this module does not retrieve; it only produces a QueryPlan.
"""

from __future__ import annotations

from .types import QueryPlan


def understand(input_text: str) -> QueryPlan:
    """Understand the user input and produce a retrieval plan.

    Args:
        input_text: A paragraph of text from the user.

    Returns:
        QueryPlan (sub-queries + criteria).

    Raises:
        RuntimeError: When the LLM is unavailable and no fallback is possible.
    """
    raise NotImplementedError  # TODO: implement, see orchestration/README.md
