"""Turn a natural-language request into a validated retrieval plan.

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

from typing import Any

from config import get_config
from llm.client import get_client
from llm.prompt import build_understanding_prompt

from .types import Criterion, QueryPlan, SubQuery

_SCHEMA_DESC = (
    "object with sub_queries (1..N objects containing query, keywords, filters) "
    "and criteria (objects containing criterion and source='extracted')"
)


def _normalise_space(value: str) -> str:
    return " ".join(value.split())


def _parse_filters(raw: Any) -> dict[str, object] | None:
    if raw in (None, {}):
        return None
    if not isinstance(raw, dict):
        raise ValueError("filters must be an object or null")

    parsed: dict[str, object] = {}
    for key in ("year_min", "year_max"):
        value = raw.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
            parsed[key] = value

    categories = raw.get("categories")
    if categories is not None:
        if not isinstance(categories, list) or not all(
            isinstance(item, str) and _normalise_space(item) for item in categories
        ):
            raise ValueError("categories must be a list of non-empty strings")
        parsed["categories"] = list(dict.fromkeys(_normalise_space(item) for item in categories))

    if (
        "year_min" in parsed
        and "year_max" in parsed
        and int(parsed["year_min"]) > int(parsed["year_max"])
    ):
        raise ValueError("year_min cannot exceed year_max")
    return parsed or None


def _parse_plan(raw: object, input_text: str, max_queries: int) -> QueryPlan:
    if not isinstance(raw, dict):
        raise ValueError("query plan must be a JSON object")
    raw_queries = raw.get("sub_queries")
    if not isinstance(raw_queries, list) or not 1 <= len(raw_queries) <= max_queries:
        raise ValueError(f"sub_queries must contain between 1 and {max_queries} items")

    queries: list[SubQuery] = []
    seen_queries: set[str] = set()
    for item in raw_queries:
        if not isinstance(item, dict):
            raise ValueError("each sub-query must be an object")
        query = item.get("query")
        if not isinstance(query, str) or not (query := _normalise_space(query)):
            raise ValueError("sub-query text must be non-empty")
        normalised_query = query.casefold()
        if normalised_query in seen_queries:
            continue
        seen_queries.add(normalised_query)

        keywords = item.get("keywords", "")
        if isinstance(keywords, list) and all(isinstance(word, str) for word in keywords):
            keywords = " ".join(keywords)
        if not isinstance(keywords, str):
            raise ValueError("keywords must be a string or list of strings")
        queries.append(
            SubQuery(
                query=query,
                keywords=_normalise_space(keywords),
                filters=_parse_filters(item.get("filters")),
            )
        )
    if not queries:
        raise ValueError("all generated sub-queries were duplicates")

    raw_criteria = raw.get("criteria", [])
    if not isinstance(raw_criteria, list):
        raise ValueError("criteria must be an array")
    criteria: list[Criterion] = []
    seen_criteria: set[str] = set()
    for item in raw_criteria:
        if not isinstance(item, dict):
            raise ValueError("each criterion must be an object")
        text = item.get("criterion")
        if not isinstance(text, str) or not (text := _normalise_space(text)):
            raise ValueError("criterion text must be non-empty")
        if text.casefold() in seen_criteria:
            continue
        seen_criteria.add(text.casefold())
        criteria.append(Criterion(criterion=text, source="extracted"))

    if not criteria:
        criteria = [Criterion(criterion=input_text, source="extracted")]
    return QueryPlan(input_text=input_text, sub_queries=queries, criteria=criteria)


def literal_plan(input_text: str) -> QueryPlan:
    """The plan without query understanding: the input as typed is the only sub-query and aspect.

    It is the fallback when the LLM fails twice, and the plan of every search with
    `orchestration.understanding.enabled=false` (D61), which makes no LLM call for it.
    """
    return QueryPlan(
        input_text=input_text,
        sub_queries=[SubQuery(query=input_text)],
        criteria=[Criterion(criterion=input_text, source="extracted")],
    )


def understand(input_text: str) -> QueryPlan:
    """Understand the user input and produce a retrieval plan.

    Args:
        input_text: A paragraph of text from the user.

    Returns:
        QueryPlan (sub-queries + criteria).

    Raises:
        RuntimeError: When the LLM is unavailable and no fallback is possible.
    """
    if not isinstance(input_text, str) or not (input_text := _normalise_space(input_text)):
        raise ValueError("input_text must be a non-empty string")

    max_queries = int(get_config()["orchestration"].get("num_sub_queries", 3))
    max_queries = max(1, min(max_queries, 5))
    client = get_client()
    messages = build_understanding_prompt(input_text)
    for _ in range(2):
        try:
            raw = client.chat_json(messages, _SCHEMA_DESC)
            return _parse_plan(raw, input_text, max_queries)
        except Exception:  # The documented fallback covers transport and schema failures.
            continue
    return literal_plan(input_text)
