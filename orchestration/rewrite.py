"""Generate supplementary retrieval queries from uncovered aspects.

Implementation requirements (see orchestration/README.md for details):
- Input: the aggregated list of missing aspects + the list of existing sub-queries (to avoid duplicates).
- Output: 1–2 supplementary SubQuery items (LLM-generated; the prompt requires semantics distinct from existing queries).
- Calls llm.client.chat_json(); failure fallback: no supplementary queries (skip rewriting for this round).
- Record the rewrite-stage latency.

Constraints: supplementary queries are added to the next round of retrieval; this module neither retrieves nor reviews.
"""

from __future__ import annotations

from typing import Any

from config import get_config
from llm.client import get_client
from llm.prompt import build_rewrite_prompt

from .types import SubQuery

_SCHEMA_DESC = "object with sub_queries containing query, keywords, and optional filters"


def _normalise_space(value: str) -> str:
    return " ".join(value.split())


def _filters(raw: Any) -> dict[str, object] | None:
    if raw in (None, {}):
        return None
    if not isinstance(raw, dict):
        raise ValueError("filters must be an object")
    parsed: dict[str, object] = {}
    for key in ("year_min", "year_max"):
        value = raw.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
            parsed[key] = value
    categories = raw.get("categories")
    if categories is not None:
        if not isinstance(categories, list) or not all(isinstance(item, str) for item in categories):
            raise ValueError("categories must be an array of strings")
        parsed["categories"] = [_normalise_space(item) for item in categories if item.strip()]
    return parsed or None


def rewrite(missing_aspects: list[str], existing_queries: list[str]) -> list[SubQuery]:
    """Generate supplementary retrieval queries from missing aspects.

    Args:
        missing_aspects: Uncovered aspects aggregated from review (deduplicated).
        existing_queries: The list of existing sub-query texts (to avoid duplication).

    Returns:
        A list of supplementary SubQuery items (0–2; 0 means no rewriting is needed).
    """
    missing = list(dict.fromkeys(_normalise_space(item) for item in missing_aspects if item.strip()))
    if not missing:
        return []

    cfg = get_config()["orchestration"].get("rewrite", {})
    max_queries = max(0, int(cfg.get("max_queries_per_round", 2)))
    if max_queries == 0:
        return []
    existing = list(dict.fromkeys(_normalise_space(item) for item in existing_queries if item.strip()))
    try:
        raw = get_client().chat_json(build_rewrite_prompt(missing, existing), _SCHEMA_DESC)
        if not isinstance(raw, dict) or not isinstance(raw.get("sub_queries"), list):
            raise ValueError("rewrite response must contain sub_queries")

        seen = {item.casefold() for item in existing}
        queries: list[SubQuery] = []
        for item in raw["sub_queries"]:
            if not isinstance(item, dict):
                raise ValueError("each supplementary query must be an object")
            text = item.get("query")
            if not isinstance(text, str) or not (text := _normalise_space(text)):
                raise ValueError("supplementary query text must be non-empty")
            if text.casefold() in seen:
                continue
            keywords = item.get("keywords", "")
            if isinstance(keywords, list) and all(isinstance(word, str) for word in keywords):
                keywords = " ".join(keywords)
            if not isinstance(keywords, str):
                raise ValueError("keywords must be a string or array of strings")
            seen.add(text.casefold())
            queries.append(
                SubQuery(
                    query=text,
                    keywords=_normalise_space(keywords),
                    filters=_filters(item.get("filters")),
                )
            )
            if len(queries) >= max_queries:
                break
        return queries
    except Exception:
        return []
