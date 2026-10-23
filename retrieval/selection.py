"""Bound rerank work and reduce repeated evidence from long papers, without changing scores."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from .query import passage_body
from .types import ScoredPassage


def select_hits(hits: list[ScoredPassage], limit: int, cfg: dict,
                *, final: bool = False) -> list[ScoredPassage]:
    """Prefer distinct papers/sections in rank order, then backfill if the pool is too small.

    Exact repeated body text within a paper is always deduplicated when enabled. Quotas are
    soft: deferred evidence can fill spare slots, preserving single-paper detail queries.
    Returned scores and IDs are unchanged; ranks are contiguous. Inputs are not modified.
    """
    if limit <= 0:
        return []
    if not cfg.get("enabled", False):
        return hits[:limit]
    prefix = "final" if final else "candidate"
    paper_cap = int(cfg.get(f"{prefix}_per_paper", 2 if final else 4))
    parent_cap = int(cfg.get(f"{prefix}_per_parent", 1 if final else 2))
    if min(paper_cap, parent_cap) <= 0:
        raise ValueError("diversity paper/parent quotas must be positive")
    paper_counts: Counter[str] = Counter()
    parent_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_text: set[tuple[str, str]] = set()
    selected, deferred = [], []
    for h in hits:
        body = " ".join(passage_body(h.text, h.title).split()).casefold()
        key = (h.paper_id, body)
        if h.passage_id in seen_ids or key in seen_text:
            continue
        seen_ids.add(h.passage_id)
        seen_text.add(key)
        parent = h.parent_section_id or h.passage_id
        if paper_counts[h.paper_id] >= paper_cap or parent_counts[parent] >= parent_cap:
            deferred.append(h)
            continue
        selected.append(h)
        paper_counts[h.paper_id] += 1
        parent_counts[parent] += 1
        if len(selected) == limit:
            break
    selected.extend(deferred[:max(0, limit - len(selected))])
    # Selection changes membership, not the relevance ordering of the retained evidence.
    positions = {h.passage_id: i for i, h in enumerate(hits)}
    selected.sort(key=lambda h: positions[h.passage_id])
    return [replace(h, rank=i) for i, h in enumerate(selected, 1)]
