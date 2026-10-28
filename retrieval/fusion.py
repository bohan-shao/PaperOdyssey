"""Reciprocal rank fusion (RRF) of the sparse and dense candidate lists.

score(p) = sum over lists of w_list / (k + rank_list(p)), with k = 60 by default
(`retrieval.fusion.rrf_k`) and weights (sparse, dense) from `retrieval.fusion.weights`
(equal weights = plain RRF; unequal = the weighted ablation variant).

Why ranks and not scores (report Q3): BM25 scores are unbounded (0 to 20+, growing with query
length and term rarity) while dense scores are cosines (about 0.3-0.9), so adding raw scores
depends on an arbitrary normalization. RRF reads only each list's ranks, so it needs no
normalization or training, and it gives the same result under any order-preserving rescaling of
either retriever's scores. A passage found by both retrievers collects two terms and rises above
one found by a single retriever; k sets how much the very top ranks dominate (k=0 gives rank 1
twice the weight of rank 2, k=60 only 1.6% more).

- Input: each hit carries its list's 1-based rank; a passage listed twice in one list counts
  once, at its best rank. A list with weight 0 contributes nothing, not even candidates.
- Output: one ScoredPassage per passage_id with score = fused score, rank 1..n, stage="hybrid",
  and the other fields of its first input hit. Ties are broken by best single rank, then
  passage_id, so runs are reproducible. `top_k` truncates (the pipeline reranks the top 50).

Constraints: pure function, no I/O; the inputs are not modified.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import math

from .types import ScoredPassage


def reciprocal_rank_fusion(
    sparse_hits: list[ScoredPassage],
    dense_hits: list[ScoredPassage],
    k: int = 60,
    weights: Sequence[float] = (1.0, 1.0),
    top_k: int | None = None,
) -> list[ScoredPassage]:
    """Fuse two retrieval result lists with RRF.

    Args:
        sparse_hits: sparse top-50 (with rank).
        dense_hits: dense top-50 (with rank).
        k: RRF constant (default 60).
        weights: (sparse weight, dense weight); equal weights by default.
        top_k: keep only the best top_k fused passages (None = all).

    Returns:
        The fused, sorted, deduplicated list (stage="hybrid").

    Raises:
        ValueError: k < 0, a rank below 1, or weights that are not two non-negative numbers
            with at least one above 0.
    """
    if not math.isfinite(k) or k < 0:
        raise ValueError(f"RRF k must be >= 0, got {k}")
    if len(weights) != 2 or any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
        raise ValueError(f"weights must be two non-negative numbers, not both 0; got {weights!r}")

    fused: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    first_hit: dict[str, ScoredPassage] = {}
    for hits, weight in zip((sparse_hits, dense_hits), weights):
        if weight == 0:
            continue
        ranks: dict[str, int] = {}
        for h in hits:
            if h.rank < 1:
                raise ValueError(f"hit {h.passage_id} has rank {h.rank}; ranks are 1-based")
            ranks[h.passage_id] = min(h.rank, ranks.get(h.passage_id, h.rank))
            first_hit.setdefault(h.passage_id, h)
        for pid, rank in ranks.items():
            fused[pid] = fused.get(pid, 0.0) + weight / (k + rank)
            best_rank[pid] = min(rank, best_rank.get(pid, rank))

    order = sorted(fused, key=lambda pid: (-fused[pid], best_rank[pid], pid))
    if top_k is not None:
        order = order[:max(0, top_k)]
    return [replace(first_hit[pid], score=fused[pid], rank=rank, stage="hybrid")
            for rank, pid in enumerate(order, start=1)]
