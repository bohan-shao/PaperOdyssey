"""RRF hybrid fusion. TODO.

Implementation requirements:
- `reciprocal_rank_fusion(sparse_hits, dense_hits, k=60)`:
  score(p) = Σ_stage 1/(k + rank_stage(p)), with each of the two inputs a top-50 list.
- Output: a list sorted by fused score in descending order and deduplicated (only one
  entry per passage_id, keeping the higher score; stage set to "hybrid").
- Support a weighted variant `fusion.weights` (sparse/dense weights, config-adjustable)
  as an ablation comparison.
- Document: the principle that RRF is insensitive to score scale (for report Q3).

Constraints: input hits must carry rank (1-based); pure function, no I/O.
"""

from __future__ import annotations

from .types import ScoredPassage


def reciprocal_rank_fusion(
    sparse_hits: list[ScoredPassage],
    dense_hits: list[ScoredPassage],
    k: int = 60,
    weights: tuple[float, float] = (1.0, 1.0),
) -> list[ScoredPassage]:
    """Fuse two retrieval result lists with RRF.

    Args:
        sparse_hits: sparse top-50 (with rank).
        dense_hits: dense top-50 (with rank).
        k: RRF constant (default 60).
        weights: (sparse weight, dense weight); equal weights by default.

    Returns:
        The fused, sorted, deduplicated list (stage="hybrid").
    """
    raise NotImplementedError  # TODO: implement; see retrieval/README.md
