"""Retrieval metrics: P@K / R@K / MRR / NDCG. TODO.

Implementation requirements (see eval/README.md; write the formula conventions into the docstring, referenced by the report):
- P@K = |hits[:K] ∩ gold| / K
- R@K = |hits[:K] ∩ gold| / |gold|
- MRR@10 = 1/rank of the first gold hit (0 if no hit within the top 10)
- NDCG@K: DCG = Σ rel_i / log2(i+1), rel_i∈{0,1} (paper-level) or graded (passage-level,
  using the graded annotation in SCHEMA.md); IDCG idealized by ranking gold
- Supports two granularities: paper-level (gold_paper_ids) and passage-level (gold_passage_ids)
- Input hits is a list of [{passage_id, paper_id, rank}] (retrieval_results JSON)

Constraint: metric implementations must be verified against hand-computed values on a few small examples before use (no test framework; see README: Sanity checks).
"""

from __future__ import annotations


def precision_at_k(hits: list[dict], gold_ids: set[str], k: int) -> float:
    """P@K (gold_ids and hits use the same granularity: paper or passage)."""
    raise NotImplementedError  # TODO: implement, see eval/README.md


def recall_at_k(hits: list[dict], gold_ids: set[str], k: int) -> float:
    """R@K."""
    raise NotImplementedError  # TODO: implement, see eval/README.md


def mrr_at_k(hits: list[dict], gold_ids: set[str], k: int = 10) -> float:
    """MRR@K."""
    raise NotImplementedError  # TODO: implement, see eval/README.md


def ndcg_at_k(hits: list[dict], gold_ids: set[str], k: int = 10, graded: bool = False) -> float:
    """NDCG@K; when graded=True, use passage-level graded relevance (SCHEMA.md §7)."""
    raise NotImplementedError  # TODO: implement, see eval/README.md


def compute_retrieval_metrics(
    hits_per_query: dict[str, list[dict]],
    gold_per_query: dict[str, dict],
    k_list: list[int] | None = None,
) -> dict:
    """Compute retrieval metrics in batch and aggregate (macro average).

    Args:
        hits_per_query: query_id -> hits (including rank/paper_id/passage_id).
        gold_per_query: query_id -> {gold_paper_ids, gold_passage_ids}.

    Returns:
        {"P@5": .., "R@5": .., "MRR@10": .., "NDCG@10": .., "n_queries": n,
         "per_granularity": {paper: {...}, passage: {...}}}.
    """
    raise NotImplementedError  # TODO: implement, see eval/README.md
