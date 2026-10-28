"""Retrieval metrics: Hit@K / P@K / R@K / MRR@K / NDCG@K (Q3).

Conventions (cited by the report):
- A ranking is the hit list ordered by `rank`, with each id once, at its best (first) position.
  At paper level (`level="paper"`, the primary granularity) a paper takes the position of its
  best-ranked passage. At passage level the passage ids are used; a passage repeated in the hits
  (e.g. hits merged across sub-queries, SCHEMA.md §8) counts once. Positions below are 1-based
  positions in that ranking.
- Hit@K = 1 if a gold id is in the top K, else 0. The eval set has one gold paper per question
  (eval/annotation/questions.py), so there Hit@K = R@K, the team's main number.
- P@K = |top K ∩ gold| / K (K is the denominator also when fewer than K results came back)
- R@K = |top K ∩ gold| / |gold|
- MRR@K = 1 / position of the first gold id in the top K (0 if there is none)
- NDCG@K: DCG = Σ_{i=1..K} rel_i / log2(i + 1); IDCG = the DCG of the gold ids in the best order
  (the min(|gold|, K) best). rel_i = 1 for a gold id (binary); with `graded=True` and gold given
  as {id: grade}, rel_i is the grade (passage-level graded relevance, SCHEMA.md §7).
- `compute_retrieval_metrics` macro-averages over the queries of the gold set. A query without hits
  (a failed or empty run) counts with every metric 0; a query without gold ids at a level is left
  out at that level (passage level: only the annotated subset). A level without any query reports
  n_queries 0 and None for every metric.
- Input hits are [{passage_id, paper_id, rank}] (retrieval_results JSON, SCHEMA.md §8).
- K must be a positive integer; empty k_list is rejected rather than replaced by defaults.

Checked against hand-computed values in tests/test_eval.py.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping


def ranking(hits: list[dict], level: str = "paper") -> list[str]:
    """Ids in rank order, each once at its best position: papers or passages."""
    if level not in ("paper", "passage"):
        raise ValueError(f"level must be 'paper' or 'passage', not {level!r}")
    key = "paper_id" if level == "paper" else "passage_id"
    return list(dict.fromkeys(h[key] for h in sorted(hits, key=lambda h: h["rank"])))


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("metric k must be a positive integer")


# The metrics on a ranking (list of ids); the public functions below take hits.


def _found(ranked: list[str], gold: Collection[str], k: int) -> int:
    _validate_k(k)
    return len(set(ranked[:k]) & set(gold))


def _mrr(ranked: list[str], gold: Collection[str], k: int) -> float:
    _validate_k(k)
    return next((1 / pos for pos, i in enumerate(ranked[:k], start=1) if i in gold), 0.0)


def _ndcg(ranked: list[str], gold: Collection[str] | Mapping[str, float], k: int, graded: bool) -> float:
    _validate_k(k)
    grades = dict(gold) if graded and isinstance(gold, Mapping) else {g: 1.0 for g in gold}
    dcg = sum(grades.get(i, 0.0) / math.log2(pos + 1) for pos, i in enumerate(ranked[:k], start=1))
    idcg = sum(g / math.log2(pos + 1) for pos, g in enumerate(sorted(grades.values(), reverse=True)[:k], start=1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(hits: list[dict], gold_ids: Collection[str], k: int, *, level: str = "paper") -> float:
    """Hit@K: 1.0 if a gold id is in the top K, else 0.0."""
    return float(_found(ranking(hits, level), gold_ids, k) > 0)


def precision_at_k(hits: list[dict], gold_ids: set[str], k: int, *, level: str = "paper") -> float:
    """P@K (gold_ids and hits use the same granularity: paper or passage)."""
    return _found(ranking(hits, level), gold_ids, k) / k


def recall_at_k(hits: list[dict], gold_ids: set[str], k: int, *, level: str = "paper") -> float:
    """R@K."""
    _validate_k(k)
    return _found(ranking(hits, level), gold_ids, k) / len(gold_ids) if gold_ids else 0.0


def mrr_at_k(hits: list[dict], gold_ids: set[str], k: int = 10, *, level: str = "paper") -> float:
    """MRR@K."""
    return _mrr(ranking(hits, level), gold_ids, k)


def ndcg_at_k(hits: list[dict], gold_ids: set[str] | Mapping[str, float], k: int = 10, graded: bool = False,
              *, level: str = "paper") -> float:
    """NDCG@K; when graded=True, use passage-level graded relevance (SCHEMA.md §7), gold as {id: grade}."""
    return _ndcg(ranking(hits, level), gold_ids, k, graded)


def _level_metrics(hits_per_query: dict[str, list[dict]], gold: dict[str, set[str]], k_list: list[int],
                   level: str) -> dict:
    names = [f"{m}@{k}" for k in k_list for m in ("Hit", "P", "R")] + ["MRR@10", "NDCG@10"]
    if not gold:
        return {"n_queries": 0, **dict.fromkeys(names)}
    sums = dict.fromkeys(names, 0.0)
    for qid, ids in gold.items():
        ranked = ranking(hits_per_query.get(qid) or [], level)  # once per query
        for k in k_list:
            found = _found(ranked, ids, k)
            sums[f"Hit@{k}"] += float(found > 0)
            sums[f"P@{k}"] += found / k
            sums[f"R@{k}"] += found / len(ids)
        sums["MRR@10"] += _mrr(ranked, ids, 10)
        sums["NDCG@10"] += _ndcg(ranked, ids, 10, graded=False)
    return {"n_queries": len(gold), **{name: total / len(gold) for name, total in sums.items()}}


def compute_retrieval_metrics(
    hits_per_query: dict[str, list[dict]],
    gold_per_query: dict[str, dict],
    k_list: list[int] | None = None,
) -> dict:
    """Compute retrieval metrics in batch and aggregate (macro average).

    Args:
        hits_per_query: query_id -> hits (including rank/paper_id/passage_id).
        gold_per_query: query_id -> {gold_paper_ids, gold_passage_ids} (eval_set.jsonl records work as they are).
        k_list: the K of Hit@K / P@K / R@K (default 1, 3, 5, 10); MRR and NDCG are at 10.

    Returns:
        {"Hit@1": .., "P@5": .., "R@5": .., "MRR@10": .., "NDCG@10": .., "n_queries": n,
         "per_granularity": {paper: {...}, passage: {...}},
         "per_query": {query_id: {"first_gold_rank": paper position or None}}}; the top-level
        numbers are the paper-level ones. `per_query` gives the failed -> succeeded examples of an ablation.
    """
    k_list = [1, 3, 5, 10] if k_list is None else k_list
    if not k_list:
        raise ValueError("k_list must not be empty")
    for k in k_list:
        _validate_k(k)
    k_list = sorted(set(k_list))
    papers = {qid: set(g.get("gold_paper_ids") or []) for qid, g in gold_per_query.items()}
    passages = {qid: set(g.get("gold_passage_ids") or []) for qid, g in gold_per_query.items()}
    papers = {qid: ids for qid, ids in papers.items() if ids}
    paper = _level_metrics(hits_per_query, papers, k_list, "paper")
    passage = _level_metrics(hits_per_query, {qid: ids for qid, ids in passages.items() if ids}, k_list, "passage")
    first_gold = {qid: next((pos for pos, pid in enumerate(ranking(hits_per_query.get(qid) or []), start=1)
                             if pid in ids), None) for qid, ids in papers.items()}
    return {
        **paper,
        "k_list": k_list,
        "per_granularity": {"paper": paper, "passage": passage},
        "per_query": {qid: {"first_gold_rank": rank} for qid, rank in first_gold.items()},
    }
