"""Retrieval metrics: Hit@K / P@K / R@K / MRR@K / NDCG@K (Q3).

Conventions (cited by the report):
- A ranking is the hit list ordered by `rank`. At paper level (`level="paper"`, the primary
  granularity) passages are reduced to papers: a paper takes the position of its best-ranked
  passage and its later passages are dropped. At passage level passage ids are used as they are.
  Positions below are 1-based positions in that ranking.
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
  out at that level (passage level: only the annotated subset).
- Input hits are [{passage_id, paper_id, rank}] (retrieval_results JSON, SCHEMA.md §8).

Checked against hand-computed values in tests/test_eval.py.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping


def ranking(hits: list[dict], level: str = "paper") -> list[str]:
    """Ids in rank order: distinct papers (best-ranked passage first) or passage ids."""
    if level not in ("paper", "passage"):
        raise ValueError(f"level must be 'paper' or 'passage', not {level!r}")
    key = "paper_id" if level == "paper" else "passage_id"
    ids = [h[key] for h in sorted(hits, key=lambda h: h["rank"])]
    return list(dict.fromkeys(ids)) if level == "paper" else ids


def _found(hits: list[dict], gold_ids: Collection[str], k: int, level: str) -> int:
    return len(set(ranking(hits, level)[:k]) & set(gold_ids))


def hit_at_k(hits: list[dict], gold_ids: Collection[str], k: int, *, level: str = "paper") -> float:
    """Hit@K: 1.0 if a gold id is in the top K, else 0.0."""
    return float(_found(hits, gold_ids, k, level) > 0)


def precision_at_k(hits: list[dict], gold_ids: set[str], k: int, *, level: str = "paper") -> float:
    """P@K (gold_ids and hits use the same granularity: paper or passage)."""
    return _found(hits, gold_ids, k, level) / k


def recall_at_k(hits: list[dict], gold_ids: set[str], k: int, *, level: str = "paper") -> float:
    """R@K."""
    return _found(hits, gold_ids, k, level) / len(gold_ids) if gold_ids else 0.0


def mrr_at_k(hits: list[dict], gold_ids: set[str], k: int = 10, *, level: str = "paper") -> float:
    """MRR@K."""
    for pos, pid in enumerate(ranking(hits, level)[:k], start=1):
        if pid in gold_ids:
            return 1 / pos
    return 0.0


def ndcg_at_k(hits: list[dict], gold_ids: set[str] | Mapping[str, float], k: int = 10, graded: bool = False,
              *, level: str = "paper") -> float:
    """NDCG@K; when graded=True, use passage-level graded relevance (SCHEMA.md §7), gold as {id: grade}."""
    grades = dict(gold_ids) if graded and isinstance(gold_ids, Mapping) else {g: 1.0 for g in gold_ids}
    dcg = sum(grades.get(pid, 0.0) / math.log2(pos + 1) for pos, pid in enumerate(ranking(hits, level)[:k], start=1))
    best = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(pos + 1) for pos, g in enumerate(best, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def _first_gold(hits: list[dict], gold_ids: Collection[str], level: str) -> int | None:
    return next((pos for pos, pid in enumerate(ranking(hits, level), start=1) if pid in gold_ids), None)


def _level_metrics(hits_per_query: dict[str, list[dict]], gold: dict[str, Collection[str]], k_list: list[int],
                   level: str) -> dict:
    out: dict = {"n_queries": len(gold)}
    if not gold:
        return out
    rows = [(hits_per_query.get(qid) or [], ids) for qid, ids in gold.items()]
    for k in k_list:
        out[f"Hit@{k}"] = sum(hit_at_k(h, g, k, level=level) for h, g in rows) / len(rows)
        out[f"P@{k}"] = sum(precision_at_k(h, g, k, level=level) for h, g in rows) / len(rows)
        out[f"R@{k}"] = sum(recall_at_k(h, g, k, level=level) for h, g in rows) / len(rows)
    out["MRR@10"] = sum(mrr_at_k(h, g, 10, level=level) for h, g in rows) / len(rows)
    out["NDCG@10"] = sum(ndcg_at_k(h, g, 10, level=level) for h, g in rows) / len(rows)
    return out


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
    k_list = sorted(set(k_list or [1, 3, 5, 10]))
    papers = {qid: set(g.get("gold_paper_ids") or []) for qid, g in gold_per_query.items()}
    passages = {qid: set(g.get("gold_passage_ids") or []) for qid, g in gold_per_query.items()}
    paper = _level_metrics(hits_per_query, {q: ids for q, ids in papers.items() if ids}, k_list, "paper")
    passage = _level_metrics(hits_per_query, {q: ids for q, ids in passages.items() if ids}, k_list, "passage")
    return {
        **paper,
        "k_list": k_list,
        "per_granularity": {"paper": paper, "passage": passage},
        "per_query": {qid: {"first_gold_rank": _first_gold(hits_per_query.get(qid) or [], ids, "paper")}
                      for qid, ids in papers.items() if ids},
    }
