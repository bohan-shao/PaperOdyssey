"""Cross-encoder reranking: BGE-reranker-v2-m3 rescores the fused candidates, top-50 -> top-10.

BM25 and the dense bi-encoder score query and passage separately; a cross-encoder reads each
(query, passage) pair together, which is much more accurate and much slower, so it only sees the
~50 fused candidates (D8, D19: cheap filtering first).

- `Reranker(model_name)`: sentence-transformers CrossEncoder; fp16 on CUDA; pairs truncated to
  `retrieval.rerank.max_length` tokens; score = sigmoid of the model's logit, in [0, 1]. One lock
  around the model (HF fast tokenizers are not thread-safe); a warm-up pair on load keeps the
  first CUDA call out of measured latencies.
- `rank_by_scores(candidates, scores, top_k)`: the ordering step on its own: sorts by the new
  score (ties: the fused rank), keeps passage_id / paper_id / text / metadata, sets rank 1..k
  and stage="rerank".
- `rerank(query, candidates, top_k)`: the module-level entry the pipeline calls (singleton).

Parent-document mode (D22) is applied by the pipeline after this step, not here: the reranker
always scores the child passage text (short and precise), and the swap to the parent section
text then works the same whether reranking is on or off (EVALUATION.md has a hybrid-without-
rerank row).

Latency: the pipeline times this call as rerank_ms (budget ~100-300 ms for 50 pairs on an A40).
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import replace

from config import get_config

from .types import ScoredPassage


def rank_by_scores(candidates: Sequence[ScoredPassage], scores: Sequence[float],
                   top_k: int) -> list[ScoredPassage]:
    """Order candidates by their rerank scores and keep the best top_k.

    Raises:
        ValueError: scores and candidates differ in length.
    """
    if len(scores) != len(candidates):
        raise ValueError(f"{len(scores)} scores for {len(candidates)} candidates")
    order = sorted(range(len(candidates)),
                   key=lambda i: (-scores[i], candidates[i].rank, candidates[i].passage_id))
    return [replace(candidates[i], score=float(scores[i]), rank=rank, stage="rerank")
            for rank, i in enumerate(order[:max(top_k, 0)], start=1)]


class Reranker:
    """Cross-encoder scoring (query, passage) pairs."""

    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 64,
                 fp16: bool = True, device: str | None = None) -> None:
        if "qwen3-reranker" in model_name.casefold():
            raise ValueError("Qwen3-Reranker requires causal yes/no logit scoring, not CrossEncoder; "
                             "use the configured BGE cross-encoder until that adapter is benchmarked")
        import torch
        from sentence_transformers import CrossEncoder

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        if fp16 and device.startswith("cuda"):
            self.model.half()
        self.name = model_name
        self.batch_size = batch_size
        self._sigmoid = torch.nn.Sigmoid()
        self._lock = threading.Lock()
        self.score("warm up", ["warm up"])  # first (CUDA) call is slow; keep it out of timings

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance of each text to the query, in [0, 1]."""
        if not texts:
            return []
        with self._lock:
            scores = self.model.predict([(query, t) for t in texts], batch_size=self.batch_size,
                                        activation_fn=self._sigmoid, show_progress_bar=False,
                                        convert_to_numpy=True)
        return [float(s) for s in scores]

    def rerank(self, query: str, candidates: Sequence[ScoredPassage],
               top_k: int = 10) -> list[ScoredPassage]:
        """Rescore the candidates with the cross-encoder and keep the best top_k."""
        if not candidates or top_k <= 0:
            return []
        return rank_by_scores(candidates, self.score(query, [c.text for c in candidates]), top_k)


_RERANKER: Reranker | None = None
_RERANKER_LOCK = threading.Lock()


def get_reranker() -> Reranker:
    """Return the process-level singleton reranker (model loaded on first use)."""
    global _RERANKER
    with _RERANKER_LOCK:
        if _RERANKER is None:
            cfg = get_config()["retrieval"]["rerank"]
            _RERANKER = Reranker(cfg["model_name"], cfg.get("max_length", 512),
                                 cfg.get("batch_size", 64), cfg.get("fp16", True), cfg.get("device"))
        return _RERANKER


def rerank(query: str, candidates: list[ScoredPassage], top_k: int = 10) -> list[ScoredPassage]:
    """Rerank candidate passages with a cross-encoder.

    Args:
        query: the search query.
        candidates: the fused top-50.
        top_k: number of results to keep (default 10).

    Returns:
        The reranked top-k list (stage="rerank", score in [0, 1]).
    """
    return get_reranker().rerank(query, candidates, top_k)
