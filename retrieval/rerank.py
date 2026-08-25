"""cross-encoder reranking. TODO.

Implementation requirements:
- Model: BGE-reranker-v2-m3 (replaceable via config `retrieval.rerank.model_name`).
- `rerank(query, candidates, top_k)`: score (query, passage.text) pairs, take the
  top_k by rerank score in descending order; keep the original passage_id/paper_id,
  update rank/score/stage="rerank".
- Parent-document mode (config `retrieval.parent_doc.enabled`): score with the child
  text, then replace it with the parent text (the content of the parent_section_id)
  in the output; passage_id keeps the child.
- Latency: record the rerank time; ~100–300ms for 50 pairs on A40 (report performance section).

Constraints: input is top-50 (fusion output), output is top-10 (pipeline guarantees order).
"""

from __future__ import annotations

from .types import ScoredPassage


def rerank(query: str, candidates: list[ScoredPassage], top_k: int = 10) -> list[ScoredPassage]:
    """Rerank candidate passages with a cross-encoder.

    Args:
        query: the search query.
        candidates: the fused top-50.
        top_k: number of results to keep (default 10).

    Returns:
        The reranked top-k list.
    """
    raise NotImplementedError  # TODO: implement; see retrieval/README.md
