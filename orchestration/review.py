"""Review module: the LLM batch-judges candidate papers against unified criteria. TODO.

Implementation requirements (see orchestration/README.md for details, D10):
- Fixed rubric items come from config `orchestration.review.rubric` (default 5 items):
  relevance to the requirement / coverage of input aspects / recency / source credibility / technical depth.
- `batch_review(candidates, criteria, input_text, round) -> list[ReviewRecord]`:
  - config.review.batch_size papers per batch (default 5); each paper is fed title + truncated abstract + top 2–3 truncated passages
    (token budget from config `review.max_tokens_per_paper`)
  - Requires a JSON array output (verdict/score/reasons/covered/missing), retry once on parse failure
- Deduplication: by paper_id (a paper hit by multiple paths is reviewed only once); deduplication happens at enqueue time (handled by loop),
  this module only reviews the batch it is given.
- verdict="reject" papers do not enter the candidate set; covered/missing are aggregated into the coverage_report (used by loop).
- Record the review-stage latency.

Constraints: does not call retrieval directly; input comes from retrieval hits aggregated by loop.
"""

from __future__ import annotations

from .types import Criterion, ReviewRecord


def batch_review(
    candidates: list[dict],
    criteria: list[Criterion],
    input_text: str,
    round_no: int,
) -> list[ReviewRecord]:
    """Batch-review candidate papers.

    Args:
        candidates: [{paper_id, title, abstract, top_passages: [str], year, categories}].
        criteria: The merged list of fixed + extracted criteria.
        input_text: The user's original input (one of the review bases).
        round_no: The current round number.

    Returns:
        One ReviewRecord per paper (verdict/score/reasons/covered/missing).
    """
    raise NotImplementedError  # TODO: implement, see orchestration/README.md
