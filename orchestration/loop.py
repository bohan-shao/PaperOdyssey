"""Parallel retrieval + review loop control (agentic RAG, D9/D11). TODO.

This module is **Paper Search** — a standalone, self-contained capability:
natural-language paper search (query understanding → parallel retrieval → review →
feedback-based rewriting). The downstream task (see `downstream/`) **calls** this
module via `run_query()` to obtain the evidence it needs (D24); this module does
not generate answers itself.

Implementation requirements (see orchestration/README.md for details):
- `run_query(input_text, query_id=None) -> FinalResult`: the only entry point of the orchestration layer.
- Flow:
  1) Query understanding → QueryPlan
  2) asyncio parallel: each sub-query → retrieval.retrieve() (**only pipeline.retrieve is allowed**)
  3) Results deduplicated by paper_id → fed in batches to review.batch_review (review workers consume in parallel,
     producer-consumer: as soon as one retrieval path finishes it is enqueued, without waiting for the others)
  4) Aggregate covered/missing → if not met and rounds remain → rewrite → next round
  5) Terminate (when any holds): coverage met / round > config.loop.max_rounds /
     passed paper count ≥ config.loop.min_papers
- Ablation switch: `orchestration.loop.enabled=false` → single-round parallel, no rewriting (Q5 baseline).
- Cache: with `loop.cache=true`, the same input_text hits the cache and directly returns the cached FinalResult
  (cache is written to data/results/final_results/, the file is the cache).
- Latency: understanding/retrieval/review/rewrite/total (milliseconds).
- Coverage report: coverage_report{total_aspects, covered_aspects, rounds,
  papers_reviewed, papers_passed}.

Constraints: this module does not implement LLM call details (goes through llm_client) or retrieval details (goes through pipeline).
"""

from __future__ import annotations

from .types import FinalResult


def run_query(input_text: str, query_id: str | None = None) -> FinalResult:
    """Run one complete Paper Search flow (agentic retrieval + review).

    Args:
        input_text: A paragraph of text from the user.
        query_id: Optional query ID (auto-generated when absent; used for the result file name).

    Returns:
        FinalResult (paper candidate set with review verdicts + coverage report + per-stage latency).
        The downstream task consumes this result; it does not contain a generated answer.
    """
    raise NotImplementedError  # TODO: implement, see orchestration/README.md
