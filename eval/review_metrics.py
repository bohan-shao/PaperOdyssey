"""Review module metrics: LLM verdicts vs. human annotation. TODO.

Implementation requirements (see eval/README.md):
- `review_accuracy(records, human_labels)`:
  - Positive class = verdict=="pass"
  - Accuracy / Precision / Recall / F1 (sklearn.metrics)
  - human_labels: paper_id -> 0/1 (human verdict after majority vote)
- `coverage_stats(coverage_reports)`: covered/total ratio, round distribution
  (share meeting the threshold in rounds 1/2/3, unmet rate), average papers_reviewed/papers_passed
- Output `data/results/metrics/review_<date>.json`

Constraint: use only the majority-vote results from eval_set.jsonl as ground truth, never a single annotator.
"""

from __future__ import annotations


def review_accuracy(records: list[dict], human_labels: dict[str, int]) -> dict:
    """Review verdicts vs. human annotation.

    Args:
        records: list of ReviewRecord (including paper_id/verdict).
        human_labels: paper_id -> 0|1 (majority-vote result).

    Returns:
        {"accuracy": .., "precision": .., "recall": .., "f1": .., "n": n}.
    """
    raise NotImplementedError  # TODO: implement, see eval/README.md


def coverage_stats(coverage_reports: list[dict]) -> dict:
    """Aggregate coverage and round distribution."""
    raise NotImplementedError  # TODO: implement, see eval/README.md
