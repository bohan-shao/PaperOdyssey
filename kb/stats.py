"""Corpus statistics (Assignment Q1 item 3; must be delivered). TODO.

Implementation requirements:
- `compute_corpus_stats(passages_path)` outputs:
  - n_documents (distinct paper_id count)
  - n_chunks (passage count)
  - n_words (word count; convention: split on whitespace into words, state it in the
    report; config offers an optional token convention)
  - n_types (unique word count, counted after lowercase normalization)
- Output `data/results/corpus_stats.json` and print an aligned table to stdout.
- Also: distribution tables by year/category (for report figures).

Constraints: every statistic must be reproducible (the same passages.jsonl yields
deterministic results).
"""

from __future__ import annotations

from pathlib import Path


def compute_corpus_stats(passages_path: str | Path) -> dict[str, int]:
    """Compute corpus statistics.

    Args:
        passages_path: data/processed/passages.jsonl.

    Returns:
        {"n_documents": int, "n_chunks": int, "n_words": int, "n_types": int}.
    """
    raise NotImplementedError  # TODO: implement; see kb/README.md


def write_stats_report(stats: dict[str, int], out_path: str | Path) -> None:
    """Write the JSON report + print the table."""
    raise NotImplementedError  # TODO: implement; see kb/README.md
