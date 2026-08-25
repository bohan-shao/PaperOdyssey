"""Chunking: section-aware fixed-size chunking (baseline) + semantic chunking / parent-document mapping (ablation). TODO.

Implementation requirements:
- Baseline (D4) `chunk_section_fixed(sections, chunk_tokens, overlap_tokens)`:
  split sections into fixed-size chunks (300–400 tokens), with 100 tokens of overlap
  between adjacent chunks; each chunk's text carries a prefix (format determined by
  config `kb.chunk.prefix`, e.g. `[Title] Section`).
- Ablation A (D21) `chunk_semantic(sections, similarity_threshold)`:
  use sentence-level embeddings (sentence-transformers; model name from config) to
  compute similarity between adjacent sentences and cut where it drops below the
  threshold; clamp chunk length to [min_tokens, max_tokens] (prevent chunks that are
  too short or too long).
- Ablation B (D22) `build_parent_map(passages, sections)`:
  generate `parent_section_id` (SCHEMA.md §2); children of the same section share the
  parent ID.
- Use one consistent token-counting convention (config `kb.chunk.tokenizer`: tiktoken
  or a transformers tokenizer; state it in the report) to avoid inconsistent
  conventions across modules.

Output: list[dict] fields per SCHEMA.md §2 (passage_id/paper_id/arxiv_id/title/
section/year/categories/chunk_index/text[+parent_section_id]).
"""

from __future__ import annotations

from collections.abc import Iterable


def chunk_section_fixed(
    paper_id: str,
    sections: Iterable[dict[str, str]],
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[dict]:
    """Section-aware fixed-size chunking (baseline).

    Args:
        paper_id: paper ID (used as the passage_id prefix).
        sections: output of parse_pdf (section/text segments).
        chunk_tokens: target number of tokens per chunk.
        overlap_tokens: number of overlapping tokens between adjacent chunks.

    Returns:
        A list of passage dicts (SCHEMA.md §2).
    """
    raise NotImplementedError  # TODO: implement; see kb/README.md


def chunk_semantic(
    paper_id: str,
    sections: Iterable[dict[str, str]],
    similarity_threshold: float,
    min_tokens: int,
    max_tokens: int,
) -> list[dict]:
    """Semantic chunking (Q3 ablation A, D21)."""
    raise NotImplementedError  # TODO: implement; see kb/README.md


def build_parent_map(passages: list[dict], sections: Iterable[dict[str, str]]) -> list[dict]:
    """Attach a parent_section_id to each passage (Q3 ablation B, D22)."""
    raise NotImplementedError  # TODO: implement; see kb/README.md
