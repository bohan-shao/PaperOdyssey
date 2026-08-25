"""Storage: passages.jsonl + papers.sqlite read/write. TODO.

Implementation requirements:
- `write_passages(passages, path)`: append-only JSONL writes, UTF-8;
  on a repeat write of the same paper, first delete that paper's old lines and then
  append (idempotent, resumable).
- `iter_passages(path)`: generator yielding dicts line by line (retrieval depends on
  this interface to build indices).
- `init_sqlite(path)` / `upsert_papers(df)` / `load_paper(paper_id)`:
  table structure per SCHEMA.md §3 (papers table); upsert keys on paper_id.
- All paths come from config.yaml `kb.store.*`.
- Large-file performance: write JSONL line by line (no one-shot pandas to_json);
  use executemany + batched transactions for SQLite.

Constraints: this module is the only persistence layer for passage/paper data; other
modules use read-only interfaces.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def write_passages(passages: list[dict], path: str | Path) -> None:
    """Append passages (idempotent; rewrite per paper_id, deduplicating)."""
    raise NotImplementedError  # TODO: implement; see kb/README.md


def iter_passages(path: str | Path) -> Iterator[dict]:
    """Read passages line by line (generator).

    Yields:
        Passage dicts (SCHEMA.md §2).
    """
    raise NotImplementedError  # TODO: implement; see kb/README.md


def init_sqlite(path: str | Path) -> None:
    """Create the papers table (SCHEMA.md §3)."""
    raise NotImplementedError  # TODO: implement; see kb/README.md


def upsert_papers(rows: list[dict], path: str | Path) -> None:
    """Batch upsert paper metadata (keyed on paper_id)."""
    raise NotImplementedError  # TODO: implement; see kb/README.md


def load_paper(paper_id: str, path: str | Path) -> dict | None:
    """Load paper metadata by paper_id; return None if it does not exist."""
    raise NotImplementedError  # TODO: implement; see kb/README.md
