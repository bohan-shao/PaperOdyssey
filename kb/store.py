"""Storage: passages.jsonl (+ sections.jsonl) + papers.sqlite read/write.

- `write_passages(passages, path)`: JSONL, UTF-8, written line by line. Idempotent per paper:
  if any incoming paper_id is already in the file, that paper's old lines are removed first
  (one streaming rewrite), then the new lines are appended. The set of paper_ids per file is
  scanned once and cached, so the normal resumable build is pure appends. A torn last line
  (crash mid-write) is truncated on the first scan.
- `write_parents` / `iter_parents` / `load_parents`: the same for `sections.jsonl`
  (parent-document units, D22).
- `iter_passages(path)`: generator (retrieval builds its indices from this).
- `init_sqlite` / `upsert_papers` / `load_paper` / `set_num_chunks`: papers table per
  SCHEMA.md §3; upsert keyed on paper_id and only overwrites the columns provided, so
  metadata upserts never reset num_chunks and vice versa. Batched executemany transactions.

Constraints: this module is the only persistence layer for passage/paper data; other modules
use the read-only interfaces. Paths come from config (`kb.store.*`) via the callers.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

PAPER_COLUMNS = [
    "paper_id", "arxiv_id", "title", "abstract", "year", "categories",
    "citation_count", "pdf_path", "num_chunks", "source",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id       TEXT PRIMARY KEY,
    arxiv_id       TEXT,
    title          TEXT,
    abstract       TEXT,
    year           INTEGER,
    categories     TEXT,      -- JSON array string
    citation_count INTEGER,
    pdf_path       TEXT,      -- relative to raw/
    num_chunks     INTEGER DEFAULT 0,
    source         TEXT DEFAULT 'arxiv'
);
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
"""

_BATCH = 1000
_paper_ids_cache: dict[Path, set[str]] = {}


# ---------------------------------------------------------------- JSONL (keyed by paper_id)


def _scan_paper_ids(path: Path) -> set[str]:
    """paper_ids present in a JSONL file (cached); truncates a torn trailing line."""
    key = path.resolve()
    if key in _paper_ids_cache:
        return _paper_ids_cache[key]
    ids: set[str] = set()
    if path.exists():
        good_bytes = 0
        with path.open("rb") as fh:
            for raw in fh:
                try:
                    ids.add(json.loads(raw)["paper_id"])
                except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                    break  # torn line from an interrupted write: drop it and everything after
                good_bytes += len(raw)
        if good_bytes < path.stat().st_size:
            with path.open("r+b") as fh:
                fh.truncate(good_bytes)
    _paper_ids_cache[key] = ids
    return ids


def _remove_papers(path: Path, paper_ids: set[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            if json.loads(line)["paper_id"] not in paper_ids:
                dst.write(line)
    tmp.replace(path)


def _write_keyed_jsonl(records: list[dict], path: str | Path) -> None:
    if not records:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    present = _scan_paper_ids(p)
    incoming = {r["paper_id"] for r in records}
    stale = incoming & present
    if stale:
        _remove_papers(p, stale)
    with p.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
    present |= incoming


def _iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def stored_paper_ids(path: str | Path) -> set[str]:
    """paper_ids already written to a passages/sections JSONL (used to resume builds)."""
    return set(_scan_paper_ids(Path(path)))


def reset_jsonl(path: str | Path) -> None:
    """Delete a JSONL output (full rebuild) and forget its cached paper_id set."""
    p = Path(path)
    p.unlink(missing_ok=True)
    _paper_ids_cache.pop(p.resolve(), None)


def write_passages(passages: list[dict], path: str | Path) -> None:
    """Append passages (idempotent; rewrite per paper_id, deduplicating)."""
    _write_keyed_jsonl(passages, path)


def iter_passages(path: str | Path) -> Iterator[dict]:
    """Read passages line by line (generator).

    Yields:
        Passage dicts (SCHEMA.md §2).
    """
    yield from _iter_jsonl(path)


def write_parents(parents: list[dict], path: str | Path) -> None:
    """Append parent-section records (sections.jsonl), idempotent per paper_id."""
    _write_keyed_jsonl(parents, path)


def iter_parents(path: str | Path) -> Iterator[dict]:
    """Read parent-section records line by line."""
    yield from _iter_jsonl(path)


def load_parents(parent_section_ids: Iterable[str], path: str | Path) -> dict[str, dict]:
    """Look up parent-section records by id (single streaming pass; for small id sets)."""
    wanted = set(parent_section_ids)
    found: dict[str, dict] = {}
    for rec in _iter_jsonl(path):
        if rec["parent_section_id"] in wanted:
            found[rec["parent_section_id"]] = rec
            if len(found) == len(wanted):
                break
    return found


# ---------------------------------------------------------------- SQLite


@contextmanager
def _connect(path: str | Path) -> Iterator[sqlite3.Connection]:
    """Connection that commits on success, rolls back on error, and is always closed."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_sqlite(path: str | Path) -> None:
    """Create the papers table (SCHEMA.md §3)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.executescript(_SCHEMA)


def _db_value(col: str, value: object) -> object:
    if col == "categories" and not isinstance(value, str) and value is not None:
        return json.dumps(list(value))
    if col in ("year", "citation_count", "num_chunks") and value is not None:
        return int(value)
    return value


def upsert_papers(rows: list[dict], path: str | Path) -> None:
    """Batch upsert paper metadata (keyed on paper_id).

    Only the columns present in each row are written; on conflict, only those columns are
    updated (e.g. a metadata upsert keeps an existing num_chunks). Unknown keys are ignored.
    """
    init_sqlite(path)
    groups: dict[tuple[str, ...], list[tuple]] = {}
    for row in rows:
        cols = tuple(c for c in PAPER_COLUMNS if c in row)
        if "paper_id" not in cols:
            raise ValueError("upsert_papers: every row needs a paper_id")
        groups.setdefault(cols, []).append(tuple(_db_value(c, row[c]) for c in cols))
    with _connect(path) as conn:
        for cols, values in groups.items():
            updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "paper_id")
            sql = (f"INSERT INTO papers ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
                   f"ON CONFLICT(paper_id) DO " + (f"UPDATE SET {updates}" if updates else "NOTHING"))
            for i in range(0, len(values), _BATCH):
                conn.executemany(sql, values[i:i + _BATCH])


def set_num_chunks(counts: dict[str, int], path: str | Path) -> None:
    """Record chunk counts per paper after chunking."""
    upsert_papers([{"paper_id": pid, "num_chunks": n} for pid, n in counts.items()], path)


def load_paper(paper_id: str, path: str | Path) -> dict | None:
    """Load paper metadata by paper_id; return None if it does not exist.

    `categories` is decoded to a list.
    """
    if not Path(path).exists():
        return None
    with _connect(path) as conn:
        row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    if row is None:
        return None
    paper = dict(row)
    try:
        paper["categories"] = json.loads(paper.get("categories") or "[]")
    except json.JSONDecodeError:
        paper["categories"] = []
    return paper


def iter_papers(path: str | Path) -> Iterator[dict]:
    """All papers (categories decoded), ordered by paper_id."""
    with _connect(path) as conn:
        for row in conn.execute("SELECT * FROM papers ORDER BY paper_id"):
            paper = dict(row)
            paper["categories"] = json.loads(paper.get("categories") or "[]")
            yield paper
