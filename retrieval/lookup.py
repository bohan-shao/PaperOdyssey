"""Passage/section lookup store: SQLite keyed by id, used to hydrate retrieval hits.

BM25 and FAISS return passage ids; everything else a ScoredPassage carries (paper_id, text,
title, year, categories, parent_section_id) comes from this store, and so does the parent
section text for parent-document mode (D22). Built from the KB outputs (passages.jsonl +
sections.jsonl) by `scripts/build_indexes.py`; lives under `paths.indices`
(config `retrieval.lookup.db_file`).

- `build_lookup(passages_path, sections_path, db_path)`: full rebuild into a temp file, then an
  atomic replace, so a crashed build never leaves a half-written store. Returns counts + time.
- `LookupStore(db_path)`: read-only, one connection shared by all threads under a lock (sparse
  and dense hydrate in parallel threads; each lookup is a sub-millisecond primary-key query).
  - `get_passages(ids)` / `get_sections(ids)`: records by id, `categories` decoded to a list.
  - `hydrate(ranked, stage)`: (passage_id, score) pairs, best first -> ScoredPassages, rank 1..n.
  - `with_parent_text(hits)`: swap each hit's text for its parent section's text.
- `get_lookup_store()`: process-level singleton at the configured path.

Constraints: primary-key lookups only; text search goes through BM25/FAISS (SQL LIKE is
forbidden). A hit id missing from the store means the indexes and the store were built from
different passages files, so it raises instead of silently dropping hits.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from itertools import islice
from pathlib import Path

from config import PROJECT_ROOT, get_config
from kb import store as kb_store

from .types import ScoredPassage
from .query import passage_body

log = logging.getLogger(__name__)

PASSAGE_COLUMNS = (
    "passage_id", "paper_id", "arxiv_id", "title", "section", "year", "categories",
    "chunk_index", "parent_section_id", "text",
)
SECTION_COLUMNS = (
    "parent_section_id", "paper_id", "arxiv_id", "title", "section", "year", "categories",
    "section_index", "text",
)

_SCHEMA = """
CREATE TABLE passages (
    passage_id        TEXT PRIMARY KEY,
    paper_id          TEXT NOT NULL,
    arxiv_id          TEXT,
    title             TEXT,
    section           TEXT,
    year              INTEGER,
    categories        TEXT,      -- JSON array string
    chunk_index       INTEGER,
    parent_section_id TEXT,
    text              TEXT NOT NULL
);
CREATE TABLE sections (
    parent_section_id TEXT PRIMARY KEY,
    paper_id          TEXT NOT NULL,
    arxiv_id          TEXT,
    title             TEXT,
    section           TEXT,
    year              INTEGER,
    categories        TEXT,      -- JSON array string
    section_index     INTEGER,
    text              TEXT NOT NULL
);
"""

_BATCH = 1000
_MAX_VARS = 500  # ids per IN (...) query; well under SQLite's bound-variable limit


# ---------------------------------------------------------------- build


def _row(rec: dict, columns: Sequence[str]) -> tuple:
    return tuple(
        json.dumps(rec.get(c) or [], ensure_ascii=False) if c == "categories" else rec.get(c)
        for c in columns
    )


def _insert_all(conn: sqlite3.Connection, table: str, columns: Sequence[str],
                records: Iterator[dict]) -> int:
    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})"
    n = 0
    while batch := [_row(r, columns) for r in islice(records, _BATCH)]:
        conn.executemany(sql, batch)
        n += len(batch)
    return n


def build_lookup(passages_path: str | Path, sections_path: str | Path,
                 db_path: str | Path) -> dict:
    """Build the lookup store from the KB outputs, replacing any existing one.

    Args:
        passages_path: processed/passages.jsonl (the same file the indexes are built from).
        sections_path: processed/sections.jsonl (parent sections).
        db_path: output SQLite file.

    Returns:
        {"num_passages", "num_sections", "size_bytes", "build_s"}.

    Raises:
        sqlite3.IntegrityError: a passage_id / parent_section_id appears twice.
    """
    t0 = time.perf_counter()
    db = Path(db_path).resolve()
    tmp = db.with_name(db.name + ".tmp")
    sources = {Path(passages_path).resolve(), Path(sections_path).resolve()}
    if db in sources or tmp in sources:
        raise ValueError("lookup output and temporary file must not overwrite KB inputs")
    db.parent.mkdir(parents=True, exist_ok=True)
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    try:
        # Safe to skip journaling: a failed build only ever leaves the temp file behind.
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.executescript(_SCHEMA)
        with conn:
            n_passages = _insert_all(conn, "passages", PASSAGE_COLUMNS,
                                     kb_store.iter_passages(passages_path))
            n_sections = _insert_all(conn, "sections", SECTION_COLUMNS,
                                     kb_store.iter_parents(sections_path))
    except BaseException:
        conn.close()
        tmp.unlink(missing_ok=True)
        raise
    conn.close()
    tmp.replace(db)
    return {
        "num_passages": n_passages,
        "num_sections": n_sections,
        "size_bytes": db.stat().st_size,
        "build_s": round(time.perf_counter() - t0, 3),
    }


# ---------------------------------------------------------------- read


class LookupStore:
    """Read-only id -> record lookups over a store built by `build_lookup`."""

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path).resolve()
        if not self.path.exists():
            raise RuntimeError(f"lookup store not found: {self.path} (run scripts/build_indexes.py)")
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _conn(self) -> sqlite3.Connection:
        """The shared connection, opened on first use; callers hold `self._lock`."""
        if self._connection is None:
            # One connection for all threads, not one per thread: orchestration runs every query
            # on fresh short-lived threads (asyncio.run per query, Streamlit reruns), and
            # per-thread connections stayed open after their threads ended.
            self._connection = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True,
                                               check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def _fetch(self, table: str, key: str, ids: Iterable[str]) -> dict[str, dict]:
        wanted = list(dict.fromkeys(ids))
        if not wanted:
            return {}
        found: dict[str, dict] = {}
        with self._lock:
            conn = self._conn()
            for i in range(0, len(wanted), _MAX_VARS):
                chunk = wanted[i:i + _MAX_VARS]
                sql = f"SELECT * FROM {table} WHERE {key} IN ({', '.join('?' * len(chunk))})"
                for row in conn.execute(sql, chunk):
                    rec = dict(row)
                    rec["categories"] = json.loads(rec["categories"] or "[]")
                    found[rec[key]] = rec
        return found

    def get_passages(self, passage_ids: Iterable[str]) -> dict[str, dict]:
        """Passage records by passage_id (SCHEMA.md section 2); unknown ids are left out."""
        return self._fetch("passages", "passage_id", passage_ids)

    def get_sections(self, parent_section_ids: Iterable[str]) -> dict[str, dict]:
        """Parent-section records by parent_section_id; unknown ids are left out."""
        return self._fetch("sections", "parent_section_id", parent_section_ids)

    def hydrate(self, ranked: Sequence[tuple[str, float]], stage: str) -> list[ScoredPassage]:
        """Turn (passage_id, score) pairs, best first, into ScoredPassages with ranks 1..n.

        Raises:
            RuntimeError: an id is not in the store (indexes and store are out of sync).
        """
        recs = self.get_passages(pid for pid, _ in ranked)
        _require(recs, (pid for pid, _ in ranked), "passage")
        hits = []
        for rank, (pid, score) in enumerate(ranked, start=1):
            r = recs[pid]
            hits.append(ScoredPassage(
                passage_id=pid, paper_id=r["paper_id"], text=r["text"], score=float(score),
                rank=rank, stage=stage, title=r["title"] or "", year=r["year"],
                categories=r["categories"], parent_section_id=r["parent_section_id"],
            ))
        return hits

    def with_parent_text(self, hits: Sequence[ScoredPassage]) -> list[ScoredPassage]:
        """Parent-document mode (D22): replace each hit's text with its parent section's text.

        passage_id, score and rank keep the child's values; hits without a parent are unchanged.

        Raises:
            RuntimeError: a parent_section_id is not in the store.
        """
        parent_ids = [h.parent_section_id for h in hits if h.parent_section_id]
        parents = self.get_sections(parent_ids)
        _require(parents, parent_ids, "parent section")
        limit = get_config()["retrieval"].get("parent_doc", {}).get("max_chars", 6000)
        return [replace(h, text=parent_excerpt(parents[h.parent_section_id]["text"], h, limit))
                if h.parent_section_id else h
                for h in hits]

    def close(self) -> None:
        """Close the connection; a later lookup opens a new one."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self) -> LookupStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _require(found: dict[str, dict], ids: Iterable[str], kind: str) -> None:
    missing = [i for i in dict.fromkeys(ids) if i not in found]
    if missing:
        raise RuntimeError(
            f"{len(missing)} {kind} id(s) not in the lookup store (e.g. {missing[:3]}); "
            "the indexes and the store are out of sync, rebuild with scripts/build_indexes.py"
        )


def parent_excerpt(parent: str, child: ScoredPassage, max_chars: int) -> str:
    """Bound a parent around the matched child; never replace evidence with an unrelated prefix.

    Zero means unbounded legacy mode. Whitespace-only differences can be anchored; ambiguous,
    missing or oversized anchors preserve the child and emit a structured warning. The budget
    is then a soft upper bound. No approximate text matching changes the evidence.
    """
    text, outcome = _parent_excerpt(parent, child, max_chars)
    log.log(logging.WARNING if outcome.startswith("fallback_") else logging.DEBUG,
            "parent excerpt %s for %s", outcome, child.passage_id,
            extra={"parent_excerpt_outcome": outcome, "passage_id": child.passage_id})
    return text


def _parent_excerpt(parent: str, child: ScoredPassage, max_chars: int) -> tuple[str, str]:
    """Return text and an observable outcome without changing the retrieval contract."""
    if max_chars < 0:
        raise ValueError("parent_doc.max_chars must be non-negative")
    if max_chars == 0 or len(parent) <= max_chars:
        return parent, "full_parent"
    body = passage_body(child.text, child.title)
    prefix, sep, _ = parent.partition("\n")
    header = prefix + sep if sep and prefix.startswith(f"[{child.title}]") else ""
    budget = max_chars - len(header)
    if not body.strip():
        return child.text, "fallback_empty"
    if len(body) > budget:
        return child.text, "fallback_oversized"
    # Sentence packing can normalize line breaks/spaces. Preserve original parent offsets.
    pattern = r"\s+".join(re.escape(word) for word in body.split())
    matches = list(islice(re.finditer(pattern, parent[len(header):]), 2))
    if not matches:
        return child.text, "fallback_unmatched"
    if len(matches) > 1:
        return child.text, "fallback_ambiguous"
    match = matches[0]
    position, end = len(header) + match.start(), len(header) + match.end()
    if end - position > budget:
        return child.text, "fallback_oversized"
    start = max(len(header), position - (budget - (end - position)) // 2)
    start = min(start, max(len(header), len(parent) - budget))
    outcome = "exact_window" if parent[position:end] == body else "whitespace_window"
    return header + parent[start:start + budget], outcome


# ---------------------------------------------------------------- config / singleton


def lookup_path(cfg: dict | None = None) -> Path:
    """The configured store path: `paths.indices` / `retrieval.lookup.db_file`."""
    cfg = cfg or get_config()
    return PROJECT_ROOT / str(cfg["paths"]["indices"]) / str(cfg["retrieval"]["lookup"]["db_file"])


_STORE: LookupStore | None = None
_STORE_LOCK = threading.Lock()


def get_lookup_store() -> LookupStore:
    """Return the process-level singleton store (opened on first use)."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = LookupStore(lookup_path())
        return _STORE
