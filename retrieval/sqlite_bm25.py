"""Disk-backed FTS5 BM25 alternative for machines without Java; no SQL LIKE retrieval.

The contentless inverted index stores title, section and body once each. Metadata lives in
separate indexed tables and filters apply before LIMIT. FTS5 uses fixed k1=1.2, b=0.75;
its negative scores are inverted to the repository's higher-is-better convention.
"""

from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path

from kb.store import iter_passages

from .query import normalize_text, passage_body
from .sparse import category_token
from .types import RetrievalFilters

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_SCHEMA = """
CREATE TABLE docs (rowid INTEGER PRIMARY KEY, passage_id TEXT UNIQUE NOT NULL, year INTEGER);
CREATE INDEX docs_year ON docs(year);
CREATE TABLE categories (docid INTEGER, category TEXT, PRIMARY KEY(category, docid));
CREATE VIRTUAL TABLE terms USING fts5(title, section, body, content='', tokenize='porter unicode61');
"""


def build(passages_path: str | Path, out: Path) -> int:
    """Stream KB passages into an FTS index; the caller handles atomic directory replacement."""
    conn = sqlite3.connect(out / "bm25.sqlite")
    n = 0
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.executescript(_SCHEMA)
        with conn:
            for n, p in enumerate(iter_passages(passages_path), 1):
                conn.execute("INSERT INTO docs VALUES (?, ?, ?)", (n, p["passage_id"], p.get("year")))
                conn.executemany("INSERT INTO categories VALUES (?, ?)",
                                 [(n, c) for c in sorted({category_token(c) for c in p.get("categories") or []})])
                title = p.get("title", "")
                fields = (title, p.get("section", ""), passage_body(p["text"], title))
                conn.execute("INSERT INTO terms(rowid, title, section, body) VALUES (?, ?, ?, ?)",
                             (n, *(normalize_text(t) for t in fields)))
            conn.execute("INSERT INTO terms(terms) VALUES ('optimize')")
        if not n:
            raise RuntimeError(f"no passages in {passages_path}")
        return n
    finally:
        conn.close()


class SQLiteBM25:
    """Read-only FTS5 search with quoted terms, global IDF, and pre-limit metadata filters."""

    def __init__(self, index_dir: Path, weights: list[float]) -> None:
        if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
            raise ValueError("sqlite field_weights must be three finite non-negative weights")
        self._weights = tuple(weights)
        self._path = (index_dir / "bm25.sqlite").resolve()

    def search(self, query: str, filters: RetrievalFilters | None, k: int) -> list[tuple[str, float]]:
        tokens = list(dict.fromkeys(_TOKEN.findall(normalize_text(query).casefold())))
        if not tokens or k <= 0:
            return []
        clauses = ["terms MATCH ?"]
        args: list[object] = [" OR ".join(f'"{t}"' for t in tokens)]
        if filters:
            if filters.year_min is not None:
                clauses.append("d.year >= ?")
                args.append(filters.year_min)
            if filters.year_max is not None:
                clauses.append("d.year <= ?")
                args.append(filters.year_max)
            if filters.categories:
                clauses.append("d.rowid IN (SELECT docid FROM categories WHERE category IN ("
                               + ",".join("?" for _ in filters.categories) + "))")
                args.extend(category_token(c) for c in filters.categories)
        sql = ("SELECT d.passage_id, bm25(terms, ?, ?, ?) AS score "
               "FROM terms JOIN docs d ON d.rowid=terms.rowid WHERE " + " AND ".join(clauses)
               + " ORDER BY score, d.rowid LIMIT ?")
        # A per-call connection keeps the index replaceable on Windows and uses bounded RAM.
        conn = sqlite3.connect(f"{self._path.as_uri()}?mode=ro", uri=True)
        try:
            rows = conn.execute(sql, (*self._weights, *args, k)).fetchall()
        finally:
            conn.close()
        return [(pid, -float(score)) for pid, score in rows]
