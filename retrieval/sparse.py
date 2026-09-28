"""BM25 sparse retrieval: Pyserini (Lucene) by default, rank-bm25 as the no-Java fallback.

- `build_index(passages_path, index_dir)`: indexes passages.jsonl with the configured backend
  (`retrieval.sparse.backend: pyserini|rankbm25`) into a temp dir, then swaps it in (a rebuild
  overwrites). Layout: `index_dir/meta.json` (backend, doc count) + `lucene/` or `rankbm25.pkl`.
- `SparseRetriever(index_dir, lookup)`: opens the index (refusing one built by the other backend)
  and serves `search(query, filters, top_k) -> (hits, latency_ms)`; hits are hydrated through
  the lookup store with stage="bm25", rank 1..n and raw BM25 scores.
- `search(...)`: the module-level entry the pipeline calls (process singleton per config).

Pyserini: text is analyzed by Anserini's default English analyzer (Porter stemming, stopwords)
and scored with BM25 (`retrieval.sparse.bm25.k1/b`). The query goes through Anserini's
bag-of-words generator, so punctuation in LLM-written sub-queries cannot break query parsing.
`year` and `categories` are extra indexed fields used only as Lucene FILTER clauses, which
restrict the candidates without changing any score. Categories are indexed as normalized tokens
(`cs.CL` -> `cs_cl`), because the analyzer would split `q-bio.NC` at the hyphen and stem
`eess.AS` to `eess.a`.

rank-bm25: lowercase alphanumeric tokens, no stemming or stopwords, so scores differ from
Lucene (note it in the report). The whole index is a pickle held in memory and each query scores
every passage in Python: fine for small corpora, slow on the full ~800k passages.

Constraints: SQL LIKE is forbidden; index and queries must use the same backend (checked on
open). Years are compared as 4-digit strings in Lucene.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from config import PROJECT_ROOT, get_config
from kb import store as kb_store

from .lookup import LookupStore, get_lookup_store
from .types import RetrievalFilters, ScoredPassage

BACKENDS = ("pyserini", "rankbm25")
_TOKEN = re.compile(r"[a-z0-9]+")


def category_token(category: str) -> str:
    """Normalize a category to one analyzer-safe token: `q-bio.NC` -> `q_bio_nc`."""
    return "_".join(_TOKEN.findall(category.lower()))


def _tokenize(text: str) -> list[str]:
    """rank-bm25 tokenizer: lowercase alphanumeric runs."""
    return _TOKEN.findall(text.lower())


# ---------------------------------------------------------------- build


def _build_pyserini(passages_path: str | Path, out: Path, threads: int) -> int:
    docs_dir = out / "_input"
    docs_dir.mkdir(parents=True)
    n = 0
    with (docs_dir / "docs.jsonl").open("w", encoding="utf-8") as fh:
        for p in kb_store.iter_passages(passages_path):
            doc = {
                "id": p["passage_id"],
                "contents": p["text"],
                "year": str(p["year"]) if p.get("year") else "",
                "categories": " ".join(category_token(c) for c in p.get("categories") or []),
            }
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "-collection", "JsonCollection", "-generator", "DefaultLuceneDocumentGenerator",
        "-input", str(docs_dir), "-index", str(out / "lucene"),
        "-fields", "year", "categories", "-threads", str(threads), "-optimize",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(docs_dir)
    if proc.returncode != 0:
        raise RuntimeError(f"Pyserini indexing failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")

    from pyserini.search.lucene import LuceneSearcher

    indexed = LuceneSearcher(str(out / "lucene")).num_docs
    if indexed != n:
        raise RuntimeError(f"Pyserini indexed {indexed} of {n} passages")
    return n


def _build_rankbm25(passages_path: str | Path, out: Path, k1: float, b: float) -> int:
    import numpy as np
    from rank_bm25 import BM25Okapi

    ids: list[str] = []
    years: list[int] = []
    corpus: list[list[str]] = []
    by_category: dict[str, list[int]] = {}
    for i, p in enumerate(kb_store.iter_passages(passages_path)):
        ids.append(p["passage_id"])
        years.append(p.get("year") or 0)
        corpus.append(_tokenize(p["text"]))
        for c in p.get("categories") or []:
            by_category.setdefault(category_token(c), []).append(i)
    if not ids:
        raise RuntimeError(f"no passages in {passages_path}")
    index = {
        "bm25": BM25Okapi(corpus, k1=k1, b=b),
        "ids": ids,
        "years": np.array(years),
        "by_category": {c: np.array(rows) for c, rows in by_category.items()},
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rankbm25.pkl").open("wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return len(ids)


def build_index(passages_path: str | Path, index_dir: str | Path, backend: str | None = None,
                threads: int | None = None) -> dict:
    """Build the BM25 index from passages.jsonl, replacing any existing one.

    Args:
        passages_path: data/processed/passages.jsonl.
        index_dir: data/indices/bm25.
        backend: "pyserini" | "rankbm25"; default `retrieval.sparse.backend`.
        threads: Lucene indexing threads (default: CPU count, at most 8).

    Returns:
        {"backend", "num_docs", "size_bytes", "build_s"}.

    Raises:
        RuntimeError: when index construction fails (e.g. backend unavailable).
    """
    t0 = time.perf_counter()
    sparse_cfg = get_config()["retrieval"]["sparse"]
    backend = backend or sparse_cfg["backend"]
    if backend not in BACKENDS:
        raise ValueError(f"unknown sparse backend {backend!r} (expected one of {BACKENDS})")
    out = Path(index_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=out.name + ".", suffix=".tmp", dir=out.parent))
    tmp.chmod(0o755)  # mkdtemp makes it owner-only; the index is shared data
    try:
        if backend == "pyserini":
            n = _build_pyserini(passages_path, tmp, threads or min(8, os.cpu_count() or 1))
        else:
            bm25 = sparse_cfg.get("bm25", {})
            n = _build_rankbm25(passages_path, tmp, bm25.get("k1", 0.9), bm25.get("b", 0.4))
        (tmp / "meta.json").write_text(json.dumps({"backend": backend, "num_docs": n}) + "\n",
                                       encoding="utf-8")
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {
        "backend": backend,
        "num_docs": n,
        "size_bytes": sum(f.stat().st_size for f in out.rglob("*") if f.is_file()),
        "build_s": round(time.perf_counter() - t0, 3),
    }


# ---------------------------------------------------------------- search backends


class _PyseriniBackend:
    def __init__(self, index_dir: Path, k1: float, b: float) -> None:
        from pyserini.analysis import Analyzer, get_lucene_analyzer
        from pyserini.pyclass import autoclass
        from pyserini.search.lucene import LuceneSearcher, querybuilder

        self._searcher = LuceneSearcher(str(index_dir / "lucene"))
        self._searcher.set_bm25(k1, b)
        self._janalyzer = get_lucene_analyzer()  # the analyzer the index was built with
        self._analyzer = Analyzer(self._janalyzer)
        self._bow = autoclass("io.anserini.search.query.BagOfWordsQueryGenerator")()
        self._term = autoclass("org.apache.lucene.index.Term")
        self._term_query = autoclass("org.apache.lucene.search.TermQuery")
        self._term_range = autoclass("org.apache.lucene.search.TermRangeQuery")
        self._qb = querybuilder
        self._lock = threading.Lock()  # one JVM search at a time (~10 ms each)

    def _filter_clauses(self, filters: RetrievalFilters | None) -> list:
        clauses = []
        if filters is None:
            return clauses
        if filters.year_min is not None or filters.year_max is not None:
            lo = str(filters.year_min) if filters.year_min is not None else None
            hi = str(filters.year_max) if filters.year_max is not None else None
            clauses.append(self._term_range.newStringRange("year", lo, hi, True, True))
        if filters.categories:
            any_of = self._qb.get_boolean_query_builder()
            for c in filters.categories:
                for term in self._analyzer.analyze(category_token(c)):
                    any_of.add(self._term_query(self._term("categories", term)),
                               self._qb.JBooleanClauseOccur.should.value)
            clauses.append(any_of.build())  # no terms at all -> matches nothing
        return clauses

    def search(self, query: str, filters: RetrievalFilters | None, k: int) -> list[tuple[str, float]]:
        q = self._bow.buildQuery("contents", self._janalyzer, query)
        clauses = self._filter_clauses(filters)
        if clauses:
            builder = self._qb.get_boolean_query_builder()
            builder.add(q, self._qb.JBooleanClauseOccur.must.value)
            for clause in clauses:
                builder.add(clause, self._qb.JBooleanClauseOccur.filter.value)
            q = builder.build()
        with self._lock:
            hits = self._searcher.search(q, k)
        return [(h.docid, float(h.score)) for h in hits]


class _RankBM25Backend:
    def __init__(self, index_dir: Path) -> None:
        with (index_dir / "rankbm25.pkl").open("rb") as fh:
            index = pickle.load(fh)
        self._bm25 = index["bm25"]
        self._ids: list[str] = index["ids"]
        self._years = index["years"]
        self._by_category = index["by_category"]

    def search(self, query: str, filters: RetrievalFilters | None, k: int) -> list[tuple[str, float]]:
        import numpy as np

        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        mask = scores > 0  # like Lucene: only passages sharing a query term
        if filters is not None:
            if filters.year_min is not None:
                mask &= self._years >= filters.year_min
            if filters.year_max is not None:
                mask &= self._years <= filters.year_max
            if filters.categories:
                in_any = np.zeros(len(self._ids), dtype=bool)
                for c in filters.categories:
                    rows = self._by_category.get(category_token(c))
                    if rows is not None:
                        in_any[rows] = True
                mask &= in_any
        rows = np.flatnonzero(mask)
        top = rows[np.argsort(-scores[rows], kind="stable")[:k]]
        return [(self._ids[i], float(scores[i])) for i in top]


# ---------------------------------------------------------------- retriever


class SparseRetriever:
    """BM25 search over one index directory; hits are hydrated through the lookup store."""

    def __init__(self, index_dir: str | Path, lookup: LookupStore, backend: str | None = None,
                 k1: float | None = None, b: float | None = None) -> None:
        sparse_cfg = get_config()["retrieval"]["sparse"]
        backend = backend or sparse_cfg["backend"]
        bm25 = sparse_cfg.get("bm25", {})
        k1 = bm25.get("k1", 0.9) if k1 is None else k1
        b = bm25.get("b", 0.4) if b is None else b
        index_dir = Path(index_dir)
        meta_path = index_dir / "meta.json"
        if not meta_path.exists():
            raise RuntimeError(f"BM25 index not found: {index_dir} "
                               "(run scripts/build_indexes.py --only sparse)")
        built_with = json.loads(meta_path.read_text(encoding="utf-8"))["backend"]
        if built_with != backend:
            raise RuntimeError(f"BM25 index {index_dir} was built with {built_with!r} but the "
                               f"configured backend is {backend!r}; rebuild the sparse index")
        self.backend = backend
        self._lookup = lookup
        self._impl = _PyseriniBackend(index_dir, k1, b) if backend == "pyserini" else _RankBM25Backend(index_dir)

    def search(self, query: str, filters: RetrievalFilters | None,
               top_k: int) -> tuple[list[ScoredPassage], int]:
        """BM25 top_k for one query (filters restrict candidates, never scores)."""
        t0 = time.perf_counter()
        ranked = self._impl.search(query, filters, top_k)
        hits = self._lookup.hydrate(ranked, stage="bm25")
        return hits, round((time.perf_counter() - t0) * 1000)


def index_path(cfg: dict | None = None) -> Path:
    """The configured index dir; `retrieval.sparse.index_dir` is relative to the data dir."""
    cfg = cfg or get_config()
    return (PROJECT_ROOT / str(cfg["paths"]["indices"])).parent / str(cfg["retrieval"]["sparse"]["index_dir"])


_RETRIEVER: SparseRetriever | None = None
_RETRIEVER_LOCK = threading.Lock()


def get_sparse_retriever() -> SparseRetriever:
    """Return the process-level singleton retriever (opened on first use)."""
    global _RETRIEVER
    with _RETRIEVER_LOCK:
        if _RETRIEVER is None:
            _RETRIEVER = SparseRetriever(index_path(), get_lookup_store())
        return _RETRIEVER


def search(
    query: str,
    filters: RetrievalFilters | None,
    top_k: int,
) -> tuple[list[ScoredPassage], int]:
    """BM25 search.

    Args:
        query: the search query (a keyword string or natural language, provided by
            the orchestration layer).
        filters: metadata filtering (None = no filtering).
        top_k: number of results to return.

    Returns:
        (hits, latency_ms).
    """
    return get_sparse_retriever().search(query, filters, top_k)
