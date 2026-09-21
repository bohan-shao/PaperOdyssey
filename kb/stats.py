"""Corpus statistics (Assignment Q1 item 3; must be delivered).

Counting conventions (state them in the report):
- n_documents: distinct paper_id in passages.jsonl.
- n_chunks: passages in passages.jsonl.
- n_words: whitespace-separated words of the **cleaned corpus text**, i.e. the parsed section
  texts in sections.jsonl without the "[title] section" prefix — chunk overlap and prefixes
  would otherwise double-count words. If sections.jsonl is unavailable, words are counted over
  passages with the prefix line stripped (overlap still included; flagged in the report as
  `word_source`).
- n_types: unique words after lowercasing and stripping surrounding punctuation
  (tokens with no letter or digit are ignored).
- Also reported: words per chunk (with overlap) and per document, and distributions by
  year / primary category / section-count — for report figures.

Every statistic is deterministic for the same input files.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from .store import iter_parents, iter_passages

_STRIP = re.compile(r"^[^\w]+|[^\w]+$")
_HAS_ALNUM = re.compile(r"\w")


def _body(text: str) -> str:
    """Drop the embedded "[title] section" prefix line."""
    if text.startswith("[") and "\n" in text:
        return text.split("\n", 1)[1]
    return text


def _words(text: str) -> Iterator[str]:
    for w in text.split():
        norm = _STRIP.sub("", w.lower())
        if norm and _HAS_ALNUM.search(norm):
            yield norm


def compute_corpus_stats(
    passages_path: str | Path,
    sections_path: str | Path | None = None,
) -> dict:
    """Compute corpus statistics.

    Args:
        passages_path: data/processed/passages.jsonl.
        sections_path: data/processed/sections.jsonl (preferred source for word/type counts).

    Returns:
        {"n_documents", "n_chunks", "n_words", "n_types", ...extra fields and distributions}.
    """
    papers: dict[str, dict] = {}
    chunk_words: list[int] = []
    passage_words = 0
    passage_types: Counter[str] = Counter()
    use_sections = sections_path is not None and Path(sections_path).exists()

    for p in iter_passages(passages_path):
        pid = p["paper_id"]
        if pid not in papers:
            cats = p.get("categories") or []
            papers[pid] = {"year": p.get("year"), "category": cats[0] if cats else "unknown", "sections": set()}
        papers[pid]["sections"].add(p.get("section", ""))
        words = list(_words(_body(p["text"])))
        chunk_words.append(len(words))
        if not use_sections:
            passage_words += len(words)
            passage_types.update(words)

    if use_sections:
        n_words = 0
        types: Counter[str] = Counter()
        doc_words: Counter[str] = Counter()
        for rec in iter_parents(sections_path):
            if rec["paper_id"] not in papers:
                continue  # sections of papers without passages are not part of the KB
            words = list(_words(_body(rec["text"])))
            n_words += len(words)
            doc_words[rec["paper_id"]] += len(words)
            types.update(words)
        word_source = "sections.jsonl (cleaned text, no overlap/prefix)"
    else:
        n_words, types, doc_words = passage_words, passage_types, Counter()
        word_source = "passages.jsonl (prefix stripped; includes chunk overlap)"

    by_year = Counter(str(v["year"]) for v in papers.values())
    by_cat = Counter(v["category"] for v in papers.values())
    per_doc = list(doc_words.values()) or [0]
    return {
        "n_documents": len(papers),
        "n_chunks": len(chunk_words),
        "n_words": n_words,
        "n_types": len(types),
        "word_source": word_source,
        "word_convention": "whitespace split; lowercase; surrounding punctuation stripped",
        "words_per_chunk_mean": round(statistics.fmean(chunk_words), 1) if chunk_words else 0.0,
        "words_per_chunk_median": statistics.median(chunk_words) if chunk_words else 0,
        "words_per_doc_mean": round(statistics.fmean(per_doc), 1),
        "chunks_per_doc_mean": round(len(chunk_words) / max(1, len(papers)), 1),
        "sections_per_doc_mean": round(statistics.fmean(len(v["sections"]) for v in papers.values()), 1)
        if papers else 0.0,
        "by_year": dict(sorted(by_year.items())),
        "by_primary_category": dict(by_cat.most_common()),
        "top_types": types.most_common(20),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_files": {"passages": str(passages_path), "sections": str(sections_path) if use_sections else None},
    }


def write_stats_report(stats: dict, out_path: str | Path) -> None:
    """Write the JSON report + print the table."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = [
        ("Documents (papers)", f"{stats['n_documents']:,}"),
        ("Chunks (passages)", f"{stats['n_chunks']:,}"),
        ("Words (tokens)", f"{stats['n_words']:,}"),
        ("Types (unique words)", f"{stats['n_types']:,}"),
        ("Words / chunk (mean)", f"{stats['words_per_chunk_mean']}"),
        ("Chunks / document (mean)", f"{stats['chunks_per_doc_mean']}"),
        ("Words / document (mean)", f"{stats['words_per_doc_mean']}"),
    ]
    width = max(len(k) for k, _ in rows)
    print("\nCorpus statistics")
    print("-" * (width + 16))
    for k, v in rows:
        print(f"{k:<{width}}  {v:>12}")
    print("-" * (width + 16))
    print(f"word source: {stats['word_source']}")
    print("by year:     " + ", ".join(f"{y}: {n}" for y, n in stats["by_year"].items()))
    print("by category: " + ", ".join(f"{c}: {n}" for c, n in list(stats["by_primary_category"].items())[:8]))
    print(f"written to {out}")
