"""Deterministic query views: preserve semantics for dense, reduce lexical repetition for BM25."""

from __future__ import annotations

import re
import unicodedata

_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−"})
# Preserve dotted versions/names and underscores for the backend's own analyzer.
_WORDS = re.compile(r"[^\W_]+(?:[._][^\W_]+)*", re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize PDF-style Unicode punctuation without expanding acronyms or inventing terms."""
    return unicodedata.normalize("NFKC", text).translate(_DASHES).replace("\u00ad", "").replace("\u200b", "")


def query_views(query: str, cfg: dict) -> tuple[str, str]:
    """Return (semantic query, lexical query); disabled means the original query for both."""
    if not cfg.get("enabled", False):
        return query, query
    semantic = " ".join(normalize_text(query).split())
    stopwords = set(cfg.get("stopwords", []))
    tokens = list(dict.fromkeys(_WORDS.findall(semantic.lower())))
    content = [t for t in tokens if t not in stopwords]
    limit = int(cfg.get("max_terms", 64))
    if limit <= 0:
        raise ValueError("query_processing.max_terms must be positive")
    # Keep all-stopword queries searchable; do not infer metadata constraints from free text.
    return semantic, " ".join((content or tokens)[:limit])


def passage_body(text: str, title: str = "") -> str:
    """Remove only the known stored title/section prefix, never an arbitrary first sentence."""
    first, sep, body = text.partition("\n")
    return body if sep and title and first.startswith(f"[{title}]") else text
