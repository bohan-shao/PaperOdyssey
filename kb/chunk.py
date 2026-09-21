"""Chunking: section-aware fixed-size chunking (baseline) + semantic chunking / parent-document mapping (ablation).

Conventions (documented in kb/README.md; state them in the report):
- **Token counting**: config `kb.chunk.tokenizer` — `tiktoken` (cl100k_base, default) or
  `transformers` (the dense model's tokenizer, `retrieval.dense.model_name`). Budgets
  (chunk_tokens / overlap_tokens / min/max) count **body tokens only**; the prefix is extra.
- **Prefix**: embedded as the first line of `text` ("[{title}] {section}\\n<body>"), so both
  BM25 and the dense encoder see paper title + section. Format: config `kb.chunk.prefix`.
- **Sentence packing**: chunks are built from whole sentences (a sentence longer than the budget
  is hard-split on token boundaries), so every chunk is readable on its own; paragraph breaks
  inside a chunk are kept as newlines. Overlap = the
  trailing sentences of the previous chunk totalling <= overlap_tokens. A trailing chunk that
  would add fewer than `min_new_tokens` new tokens is folded into the previous one's overlap.
- Chunks never cross section or paper boundaries; `chunk_index` is monotonic per paper.

Output: list[dict] per SCHEMA.md §2 (passage_id/paper_id/arxiv_id/title/section/year/
categories/chunk_index/text) + `parent_section_id` (always attached, D22 — lets the
parent-document ablation run without rebuilding the KB). Parent texts come from
`build_parent_records()` and are stored in `processed/sections.jsonl`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

DEFAULT_PREFIX = "[{title}] {section}"

# ---------------------------------------------------------------- tokenizer


@dataclass(frozen=True)
class Tokenizer:
    """Minimal encode/decode pair so both token conventions share one interface."""

    name: str
    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]

    def count(self, text: str) -> int:
        return len(self.encode(text))


@lru_cache(maxsize=4)
def get_tokenizer(kind: str = "tiktoken", model_name: str = "cl100k_base") -> Tokenizer:
    """Build a tokenizer.

    Args:
        kind: "tiktoken" (model_name = encoding name, default cl100k_base) or
            "transformers" (model_name = HF model id, e.g. BAAI/bge-large-en-v1.5).
    """
    if kind == "tiktoken":
        import tiktoken

        enc = tiktoken.get_encoding(model_name if model_name else "cl100k_base")
        return Tokenizer(f"tiktoken:{enc.name}", lambda t: enc.encode(t, disallowed_special=()), enc.decode)
    if kind == "transformers":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        return Tokenizer(
            f"transformers:{model_name}",
            lambda t: tok.encode(t, add_special_tokens=False),
            lambda ids: tok.decode(ids, skip_special_tokens=True),
        )
    raise ValueError(f"unknown tokenizer kind: {kind!r} (expected tiktoken | transformers)")


def tokenizer_from_config(cfg: dict) -> Tokenizer:
    """Tokenizer per config `kb.chunk.tokenizer` (transformers uses retrieval.dense.model_name)."""
    kind = cfg["kb"]["chunk"].get("tokenizer", "tiktoken")
    if kind == "transformers":
        return get_tokenizer(kind, cfg["retrieval"]["dense"]["model_name"])
    return get_tokenizer(kind, cfg["kb"]["chunk"].get("tiktoken_encoding", "cl100k_base"))


# ---------------------------------------------------------------- sentences

_ABBREV = {
    "e.g", "i.e", "et al", "al", "etc", "fig", "figs", "eq", "eqs", "sec", "secs", "tab", "cf",
    "vs", "resp", "approx", "dr", "prof", "mr", "ms", "vol", "pp", "ref", "eds", "ed",
}
_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter tolerant of academic abbreviations (et al., e.g., Fig. 3)."""
    out: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        pieces = _SPLIT_RE.split(para)
        buf = ""
        for piece in pieces:
            buf = f"{buf} {piece}".strip() if buf else piece.strip()
            last = buf.rsplit(None, 1)[-1].rstrip(".").lower() if buf else ""
            # keep going when the "sentence" ended on an abbreviation or a single initial ("J.")
            if last in _ABBREV or re.fullmatch(r"[a-z]", last) or buf.endswith(("et al.", "e.g.", "i.e.")):
                continue
            out.append(buf)
            buf = ""
        if buf:
            out.append(buf)
    return out


# ---------------------------------------------------------------- shared helpers


def _meta(meta: dict | None) -> dict:
    meta = meta or {}
    return {
        "arxiv_id": str(meta.get("arxiv_id") or ""),
        "title": str(meta.get("title") or ""),
        "year": int(meta["year"]) if meta.get("year") not in (None, "") else None,
        "categories": list(meta.get("categories") or []),
    }


def parent_section_id(paper_id: str, section_index: int) -> str:
    return f"{paper_id}_S{section_index:03d}"


def _make_passage(paper_id: str, meta: dict, section: str, section_index: int,
                  chunk_index: int, body: str, prefix_fmt: str) -> dict:
    prefix = prefix_fmt.format(title=meta["title"], section=section or "Body").strip()
    return {
        "passage_id": f"P{paper_id}_{chunk_index:04d}",
        "paper_id": paper_id,
        "arxiv_id": meta["arxiv_id"],
        "title": meta["title"],
        "section": section,
        "year": meta["year"],
        "categories": meta["categories"],
        "chunk_index": chunk_index,
        "text": f"{prefix}\n{body}" if prefix else body,
        "parent_section_id": parent_section_id(paper_id, section_index),
    }


Unit = tuple[str, int, bool]  # (sentence text, token count, starts a new paragraph)


def _sentence_units(text: str, tok: Tokenizer, max_tokens: int) -> list[Unit]:
    """Sentences with token counts; sentences over max_tokens are hard-split on token boundaries."""
    units: list[Unit] = []
    for para in text.split("\n"):
        first = True
        for sent in split_sentences(para):
            ids = tok.encode(sent)
            if len(ids) <= max_tokens:
                units.append((sent, len(ids), first))
                first = False
                continue
            for i in range(0, len(ids), max_tokens):
                # token boundaries can split a multi-byte character; drop the replacement glyph
                piece = tok.decode(ids[i:i + max_tokens]).replace("\ufffd", "").strip()
                if piece:
                    units.append((piece, len(ids[i:i + max_tokens]), first))
                    first = False
    return units


def _join(units: list[Unit]) -> str:
    """Join sentences, keeping paragraph breaks as newlines (readable chunks)."""
    out = ""
    for k, (sent, _, para_start) in enumerate(units):
        out += sent if k == 0 else ("\n" if para_start else " ") + sent
    return out


def _pack(units: list[Unit], chunk_tokens: int, overlap_tokens: int,
          min_new_tokens: int) -> list[str]:
    """Greedy sentence packing with sentence-aligned overlap."""
    chunks: list[str] = []
    i = 0
    n = len(units)
    while i < n:
        # overlap: walk back from i over previous sentences while within overlap budget
        start = i
        if chunks and overlap_tokens > 0:
            carried = 0
            while start > 0 and carried + units[start - 1][1] <= overlap_tokens:
                start -= 1
                carried += units[start][1]
        total = sum(u[1] for u in units[start:i])
        j = i
        while j < n and (j == i or total + units[j][1] <= chunk_tokens):
            total += units[j][1]
            j += 1
        new_tokens = sum(u[1] for u in units[i:j])
        if chunks and j == n and new_tokens < min_new_tokens:
            # tiny tail: append to the previous chunk instead of emitting a mostly-overlap chunk
            chunks[-1] += ("\n" if units[i][2] else " ") + _join(units[i:j])
            break
        chunks.append(_join(units[start:j]))
        i = j
    return chunks


# ---------------------------------------------------------------- baseline (D4)


def chunk_section_fixed(
    paper_id: str,
    sections: Iterable[dict[str, str]],
    chunk_tokens: int,
    overlap_tokens: int,
    *,
    meta: dict | None = None,
    prefix_fmt: str = DEFAULT_PREFIX,
    tokenizer: Tokenizer | None = None,
    min_new_tokens: int = 50,
) -> list[dict]:
    """Section-aware fixed-size chunking (baseline).

    Args:
        paper_id: paper ID (used as the passage_id prefix).
        sections: output of parse_pdf (section/text segments).
        chunk_tokens: target number of body tokens per chunk.
        overlap_tokens: overlapping tokens between adjacent chunks (sentence-aligned, <=).
        meta: paper metadata {arxiv_id, title, year, categories}.
        prefix_fmt: chunk prefix format (config `kb.chunk.prefix`).
        tokenizer: token-counting convention (default tiktoken cl100k_base).
        min_new_tokens: a final chunk adding fewer new tokens is merged into the previous one.

    Returns:
        A list of passage dicts (SCHEMA.md §2 + parent_section_id).
    """
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")
    tok = tokenizer or get_tokenizer()
    m = _meta(meta)
    passages: list[dict] = []
    for s_idx, sec in enumerate(sections):
        units = _sentence_units(sec["text"], tok, chunk_tokens)
        for body in _pack(units, chunk_tokens, overlap_tokens, min_new_tokens):
            passages.append(_make_passage(paper_id, m, sec["section"], s_idx, len(passages), body, prefix_fmt))
    return passages


# ---------------------------------------------------------------- ablation A: semantic (D21)


@lru_cache(maxsize=2)
def _sentence_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def chunk_semantic(
    paper_id: str,
    sections: Iterable[dict[str, str]],
    similarity_threshold: float,
    min_tokens: int,
    max_tokens: int,
    *,
    meta: dict | None = None,
    prefix_fmt: str = DEFAULT_PREFIX,
    tokenizer: Tokenizer | None = None,
    model_name: str = "BAAI/bge-small-en-v1.5",
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> list[dict]:
    """Semantic chunking (Q3 ablation A, D21).

    Within each section, adjacent sentences are embedded (normalized) and a boundary is placed
    where cosine similarity drops below `similarity_threshold`, subject to the chunk holding at
    least `min_tokens`; a chunk is also closed before exceeding `max_tokens`. A final chunk under
    `min_tokens` is merged into the previous chunk of the same section. No overlap.

    Args:
        embed_fn: optional override (list[str] -> L2-normalized [n, d] array), e.g. for tests;
            defaults to sentence-transformers `model_name` (config `kb.chunk.semantic_model`).
    """
    tok = tokenizer or get_tokenizer()
    m = _meta(meta)
    if embed_fn is None:
        encoder = _sentence_encoder(model_name)
        embed_fn = lambda sents: encoder.encode(sents, normalize_embeddings=True, batch_size=64)  # noqa: E731

    passages: list[dict] = []
    for s_idx, sec in enumerate(sections):
        units = _sentence_units(sec["text"], tok, max_tokens)
        if not units:
            continue
        sims = np.ones(max(0, len(units) - 1))
        if len(units) > 1:
            emb = np.asarray(embed_fn([u[0] for u in units]))
            sims = (emb[:-1] * emb[1:]).sum(axis=1)
        groups: list[list[Unit]] = [[units[0]]]
        size = units[0][1]
        for k in range(1, len(units)):
            n_tok = units[k][1]
            topic_shift = sims[k - 1] < similarity_threshold and size >= min_tokens
            if topic_shift or size + n_tok > max_tokens:
                groups.append([])
                size = 0
            groups[-1].append(units[k])
            size += n_tok
        if len(groups) > 1 and sum(u[1] for u in groups[-1]) < min_tokens:
            tail = groups.pop()
            groups[-1].extend(tail)  # may slightly exceed max_tokens; keeps the tail self-contained
        for g in groups:
            body = _join(g)
            passages.append(_make_passage(paper_id, m, sec["section"], s_idx, len(passages), body, prefix_fmt))
    return passages


# ---------------------------------------------------------------- ablation B: parent documents (D22)


def build_parent_records(
    paper_id: str,
    sections: Iterable[dict[str, str]],
    *,
    meta: dict | None = None,
    prefix_fmt: str = DEFAULT_PREFIX,
) -> list[dict]:
    """Parent units (one per parsed section) for `processed/sections.jsonl`.

    Record: {parent_section_id, paper_id, arxiv_id, title, section, year, categories,
             section_index, text} — `text` carries the same prefix as passages.
    """
    m = _meta(meta)
    records = []
    for s_idx, sec in enumerate(sections):
        prefix = prefix_fmt.format(title=m["title"], section=sec["section"] or "Body").strip()
        records.append({
            "parent_section_id": parent_section_id(paper_id, s_idx),
            "paper_id": paper_id,
            "arxiv_id": m["arxiv_id"],
            "title": m["title"],
            "section": sec["section"],
            "year": m["year"],
            "categories": m["categories"],
            "section_index": s_idx,
            "text": f"{prefix}\n{sec['text']}" if prefix else sec["text"],
        })
    return records


def build_parent_map(passages: list[dict], sections: Iterable[dict[str, str]]) -> list[dict]:
    """Attach a parent_section_id to each passage (Q3 ablation B, D22).

    The chunkers already attach `parent_section_id`; this function validates it against the
    parsed sections (and fills it by section-name match for passages produced elsewhere).
    Children of the same section share the parent ID.
    """
    secs = list(sections)
    by_name: dict[str, int] = {}
    for idx, sec in enumerate(secs):
        by_name.setdefault(sec["section"], idx)
    out = []
    for p in passages:
        q = dict(p)
        pid = q.get("parent_section_id")
        valid = pid and pid.startswith(f"{q['paper_id']}_S") and int(pid.rsplit("_S", 1)[1]) < len(secs)
        if not valid:
            if q.get("section") not in by_name:
                raise ValueError(f"{q['passage_id']}: section {q.get('section')!r} not in parsed sections")
            q["parent_section_id"] = parent_section_id(q["paper_id"], by_name[q["section"]])
        out.append(q)
    return out
