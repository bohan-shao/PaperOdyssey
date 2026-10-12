"""Passages from processed/sections.jsonl instead of the PDFs (DESIGN.md D36).

The parse is the same in every chunk mode (build_kb.py), and sections.jsonl stores every parsed
section of every built paper, in order, with its text after the "[title] section" prefix line
(kb.chunk.build_parent_records). So the passages of any chunk mode can be made from it on one
machine, also after a shard merge (D35), when no machine holds every PDF. Rebuilding the
section_fixed passages this way gives the build's passages byte for byte.

Papers are read in file order and written in that order. Resumable: papers already in the
passages file are skipped, except the last one, which an interrupted run may have left incomplete
and which is rebuilt (`store.write_passages` replaces a paper's lines).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from . import chunk, store


def _papers(sections_path: Path) -> Iterator[list[dict]]:
    """The records of each paper, in file order (a paper's records are written together)."""
    seen: set[str] = set()
    recs: list[dict] = []
    for r in store.iter_parents(sections_path):
        if recs and r["paper_id"] != recs[0]["paper_id"]:
            yield recs
            recs = []
        if not recs:
            if r["paper_id"] in seen:
                raise ValueError(f"{sections_path}: the sections of {r['paper_id']} are not together")
            seen.add(r["paper_id"])
        recs.append(r)
    if recs:
        yield recs


def paper_count(sections_path: str | Path) -> int:
    """Papers in a sections.jsonl (for progress bars)."""
    return sum(1 for _ in _papers(Path(sections_path)))


def _last_paper_id(path: Path) -> str | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as fh:
        fh.seek(max(0, path.stat().st_size - 1_000_000))
        lines = fh.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)["paper_id"]
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
            continue  # a torn last line
    return None


def rechunk_from_sections(
    sections_path: str | Path,
    passages_path: str | Path,
    cfg: dict,
    mode: str,
    *,
    batch_papers: int = 200,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    progress: Callable[[int], None] | None = None,
) -> dict:
    """Write the `mode` passages of every paper in sections.jsonl to passages_path.

    Args:
        sections_path: processed/sections.jsonl of a built (or merged) KB.
        passages_path: the passages file to write (an ablation file, e.g. passages_semantic.jsonl).
        cfg: `config.get_config()`; chunk settings from `kb.chunk`.
        mode: "section_fixed" or "semantic".
        batch_papers: papers whose sentences are embedded together (semantic mode).
        embed_fn: sentence embedding override for tests (see `chunk.chunk_semantic`).
        progress: called with the number of papers done after each batch.

    Returns:
        {"papers", "passages"} written by this run, "skipped" (already in the file) and, in semantic
        mode with the default encoder, "embedding_device".
    """
    sections_path, passages_path = Path(sections_path), Path(passages_path)
    if sections_path.resolve() == passages_path.resolve():
        raise ValueError("passages_path is the sections file")
    ch = cfg["kb"]["chunk"]
    tok = chunk.tokenizer_from_config(cfg)
    prefix = ch["prefix"]
    last = _last_paper_id(passages_path)
    done = store.stored_paper_ids(passages_path) - {last}
    result: dict = {"papers": 0, "passages": 0, "skipped": 0}
    if mode == "semantic" and embed_fn is None:
        encoder = chunk._sentence_encoder(ch.get("semantic_model", "BAAI/bge-small-en-v1.5"))
        result["embedding_device"] = str(encoder.device)

    def flush(batch: list[tuple[str, list[dict], dict]]) -> None:
        if mode == "semantic":
            per_paper = chunk.chunk_semantic_batch(
                batch, ch["semantic_threshold"], ch.get("semantic_min_tokens", 100),
                ch.get("semantic_max_tokens", 500), prefix_fmt=prefix, tokenizer=tok,
                model_name=ch.get("semantic_model", "BAAI/bge-small-en-v1.5"), embed_fn=embed_fn)
        elif mode == "section_fixed":
            per_paper = [chunk.chunk_section_fixed(pid, secs, ch["chunk_tokens"], ch["overlap_tokens"], meta=meta,
                                                   prefix_fmt=prefix, tokenizer=tok,
                                                   min_new_tokens=ch.get("min_new_tokens", 50))
                         for pid, secs, meta in batch]
        else:
            raise ValueError(f"unknown chunk mode {mode!r}")
        passages = [p for ps in per_paper for p in ps]
        store.write_passages(passages, passages_path)
        result["papers"] += len(batch)
        result["passages"] += len(passages)
        if progress:
            progress(len(batch))

    batch: list[tuple[str, list[dict], dict]] = []
    for recs in _papers(sections_path):
        pid = recs[0]["paper_id"]
        if pid in done:
            result["skipped"] += 1
            continue
        meta = {k: recs[0][k] for k in ("arxiv_id", "title", "year", "categories")}
        batch.append((pid, chunk.sections_from_parents(recs, prefix), meta))
        if len(batch) >= batch_papers:
            flush(batch)
            batch = []
    if batch:
        flush(batch)
    return result
