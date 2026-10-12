"""Merge KB shard builds (one per KB member) into one knowledge base (DESIGN.md D35).

Each shard is a member's data/ folder, built from the shared raw/metadata.csv with
`download_papers.py --shard I/N` and `build_kb.py --rebuild` (kb/README.md). Shards and output have the
layout of one build: folders named like the last part of config `paths.raw` / `paths.processed` /
`paths.results`, files named by `kb.store.*` and `kb.stats.out_file`:
    raw/metadata.csv (every shard's pdf_path), raw/parse_failures.jsonl, raw/download_failures.jsonl,
    processed/passages.jsonl, processed/sections.jsonl, processed/papers.sqlite (num_chunks),
    results/parse_quality.jsonl, results/parse_qa_sample.jsonl/.csv (from shard `qa_from`, the
    reviewed sample), results/corpus_stats.json (chunking settings as recorded by the shard builds).

Lines are copied in shard order, unchanged except that every line ends with LF: a build on Windows
writes CRLF, and a JSON line cannot hold a raw CR, so no record changes.

Refuses (MergeError, before anything is written) when the output already has processed outputs,
`qa_from` is not a shard, a shard lacks metadata.csv, passages, sections or corpus statistics, the
shards were not built from the same selection (a metadata column other than pdf_path differs), a
paper has a PDF in two shards, a shard holds passages or sections of a paper without a PDF in its
own metadata.csv (a wrong or stale build folder), or the shards were chunked differently.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import store
from .fetch import read_metadata_csv, write_metadata_csv
from .stats import compute_corpus_stats, parse_summary, write_qa_csv, write_stats_report


class MergeError(Exception):
    """The shards cannot be merged; nothing was written."""


def _paper_ids(path: Path) -> Counter:
    with path.open(encoding="utf-8") as fh:
        return Counter(json.loads(line)["paper_id"] for line in fh if line.strip())


def _concat(sources: list[Path], dest: Path) -> None:
    """The lines of every existing source, in order, each ending with LF."""
    with dest.open("wb") as out:
        for src in sources:
            if src.exists():
                with src.open("rb") as fh:
                    for line in fh:
                        out.write(line.rstrip(b"\r\n") + b"\n")


def merge_shards(shards: list[Path], out: Path, cfg: dict, qa_from: int = 0) -> dict:
    """Merge shard data/ folders into `out` (see the module docstring).

    Args:
        shards: each member's data/ folder, in the order their lines are written.
        out: output data/ folder; must not hold passages, sections or papers.sqlite yet.
        cfg: `config.get_config()`.
        qa_from: position in `shards` of the shard whose parse QA sample is kept.

    Returns:
        The merged corpus statistics (also written to results/corpus_stats.json).

    Raises:
        MergeError: the shards cannot be merged (module docstring); nothing was written.
    """
    shards, out = [Path(s) for s in shards], Path(out)
    raw, processed, results = (Path(Path(cfg["paths"][k]).name) for k in ("raw", "processed", "results"))
    kb = cfg["kb"]
    mode = kb["chunk"].get("mode", "section_fixed")
    stats_file = results / kb["stats"]["out_file"]
    if mode != "section_fixed":  # named like build_kb.py names it
        stats_file = stats_file.with_name(f"{stats_file.stem}_{mode}{stats_file.suffix}")
    metadata, qa = raw / "metadata.csv", results / "parse_qa_sample.jsonl"
    passages = processed / kb["store"]["passages_file"]
    sections = processed / kb["store"].get("sections_file", "sections.jsonl")
    sqlite = processed / kb["store"]["sqlite_file"]

    if any((out / f).exists() for f in (passages, sections, sqlite)):
        raise MergeError(f"{out / processed} already has outputs; pick an empty folder")
    if not 0 <= qa_from < len(shards):
        raise MergeError(f"qa_from={qa_from} is not a shard (0 to {len(shards) - 1})")
    for s in shards:
        missing = [str(f) for f in (metadata, passages, sections, stats_file) if not (s / f).is_file()]
        if missing:
            raise MergeError(f"{s}: missing {', '.join(missing)}")

    metas = [read_metadata_csv(s / metadata) for s in shards]
    cols = [c for c in metas[0].columns if c != "pdf_path"]
    for s, m in zip(shards[1:], metas[1:]):
        if list(m.columns) != list(metas[0].columns) or not m[cols].equals(metas[0][cols]):
            raise MergeError(f"{s}: metadata.csv is not the same selection as {shards[0]} (columns other than pdf_path differ)")
    merged = metas[0].copy()
    for s, m in zip(shards[1:], metas[1:]):
        both = (merged.pdf_path != "") & (m.pdf_path != "")
        if both.any():
            raise MergeError(f"{s}: {int(both.sum())} papers have a PDF in two shards, e.g. {merged.paper_id[both].iloc[0]}")
        merged.loc[m.pdf_path != "", "pdf_path"] = m.pdf_path[m.pdf_path != ""]

    counts: Counter = Counter()  # PDFs are in one shard only, so no paper is counted twice
    for s, m in zip(shards, metas):
        own = set(m.paper_id[m.pdf_path != ""])
        c = _paper_ids(s / passages)
        stray = (c.keys() | _paper_ids(s / sections).keys()) - own
        if stray:
            raise MergeError(f"{s}: passages or sections of {len(stray)} papers without a PDF in its metadata.csv, "
                             f"e.g. {min(stray)} (a wrong or stale build folder?)")
        counts.update(c)

    chunking = [json.loads((s / stats_file).read_text("utf-8")).get("chunking") for s in shards]
    for s, c in zip(shards, chunking):
        if not c:
            raise MergeError(f"{s}: {stats_file} records no chunking settings")
        if c != chunking[0]:
            raise MergeError(f"{s} was chunked with {c}, {shards[0]} with {chunking[0]}")

    for d in (raw, processed, results):
        (out / d).mkdir(parents=True, exist_ok=True)
    write_metadata_csv(merged, out / metadata)
    for rel in (passages, sections, raw / "parse_failures.jsonl", raw / "download_failures.jsonl",
                results / "parse_quality.jsonl"):
        _concat([s / rel for s in shards], out / rel)

    with_pdf = merged[merged.pdf_path != ""]
    store.upsert_papers([{**r, "source": "arxiv"} for r in with_pdf.to_dict("records")], out / sqlite)
    store.set_num_chunks(dict(counts), out / sqlite)

    if (shards[qa_from] / qa).exists():
        _concat([shards[qa_from] / qa], out / qa)
        write_qa_csv(out / qa, (out / qa).with_suffix(".csv"))

    stats = compute_corpus_stats(out / passages, out / sections)
    stats["chunking"] = chunking[0]
    stats["pipeline"] = parse_summary(out / raw, stats["n_documents"])
    write_stats_report(stats, out / stats_file)
    return stats
