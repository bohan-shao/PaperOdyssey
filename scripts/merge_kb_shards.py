"""Merge KB shard builds (one per KB member) into one knowledge base.

Usage:
    python scripts/merge_kb_shards.py --out OUT_DIR [--qa-from K] SHARD_DIR [SHARD_DIR ...]

Each SHARD_DIR is a member's `data/` folder (processed/, results/, raw/), built from the shared
raw/metadata.csv with `download_papers.py --shard I/N` and `build_kb.py --rebuild` (kb/README.md).
OUT_DIR gets the same layout as a single build:
    raw/metadata.csv (every shard's pdf_path), raw/parse_failures.jsonl, raw/download_failures.jsonl,
    processed/passages.jsonl, processed/sections.jsonl, processed/papers.sqlite (num_chunks),
    results/parse_quality.jsonl, results/parse_qa_sample.jsonl/.csv (from shard --qa-from, the
    reviewed sample), results/corpus_stats.json.

Refuses (exit 1, nothing written) when the shards were not built from the same selection (any
metadata column other than pdf_path differs), when two shards hold the same paper, or when OUT_DIR
already has processed outputs. Passage and section lines are copied unchanged, in shard order.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_kb import _parse_summary, _write_qa_csv  # noqa: E402
from config import get_config  # noqa: E402
from kb import store  # noqa: E402
from kb.fetch import read_metadata_csv, write_metadata_csv  # noqa: E402
from kb.stats import compute_corpus_stats, write_stats_report  # noqa: E402


def _paper_ids(path: Path) -> Counter:
    with path.open(encoding="utf-8") as fh:
        return Counter(json.loads(line)["paper_id"] for line in fh if line.strip())


def _concat(sources: list[Path], dest: Path) -> None:
    with dest.open("wb") as out:
        for src in sources:
            if src.exists():
                with src.open("rb") as fh:
                    shutil.copyfileobj(fh, out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge KB shard builds into one knowledge base")
    parser.add_argument("shards", nargs="+", type=Path, help="each member's data/ folder")
    parser.add_argument("--out", required=True, type=Path, help="output data/ folder")
    parser.add_argument("--qa-from", type=int, default=0, help="shard (position in the list) whose QA sample is kept")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = get_config(args.config)

    out = args.out
    if any((out / "processed" / f).exists() for f in ("passages.jsonl", "sections.jsonl", "papers.sqlite")):
        sys.exit(f"{out}/processed already has outputs; pick an empty folder")

    metas = [read_metadata_csv(s / "raw/metadata.csv") for s in args.shards]
    cols = [c for c in metas[0].columns if c != "pdf_path"]
    for s, m in zip(args.shards[1:], metas[1:]):
        if list(m.columns) != list(metas[0].columns) or not m[cols].equals(metas[0][cols]):
            sys.exit(f"{s}: metadata.csv is not the same selection as {args.shards[0]} (columns other than pdf_path differ)")
    merged = metas[0].copy()
    for s, m in zip(args.shards[1:], metas[1:]):
        both = (merged.pdf_path != "") & (m.pdf_path != "")
        if both.any():
            sys.exit(f"{s}: {int(both.sum())} papers have a PDF in two shards, e.g. {merged.paper_id[both].iloc[0]}")
        merged.loc[m.pdf_path != "", "pdf_path"] = m.pdf_path[m.pdf_path != ""]

    counts: Counter = Counter()
    for s in args.shards:
        c = _paper_ids(s / "processed/passages.jsonl")
        if counts.keys() & c.keys():
            sys.exit(f"{s}: papers already built by an earlier shard, e.g. {sorted(counts.keys() & c.keys())[0]}")
        counts.update(c)

    for d in ("raw", "processed", "results"):
        (out / d).mkdir(parents=True, exist_ok=True)
    write_metadata_csv(merged, out / "raw/metadata.csv")
    for rel in ("processed/passages.jsonl", "processed/sections.jsonl", "raw/parse_failures.jsonl",
                "raw/download_failures.jsonl", "results/parse_quality.jsonl"):
        _concat([s / rel for s in args.shards], out / rel)

    sqlite = out / "processed/papers.sqlite"
    with_pdf = merged[merged.pdf_path != ""]
    store.upsert_papers([{**r, "source": "arxiv"} for r in with_pdf.to_dict("records")], sqlite)
    store.set_num_chunks(dict(counts), sqlite)

    qa = args.shards[args.qa_from] / "results/parse_qa_sample.jsonl"
    if qa.exists():
        shutil.copy(qa, out / "results/parse_qa_sample.jsonl")
        _write_qa_csv(out / "results/parse_qa_sample.jsonl", out / "results/parse_qa_sample.csv")

    kb = cfg["kb"]
    stats = compute_corpus_stats(out / "processed/passages.jsonl", out / "processed/sections.jsonl")
    stats["chunking"] = {"mode": "section_fixed", **{k: kb["chunk"].get(k) for k in (
        "chunk_tokens", "overlap_tokens", "min_new_tokens", "tokenizer", "tiktoken_encoding", "prefix")}}
    stats["pipeline"] = _parse_summary(out / "raw", stats["n_documents"])
    stats["shards"] = [str(s) for s in args.shards]
    write_stats_report(stats, out / "results" / kb["stats"]["out_file"])
    pipe = stats["pipeline"]
    print(f"merged {len(args.shards)} shards: {pipe['papers_selected']} selected -> {pipe['pdfs_available']} PDFs -> "
          f"{pipe['papers_built']} built, {pipe['papers_failed']} failed; {sum(counts.values())} passages")


if __name__ == "__main__":
    main()
