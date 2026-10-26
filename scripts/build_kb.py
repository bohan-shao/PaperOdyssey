"""Build the knowledge base from downloaded PDFs: parse -> clean -> chunk -> store -> stats.

Usage:
    python scripts/build_kb.py --config config.yaml [--limit N] [--rebuild] [--workers W]
        [--chunk-mode section_fixed|semantic] [--passages-file NAME] [--from-sections] [--stats-only]

Pipeline (per paper, parallel over processes):
    kb.parse.parse_pdf -> kb.parse.quality_check -> (+ metadata abstract if no Abstract section; the
    paper's own copy is then dropped from leading front matter, kb.parse.drop_leading_abstract)
    -> kb.chunk.chunk_section_fixed | chunk_semantic -> kb.store.write_passages / write_parents
    -> kb.store.set_num_chunks; finally kb.stats.compute_corpus_stats -> results/corpus_stats.json

Outputs:
    processed/passages.jsonl, processed/sections.jsonl, processed/papers.sqlite (num_chunks),
    raw/parse_failures.jsonl (parse errors + quality rejects), results/parse_quality.jsonl
    (per-paper quality scores), results/parse_qa_sample.jsonl (kb.parse.qa_sample random papers
    with their first sections, for the manual quality review reported in Q1),
    results/corpus_stats.json.

Resumable: papers already in passages.jsonl are skipped unless --rebuild; papers that failed
before (raw/parse_failures.jsonl) are skipped unless --retry-failed. The corpus funnel
(selected -> PDFs -> built/failed, failure rate and reasons) is written into corpus_stats.json.
Ablation (D21): `--chunk-mode semantic --passages-file passages_semantic.jsonl` builds a
separate passages file without touching the baseline: papers.sqlite num_chunks, sections.jsonl
and the parse quality/QA files keep the baseline's values.
`--from-sections` (D36) chunks the parsed sections stored in processed/sections.jsonl instead of
the PDFs (kb.rechunk), so passages can be built after a shard merge (D35), when no machine holds
every PDF: an ablation file with `--passages-file`, or, with `--rebuild` and no `--passages-file`,
the baseline passages.jsonl and papers.sqlite num_chunks again (e.g. after a chunker fix, D37);
sections.jsonl and the parse quality/QA files are never changed.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm  # noqa: E402

from config import PROJECT_ROOT, get_config  # noqa: E402
from kb import chunk, rechunk, store  # noqa: E402
from kb.fetch import read_metadata_csv  # noqa: E402
from kb.parse import ParseOptions, PDFParseError, drop_leading_abstract, parse_pdf, quality_check  # noqa: E402
from kb.stats import compute_corpus_stats, parse_summary, write_qa_csv, write_stats_report  # noqa: E402

log = logging.getLogger("build_kb")


def process_paper(row: dict, raw_dir: str, cfg: dict, mode: str) -> dict:
    """Parse + quality-check + chunk one paper (runs in a worker process)."""
    kb = cfg["kb"]
    opts = ParseOptions.from_config(kb.get("parse"))
    pdf = Path(raw_dir) / row["pdf_path"]
    try:
        sections = parse_pdf(pdf, opts)
    except PDFParseError as exc:
        return {"paper_id": row["paper_id"], "error": f"parse: {exc}"}
    except Exception as exc:  # noqa: BLE001 (one bad PDF must not stop the build)
        return {"paper_id": row["paper_id"], "error": f"parse: {type(exc).__name__}: {exc}"}
    quality = {"paper_id": row["paper_id"], **quality_check(row["pdf_path"], sections, opts)}
    if not quality["ok"]:
        return {"paper_id": row["paper_id"], "error": "quality check failed", "quality": quality}

    # The metadata abstract is authoritative; add it when the parser found no Abstract section
    # (and drop the paper's own copy from leading front matter, so it is not there twice).
    if row.get("abstract") and not any(s["section"].lower() == "abstract" for s in sections):
        try:
            rest = drop_leading_abstract(sections, row["abstract"])
        except Exception:  # noqa: BLE001 (a cleanup step must not stop the build; keep the text as parsed)
            rest = sections
        sections = [{"section": "Abstract", "text": row["abstract"]}, *rest]

    meta = {k: row.get(k) for k in ("arxiv_id", "title", "year", "categories")}
    ch = kb["chunk"]
    try:
        tok = chunk.tokenizer_from_config(cfg)
        if mode == "semantic":
            passages = chunk.chunk_semantic(
                row["paper_id"], sections, ch["semantic_threshold"],
                ch.get("semantic_min_tokens", 100), ch.get("semantic_max_tokens", 500),
                meta=meta, prefix_fmt=ch["prefix"], tokenizer=tok,
                model_name=ch.get("semantic_model", "BAAI/bge-small-en-v1.5"),
            )
        else:
            passages = chunk.chunk_section_fixed(
                row["paper_id"], sections, ch["chunk_tokens"], ch["overlap_tokens"],
                meta=meta, prefix_fmt=ch["prefix"], tokenizer=tok,
                min_new_tokens=ch.get("min_new_tokens", 50),
            )
        parents = chunk.build_parent_records(row["paper_id"], sections, meta=meta, prefix_fmt=ch["prefix"])
    except Exception as exc:  # noqa: BLE001 (one bad paper must not stop the build)
        return {"paper_id": row["paper_id"], "error": f"chunk: {type(exc).__name__}: {exc}", "quality": quality}
    return {"paper_id": row["paper_id"], "passages": passages, "parents": parents,
            "quality": quality, "sections": sections}


def _append_jsonl(path: Path, records: list[dict]) -> None:
    if records:
        with path.open("a", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build passages.jsonl / sections.jsonl / papers.sqlite from PDFs")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=0, help="process at most N papers in total (trial runs)")
    parser.add_argument("--rebuild", action="store_true", help="delete existing outputs and rebuild from scratch")
    parser.add_argument("--retry-failed", action="store_true",
                        help="retry papers listed in raw/parse_failures.jsonl (skipped by default)")
    parser.add_argument("--workers", type=int, default=0, help="parallel processes (default kb.parse.workers)")
    parser.add_argument("--chunk-mode", choices=["section_fixed", "semantic"], default=None,
                        help="override kb.chunk.mode (Q3 ablation)")
    parser.add_argument("--passages-file", default=None, help="override kb.store.passages_file (ablation outputs)")
    parser.add_argument("--from-sections", action="store_true",
                        help="chunk the parsed sections in processed/sections.jsonl instead of the PDFs")
    parser.add_argument("--stats-only", action="store_true", help="only recompute corpus statistics")
    args = parser.parse_args()

    cfg = get_config(args.config)
    kb = cfg["kb"]
    paths = {k: PROJECT_ROOT / str(v) for k, v in cfg["paths"].items()}
    raw, processed, results = paths["raw"], paths["processed"], paths["results"]
    for d in (raw, processed, results):
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(raw / "build_kb.log", encoding="utf-8"),
                                  logging.StreamHandler()])

    mode = args.chunk_mode or kb["chunk"].get("mode", "section_fixed")
    passages_path = processed / (args.passages_file or kb["store"]["passages_file"])
    sections_path = processed / kb["store"].get("sections_file", "sections.jsonl")
    sqlite_path = processed / kb["store"]["sqlite_file"]
    stats_path = results / kb["stats"]["out_file"]
    if mode != "section_fixed":
        stats_path = stats_path.with_name(f"{stats_path.stem}_{mode}{stats_path.suffix}")
    failures_path = raw / "parse_failures.jsonl"
    qa_path = results / "parse_qa_sample.jsonl"

    # an ablation (--passages-file) writes only its own passages file; the shared outputs
    # (papers.sqlite num_chunks, parse quality/QA files, sections.jsonl) stay the baseline's
    ablation = args.passages_file is not None

    rechunked: dict = {}
    if args.from_sections and not args.stats_only:
        if args.rebuild:
            store.reset_jsonl(passages_path)
        t0 = time.time()
        with tqdm(total=rechunk.paper_count(sections_path), desc=f"{mode} from sections", unit="paper") as bar:
            rechunked = rechunk.rechunk_from_sections(sections_path, passages_path, cfg, mode, progress=bar.update)
        if not ablation:  # the baseline's num_chunks follow its passages (0 for papers without any)
            store.reset_num_chunks(sqlite_path)
            store.set_num_chunks(dict(Counter(p["paper_id"] for p in store.iter_passages(passages_path))), sqlite_path)
        log.info("from %s: %d papers (%d chunks), %d already in %s, in %.0fs", sections_path.name,
                 rechunked["papers"], rechunked["passages"], rechunked["skipped"], passages_path.name, time.time() - t0)
    elif not args.stats_only:
        meta = read_metadata_csv(raw / "metadata.csv")
        meta = meta[meta["pdf_path"] != ""]
        if args.rebuild:
            store.reset_jsonl(passages_path)
            if not ablation:  # an ablation rebuild must not wipe the baseline's shared outputs
                store.reset_jsonl(sections_path)
                store.reset_num_chunks(sqlite_path)  # counted again as papers are built (failed/skipped stay 0)
                for f in (failures_path, results / "parse_quality.jsonl", qa_path):
                    f.unlink(missing_ok=True)
        if args.retry_failed:
            failures_path.unlink(missing_ok=True)
        done = store.stored_paper_ids(passages_path)
        failed = {r["paper_id"] for r in store.read_records(failures_path)}
        todo = meta[~meta["paper_id"].isin(done | failed)]
        if args.limit:
            todo = todo.head(max(0, args.limit - len(done)))
        log.info("mode=%s: %d papers with PDFs, %d already built, %d previously failed (skipped), %d to process",
                 mode, len(meta), len(done), len(failed), len(todo))

        store.upsert_papers([{**r, "source": "arxiv"} for r in meta.to_dict("records")], sqlite_path)
        sections_done = store.stored_paper_ids(sections_path)
        workers = args.workers or kb.get("parse", {}).get("workers", 4)
        qa_need = 0 if ablation else max(0, kb.get("parse", {}).get("qa_sample", 50) - len(store.read_records(qa_path)))
        rng = random.Random(cfg.get("seed", 42))
        qa_ids = set(rng.sample(sorted(todo["paper_id"]), min(qa_need, len(todo))))
        n_ok = n_fail = n_chunks = 0
        t0 = time.time()
        rows = todo.to_dict("records")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_paper, r, str(raw), cfg, mode) for r in rows]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="parse+chunk", unit="paper"):
                res = fut.result()
                pid = res["paper_id"]
                if "quality" in res and not ablation:
                    _append_jsonl(results / "parse_quality.jsonl", [res["quality"]])
                if "error" not in res and not res["passages"]:
                    res["error"] = "chunk: no passages produced"
                if "error" in res:
                    n_fail += 1
                    _append_jsonl(failures_path, [{
                        "paper_id": pid, "error": res["error"], "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }])
                    continue
                # parents first: a paper counts as built once its passages are written (resume marker)
                if pid not in sections_done:
                    store.write_parents(res["parents"], sections_path)  # same parse in every chunk mode
                store.write_passages(res["passages"], passages_path)
                if not ablation:
                    store.set_num_chunks({pid: len(res["passages"])}, sqlite_path)
                n_ok += 1
                n_chunks += len(res["passages"])
                if pid in qa_ids:
                    _append_jsonl(qa_path, [{
                        "paper_id": pid, "pdf_path": res["quality"]["pdf_path"], "quality": res["quality"],
                        "sections": [{"section": s["section"], "text": s["text"][:1500]} for s in res["sections"]],
                        "review": {"ok": None, "notes": ""},  # filled in by the manual reviewer
                    }])
        elapsed = time.time() - t0
        log.info("this run: built %d papers (%d chunks), %d failed, in %.0fs",
                 n_ok, n_chunks, n_fail, elapsed)

    if qa_path.exists():
        write_qa_csv(qa_path, qa_path.with_suffix(".csv"))

    stats = compute_corpus_stats(passages_path, sections_path)
    settings = (("semantic_threshold", "semantic_model", "semantic_min_tokens", "semantic_max_tokens") if mode == "semantic"
                else ("chunk_tokens", "overlap_tokens", "min_new_tokens"))
    stats["chunking"] = {"mode": mode, **{k: kb["chunk"].get(k) for k in (
        *settings, "tokenizer", "tiktoken_encoding", "prefix")}}
    if args.from_sections:
        stats["chunking"]["source"] = sections_path.name
        if "embedding_device" in rechunked:
            stats["chunking"]["embedding_device"] = rechunked["embedding_device"]
    stats["pipeline"] = parse_summary(raw, stats["n_documents"])
    write_stats_report(stats, stats_path)
    pipe = stats["pipeline"]
    print(f"pipeline: {pipe.get('papers_selected', '?')} selected -> {pipe.get('pdfs_available', '?')} PDFs -> "
          f"{pipe['papers_built']} built, {pipe['papers_failed']} failed ({pipe['failure_rate']:.1%}) "
          f"{pipe['failure_reasons']}")


if __name__ == "__main__":
    main()
