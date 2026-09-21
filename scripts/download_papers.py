"""Download papers: metadata -> citations -> filter -> PDFs.

Usage:
    python scripts/download_papers.py --config config.yaml [--limit N] [--resume]
        [--metadata-only] [--skip-citations] [--shard I/N]

Pipeline: kb.fetch.fetch_arxiv_metadata -> kb.fetch.enrich_citations -> kb.fetch.select_papers
    -> raw/metadata.csv -> kb.fetch.download_pdfs -> raw/metadata.csv (pdf_path) -> kb.store.upsert_papers

- Idempotent/resumable: arXiv slices and OpenAlex lookups are cached under raw/, existing PDFs
  are skipped. `--resume` reuses an existing raw/metadata.csv (no re-selection) and only
  downloads missing PDFs.
- `--limit N`: small trial run (N papers of metadata, no OpenAlex quota worries).
- Failed arXiv slices / OpenAlex batches stop the run before selection (exit 1) instead of
  selecting from partial data; rerun to retry only what failed (everything else is cached).
  `--allow-missing-citations` accepts unresolved OpenAlex lookups as uncited.
- `--shard I/N` (implies --resume): download only PDFs whose position in metadata.csv is I mod N
  (0-based), so the 3 KB members can split the ~17 h full download (arXiv rate limits are per IP).
  Everyone must use the same metadata.csv: run `--metadata-only` once, share raw/metadata.csv,
  then each member runs one shard. Afterwards, collect all PDFs in one raw/pdf/ and run
  `--resume` there to record every pdf_path (and retry failures).
- Logs to data/raw/download.log.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT, get_config  # noqa: E402
from kb import fetch  # noqa: E402
from kb.store import upsert_papers  # noqa: E402


def _setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the arXiv paper corpus")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=0, help="0=full run (use a small value for trial runs)")
    parser.add_argument("--resume", action="store_true",
                        help="reuse existing raw/metadata.csv and only download missing PDFs")
    parser.add_argument("--metadata-only", action="store_true", help="stop after writing metadata.csv")
    parser.add_argument("--skip-citations", action="store_true",
                        help="skip OpenAlex (citation_count = default); for quick trials")
    parser.add_argument("--allow-missing-citations", action="store_true",
                        help="select papers even if some OpenAlex lookups failed (they count as uncited)")
    parser.add_argument("--shard", default="", help="I/N: download only every N-th PDF starting at I")
    args = parser.parse_args()

    cfg = get_config(args.config)
    kb = cfg["kb"]
    flt, fcfg, dl = kb["filter"], kb.get("fetch", {}), kb["download"]
    raw = PROJECT_ROOT / str(cfg["paths"]["raw"])
    processed = PROJECT_ROOT / str(cfg["paths"]["processed"])
    raw.mkdir(parents=True, exist_ok=True)
    _setup_logging(raw / "download.log")
    log = logging.getLogger("download_papers")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    contact = fetch.clean_setting(fcfg.get("contact_email", ""))
    metadata_csv = raw / "metadata.csv"

    shard = None
    if args.shard:
        try:
            shard = tuple(int(x) for x in args.shard.split("/"))
            if len(shard) != 2 or not 0 <= shard[0] < shard[1]:
                raise ValueError
        except ValueError:
            parser.error(f"--shard must be I/N with 0 <= I < N, got {args.shard!r}")
        if not metadata_csv.exists():
            # every member must download from the SAME selection: never re-select on a shard machine
            parser.error(f"--shard needs the shared {metadata_csv} (copy it from the machine that "
                         "ran --metadata-only)")
        args.resume = True
    if args.resume and not metadata_csv.exists():
        log.warning("--resume: %s not found, running the full fetch + selection", metadata_csv)

    if args.resume and metadata_csv.exists():
        selected = fetch.read_metadata_csv(metadata_csv)
        log.info("resume: %d papers from %s", len(selected), metadata_csv)
    else:
        try:
            pool = fetch.fetch_arxiv_metadata(
                flt["categories"], flt["year_min"], flt["year_max"], args.limit,
                per_slice=fcfg.get("per_slice", 150),
                page_size=fcfg.get("page_size", 100),
                request_interval_s=fcfg.get("request_interval_s", 3.0),
                cache_dir=raw / "arxiv_slices",
                contact_email=contact,
                seed=cfg.get("seed", 42),
            )
        except fetch.ArxivFetchIncomplete as exc:
            log.error("%s", exc)
            sys.exit(1)
        log.info("candidate pool: %d papers", len(pool))
        if args.skip_citations:
            pool = pool.assign(citation_count=int(fcfg.get("citation_default", 0)))
        else:
            try:
                pool = fetch.enrich_citations(
                    pool,
                    batch_size=fcfg.get("openalex_batch", 50),
                    api_key=fcfg.get("openalex_api_key", ""),
                    contact_email=contact,
                    citation_default=fcfg.get("citation_default", 0),
                    cache_path=raw / "openalex_citations.jsonl",
                    allow_missing=args.allow_missing_citations,
                )
            except (fetch.OpenAlexQuotaError, fetch.CitationLookupIncomplete) as exc:
                # never select papers from partial citation data; the caches make the rerun cheap
                log.error("%s", exc)
                sys.exit(1)
        target = min(flt["target_papers"], args.limit) if args.limit else flt["target_papers"]
        selected = fetch.select_papers(
            pool, citation_min=flt["citation_min"], target_papers=target,
            year_min=flt["year_min"], year_max=flt["year_max"], categories=flt["categories"],
        )
        selected = selected.assign(pdf_path="")
        fetch.write_metadata_csv(selected, metadata_csv)
        log.info("selected %d papers (pool %d, citation_min %d) -> %s",
                 len(selected), len(pool), flt["citation_min"], metadata_csv)
        if len(selected) < 10_000 and not args.limit:
            log.warning("only %d papers selected; Q1 requires >= 10,000 documents "
                        "(raise kb.fetch.per_slice or lower kb.filter.citation_min)", len(selected))

    if args.metadata_only:
        return

    todo = selected
    if shard:
        i, n = shard
        todo = selected.iloc[i::n]
        log.info("shard %d/%d: %d papers", i, n, len(todo))

    done = fetch.download_pdfs(
        todo, raw,
        max_concurrency=dl.get("max_concurrency", 4),
        request_interval_s=dl.get("request_interval_s", 3.0),
        contact_email=contact,
        failures_path=raw / "download_failures.jsonl",
    )
    if shard:
        # merge only this shard's results; other shards' pdf_path values are preserved
        paths = dict(zip(done["paper_id"], done["pdf_path"], strict=True))
        selected["pdf_path"] = [paths.get(pid) or old
                                for pid, old in zip(selected["paper_id"], selected["pdf_path"], strict=True)]
    else:
        selected = done
    fetch.write_metadata_csv(selected, metadata_csv)

    upsert_papers(
        [{**r, "source": "arxiv"} for r in selected.to_dict("records")],
        processed / kb["store"]["sqlite_file"],
    )
    n_ok = int((selected["pdf_path"] != "").sum())
    log.info("PDFs available: %d/%d; metadata.csv and papers.sqlite updated", n_ok, len(selected))


if __name__ == "__main__":
    main()
