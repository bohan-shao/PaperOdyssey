"""Unified evaluation entry point (EVALUATION.md section 9).

Usage:
    python scripts/run_eval.py --config config.yaml --stage retrieval
        [--configs bm25 dense hybrid_rrf hybrid_rerank] [--limit N] [--tag TAG]
    python scripts/run_eval.py --config config.yaml --stage review
        [--split all|dev|eval] [--limit N] [--tag TAG] [--resume] [--workers N]
    python scripts/run_eval.py --config config.yaml --stage all
    python scripts/run_eval.py --config config.yaml --stage retrieval --ablation query_views diversity
        [--split all|dev|eval] [--limit N] [--tag TAG] [--resume] [--dry-run]
    python scripts/run_eval.py --config config.yaml --stage review --ablation loop hyde --workers 8

Stages:
- retrieval: eval.retrieval_eval.run_retrieval_eval. Every eval-set question goes through
  retrieve() under the four EVALUATION.md section 2 configurations (switched in memory; config.yaml
  is never modified): hits -> results/retrieval_results/<configuration>_<tag>.jsonl (SCHEMA.md
  section 8), metrics per split -> results/metrics/retrieval_<tag>.json (with a config snapshot and
  dependency versions), and a summary on stdout. Needs the built lookup/BM25/FAISS indexes and the
  models (a GPU for the dense encoder and the reranker); run it in a fresh process.
- review: eval.review_eval.run_review_eval runs Paper Search with full pass/reject traces and
  online caching disabled. The runner saves resumable results and its report under
  results/review_runs/<tag>/; --split and --resume apply only to this stage. A partial or failed
  report prints its summary to stderr and exits 1. Resume requires matching run fingerprints.
  --workers N runs up to N questions at once in this process (D48; default 1). It applies to the
  review stage and to the review cells of an ablation, and a run may resume with another value.
- all (the default): retrieval, then review. --configs affects retrieval; --split and --resume
  affect review.
- ablation: one or two named factors, with an explicit --stage. Runs baseline/A (one factor)
  or baseline/A/B/AB (two factors), with selected factors off in the baseline. --split and
  --resume also work for retrieval ablations. --dry-run prints the plan without writing files.
  Reports go under ablation_runs/<tag>/ and to metrics/ablation_<factors>_<tag>.json.
  Generation/RAGAS stages remain unimplemented.

The tag defaults to today's date; with --limit it gets a `_first<N>` suffix, so a quick check
never overwrites a full run's files. Reproducibility: the retrieval stage keeps no cache between
runs (retrieval makes no LLM calls); it saves every question's hits, from which the metrics can be
recomputed, and the report records the eval set's SHA-256, the index manifests and the config.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config, get_path  # noqa: E402
from eval.ablation import FACTORS, format_summary as format_ablation_summary, run_ablation  # noqa: E402
from eval.retrieval_eval import CONFIGS, format_summary, run_retrieval_eval  # noqa: E402
from eval.review_eval import MAX_WORKERS, format_summary as format_review_summary, run_review_eval  # noqa: E402


def _write_report(path: Path, report: dict) -> None:
    """Publish a complete JSON file atomically, retaining the old file if writing fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation entry point")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--stage", choices=["retrieval", "review", "all"], default=None,
        help="stage to evaluate (default: all; must be explicit with --ablation)",
    )
    parser.add_argument("--ablation", nargs="+", choices=list(FACTORS), default=None,
                        help="one or two named factors; baseline disables all selected factors")
    parser.add_argument("--dry-run", action="store_true",
                        help="ablation only: validate and print the experiment plan without running or saving it")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=None,
                        help="retrieval stage: the configurations to run, in this order (default: all four)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N selected questions per stage (a quick check)")
    parser.add_argument("--tag", default=None,
                        help="file-name suffix of the outputs (default: today's date, plus _first<N> with --limit)")
    parser.add_argument("--split", choices=["all", "dev", "eval"], default=None,
                        help="review or ablation: select a split before applying --limit (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="review or ablation: resume the same tag with matching run fingerprints")
    parser.add_argument("--workers", type=int, default=1,
                        help=f"review stage or review ablation cells: questions run at once (1-{MAX_WORKERS}, "
                             "default 1; D48)")
    args = parser.parse_args(argv)
    if args.ablation is not None:
        if args.stage is None:
            parser.error("--ablation requires an explicit --stage retrieval, review or all")
        if args.configs is not None:
            parser.error("--configs cannot be combined with --ablation")
    elif args.dry_run:
        parser.error("--dry-run requires --ablation")
    args.stage = args.stage or "all"
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.ablation is None and args.stage == "retrieval" and (args.split is not None or args.resume):
        parser.error("--split and --resume apply only to the review stage (or its part of --stage all)")
    if args.stage == "review" and args.configs is not None:
        parser.error("--configs applies only to the retrieval stage (or its part of --stage all)")
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be from 1 to {MAX_WORKERS}")
    if args.stage == "retrieval" and args.workers != 1:
        parser.error("--workers applies only to the review stage (retrieval makes no LLM calls)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = get_config(args.config)
    tag = args.tag or date.today().isoformat() + (f"_first{args.limit}" if args.limit else "")
    eval_set = get_path("eval") / cfg["eval"]["annotation"]["eval_set_file"]
    if args.ablation is not None:
        try:
            report = run_ablation(eval_set, get_path("results") / "ablation_runs", tag,
                                  factors=args.ablation, stage=args.stage, split=args.split or "all",
                                  limit=args.limit, resume=args.resume, dry_run=args.dry_run,
                                  workers=args.workers)
            out = get_path("results") / "metrics" / f"ablation_{'_'.join(args.ablation)}_{tag}.json"
            if not args.dry_run:
                _write_report(out, report)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Ablation {tag}: failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        successful = report.get("status") in ("complete", "planned")
        summary = format_ablation_summary(report)
        if not args.dry_run:
            summary += f"\n\nreport -> {out}"
        print("\n" + summary, file=sys.stdout if successful else sys.stderr)
        if not successful:
            if report.get("error"):
                print(f"ablation error: {report['error']}", file=sys.stderr)
            raise SystemExit(1)
        return
    if args.stage in ("retrieval", "all"):
        configs = args.configs if args.configs is not None else list(CONFIGS)
        report = run_retrieval_eval(eval_set, get_path("results") / "retrieval_results", tag, configs, args.limit)
        out = get_path("results") / "metrics" / f"retrieval_{tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("\n" + format_summary(report) + f"\n\nreport -> {out}")
    if args.stage in ("review", "all"):
        try:
            report = run_review_eval(eval_set, get_path("results") / "review_runs", tag,
                                     limit=args.limit, split=args.split or "all", resume=args.resume,
                                     workers=args.workers)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Review {tag}: failed before a report was returned: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        out = get_path("results") / "review_runs" / tag / "report.json"
        complete = report.get("status") == "complete"
        stream = sys.stdout if complete else sys.stderr
        print("\n" + format_review_summary(report) + f"\n\nreport -> {out}", file=stream)
        if not complete:
            if report.get("error"):
                print(f"review error: {report['error']}", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
