"""Unified evaluation entry point (EVALUATION.md section 9).

Usage:
    python scripts/run_eval.py --config config.yaml --stage retrieval
        [--configs bm25 dense hybrid_rrf hybrid_rerank] [--limit N] [--tag TAG]
    python scripts/run_eval.py --config config.yaml --stage review
        [--split all|dev|eval] [--limit N] [--tag TAG] [--resume]
    python scripts/run_eval.py --config config.yaml --stage all

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
- all (the default): retrieval, then review. --configs affects retrieval; --split and --resume
  affect review. --ablation and generation/RAGAS stages remain unimplemented.

The tag defaults to today's date; with --limit it gets a `_first<N>` suffix, so a quick check
never overwrites a full run's files. Reproducibility: the retrieval stage keeps no cache between
runs (retrieval makes no LLM calls); it saves every question's hits, from which the metrics can be
recomputed, and the report records the eval set's SHA-256, the index manifests and the config.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config, get_path  # noqa: E402
from eval.retrieval_eval import CONFIGS, format_summary, run_retrieval_eval  # noqa: E402
from eval.review_eval import format_summary as format_review_summary, run_review_eval  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation entry point")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--stage", choices=["retrieval", "review", "all"], default="all"
    )
    parser.add_argument("--ablation", default=None,
                        help="ablation config name (corresponds to a config override); not implemented yet")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=None,
                        help="retrieval stage: the configurations to run, in this order (default: all four)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N selected questions per stage (a quick check)")
    parser.add_argument("--tag", default=None,
                        help="file-name suffix of the outputs (default: today's date, plus _first<N> with --limit)")
    parser.add_argument("--split", choices=["all", "dev", "eval"], default=None,
                        help="review stage: select a split before applying --limit (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="review stage: resume the same tag with matching data/config/code fingerprints")
    args = parser.parse_args(argv)
    if args.ablation is not None:
        parser.error("--ablation is not implemented yet")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.stage == "retrieval" and (args.split is not None or args.resume):
        parser.error("--split and --resume apply only to the review stage (or its part of --stage all)")
    if args.stage == "review" and args.configs is not None:
        parser.error("--configs applies only to the retrieval stage (or its part of --stage all)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = get_config(args.config)
    tag = args.tag or date.today().isoformat() + (f"_first{args.limit}" if args.limit else "")
    eval_set = get_path("eval") / cfg["eval"]["annotation"]["eval_set_file"]
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
                                     limit=args.limit, split=args.split or "all", resume=args.resume)
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
