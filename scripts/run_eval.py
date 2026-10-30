"""Unified evaluation entry point (EVALUATION.md section 9).

Usage:
    python scripts/run_eval.py --config config.yaml --stage retrieval
        [--configs bm25 dense hybrid_rrf hybrid_rerank] [--limit N] [--tag TAG]

Stages:
- retrieval: eval.retrieval_eval.run_retrieval_eval. Every eval-set question goes through
  retrieve() under the four EVALUATION.md section 2 configurations (switched in memory; config.yaml
  is never modified): hits -> results/retrieval_results/<configuration>_<tag>.jsonl (SCHEMA.md
  section 8), metrics per split -> results/metrics/retrieval_<tag>.json (with a config snapshot and
  dependency versions), and a summary on stdout. Needs the built lookup/BM25/FAISS indexes and the
  models (a GPU for the dense encoder and the reranker); run it in a fresh process.
- review, all: not implemented yet (eval/review_metrics.py is a stub). --ablation (a named config
  override -> metrics/ablation_<innovation>_<date>.json) is not implemented yet either.
  Generation/RAGAS stages remain pending (Q4).

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation entry point")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--stage", choices=["retrieval", "review", "all"], default="all"
    )
    parser.add_argument("--ablation", default=None,
                        help="ablation config name (corresponds to a config override); not implemented yet")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS),
                        help="retrieval stage: the configurations to run, in this order (default: all four)")
    parser.add_argument("--limit", type=int, default=None,
                        help="retrieval stage: only the first N questions (a quick check)")
    parser.add_argument("--tag", default=None,
                        help="file-name suffix of the outputs (default: today's date, plus _first<N> with --limit)")
    args = parser.parse_args(argv)
    if args.stage != "retrieval":
        parser.error(f"--stage {args.stage} is not implemented yet (the review stage, "
                     "eval/review_metrics.py, is a stub); use --stage retrieval")
    if args.ablation is not None:
        parser.error("--ablation is not implemented yet")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = get_config(args.config)
    tag = args.tag or date.today().isoformat() + (f"_first{args.limit}" if args.limit else "")
    eval_set = get_path("eval") / cfg["eval"]["annotation"]["eval_set_file"]
    report = run_retrieval_eval(eval_set, get_path("results") / "retrieval_results", tag, args.configs, args.limit)
    out = get_path("results") / "metrics" / f"retrieval_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n" + format_summary(report) + f"\n\nreport -> {out}")


if __name__ == "__main__":
    main()
