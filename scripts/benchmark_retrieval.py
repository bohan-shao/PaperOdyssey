"""Benchmark retrieval-core latency (Q2 latency, EVALUATION.md section 7).

Usage:
    python scripts/benchmark_retrieval.py [--config config.yaml] [--queries 100] [--concurrency 1 4 8]
        [--ef 32 64 128 256] [--depths 10 20 30 50] [--protocol legacy_50_v2|production] [--out PATH]

Pipeline: eval.latency.run_benchmark (needs the lookup, BM25 and FAISS indexes and the models;
run inside Docker) -> results/metrics/latency_retrieval_<date>.json (latency_retrieval_production_<date>.json
with --protocol production) + a summary on stdout.
Measures retrieve() only, no LLM stages. Uses config.yaml; configurations are switched in memory
and config.yaml is never modified. Numbers depend on corpus size: rerun after the full KB build.
--protocol production times the configured settings (D51, the deployed system for Q2); the default
legacy_50_v2 fixes the ranking settings for comparisons across config changes. With production,
dense.top_k is 150 and FAISS HNSW searches with max(efSearch, k), so pass --ef values >= 150.
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
from eval.latency import PROTOCOLS, format_summary, run_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval-core latency")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--queries", type=int, default=100, help="paper titles sampled as queries")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--ef", type=int, nargs="+", default=[32, 64, 128, 256], help="efSearch values")
    parser.add_argument("--depths", type=int, nargs="+", default=[10, 20, 30, 50], help="rerank depths")
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), default="legacy_50_v2",
                        help="legacy_50_v2 (default): fixed ranking settings; production: the configured settings (D51)")
    parser.add_argument("--out", type=Path, default=None,
                        help="report path (default results/metrics/latency_retrieval_[production_]<date>.json)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    get_config(args.config)
    report = run_benchmark(args.queries, args.concurrency, args.ef, args.depths, args.protocol)
    prefix = "latency_retrieval_production" if args.protocol == "production" else "latency_retrieval"
    out = args.out or get_path("results") / "metrics" / f"{prefix}_{date.today().isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n" + format_summary(report) + f"\n\nreport -> {out}")


if __name__ == "__main__":
    main()
