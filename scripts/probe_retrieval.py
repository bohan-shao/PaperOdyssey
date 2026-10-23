"""Run a small known-source retrieval comparison; see eval/README.md for limitations."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config, get_path  # noqa: E402
from eval.retrieval_probe import run_probe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--papers", type=int, default=300)
    parser.add_argument("--queries", type=int, default=60)
    parser.add_argument("--hybrid", action="store_true", help="Use the configured small probe embedding model")
    parser.add_argument("--rerank", action="store_true", help="Use the configured small probe cross-encoder")
    args = parser.parse_args()
    get_config(args.config)
    report = run_probe(args.papers, args.queries, hybrid=args.hybrid, reranking=args.rerank)
    mode = "hybrid_rerank" if args.hybrid and args.rerank else "hybrid" if args.hybrid else "sparse"
    out = get_path("results") / "metrics" / f"ablation_retrieval_probe_{mode}_{date.today().isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, row in report["rows"].items():
        print(name, json.dumps(row["metrics"]), "p50_ms=", row["latency"]["total_ms"]["p50"])
    print(out)


if __name__ == "__main__":
    main()
