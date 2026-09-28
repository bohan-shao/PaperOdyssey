"""Unified evaluation entry point. TODO.

Usage:
    python scripts/run_eval.py --config config.yaml --stage retrieval|review|all
    python scripts/run_eval.py --config config.yaml --stage retrieval --ablation rerank_off

Pipeline: run the corresponding metrics per docs/EVALUATION.md; results land in data/results/metrics/.
Constraints: each run outputs a config snapshot + dependency versions; same-query caching ensures reproducibility.
Note: paper QA is implemented; generation/ragas metric stages remain pending (Q4).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation entry point")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--stage", choices=["retrieval", "review", "all"], default="all"
    )
    parser.add_argument("--ablation", default=None, help="ablation config name (corresponds to a config override)")
    args = parser.parse_args()
    raise NotImplementedError  # TODO: implement, see docs/EVALUATION.md


if __name__ == "__main__":
    main()
