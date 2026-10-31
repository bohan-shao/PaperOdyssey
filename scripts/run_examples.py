"""Run the UI example inputs through Paper Search + cited answer (Q2 five queries, D51).

Usage:
    python scripts/run_examples.py [--config config.yaml] [--out PATH]

Pipeline: eval.examples.run_examples (needs the built indexes, the models and an LLM key; start it
in a fresh process) -> results/metrics/examples_<date>.json + one line per input on stdout. Uses
config.yaml's settings and `ui.example_inputs` with the search cache switched off in memory;
config.yaml is never modified. Exits 1 if any input failed (the report still lists every input).
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
from eval.examples import format_summary, run_examples  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the UI example inputs through the deployed system")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", type=Path, default=None,
                        help="report path (default results/metrics/examples_<date>.json)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    get_config(args.config)
    report = run_examples()
    out = args.out or get_path("results") / "metrics" / f"examples_{date.today().isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n" + format_summary(report) + f"\n\nreport -> {out}")
    if report["summary"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
