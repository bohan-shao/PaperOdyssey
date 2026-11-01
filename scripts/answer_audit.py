"""Human audit of Paper QA answers (Q4 answer quality, D54).

Usage:
    python scripts/answer_audit.py sheets --run RUN_ROW_DIR [--config config.yaml] [--out DIR]
    python scripts/answer_audit.py score [--config config.yaml] [--out DIR] [--tag TAG]

sheets: eval.answer_audit.write_sheets samples `eval.answer_audit.sizes` answers of one finished
answer-run row (e.g. results/answer_runs/loop-hyde_2026-10-30/rows/a, the configured defaults)
by LLM-judge stratum and writes, under paths.eval/eval.answer_audit.dir (or --out), one marking
sheet answer_audit_<annotator>.csv and one reading packet answer_audit_<annotator>.html per
annotator, plus the manifest answer_audit.json. It never overwrites an existing audit.
score: eval.answer_audit.score_sheets reads the marked sheets back -> results/metrics/
answer_audit_<tag>.json (tag defaults to today's date) + a summary on stdout. Marking rules:
eval/annotation/ANSWER_AUDIT.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config, get_path  # noqa: E402
from eval.answer_audit import format_summary, score_sheets, write_sheets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Human audit of Paper QA answers")
    parser.add_argument("command", choices=["sheets", "score"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run", type=Path, default=None, help="sheets: an answer-run row directory")
    parser.add_argument("--out", type=Path, default=None, help="audit folder (default paths.eval/eval.answer_audit.dir)")
    parser.add_argument("--tag", default=None, help="score: report suffix (default today's date)")
    args = parser.parse_args()
    cfg = get_config(args.config)
    settings = cfg["eval"]["answer_audit"]
    out_dir = args.out or get_path("eval") / settings["dir"]
    try:
        if args.command == "sheets":
            if args.run is None:
                parser.error("sheets needs --run RUN_ROW_DIR")
            manifest = write_sheets(args.run, out_dir, cfg["eval"]["annotation"]["annotators"], settings["sizes"],
                                    per_answer=settings["per_answer"], split=settings["split"], seed=cfg.get("seed", 42))
            print(f"{len(manifest['items'])} answers sampled {manifest['sampled']} of {manifest['population']}; "
                  f"answers per annotator {manifest['workload']} -> {out_dir}")
        else:
            report = score_sheets(out_dir)
            out = get_path("results") / "metrics" / f"answer_audit_{args.tag or date.today().isoformat()}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(format_summary(report) + f"\n\nreport -> {out}")
    except (OSError, ValueError, KeyError) as exc:
        print(f"answer audit {args.command}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
