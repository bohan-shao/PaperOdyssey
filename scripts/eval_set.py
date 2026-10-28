"""Build the evaluation set: LLM-drafted questions checked by three annotators (eval.annotation.questions, D39).

Usage:
    python scripts/eval_set.py draft --n 1000 [--workers 8] [--papers PATH]   # needs an LLM key in .env
    python scripts/eval_set.py sheets [--round 1] [--per-question 2]
    python scripts/eval_set.py build
    python scripts/eval_set.py tiebreak [--round 1]                         # after build reports ties
    python scripts/eval_set.py rewrites                                      # after filling to_rewrite.csv

File names and the annotators come from config `eval.annotation` (files under `paths.eval`):
drafts.jsonl, annotation/round<R>_part<P>_<annotator>.csv, annotation/to_rewrite.csv,
eval_set.jsonl; the agreement report goes to `paths.results`/metrics/iaa_<date>.json. `draft`
samples from the configured papers.sqlite (--papers for another one, or a KB folder holding it)
with the config `seed`; rerun it to finish after an interruption, or with a larger --n to add
questions.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT, get_config  # noqa: E402
from eval.annotation import questions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the evaluation set (LLM-drafted, human-checked questions)")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("draft", help="draft one question per sampled paper with the LLM")
    p.add_argument("--n", type=int, required=True, help="papers to sample (query ids Q0001..)")
    p.add_argument("--workers", type=int, default=8, help="parallel LLM calls")
    p.add_argument("--papers", type=Path, default=None, help="papers.sqlite or a folder holding it (default from config)")
    p = sub.add_parser("sheets", help="write one marking sheet per annotator (questions without a sheet yet)")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--per-question", type=int, default=None,
                   help="annotators per question (default: all; e.g. 2 of 5, balanced over the pairs)")
    sub.add_parser("build", help="majority vote -> eval_set.jsonl, agreement report, to_rewrite.csv")
    p = sub.add_parser("tiebreak", help="sheets for one more annotator on each evenly split question")
    p.add_argument("--round", type=int, default=1)
    sub.add_parser("rewrites", help="add the new questions of to_rewrite.csv as the next round")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg = get_config(args.config)
    paths = {k: PROJECT_ROOT / str(v) for k, v in cfg["paths"].items()}

    try:
        run(args, cfg, paths)
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError, sqlite3.Error) as exc:  # RuntimeError: LLMError
        sys.exit(f"error: {exc}")


def run(args: argparse.Namespace, cfg: dict, paths: dict[str, Path]) -> None:
    names = cfg["eval"]["annotation"]
    drafts = paths["eval"] / names["drafts_file"]
    sheet_dir = paths["eval"] / names["sheets_dir"]
    rewrite = sheet_dir / names["rewrite_file"]
    annotators = tuple(names["annotators"])
    if args.cmd == "draft":
        from tqdm import tqdm

        sqlite_file = cfg["kb"]["store"]["sqlite_file"]
        db = args.papers or paths["processed"] / sqlite_file
        papers = questions.sample_papers(args.n, cfg["seed"], db / sqlite_file if db.is_dir() else db)
        with tqdm(total=len(papers), desc="drafting", unit="question") as bar:
            result = questions.draft_questions(papers, drafts, workers=args.workers, progress=bar.update)
        print(f"{result['drafted']} drafted, {result['skipped']} already there, {result['failed']} failed "
              f"(rerun to retry) -> {drafts}")
    elif args.cmd == "sheets":
        for path in questions.write_sheets(drafts, sheet_dir, args.round, annotators, args.per_question):
            print(path)
    elif args.cmd == "tiebreak":
        for path in questions.write_tiebreak_sheets(drafts, sheet_dir, args.round, annotators):
            print(path)
    elif args.cmd == "build":
        summary = questions.build_eval_set(
            drafts, sheet_dir, paths["eval"] / names["eval_set_file"],
            paths["results"] / "metrics" / f"iaa_{date.today().isoformat()}.json", rewrite,
            annotators, cfg["eval"].get("dev_ratio", 0.1), cfg["seed"])
        print(json.dumps(summary, indent=2))
    else:
        print(f"{questions.apply_rewrites(rewrite, drafts)} new questions added to {drafts}; "
              "run `sheets --round <next round>` for them")


if __name__ == "__main__":
    main()
