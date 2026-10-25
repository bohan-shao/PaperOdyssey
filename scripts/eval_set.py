"""Build the evaluation set: LLM-drafted questions checked by three annotators (eval.annotation.questions, D39).

Usage:
    python scripts/eval_set.py draft --n 1000 [--workers 8] [--papers PATH]   # needs an LLM key in .env
    python scripts/eval_set.py sheets [--round 1] [--annotators A B C]
    python scripts/eval_set.py build [--annotators A B C]
    python scripts/eval_set.py rewrites                                      # after filling to_rewrite.csv

Files (under config `paths.eval`): drafts.jsonl, annotation/round<R>_part<P>_<annotator>.csv,
annotation/to_rewrite.csv, eval_set.jsonl; the agreement report goes to
`paths.results`/metrics/iaa_<date>.json. `draft` samples from the configured papers.sqlite
(--papers to use another, e.g. a merged KB folder) with the config `seed`; rerun it to finish
after an interruption, or with a larger --n to add questions.
"""

from __future__ import annotations

import argparse
import json
import logging
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
    p.add_argument("--papers", type=Path, default=None, help="papers.sqlite (default from config)")
    p = sub.add_parser("sheets", help="write one marking sheet per annotator (questions without a sheet yet)")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--annotators", nargs="+", default=list(questions.ANNOTATORS))
    p = sub.add_parser("build", help="majority vote -> eval_set.jsonl, agreement report, to_rewrite.csv")
    p.add_argument("--annotators", nargs="+", default=list(questions.ANNOTATORS))
    sub.add_parser("rewrites", help="add the new questions of to_rewrite.csv as the next round")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg = get_config(args.config)
    paths = {k: PROJECT_ROOT / str(v) for k, v in cfg["paths"].items()}

    try:
        run(args, cfg, paths)
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as exc:  # RuntimeError: LLMError
        sys.exit(f"error: {exc}")


def run(args: argparse.Namespace, cfg: dict, paths: dict[str, Path]) -> None:
    drafts = paths["eval"] / "drafts.jsonl"
    sheet_dir = paths["eval"] / "annotation"
    rewrite = sheet_dir / "to_rewrite.csv"
    if args.cmd == "draft":
        from tqdm import tqdm

        papers = questions.sample_papers(args.n, cfg["seed"],
                                         args.papers or paths["processed"] / cfg["kb"]["store"]["sqlite_file"])
        with tqdm(total=len(papers), desc="drafting", unit="question") as bar:
            result = questions.draft_questions(papers, drafts, workers=args.workers, progress=bar.update)
        print(f"{result['drafted']} drafted, {result['skipped']} already there, {result['failed']} failed "
              f"(rerun to retry) -> {drafts}")
    elif args.cmd == "sheets":
        for path in questions.write_sheets(drafts, sheet_dir, args.round, tuple(args.annotators)):
            print(path)
    elif args.cmd == "build":
        summary = questions.build_eval_set(
            drafts, sheet_dir, paths["eval"] / "eval_set.jsonl",
            paths["results"] / "metrics" / f"iaa_{date.today().isoformat()}.json", rewrite,
            tuple(args.annotators), cfg["eval"].get("dev_ratio", 0.1), cfg["seed"])
        print(json.dumps(summary, indent=2))
    else:
        print(f"{questions.apply_rewrites(rewrite, drafts)} new questions added to {drafts}; "
              "run `sheets --round <next round>` for them")


if __name__ == "__main__":
    main()
