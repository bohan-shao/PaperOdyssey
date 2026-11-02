"""Challenge set: comparison, two-part, date-bound and out-of-KB questions (Q3/Q5, D56).

Usage:
    python scripts/challenge_set.py draft --papers PAPERS_SQLITE [--config config.yaml]
    python scripts/challenge_set.py sheets
    python scripts/challenge_set.py build
    python scripts/challenge_set.py score --searches ABLATION_DIR [--answers ANSWER_RUN_DIR] [--judge-parts] [--tag TAG]

draft: eval.challenge.draft_challenge, eval.challenge.per_kind questions per kind from the full KB's
papers.sqlite (papers that are gold in the main eval set are left out) -> paths.eval/<challenge.dir>/
drafts.jsonl (LLM calls; never overwritten). sheets: one marking sheet per eval.challenge.annotators
(challenge_<annotator>.csv; guide eval/annotation/CHALLENGE.md). build: keeps the items every
annotator accepted -> paths.eval/<challenge.set_file> + metrics/challenge_iaa_<date>.json. Run it like
the main set, e.g. run_eval.py --eval-set data/eval/challenge_set.jsonl --stage review --ablation loop
re_review, then --stage answer on that ablation. score: per row and kind -> metrics/challenge_<tag>.json;
--judge-parts adds one answer-judge call (eval.answer.judge_model) per answered two_part / comparison
question: whether the answer addresses each part with a cited statement.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config, get_path  # noqa: E402
from eval.challenge import (  # noqa: E402
    build_challenge_set,
    candidate_papers,
    draft_challenge,
    format_summary,
    part_judge_client,
    read_jsonl,
    score_challenge,
    write_sheets,
)
from eval.retrieval_eval import load_eval_set  # noqa: E402


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Challenge set for the agentic search")
    parser.add_argument("command", choices=["draft", "sheets", "build", "score"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--papers", type=Path, default=None, help="draft: papers.sqlite of the full KB")
    parser.add_argument("--searches", type=Path, default=None, help="score: a review ablation on the challenge set")
    parser.add_argument("--answers", type=Path, default=None, help="score: the answer run of that ablation")
    parser.add_argument("--judge-parts", action="store_true",
                        help="score: LLM check that each answer covers every part (needs --answers)")
    parser.add_argument("--tag", default=None, help="score: report suffix (default today's date)")
    args = parser.parse_args()
    cfg = get_config(args.config)
    settings = cfg["eval"]["challenge"]
    folder = get_path("eval") / settings["dir"]
    drafts_path = folder / "drafts.jsonl"
    try:
        if args.command == "draft":
            if args.papers is None:
                parser.error("draft needs --papers PAPERS_SQLITE")
            if drafts_path.exists():
                raise ValueError(f"{drafts_path} exists; delete it only if no sheet was handed out")
            from llm.client import get_client

            main_set = get_path("eval") / cfg["eval"]["annotation"]["eval_set_file"]
            exclude = {pid for q in load_eval_set(main_set) for pid in q["gold_paper_ids"]}
            papers = candidate_papers(args.papers, exclude, cfg.get("seed", 42))
            drafts = draft_challenge(papers, get_client(), per_kind=settings["per_kind"], seed=cfg.get("seed", 42),
                                     source=f"llm:{cfg['llm']['model']}")
            drafts_path.parent.mkdir(parents=True, exist_ok=True)
            drafts_path.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in drafts), encoding="utf-8")
            print(f"{len(drafts)} questions -> {drafts_path}")
        elif args.command in ("sheets", "build"):
            drafts = read_jsonl(drafts_path)
            if args.command == "sheets":
                paths = write_sheets(drafts, folder, settings["annotators"])
                print(f"{len(drafts)} questions per sheet -> " + ", ".join(str(p) for p in paths))
            else:
                out = get_path("eval") / settings["set_file"]
                report = build_challenge_set(drafts, folder, settings["annotators"], out)
                metrics = get_path("results") / "metrics" / f"challenge_iaa_{date.today().isoformat()}.json"
                _write_json(metrics, report)
                agreement = ("none (one annotator)" if report["percentage_agreement"] is None else
                             f"{report['percentage_agreement']:.1%}, Fleiss kappa {report['fleiss_kappa']}")
                print(f"accepted {report['accepted']} of {report['drafted']} {report['by_kind']}; agreement "
                      f"{agreement} -> {out}, {metrics}")
        else:
            if args.searches is None:
                parser.error("score needs --searches ABLATION_DIR")
            client = part_judge_client(cfg) if args.judge_parts else None
            report = score_challenge(args.searches, args.answers, client)
            out = get_path("results") / "metrics" / f"challenge_{args.tag or date.today().isoformat()}.json"
            _write_json(out, report)
            print(format_summary(report) + f"\n\nreport -> {out}")
    except (OSError, ValueError, KeyError) as exc:
        print(f"challenge {args.command}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
