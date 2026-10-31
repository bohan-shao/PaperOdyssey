"""Offline tests for eval/challenge.py (fake LLM, hand-made runs). Run: python -m unittest tests.test_challenge"""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from eval import challenge
from eval.retrieval_eval import load_eval_set

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def paper(pid: str, year: int = 2020, text: str = "") -> dict:
    return {"paper_id": pid, "title": f"Title {pid}", "year": year,
            "abstract": text or f"Abstract of {pid} about topic {pid}."}


class FakeClient:
    """Answers by the system prompt; records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def chat_json(self, messages, schema):
        system, payload = messages[0]["content"], json.loads(messages[1]["content"])
        if system == challenge.COMPARISON:
            kind = "comparison"
            pids = (payload["paper_a"]["title"], payload["paper_b"]["title"])
            reply = ({"skip": True} if pids == ("Title P1", "Title P2")
                     else {"question": f"How do {pids[0]}  and {pids[1]} differ?", "aspects": ["a", "b"]})
        elif system == challenge.TWO_PART:
            kind, reply = "two_part", {"question": f"How does {payload['title']} work and how is it tested?",
                                       "parts": ["how", "tested"]}
        elif system == challenge.DATE:
            kind = "date"
            # The first date candidate forgets its bound and must be replaced.
            bound = "" if not any(c[0] == "date" for c in self.calls) else payload["time_bound"]
            reply = {"question": f"Which work on {payload['title']} appeared {bound}?".replace("  ", " ")}
        else:
            kind, reply = "out_of_kb", {"question": f"What drives change in {payload['field']}?"}
        self.calls.append((kind, payload))
        return reply


PAIRS = [(paper("P1"), paper("P2")), (paper("P3"), paper("P4")), (paper("P5"), paper("P6"))]
PAPERS = [paper(f"P{i}", 2015 + i) for i in range(1, 13)]


class DraftTest(unittest.TestCase):
    def test_draft_makes_each_kind_skips_bad_replies_and_never_reuses_a_paper(self) -> None:
        client = FakeClient()
        drafts = challenge.draft_challenge(PAPERS, client, per_kind=2, seed=42, source="llm:test",
                                           pairs=lambda papers: iter(PAIRS))
        self.assertEqual([d["challenge_id"] for d in drafts], [f"C{i:03d}" for i in range(1, 9)])
        self.assertEqual([d["kind"] for d in drafts], ["comparison"] * 2 + ["two_part"] * 2 + ["date"] * 2 + ["out_of_kb"] * 2)
        self.assertEqual([d["gold_paper_ids"] for d in drafts[:2]], [["P3", "P4"], ["P5", "P6"]])  # P1/P2 skipped
        self.assertEqual(drafts[0]["question"], "How do Title P3 and Title P4 differ?")  # whitespace normalised
        golds = [pid for d in drafts for pid in d["gold_paper_ids"]]
        self.assertEqual(len(golds), len(set(golds)))
        for d in drafts[4:6]:
            bound, year = d["time_bound"], d["papers"][0]["year"]
            self.assertIn(bound["phrase"], d["question"])
            self.assertEqual(bound.get("year_min", bound.get("year_max")), year)
        self.assertEqual([d["answerable"] for d in drafts], [True] * 6 + [False] * 2)
        self.assertEqual(drafts[6]["gold_paper_ids"], [])
        self.assertEqual(sum(c[0] == "date" for c in client.calls), 3)  # one replaced for a missing bound

    def test_too_few_drafts_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "comparison"):
            challenge.draft_challenge(PAPERS, FakeClient(), per_kind=3, pairs=lambda papers: iter(PAIRS[:2]))

    @unittest.skipUnless(HAS_SKLEARN, "needs scikit-learn")
    def test_related_pairs_join_papers_on_one_topic(self) -> None:
        texts = {"A1": "graph neural networks message passing node classification graphs",
                 "A2": "graph neural networks for node classification with attention on graphs",
                 "B1": "speech recognition acoustic models transducer audio",
                 "B2": "acoustic models for streaming speech recognition of audio"}
        papers = [paper(pid, text=t) for pid, t in texts.items()]
        pairs = {tuple(sorted((a["paper_id"], b["paper_id"]))) for a, b in challenge.related_pairs(papers, low=0.1, high=0.95)}
        self.assertEqual(pairs, {("A1", "A2"), ("B1", "B2")})


class SheetsAndBuildTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.dir = Path(self.directory.name)
        self.drafts = challenge.draft_challenge(PAPERS, FakeClient(), per_kind=1, pairs=lambda papers: iter(PAIRS[1:]))
        challenge.write_sheets(self.drafts, self.dir, ["A", "B"])

    def mark(self, annotator: str, fits: dict[str, str], edit=None) -> None:
        path = self.dir / f"challenge_{annotator}.csv"
        with path.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["fits"] = fits.get(row["challenge_id"], "")
            if edit:
                edit(row)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=challenge.SHEET_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def test_build_keeps_items_both_accept_and_reports_agreement(self) -> None:
        self.mark("A", {"C001": "y", "C002": "y", "C003": "y", "C004": "y"})
        self.mark("B", {"C001": "y", "C002": "n", "C003": "Y", "C004": "yes"})
        out = self.dir / "challenge_set.jsonl"
        report = challenge.build_challenge_set(self.drafts, self.dir, ["A", "B"], out)
        self.assertEqual((report["accepted"], report["rejected"]), (3, ["C002"]))
        self.assertEqual(report["percentage_agreement"], 0.75)
        # Fleiss, 2 raters, 4 items: P-bar 0.75; p_yes 7/8, p_no 1/8 -> P_e 50/64 -> kappa (0.75 - 0.78125) / 0.21875
        self.assertAlmostEqual(report["fleiss_kappa"], (0.75 - 50 / 64) / (1 - 50 / 64))
        questions = load_eval_set(out)  # the eval-set loader accepts it, including the no-gold item
        self.assertEqual([q["query_id"] for q in questions], ["C001", "C003", "C004"])
        unanswerable = questions[-1]
        self.assertEqual((unanswerable["gold_paper_ids"], unanswerable["answerable"], unanswerable["question_type"]),
                         ([], False, "out_of_kb"))
        self.assertEqual(questions[1]["time_bound"], self.drafts[2]["time_bound"])
        with self.assertRaisesRegex(ValueError, "exist"):
            challenge.write_sheets(self.drafts, self.dir, ["A"])

    def test_marks_and_questions_are_checked(self) -> None:
        self.mark("B", {"C001": "y", "C002": "y", "C003": "y", "C004": "y"})
        self.mark("A", {"C001": "y", "C002": "y", "C003": "y"})
        with self.assertRaisesRegex(ValueError, "fits must be y or n"):
            challenge.build_challenge_set(self.drafts, self.dir, ["A", "B"], self.dir / "out.jsonl")
        self.mark("A", {"C001": "y", "C002": "y", "C003": "y", "C004": "y"},
                  edit=lambda row: row.update(question=row["question"] + " Really?") if row["challenge_id"] == "C002" else None)
        with self.assertRaisesRegex(ValueError, "question of C002 was changed"):
            challenge.build_challenge_set(self.drafts, self.dir, ["A", "B"], self.dir / "out.jsonl")


class LoaderTest(unittest.TestCase):
    def test_only_an_unanswerable_question_may_have_no_gold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "set.jsonl"
            base = {"query_id": "Q1", "input_text": "q", "gold_paper_ids": [], "split": "eval"}
            path.write_text(json.dumps({**base, "answerable": False}) + "\n", encoding="utf-8")
            self.assertEqual(len(load_eval_set(path)), 1)
            for record in (base, {**base, "answerable": True}):
                path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "empty input_text or gold_paper_ids"):
                    load_eval_set(path)


class ScoreTest(unittest.TestCase):
    def test_hand_computed_scores_per_row_and_kind(self) -> None:
        questions = [
            {"query_id": "C001", "question_type": "comparison", "gold_paper_ids": ["A", "B"], "answerable": True},
            {"query_id": "C002", "question_type": "comparison", "gold_paper_ids": ["C", "D"], "answerable": True},
            {"query_id": "C003", "question_type": "date", "gold_paper_ids": ["E"], "answerable": True,
             "time_bound": {"phrase": "since 2021", "year_min": 2021}},
            {"query_id": "C004", "question_type": "out_of_kb", "gold_paper_ids": [], "answerable": False},
        ]
        searches = {  # query_id -> (passing papers, rounds, re_reviews, total_ms, tokens)
            "C001": (["A", "B", "X"], 2, 1, 4000, 300), "C002": (["C"], 1, 0, 2000, 100),
            "C003": (["E", "Y"], 1, 0, 3000, 200), "C004": ([], 3, 0, 9000, 900)}
        answers = {  # query_id -> (status, [(number, paper_id, year)], cited numbers)
            "C001": ("answered", [(1, "A", 2020), (2, "B", 2021)], [1, 2]),
            "C002": ("answered", [(1, "C", 2020)], [1]),
            "C003": ("answered", [(1, "E", 2022), (2, "Y", 2019)], [1, 2]),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            searches_dir, answers_dir = root / "ablation", root / "answers"
            searches_dir.mkdir()
            (searches_dir / "selected_eval.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions), encoding="utf-8")
            attempts = searches_dir / "rows" / "a" / "review" / "attempts"
            attempts.mkdir(parents=True)
            for n, (qid, (passing, rounds, re_reviews, ms, tokens)) in enumerate(searches.items(), 1):
                attempt = {"status": "complete", "question": {"query_id": qid}, "usage": {"prompt_tokens": tokens, "completion_tokens": 0},
                           "result": {"papers": [{"paper_id": p} for p in passing], "latency_by_stage": {"total_ms": ms},
                                      "coverage_report": {"rounds": rounds, "re_reviews": re_reviews}}}
                (attempts / f"{n:06d}.json").write_text(json.dumps(attempt), encoding="utf-8")
            (answers_dir / "rows" / "a").mkdir(parents=True)
            (answers_dir / "rows" / "a" / "answers.jsonl").write_text("".join(json.dumps({
                "query_id": qid, "status": status, "cited_source_numbers": cited,
                "sources": [{"number": n, "paper_id": p, "year": y} for n, p, y in sources]}) + "\n"
                for qid, (status, sources, cited) in answers.items()), encoding="utf-8")
            report = challenge.score_challenge(searches_dir, answers_dir)
        row = report["rows"]["a"]["by_kind"]
        comparison = row["comparison"]
        self.assertEqual((comparison["found"], comparison["all_found"]), (0.75, 0.5))   # 2/2 and 1/2 gold found
        self.assertEqual((comparison["cited"], comparison["all_cited"]), (0.75, 0.5))
        self.assertEqual((comparison["rounds"], comparison["re_reviews"], comparison["search_ms"]), (1.5, 0.5, 3000))
        date = row["date"]
        self.assertEqual((date["found"], date["cited"]), (1.0, 1.0))
        self.assertEqual((date["answers_citing_outside_bound"], date["cited_sources_outside_bound"]), (1.0, 0.5))
        out_of_kb = row["out_of_kb"]
        self.assertEqual((out_of_kb["n"], out_of_kb["answered_run"], out_of_kb["abstained"], out_of_kb["found"]), (1, 0, None, None))
        self.assertEqual(report["by_kind_n"], {"comparison": 2, "two_part": 0, "date": 1, "out_of_kb": 1})
        self.assertIn("comparison n=2", challenge.format_summary(report))


if __name__ == "__main__":
    unittest.main()
