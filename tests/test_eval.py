"""Offline tests for the eval metrics, IAA and eval-set tooling. Run: python -m unittest tests.test_eval

Expected metric values are worked out by hand in the comments; the LLM is a fake client.
"""

from __future__ import annotations

import csv
import json
import math
import tempfile
import threading
import unittest
from pathlib import Path

from eval import retrieval_metrics as rm
from eval.annotation import iaa, questions
from kb import store


def hit(passage_id: str, rank: int) -> dict:
    return {"passage_id": passage_id, "paper_id": passage_id.split("_")[0], "rank": rank}


# paper ranking P1, P2, P3, P4 (P1_1 is a second passage of P1); given out of order on purpose
HITS = [hit("P3_0", 4), hit("P1_0", 1), hit("P1_1", 3), hit("P2_0", 2), hit("P4_0", 5)]


class RetrievalMetricsTest(unittest.TestCase):
    def test_paper_ranking_keeps_each_papers_best_passage(self) -> None:
        self.assertEqual(rm.ranking(HITS), ["P1", "P2", "P3", "P4"])
        self.assertEqual(rm.ranking(HITS, "passage"), ["P1_0", "P2_0", "P1_1", "P3_0", "P4_0"])

    def test_hand_computed_paper_level(self) -> None:
        gold = {"P3", "P9"}  # P3 at position 3, P9 not retrieved
        self.assertEqual(rm.hit_at_k(HITS, gold, 1), 0.0)
        self.assertEqual(rm.hit_at_k(HITS, gold, 3), 1.0)
        self.assertAlmostEqual(rm.precision_at_k(HITS, gold, 3), 1 / 3)
        self.assertAlmostEqual(rm.precision_at_k(HITS, gold, 10), 1 / 10)  # K is the denominator
        self.assertAlmostEqual(rm.recall_at_k(HITS, gold, 3), 1 / 2)
        self.assertAlmostEqual(rm.mrr_at_k(HITS, gold), 1 / 3)
        self.assertEqual(rm.mrr_at_k(HITS, gold, 2), 0.0)
        # DCG = 1/log2(4) = 0.5; IDCG = 1 + 1/log2(3) = 1.63093
        self.assertAlmostEqual(rm.ndcg_at_k(HITS, gold), 0.3065736, places=6)

    def test_hand_computed_passage_level_and_graded(self) -> None:
        self.assertAlmostEqual(rm.precision_at_k(HITS, {"P1_1"}, 3, level="passage"), 1 / 3)
        self.assertAlmostEqual(rm.mrr_at_k(HITS, {"P1_1"}, level="passage"), 1 / 3)
        # ranking P1_0 (grade 1), P2_0, P1_1 (grade 2): DCG = 1 + 2/log2(4) = 2; IDCG = 2 + 1/log2(3)
        graded = {"P1_1": 2, "P1_0": 1}
        self.assertAlmostEqual(rm.ndcg_at_k(HITS, graded, graded=True, level="passage"), 0.7601875, places=6)

    def test_batch_counts_a_query_without_hits_as_a_miss(self) -> None:
        gold = {"q1": {"gold_paper_ids": ["P3"], "gold_passage_ids": ["P3_0"]},
                "q2": {"gold_paper_ids": ["P5"], "gold_passage_ids": []}}
        out = rm.compute_retrieval_metrics({"q1": HITS, "q_unknown": HITS}, gold, [1, 3])
        self.assertEqual(out["n_queries"], 2)
        self.assertEqual(out["Hit@1"], 0.0)
        self.assertEqual(out["Hit@3"], 0.5)  # q1 hit at 3, q2 has no hits
        self.assertAlmostEqual(out["MRR@10"], (1 / 3 + 0) / 2)
        self.assertEqual(out["per_granularity"]["passage"]["n_queries"], 1)  # only q1 has passage gold
        self.assertAlmostEqual(out["per_granularity"]["passage"]["MRR@10"], 1 / 4)
        self.assertEqual(out["per_query"], {"q1": {"first_gold_rank": 3}, "q2": {"first_gold_rank": None}})


def item(qid: str, *labels: bool) -> dict:
    return {"query_id": qid, "votes": [["P"] if v else [] for v in labels], "paper_ids": ["P"]}


class IAATest(unittest.TestCase):
    def test_fleiss_matches_the_wikipedia_example(self) -> None:
        # en.wikipedia.org/wiki/Fleiss'_kappa worked example: P-bar 0.378, P_e 0.213, kappa 0.210
        counts = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
                  [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
        p_bar, kappa = iaa._fleiss(counts)
        self.assertAlmostEqual(p_bar, 0.378022, places=6)
        self.assertAlmostEqual(kappa, 0.209931, places=6)

    def test_hand_computed_three_annotators(self) -> None:
        items = [item("a", True, True, True), item("b", True, True, False), item("c", False, False, False),
                 item("d", True, True, True)]
        self.assertEqual(iaa.percentage_agreement(items), 0.75)
        # P-bar = (1 + 1/3 + 1 + 1) / 4 = 5/6; p_yes = 8/12, P_e = 4/9 + 1/9 = 5/9; kappa = (5/6 - 5/9) / (4/9) = 0.625
        self.assertAlmostEqual(iaa.fleiss_kappa(items), 0.625)
        with tempfile.TemporaryDirectory() as tmp:
            report = iaa.compute_iaa_report(items, str(Path(tmp) / "iaa.json"), extra={"eval_set": "x"})
            self.assertEqual(json.loads((Path(tmp) / "iaa.json").read_text())["eval_set"], "x")
        self.assertAlmostEqual(report["pairwise_agreement"], 5 / 6)
        self.assertEqual((report["n_items"], report["n_units"], report["n_annotators"]), (4, 4, 3))

    def test_without_paper_ids_units_are_the_voted_papers(self) -> None:
        items = [{"query_id": "a", "votes": [["P1", "P2"], ["P1"], ["P1"]]}]  # P1 all agree, P2 one of three
        self.assertEqual(iaa.percentage_agreement(items), 0.5)

    def test_all_labels_the_same_has_no_kappa(self) -> None:
        items = [item("a", True, True, True), item("b", True, True, True)]
        self.assertEqual(iaa.percentage_agreement(items), 1.0)
        self.assertTrue(math.isnan(iaa.fleiss_kappa(items)))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(iaa.compute_iaa_report(items, str(Path(tmp) / "iaa.json"))["fleiss_kappa"])

    def test_every_item_needs_the_same_number_of_votes(self) -> None:
        with self.assertRaises(ValueError):
            iaa.percentage_agreement([item("a", True, True, True), item("b", True, True)])


class FakeClient:
    model = "fake-llm"

    def __init__(self, fail_titles: set[str] = frozenset()) -> None:
        self.fail_titles = set(fail_titles)
        self.calls = 0
        self._lock = threading.Lock()

    def chat_json(self, messages: list[dict], schema_desc: str) -> dict:
        with self._lock:
            self.calls += 1
        title = json.loads(messages[1]["content"])["title"]
        if title in self.fail_titles:
            raise RuntimeError("provider failure")
        return {"question": f"  Which work studies\n {title.lower()}? ", "type": "Survey"}


def papers_db(path: Path) -> None:
    rows = [{"paper_id": f"{i:04d}", "arxiv_id": f"{i:04d}v1", "title": f"Topic {i}", "year": 2020,
             "abstract": "x" * 250, "num_chunks": 3} for i in range(30)]
    rows[0]["num_chunks"] = 0  # not built
    rows[1]["abstract"] = "too short"
    store.upsert_papers(rows, path)


def mark(path: Path, fits: dict[str, str]) -> None:
    """Fill the fits column of a marking sheet the way an annotator would."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["fits"] = fits.get(row["query_id"], row["fits"])
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=questions.SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class QuestionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.db = self.dir / "papers.sqlite"
        papers_db(self.db)
        self.drafts = self.dir / "eval" / "drafts.jsonl"
        self.sheets = self.dir / "eval" / "annotation"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def build(self) -> dict:
        return questions.build_eval_set(self.drafts, self.sheets, self.dir / "eval" / "eval_set.jsonl",
                                        self.dir / "iaa.json", self.sheets / "to_rewrite.csv")

    def test_sample_is_seeded_built_papers_and_a_larger_n_extends_it(self) -> None:
        five, ten = questions.sample_papers(5, 42, self.db), questions.sample_papers(10, 42, self.db)
        self.assertEqual([p["paper_id"] for p in five], [p["paper_id"] for p in ten[:5]])
        every = {p["paper_id"] for p in questions.sample_papers(28, 42, self.db)}
        self.assertNotIn("0000", every)
        self.assertNotIn("0001", every)
        with self.assertRaises(ValueError):
            questions.sample_papers(29, 42, self.db)

    def test_draft_retries_failures_on_a_rerun_and_refuses_another_sample(self) -> None:
        papers = questions.sample_papers(6, 42, self.db)
        first = questions.draft_questions(papers, self.drafts, FakeClient({papers[2]["title"]}), workers=3)
        self.assertEqual(first, {"drafted": 5, "skipped": 0, "failed": 1})
        client = FakeClient()
        self.assertEqual(questions.draft_questions(papers, self.drafts, client, workers=3),
                         {"drafted": 1, "skipped": 5, "failed": 0})
        self.assertEqual(client.calls, 1)
        drafts = questions.latest_drafts(self.drafts)
        self.assertEqual(list(drafts), [f"Q{i:04d}" for i in range(1, 7)])
        q3 = drafts["Q0003"]
        self.assertEqual(q3["paper_id"], papers[2]["paper_id"])
        self.assertEqual(q3["question"], f"Which work studies {papers[2]['title'].lower()}?")
        self.assertEqual((q3["question_type"], q3["source"], q3["round"]), ("survey", "llm:fake-llm", 1))
        with self.assertRaises(ValueError):
            questions.draft_questions(questions.sample_papers(6, 7, self.db), self.drafts, FakeClient())

    def test_a_failing_first_call_stops_the_draft(self) -> None:
        papers = questions.sample_papers(4, 42, self.db)
        client = FakeClient({papers[0]["title"]})  # e.g. no API key
        with self.assertRaises(RuntimeError):
            questions.draft_questions(papers, self.drafts, client, workers=2)
        self.assertEqual(client.calls, 1)
        self.assertFalse(self.drafts.exists())

    def test_marking_rounds_build_the_eval_set(self) -> None:
        questions.draft_questions(questions.sample_papers(3, 42, self.db), self.drafts, FakeClient(), workers=2)
        a, b, c = questions.write_sheets(self.drafts, self.sheets)  # the pilot
        self.assertEqual(a.name, "round1_part1_A.csv")
        with self.assertRaises(ValueError):  # every question already has a sheet
            questions.write_sheets(self.drafts, self.sheets)
        questions.draft_questions(questions.sample_papers(4, 42, self.db), self.drafts, FakeClient(), workers=2)
        a2, b2, c2 = questions.write_sheets(self.drafts, self.sheets)  # the rest: only the new question
        self.assertEqual(a2.name, "round1_part2_A.csv")
        self.assertEqual(list(questions.read_sheet(a2)), [("Q0004", 1)])
        # Q0001 accepted 3/3, Q0002 accepted 2/3, Q0003 rejected 1/3, Q0004 not marked by C yet
        mark(a, {"Q0001": "y", "Q0002": "Yes", "Q0003": "n"})
        mark(b, {"Q0001": "y", "Q0002": "n", "Q0003": "n"})
        mark(c, {"Q0001": "Y", "Q0002": "y", "Q0003": "y"})
        mark(a2, {"Q0004": "y"})
        mark(b2, {"Q0004": "y"})
        summary = self.build()
        self.assertEqual({k: summary[k] for k in ("questions", "pending", "accepted", "rejected")},
                         {"questions": 4, "pending": 1, "accepted": 2, "rejected": 1})
        # 3 fully marked units: (3 yes), (2 yes), (1 yes) -> 1 of 3 unanimous
        self.assertAlmostEqual(summary["iaa"]["percentage_agreement"], 1 / 3)
        records = store.read_records(self.dir / "eval" / "eval_set.jsonl")
        self.assertEqual([r["query_id"] for r in records], ["Q0001", "Q0002"])
        q2 = records[1]
        self.assertEqual(q2["gold_paper_ids"], q2["majority_paper_ids"])
        self.assertEqual([x["papers"] for x in q2["annotations"]], [q2["gold_paper_ids"], [], q2["gold_paper_ids"]])
        self.assertIn(q2["split"], ("dev", "eval"))

        # the rejected question is rewritten by a person and marked again by everyone
        rewrite = self.sheets / "to_rewrite.csv"
        with rewrite.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual([r["query_id"] for r in rows], ["Q0003"])
        rows[0]["new_question"] = "A better  question?"
        with rewrite.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=questions.REWRITE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(FileExistsError):  # a filled-in rewrite list is never overwritten
            self.build()
        self.assertEqual(questions.apply_rewrites(rewrite, self.drafts), 1)
        self.assertEqual(questions.apply_rewrites(rewrite, self.drafts), 0)  # already applied
        sheets2 = questions.write_sheets(self.drafts, self.sheets, round_=2)
        self.assertEqual(sheets2[0].name, "round2_part1_A.csv")
        for path in sheets2:
            mark(path, {"Q0003": "y"})
        mark(c2, {"Q0004": "n"})
        summary = self.build()
        self.assertEqual((summary["accepted"], summary["pending"], summary["rejected"]), (4, 0, 0))
        self.assertEqual(summary["iaa"]["n_items"], 5)  # round 1 of Q0001-Q0004 and round 2 of Q0003
        q3 = store.read_records(self.dir / "eval" / "eval_set.jsonl")[2]
        self.assertEqual((q3["input_text"], q3["round"], q3["question_source"]), ("A better question?", 2, "human"))

    def test_a_mark_other_than_y_or_n_is_an_error(self) -> None:
        questions.draft_questions(questions.sample_papers(2, 42, self.db), self.drafts, FakeClient(), workers=1)
        a, _, _ = questions.write_sheets(self.drafts, self.sheets)
        mark(a, {"Q0001": "maybe"})
        with self.assertRaises(ValueError):
            questions.read_sheet(a)

    def test_split_is_stable_per_query(self) -> None:
        splits = [questions._split(f"Q{i:04d}", 42, 0.1) for i in range(1, 1001)]
        self.assertEqual(splits, [questions._split(f"Q{i:04d}", 42, 0.1) for i in range(1, 1001)])
        self.assertTrue(70 <= splits.count("dev") <= 130)


if __name__ == "__main__":
    unittest.main()
