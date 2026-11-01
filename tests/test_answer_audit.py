"""Offline tests for eval/answer_audit.py with hand-computed scores. Run: python -m unittest tests.test_answer_audit"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from eval import answer_audit as audit


def answer_row(qid: str, faithfulness: float | None, status: str = "answered") -> dict:
    return {"query_id": qid, "split": "eval", "input_text": f"Question {qid} <about> X?", "gold_paper_ids": ["1"],
            "status": status, "answer": f"Answer for {qid} & more [1].", "cited_source_numbers": [1],
            "sources": [{"number": 1}, {"number": 2}], "faithfulness": faithfulness, "answer_relevance": 0.7,
            "context_relevance": 0.5, "context_precision": 1.0, "gold_in_sources": 1.0, "gold_cited": 1.0}


def write_run(root: Path, rows: list[dict]) -> Path:
    run = root / "rows" / "a"
    (run / "answers").mkdir(parents=True)
    (run / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for r in rows:
        sources = [{"number": n, "paper_id": "1", "passage_id": f"P1_{n}", "title": f"Paper {n}", "year": 2020,
                    "section": "Method", "url": "https://arxiv.org/abs/1", "text": f"Excerpt {n} of {r['query_id']}"}
                   for n in (1, 2)]
        checkpoint = {"status": "complete", "answer": {"answer": r["answer"], "status": r["status"],
                                                       "sources": sources, "cited_source_numbers": [1]}}
        (run / "answers" / f"{r['query_id']}.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    return run


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fill(out: Path, annotator: str, marks: dict[str, tuple[str, str, str]]) -> None:
    """Write marks keyed by query_id into an annotator's sheet."""
    path = out / f"answer_audit_{annotator}.csv"
    rows = read(path)
    for row in rows:
        row["correct"], row["complete"], row["supported"] = marks[row["query_id"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


MIXED = ([answer_row(f"P{i}", 1.0) for i in range(6)] + [answer_row(f"M{i}", 0.9) for i in range(3)]
         + [answer_row("L0", 0.5), answer_row("L1", None), answer_row("N0", None, "insufficient_evidence")])


class SamplingAndSheetsTest(unittest.TestCase):
    def test_stratified_sample_is_deterministic_and_keeps_its_size(self) -> None:
        sizes = {"perfect": 2, "mostly": 2, "low": 2, "no_answer": 2}
        picked = audit.sample(MIXED, sizes, seed=42)
        counts = {s: sum(g == s for g, _ in picked) for s in audit.STRATA}
        self.assertEqual(counts, {"perfect": 3, "mostly": 2, "low": 2, "no_answer": 1})  # no_answer short by 1
        self.assertEqual(picked, audit.sample(list(reversed(MIXED)), sizes, seed=42))  # file order does not matter
        self.assertEqual(audit.stratum(answer_row("x", 0.8)), "mostly")
        self.assertEqual(audit.stratum(answer_row("x", 0.79)), "low")

    def test_sheets_packets_and_manifest_hide_the_judge_and_never_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, out = write_run(root, MIXED), root / "audit"
            manifest = audit.write_sheets(run, out, ["A", "B", "C", "D"],
                                          {"perfect": 2, "mostly": 2, "low": 2, "no_answer": 2}, seed=42)
            self.assertEqual(manifest["population"], {"perfect": 6, "mostly": 3, "low": 2, "no_answer": 1})
            self.assertEqual(len(manifest["items"]), 8)
            self.assertTrue(all(len(set(i["annotators"])) == 2 for i in manifest["items"]))
            self.assertEqual(sum(manifest["workload"].values()), 16)
            for annotator in "ABCD":
                rows = read(out / f"answer_audit_{annotator}.csv")
                self.assertEqual(len(rows), manifest["workload"][annotator])
                self.assertEqual(list(rows[0]) if rows else audit.COLUMNS, audit.COLUMNS)  # no judge columns
                page = (out / f"answer_audit_{annotator}.html").read_text(encoding="utf-8")
                self.assertNotIn("faithfulness", page)
                for row in rows:
                    self.assertIn(f"Answer for {row['query_id']} &amp; more [1].", page)
                    self.assertIn(f"Excerpt 2 of {row['query_id']}", page)
                    self.assertIn("Question " + row["query_id"] + " &lt;about&gt; X?", page)
            self.assertIn("class='cited'", page)
            with self.assertRaisesRegex(ValueError, "exists"):
                audit.write_sheets(run, out, ["A", "B"], {"perfect": 1})


class ScoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        rows = [answer_row(f"P{i}", 1.0) for i in range(4)] + [answer_row("L0", 0.5), answer_row("L1", 0.5)]
        self.out = root / "audit"
        self.manifest = audit.write_sheets(write_run(root, rows), self.out, ["A", "B"], {"perfect": 2, "low": 2})
        self.perfect = [i["query_id"] for i in self.manifest["items"] if i["stratum"] == "perfect"]

    def marks(self) -> tuple[dict, dict]:
        p1, p2 = self.perfect
        a = {p1: ("y", "y", "y"), p2: ("y", "y", "y"), "L0": ("n", "n", "p"), "L1": ("p", "p", "n")}
        b = {p1: ("y", "y", "y"), p2: ("p", "y", "y"), "L0": ("n", "n", "p"), "L1": ("n", "p", "n")}
        return a, b

    def test_hand_computed_scores_agreement_and_judge_comparison(self) -> None:
        a, b = self.marks()
        fill(self.out, "A", a)
        fill(self.out, "B", b)
        report = audit.score_sheets(self.out)
        correct = report["criteria"]["correct"]
        # per answer: perfect 1, 0.75; low 0, 0.25 -> strata 0.875 / 0.125; population 4 perfect, 2 low
        self.assertEqual(correct["by_stratum"], {"perfect": 0.875, "low": 0.125})
        self.assertAlmostEqual(correct["weighted_mean"], (4 * 0.875 + 2 * 0.125) / 6)
        self.assertAlmostEqual(correct["sample_mean"], 0.5)
        self.assertEqual(correct["exact_agreement"], 0.5)
        # Fleiss: P-bar 0.5; p_y = p_n = 3/8, p_p = 2/8 -> P_e = 0.34375 -> kappa 0.15625 / 0.65625
        self.assertAlmostEqual(correct["fleiss_kappa"], 0.15625 / 0.65625)
        self.assertEqual(report["criteria"]["complete"]["exact_agreement"], 1.0)
        # judge faithfulness 1, 1, 0.5, 0.5 vs human supported 1, 1, 0.5, 0 -> rho = 4 / sqrt(4 * 4.5)
        self.assertAlmostEqual(report["judge_vs_human"]["spearman_faithfulness_vs_supported"], 4 / 18 ** 0.5)
        self.assertEqual(report["judge_vs_human"]["judge_faithfulness_by_stratum"], {"perfect": 1.0, "low": 0.5})
        self.assertIn("correct", audit.format_summary(report))
        self.assertIn("stratified sample", audit.format_summary(report))
        self.assertIn("stratified sample", report["judge_vs_human"]["scope"])

    def test_marks_are_checked(self) -> None:
        a, b = self.marks()
        fill(self.out, "B", b)
        bad_cases = [({**a, "L0": ("maybe", "n", "p")}, "must be y, p or n"),
                     ({**a, "L0": ("", "n", "p")}, "must be y, p or n")]
        for marks, message in bad_cases:
            fill(self.out, "A", marks)
            with self.assertRaisesRegex(ValueError, message):
                audit.score_sheets(self.out)
        path = self.out / "answer_audit_A.csv"
        fill(self.out, "A", a)
        rows = read(path)
        for changed, message in ((rows[:-1], "not marked"), ([{**rows[0], "query_id": "X"}, *rows[1:]], "query_id changed"),
                                 ([*rows, rows[0]], "unexpected or repeated")):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=audit.COLUMNS)
                writer.writeheader()
                writer.writerows(changed)
            with self.assertRaisesRegex(ValueError, message):
                audit.score_sheets(self.out)


class SpearmanTest(unittest.TestCase):
    def test_ties_and_constant_input(self) -> None:
        self.assertAlmostEqual(audit.spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertIsNone(audit.spearman([1, 1, 1], [1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
