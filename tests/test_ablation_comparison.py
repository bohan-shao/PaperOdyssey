"""Hand-calculated paired comparisons, incomplete runs, and factorial contrasts."""

from __future__ import annotations

import copy
import json
import math
import unittest

from eval.ablation_comparison import compare_rows, interaction


def question(qid: str, split: str = "dev") -> dict:
    return {"query_id": qid, "split": split, "input_text": f"Find source {qid}.",
            "gold_paper_ids": [f"paper-{qid}"]}


def observation(rank: int | None, *, stage: str = "retrieval", wall: float = 100,
                cold: bool = False, recall: float | None = None,
                covered: bool | None = None, rounds: int | None = None) -> dict:
    return {"first_gold_rank": rank,
            "source_success": rank is not None and (stage == "review" or rank <= 10),
            "source_recall": recall, "wall_ms": wall, "cold_start": cold,
            "coverage_met": covered, "rounds": rounds}


class PairedComparisonTest(unittest.TestCase):
    def test_only_common_completed_queries_count_and_splits_stay_separate(self) -> None:
        questions = [question(f"Q{i}") for i in range(1, 5)] + [question("Q5", "eval")]
        rows = {
            "baseline": {"Q1": observation(None, wall=100, cold=True), "Q2": observation(2, wall=200),
                         "Q3": observation(12, wall=300), "Q5": observation(1)},
            "A": {"Q1": observation(1, wall=150), "Q2": observation(None, wall=250),
                  "Q4": observation(1), "Q5": observation(3)},
        }
        original = copy.deepcopy((questions, rows))
        report = compare_rows(questions, rows, [("baseline", "A")], stage="retrieval")[0]
        dev = report["splits"]["dev"]
        self.assertEqual((dev["n_selected"], dev["n_paired"]), (4, 2))
        self.assertEqual(dev["unpaired_query_ids"], ["Q3", "Q4"])
        self.assertEqual(dev["paired_query_ids"], ["Q1", "Q2"])
        self.assertEqual(dev["metrics"]["Hit@1"], {"n": 2, "before": 0.0, "after": 0.5, "delta": 0.5})
        self.assertEqual(dev["metrics"]["MRR@10"], {"n": 2, "before": 0.25, "after": 0.5, "delta": 0.25})
        self.assertEqual(dev["metrics"]["wall_ms"], {"n": 2, "before": 150.0, "after": 200.0, "delta": 50.0})
        self.assertEqual((dev["fixed"], dev["broken"], dev["moved_up"], dev["moved_down"]), (1, 1, 1, 1))
        self.assertEqual(dev["fixed_query_ids"], ["Q1"])
        self.assertEqual(dev["broken_query_ids"], ["Q2"])
        self.assertEqual(dev["fixed_examples"][0], {
            "query_id": "Q1", "input_text": "Find source Q1.", "gold_paper_ids": ["paper-Q1"],
            "rank_before": None, "rank_after": 1,
        })
        self.assertEqual(dev["cold_start_counts"], {"baseline": 1, "A": 0})
        self.assertIn("not a controlled latency benchmark", dev["timing_note"])
        held_out = report["splits"]["eval"]
        self.assertEqual(held_out["n_paired"], 1)
        self.assertEqual(held_out["metrics"]["Hit@1"]["delta"], -1)
        self.assertAlmostEqual(held_out["metrics"]["MRR@10"]["delta"], -2 / 3)
        self.assertEqual((questions, rows), original)
        json.dumps(report, allow_nan=False)

    def test_top10_boundary_and_mrr_cutoff(self) -> None:
        questions = [question("Q1"), question("Q2"), question("Q3")]
        rows = {"before": {"Q1": observation(11), "Q2": observation(10), "Q3": observation(None)},
                "after": {"Q1": observation(10), "Q2": observation(11), "Q3": observation(None)}}
        result = compare_rows(questions, rows, [("before", "after")], stage="retrieval")[0]["splits"]["dev"]
        self.assertEqual((result["fixed"], result["broken"]), (1, 1))
        self.assertAlmostEqual(result["metrics"]["MRR@10"]["before"], 0.1 / 3)
        self.assertEqual(result["metrics"]["Hit@10"]["delta"], 0.0)

    def test_review_acceptance_is_separate_from_top10_ranking(self) -> None:
        questions = [question("Q1"), question("Q2")]
        questions[0]["gold_paper_ids"].append("second-source")
        rows = {
            "baseline": {
                "Q1": observation(None, stage="review", recall=0, covered=False, rounds=3),
                "Q2": observation(12, stage="review", recall=1, covered=None, rounds=2),
            },
            "A": {
                "Q1": observation(11, stage="review", recall=0.5, covered=True, rounds=1),
                "Q2": observation(2, stage="review", recall=1, covered=True, rounds=1),
            },
        }
        result = compare_rows(questions, rows, [("baseline", "A")], stage="review")[0]["splits"]["dev"]
        self.assertEqual(result["fixed_query_ids"], ["Q1"])
        self.assertEqual(result["top10_fixed_query_ids"], ["Q2"])
        self.assertEqual(result["metrics"]["source_success_rate"],
                         {"n": 2, "before": 0.5, "after": 1.0, "delta": 0.5})
        self.assertEqual(result["metrics"]["source_recall"],
                         {"n": 2, "before": 0.5, "after": 0.75, "delta": 0.25})
        self.assertEqual(result["metrics"]["coverage_met_rate"],
                         {"n": 1, "before": 0.0, "after": 1.0, "delta": 1.0})
        self.assertEqual(result["metrics"]["mean_rounds"]["delta"], -1.5)
        self.assertFalse(any(name.startswith(("P@", "NDCG")) for name in result["metrics"]))

    def test_missing_optional_values_never_use_different_denominators(self) -> None:
        questions = [question("Q")]
        rows = {"before": {"Q": observation(1, stage="review")},
                "after": {"Q": observation(1, stage="review", recall=1, covered=True, rounds=1)}}
        result = compare_rows(questions, rows, [("before", "after")], stage="review")[0]["splits"]["dev"]
        for name in ("source_recall", "coverage_met_rate", "mean_rounds"):
            self.assertEqual(result["metrics"][name], {"n": 0, "before": None, "after": None, "delta": None})

    def test_empty_or_unpaired_splits_have_null_metrics_not_zero_scores(self) -> None:
        result = compare_rows([question("Q")], {"before": {"Q": observation(1)}, "after": {}},
                              [("before", "after")], stage="retrieval")[0]
        self.assertEqual(result["splits"]["dev"]["unpaired_query_ids"], ["Q"])
        for split in ("dev", "eval"):
            row = result["splits"][split]
            self.assertEqual((row["n_paired"], row["fixed"], row["broken"]), (0, 0, 0))
            self.assertIsNone(row["metrics"]["Hit@10"]["delta"])
        empty = compare_rows([], {"before": {}, "after": {}}, [("before", "after")], stage="retrieval")
        self.assertEqual(empty[0]["splits"]["dev"]["n_selected"], 0)

    def test_all_changed_ids_remain_when_examples_are_capped(self) -> None:
        questions = [question(f"Q{i}") for i in range(7)]
        rows = {"baseline": {q["query_id"]: observation(None) for q in questions},
                "A": {q["query_id"]: observation(1) for q in questions}}
        result = compare_rows(questions, rows, [("baseline", "A")], stage="retrieval")[0]["splits"]["dev"]
        self.assertEqual(result["fixed"], 7)
        self.assertEqual(result["fixed_query_ids"], [q["query_id"] for q in questions])
        self.assertEqual(len(result["fixed_examples"]), 5)

    def test_invalid_observations_are_rejected(self) -> None:
        bad_values = {
            "first_gold_rank": [0, -1, 1.5, True, math.nan, "1"],
            "wall_ms": [-1, math.inf, math.nan, True, None],
            "source_success": [1, None, "true"],
            "cold_start": [1, None],
            "source_recall": [-0.1, 1.1, math.nan, True],
            "coverage_met": [1, "yes"],
            "rounds": [0, -1, True, 1.5],
        }
        for field, values in bad_values.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    compare_rows([question("Q")], {"A": {"Q": {**observation(1), field: value}},
                                                    "B": {"Q": observation(1)}}, [("A", "B")], stage="retrieval")
        with self.assertRaisesRegex(ValueError, "disagree"):
            compare_rows([question("Q")], {"A": {"Q": {**observation(1), "source_success": False}}, "B": {}},
                         [("A", "B")], stage="retrieval")

    def test_bad_identity_and_edges_are_rejected(self) -> None:
        qs, rows = [question("Q")], {"A": {"Q": observation(1)}, "B": {}}
        cases = [([*qs, *qs], rows, [("A", "B")]),
                 ([{**qs[0], "split": "test"}], rows, [("A", "B")]),
                 (qs, {"A": {"unknown": observation(1)}, "B": {}}, [("A", "B")]),
                 (qs, rows, [("A", "missing")]), (qs, rows, [("A", "A")]),
                 (qs, rows, [("A", "B"), ("A", "B")]), (qs, rows, [("A",)])]
        for questions, values, edges in cases:
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                compare_rows(questions, values, edges, stage="retrieval")
        with self.assertRaises(ValueError):
            compare_rows(qs, rows, [("A", "B")], stage="all")


class FactorialInteractionTest(unittest.TestCase):
    def test_contrast_uses_four_way_intersection_not_independent_row_means(self) -> None:
        qs = [question("Q1"), question("Q2"), question("Q3", "eval")]
        rows = {
            "base": {"Q1": observation(None), "Q2": observation(1)},
            "a": {"Q1": observation(2), "Q2": observation(1)},
            "b": {"Q1": observation(None), "Q2": observation(1)},
            "ab": {"Q1": observation(1)},
        }
        result = interaction(qs, rows, ("base", "a", "b", "ab"), stage="retrieval")
        dev = result["splits"]["dev"]
        self.assertEqual((dev["n_selected"], dev["n_paired"]), (2, 1))
        self.assertEqual(dev["unpaired_query_ids"], ["Q2"])
        self.assertEqual(dev["metrics"]["Hit@10"]["interaction"], 0.0)
        self.assertEqual(dev["metrics"]["MRR@10"],
                         {"n": 1, "baseline": 0.0, "A": 0.5, "B": 0.0, "AB": 1.0, "interaction": 0.5})
        self.assertIsNone(result["splits"]["eval"]["metrics"]["Hit@10"]["interaction"])
        json.dumps(result, allow_nan=False)

    def test_review_contrast_includes_known_source_recall_not_precision(self) -> None:
        rows = {name: {"Q": observation(rank, stage="review", recall=recall, covered=covered)}
                for name, rank, recall, covered in (("base", None, 0, False), ("a", 1, 0.5, True),
                                                    ("b", 2, 0.5, None), ("ab", 1, 1, True))}
        metrics = interaction([question("Q")], rows, ("base", "a", "b", "ab"),
                              stage="review")["splits"]["dev"]["metrics"]
        self.assertEqual(metrics["source_recall"]["interaction"], 0.0)
        self.assertEqual(metrics["source_success_rate"]["interaction"], -1.0)
        self.assertEqual(metrics["coverage_met_rate"]["n"], 0)
        self.assertIsNone(metrics["coverage_met_rate"]["interaction"])
        self.assertNotIn("precision", metrics)

    def test_four_distinct_existing_rows_are_required(self) -> None:
        rows = {name: {} for name in ("base", "a", "b", "ab")}
        for names in (("base", "a", "b"), ("base", "a", "b", "missing"), ("base", "a", "b", "b")):
            with self.subTest(names=names), self.assertRaises(ValueError):
                interaction([], rows, names, stage="retrieval")


if __name__ == "__main__":
    unittest.main()
