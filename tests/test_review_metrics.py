"""Hand-computed review classification and coverage examples; no APIs or retrieval needed."""

from __future__ import annotations

import json
import unittest

from eval.review_metrics import coverage_stats, review_accuracy


def review(paper_id: str, verdict: str) -> dict:
    return {"paper_id": paper_id, "verdict": verdict}


def coverage(total: int = 2, covered: int = 2, rounds: int = 1, reviewed: int = 4,
             passed: int = 2, reason: str = "coverage_met") -> dict:
    return {"total_aspects": total, "covered_aspects": covered, "rounds": rounds,
            "papers_reviewed": reviewed, "papers_passed": passed, "termination_reason": reason}


class ReviewAccuracyTest(unittest.TestCase):
    def test_mixed_labels_ignore_unknown_and_unreviewed_papers(self) -> None:
        records = [review("A", "pass"), review("B", "pass"), review("C", "reject"),
                   review("D", "pass"), review("E", "pass"), review("F", "reject"),
                   review("U", "pass"), review("V", "reject")]
        labels = {"A": 1, "B": 1, "C": 1, "D": 0, "E": 0, "F": 0, "unreviewed": 0}
        result = review_accuracy(records, labels)
        # TP=2, TN=1, FP=2, FN=1; unknown U/V and the unreviewed paper are not examples.
        self.assertEqual(result["confusion"], {"tp": 2, "tn": 1, "fp": 2, "fn": 1})
        self.assertEqual((result["n"], result["n_records"], result["n_unlabeled"], result["n_labels_unreviewed"]),
                         (6, 8, 2, 1))
        self.assertEqual(result["support"], {"positive": 3, "negative": 3})
        self.assertEqual(result["label_classes"], [0, 1])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f1"], 4 / 7)

    def test_positive_only_labels_expose_missing_negative_support(self) -> None:
        result = review_accuracy([review("A", "pass"), review("B", "reject"), review("unknown", "pass")],
                                 {"A": 1, "B": 1})
        self.assertEqual(result["confusion"], {"tp": 1, "tn": 0, "fp": 0, "fn": 1})
        self.assertEqual(result["support"], {"positive": 2, "negative": 0})
        self.assertEqual(result["label_classes"], [1])
        self.assertEqual(result["n_unlabeled"], 1)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["precision"], 1.0)  # conditional on labeled subset, not all passed papers
        self.assertEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 2 / 3)

    def test_zero_denominators_and_empty_inputs(self) -> None:
        for records, labels, counts in (
            ([], {}, (0, 0, 0)),
            ([review("unknown", "pass")], {}, (0, 1, 0)),
            ([], {"unreviewed": 1}, (0, 0, 1)),
        ):
            with self.subTest(records=records, labels=labels):
                result = review_accuracy(records, labels)
                self.assertEqual((result["n"], result["n_unlabeled"], result["n_labels_unreviewed"]), counts)
                self.assertEqual(result["label_classes"], [])
                for metric in ("accuracy", "precision", "recall", "f1"):
                    self.assertIsNone(result[metric])
                json.dumps(result, allow_nan=False)
        negative = review_accuracy([review("A", "reject")], {"A": 0})
        self.assertEqual(negative["accuracy"], 1.0)
        for metric in ("precision", "recall", "f1"):
            self.assertIsNone(negative[metric])
        missed_positive = review_accuracy([review("A", "reject")], {"A": 1})
        self.assertIsNone(missed_positive["precision"])
        self.assertEqual(missed_positive["recall"], 0.0)
        self.assertEqual(missed_positive["f1"], 0.0)
        false_positive = review_accuracy([review("A", "pass")], {"A": 0})
        self.assertIsNone(false_positive["recall"])
        self.assertEqual(false_positive["precision"], 0.0)
        self.assertEqual(false_positive["f1"], 0.0)

    def test_invalid_labels_are_rejected_even_without_a_review(self) -> None:
        for label in (-1, 2, 0.0, 1.0, True, False, "1", None, []):
            with self.subTest(label=label), self.assertRaises(ValueError):
                review_accuracy([], {"A": label})
        for paper_id in ("", " ", 1, None):
            with self.subTest(paper_id=paper_id), self.assertRaises(ValueError):
                review_accuracy([], {paper_id: 1})

    def test_invalid_records_and_duplicate_ids_are_rejected(self) -> None:
        invalid = [None, [], {}, {"paper_id": "A"}, {"verdict": "pass"},
                   review("", "pass"), review(" ", "pass"), review(None, "pass"),
                   review("A", None), review("A", "PASS"), review("A", True), review("A", ["pass"])]
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError):
                review_accuracy([record], {})  # unlabeled does not make bad data valid
        for second in ("pass", "reject"):
            for labels in ({}, {"A": 1}):
                with self.subTest(second=second, labels=labels), self.assertRaisesRegex(ValueError, "duplicate"):
                    review_accuracy([review("A", "pass"), review("A", second)], labels)
        for records, labels in (({}, {}), ([], []), (None, {}), ([], None)):
            with self.subTest(records=records, labels=labels), self.assertRaises(ValueError):
                review_accuracy(records, labels)


class CoverageStatsTest(unittest.TestCase):
    def test_partial_coverage_rounds_and_termination_reasons(self) -> None:
        reports = [coverage(), coverage(4, 1, 2, 8, 5, "min_papers"),
                   coverage(2, 0, 3, 6, 0, "max_rounds"), coverage(1, 1, 3, 2, 1, "coverage_met")]
        result = coverage_stats(reports)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["n_coverage_defined"], 4)
        self.assertEqual(result["n_zero_aspects"], 0)
        # Per-query coverage: 1, 1/4, 0, 1 => macro 9/16; pooled 4/9 aspects.
        self.assertEqual((result["covered_aspects"], result["total_aspects"]), (4, 9))
        self.assertAlmostEqual(result["coverage_ratio"], 4 / 9)
        self.assertEqual(result["mean_coverage"], 9 / 16)
        self.assertEqual(result["coverage_met"], 2)
        self.assertEqual(result["coverage_met_rate"], 0.5)
        self.assertEqual(result["unmet_rate"], 0.5)
        self.assertEqual(result["round_distribution"], {"1": 1, "2": 1, "3": 2})
        self.assertEqual(result["coverage_met_round_distribution"], {"1": 1, "3": 1})
        self.assertEqual(result["coverage_met_round_rates"], {"1": 0.25, "2": 0.0, "3": 0.25})
        self.assertEqual(result["termination_reasons"], {"coverage_met": 2, "min_papers": 1, "max_rounds": 1})
        self.assertEqual(result["mean_rounds"], 9 / 4)
        self.assertEqual(result["mean_papers_reviewed"], 5)
        self.assertEqual(result["mean_papers_passed"], 2)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_coverage_success_is_not_inferred_from_termination(self) -> None:
        # Even an inconsistent reason cannot turn partial coverage into success or hide full coverage.
        result = coverage_stats([coverage(2, 1, reason="coverage_met"), coverage(reason="loop_disabled")])
        self.assertEqual(result["coverage_met"], 1)
        self.assertEqual(result["coverage_met_rate"], 0.5)
        self.assertEqual(result["termination_reasons"], {"coverage_met": 1, "loop_disabled": 1})
        partial = coverage_stats([coverage(2, 1, reviewed=5, passed=5, reason="min_papers")])
        self.assertEqual(partial["coverage_met"], 0)
        self.assertEqual(partial["unmet_rate"], 1.0)

    def test_empty_and_zero_aspect_reports_do_not_claim_coverage(self) -> None:
        empty = coverage_stats([])
        self.assertEqual(empty["n"], 0)
        self.assertEqual(empty["coverage_met"], 0)
        for metric in ("coverage_ratio", "mean_coverage", "coverage_met_rate", "unmet_rate",
                       "mean_rounds", "mean_papers_reviewed", "mean_papers_passed"):
            self.assertIsNone(empty[metric])
        for key in ("round_distribution", "coverage_met_round_distribution", "coverage_met_round_rates",
                    "termination_reasons"):
            self.assertEqual(empty[key], {})
        zero = coverage(0, 0, 1, 0, 0, "rewrite_exhausted")
        result = coverage_stats([zero])
        self.assertEqual(result["n_zero_aspects"], 1)
        self.assertEqual(result["n_coverage_defined"], 0)
        self.assertEqual(result["coverage_met"], 0)
        for metric in ("coverage_ratio", "mean_coverage", "coverage_met_rate", "unmet_rate"):
            self.assertIsNone(result[metric])
        self.assertEqual(result["coverage_met_round_rates"], {"1": None})
        mixed = coverage_stats([zero, coverage(2, 1, reason="min_papers")])
        self.assertEqual(mixed["mean_coverage"], 0.5)
        self.assertEqual(mixed["unmet_rate"], 1.0)  # only the one query with defined coverage
        self.assertEqual(mixed["mean_papers_reviewed"], 2.0)  # both runs remain in runtime statistics

    def test_invalid_counts_and_missing_fields_are_rejected(self) -> None:
        for key in ("total_aspects", "covered_aspects", "rounds", "papers_reviewed", "papers_passed"):
            for value in (-1, 1.0, True, "1", None):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    coverage_stats([{**coverage(), key: value}])
            report = coverage()
            del report[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                coverage_stats([report])
        for report in (coverage(1, 2), coverage(reviewed=1, passed=2), coverage(rounds=0),
                       coverage(reason="unknown"), coverage(reason=None), coverage(reason=[]), {}, None):
            with self.subTest(report=report), self.assertRaises(ValueError):
                coverage_stats([report])
        for reports in ({}, None, ()):
            with self.subTest(reports=reports), self.assertRaises(ValueError):
                coverage_stats(reports)


if __name__ == "__main__":
    unittest.main()
