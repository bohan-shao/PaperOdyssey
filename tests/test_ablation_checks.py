"""Review-resume checkpoint corruption checks; no model or provider calls."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.ablation_checks import validate_review_attempts


def question(qid: str = "Q1") -> dict:
    return {"query_id": qid, "input_text": "Find a source.", "gold_paper_ids": ["P1"], "split": "dev"}


def attempt(qid: str = "Q1", status: str = "complete") -> dict:
    q = question(qid)
    row = {"question": q, "status": status, "cold_start": True,
           "usage": {"calls": 2, "prompt_tokens": 100, "completion_tokens": 50}, "wall_ms": 12.5}
    record = {"paper_id": "P1", "verdict": "pass", "score": 4, "reasons": "relevant",
              "covered": ["aspect"], "missing": [], "round": 1, "evidence": {}}
    if status == "complete":
        row["result"] = {
            "query_id": qid, "input_text": q["input_text"], "reviews": [record], "papers": [record],
            "retrieval_ranks": {"P1": 1},
            "coverage_report": {"total_aspects": 1, "covered_aspects": 1, "rounds": 1,
                                "papers_reviewed": 1, "papers_passed": 1,
                                "termination_reason": "coverage_met", "cache_hit": False},
        }
    elif status == "failed":
        row["error"] = {"query_id": qid, "type": "OrchestrationError"}
    return row


class ReviewCheckpointValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "attempts").mkdir()

    def write(self, row: object, number: int = 1) -> Path:
        path = self.root / "attempts" / f"{number:06d}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        return path

    def test_valid_retry_history_and_interrupted_attempts(self) -> None:
        rows = [attempt(status="running"), attempt(status="failed"), attempt(), attempt("Q2", "running")]
        del rows[0]["usage"]
        del rows[0]["wall_ms"]
        for number, row in enumerate(rows, 1):
            self.write(row, number)
        with patch("llm.client.get_client", side_effect=AssertionError("must not create a client")):
            validate_review_attempts(self.root, [question(), question("Q2")])
        # Validation must not rewrite even an interrupted record.
        self.assertEqual(json.loads((self.root / "attempts/000001.json").read_text()), rows[0])

    def test_new_directory_is_valid_without_creating_anything(self) -> None:
        path = self.root / "not-created"
        validate_review_attempts(path, [question()])
        self.assertFalse(path.exists())

    def test_rejects_negative_fractional_boolean_and_missing_usage(self) -> None:
        for status in ("complete", "failed", "running"):
            for key in ("calls", "prompt_tokens", "completion_tokens"):
                for value in (-1, 1.0, True, math.nan, math.inf, "1", None):
                    with self.subTest(status=status, key=key, value=value):
                        row = attempt(status=status)
                        row["usage"][key] = value
                        self.write(row)
                        with self.assertRaises(ValueError):
                            validate_review_attempts(self.root, [question()])
        for status in ("complete", "failed"):
            row = attempt(status=status)
            del row["usage"]
            self.write(row)
            with self.subTest(missing_usage=status), self.assertRaises(ValueError):
                validate_review_attempts(self.root, [question()])
        row = attempt()
        del row["usage"]["completion_tokens"]
        self.write(row)
        with self.assertRaises(ValueError):
            validate_review_attempts(self.root, [question()])

    def test_rejects_invalid_or_missing_wall_time(self) -> None:
        for status in ("complete", "failed", "running"):
            for value in (-1, math.inf, math.nan, True, "1", None, 10 ** 400):
                row = attempt(status=status)
                row["wall_ms"] = value
                self.write(row)
                with self.subTest(status=status, value=value), self.assertRaises(ValueError):
                    validate_review_attempts(self.root, [question()])
        for status in ("complete", "failed"):
            row = attempt(status=status)
            del row["wall_ms"]
            self.write(row)
            with self.assertRaises(ValueError):
                validate_review_attempts(self.root, [question()])

    def test_sequence_and_completed_query_are_immutable(self) -> None:
        self.write(attempt(), 2)
        with self.assertRaisesRegex(ValueError, "sequence"):
            validate_review_attempts(self.root, [question()])
        self.write(attempt(), 1)
        for status in ("complete", "failed", "running"):
            self.write(attempt(status=status), 2)
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "already completed"):
                validate_review_attempts(self.root, [question()])

    def test_question_status_and_cold_start_are_checked(self) -> None:
        cases = [None, [], {"question": None}, {**attempt(), "status": "unknown"},
                 {**attempt(), "cold_start": 1}, {**attempt(), "question": question("Q-other")},
                 {**attempt(), "question": {**question(), "input_text": "Changed question"}}]
        for row in cases:
            self.write(row)
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_review_attempts(self.root, [question()])

    def test_failed_error_must_match_query_and_have_a_type(self) -> None:
        for error in (None, [], {}, {"query_id": "wrong", "type": "Error"},
                      {"query_id": "Q1", "type": ""}, {"query_id": "Q1", "type": 1}):
            self.write({**attempt(status="failed"), "error": error})
            with self.subTest(error=error), self.assertRaises(ValueError):
                validate_review_attempts(self.root, [question()])

    def test_complete_trace_uses_existing_result_validation(self) -> None:
        variants = []
        row = attempt()
        row["result"]["reviews"] = []
        variants.append(row)
        row = attempt()
        row["result"]["coverage_report"]["cache_hit"] = True
        variants.append(row)
        row = attempt()
        row["result"]["query_id"] = "wrong"
        variants.append(row)
        variants += [{**attempt(), "result": None}, {**attempt(), "result": {}}]
        for row in variants:
            self.write(row)
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "result"):
                validate_review_attempts(self.root, [question()])

    def test_invalid_selected_question_ids_are_rejected(self) -> None:
        for questions in (None, {}, [None], [question(), question()], [{"query_id": " "}]):
            with self.subTest(questions=questions), self.assertRaises(ValueError):
                validate_review_attempts(self.root, questions)


if __name__ == "__main__":
    unittest.main()
