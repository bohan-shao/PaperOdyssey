"""Offline tests for eval/examples.py (no models, no indexes, no LLM). Run: python -m unittest tests.test_examples"""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from downstream import DownstreamError
from downstream.types import DownstreamResult, EvidenceSource
from eval import examples
from llm import client as llm_client
from orchestration.types import FinalResult, QueryPlan, ReviewRecord


def make_config() -> dict:
    return {"orchestration": {"loop": {"enabled": True, "cache": True}},
            "llm": {"model": "m", "api_key": "secret-key"},
            "ui": {"example_inputs": ["first input", "second input"]}}


def fake_result(text: str) -> DownstreamResult:
    paper = ReviewRecord("2101.00001", "pass", 5, covered=["aspect"], round=1,
                         evidence={"title": "A Paper", "top_passages": ["P2101.00001_0001"]})
    search = FinalResult("Q1", text, QueryPlan(text), [paper],
                         {"rounds": 1, "papers_reviewed": 3, "papers_passed": 1, "termination_reason": "coverage_met"},
                         {"understanding_ms": 10, "retrieval_ms": 20, "review_ms": 30, "total_ms": 70})
    source = EvidenceSource(1, "2101.00001", "P2101.00001_0001", "A Paper", 2021, "Method",
                            "https://arxiv.org/abs/2101.00001", "excerpt text")
    return DownstreamResult("Q1", text, "It works [1].", "answered", [source], [1], search,
                            {"search_ms": 70, "evidence_ms": 1, "generation_ms": 29, "total_ms": 100})


class RunExamplesTest(unittest.TestCase):
    def test_every_input_is_answered_live_with_its_results_timings_and_usage(self) -> None:
        cfg = make_config()
        original = copy.deepcopy(cfg)
        calls = []

        def fake_answer(text: str) -> DownstreamResult:
            calls.append((text, cfg["orchestration"]["loop"]["cache"]))
            llm_client._count_in_scope(100, 20)  # one LLM response inside this input's scope
            if text == "second input":
                raise DownstreamError("Could not produce a valid cited answer.")
            return fake_result(text)

        with patch.object(examples, "get_config", return_value=cfg), \
                patch.object(examples, "answer", side_effect=fake_answer), \
                patch.object(examples, "warm_up") as warm, \
                patch.object(examples, "require_fresh_runtime") as fresh, \
                patch.object(examples, "clear_runtime") as clear, \
                patch.object(examples, "index_manifests", return_value={}), \
                patch.object(examples, "environment", return_value={}):
            report = examples.run_examples()
        fresh.assert_called_once_with()
        warm.assert_called_once_with()
        clear.assert_called_once_with(release_models=True)
        self.assertEqual(calls, [("first input", False), ("second input", False)])  # cache off, live
        self.assertEqual(cfg, original)  # restored afterwards
        first, second = report["examples"]
        self.assertEqual((first["status"], first["answer"], first["cited_source_numbers"]), ("answered", "It works [1].", [1]))
        self.assertEqual(first["papers"], [{"paper_id": "2101.00001", "title": "A Paper", "score": 5}])
        self.assertEqual(first["sources"][0]["text"], "excerpt text")
        self.assertEqual(first["latency_ms"]["search"]["review_ms"], 30)
        self.assertEqual(first["latency_ms"]["answer"]["total_ms"], 100)
        self.assertEqual(first["usage"], {"calls": 1, "prompt_tokens": 100, "completion_tokens": 20})
        self.assertEqual(second["status"], "failed")
        self.assertIn("Could not produce a valid cited answer", second["error"])
        self.assertEqual(second["usage"]["calls"], 1)  # spend of a failed input still counts
        self.assertEqual(report["summary"]["failed"], 1)
        self.assertEqual(report["summary"]["total_ms"], {"n": 1, "mean": 100, "p50": 100, "p95": 100, "max": 100})
        self.assertEqual(report["config_snapshot"]["llm"]["api_key"], "<redacted>")
        text = examples.format_summary(report)
        self.assertIn("1. 0.1 s (search 0.1, answer 0.0), answered, 1 papers, 1 cited, 120 tokens", text)
        self.assertIn("2. FAILED after", text)

    def test_invalid_inputs_fail_before_anything_loads(self) -> None:
        with patch.object(examples, "get_config", return_value=make_config()), \
                patch.object(examples, "require_fresh_runtime") as fresh:
            for inputs in ([], ["ok", "  "], [3]):
                with self.assertRaises(ValueError):
                    examples.run_examples(inputs)
        fresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
