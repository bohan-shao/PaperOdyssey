"""Offline Paper Search evaluation: honest gold accounting, checkpoints and resumability."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eval import review_eval
from orchestration.types import FinalResult, QueryPlan, ReviewRecord, passing_order_key


QUESTIONS = [
    {"query_id": "Q1", "input_text": "first question", "gold_paper_ids": ["A"], "split": "eval"},
    {"query_id": "Q2", "input_text": "second question", "gold_paper_ids": ["B"], "split": "eval"},
    {"query_id": "Q3", "input_text": "third question", "gold_paper_ids": ["C"], "split": "eval"},
    {"query_id": "Q4", "input_text": "dev question", "gold_paper_ids": ["D"], "split": "dev"},
]


class FakeClient:
    """A monotonic process counter whose pre-existing usage must not enter a new run."""

    def __init__(self) -> None:
        self.totals = {"calls": 19, "prompt_tokens": 1000, "completion_tokens": 500}

    def spend(self) -> None:
        for key, amount in (("calls", 1), ("prompt_tokens", 11), ("completion_tokens", 7)):
            self.totals[key] += amount

    def usage_summary(self) -> dict:
        return dict(self.totals)


def result_for(question: dict) -> FinalResult:
    """A passes behind unknown X; B is rejected; C is never reviewed; dev D passes."""
    traces = {
        "Q1": [ReviewRecord("A", "pass", 4, round=2), ReviewRecord("X", "pass", 5)],
        "Q2": [ReviewRecord("B", "reject", 2), ReviewRecord("Y", "pass", 4)],
        "Q3": [ReviewRecord("Z", "reject", 1, round=3)],
        "Q4": [ReviewRecord("D", "pass", 5)],
    }
    rounds, covered, reason = {
        "Q1": (2, 2, "coverage_met"), "Q2": (1, 1, "min_papers"),
        "Q3": (3, 0, "max_rounds"), "Q4": (1, 2, "coverage_met"),
    }[question["query_id"]]
    reviews = traces[question["query_id"]]
    ranks = {r.paper_id: rank for rank, r in enumerate(reviews, 1)}
    papers = sorted((r for r in reviews if r.verdict == "pass"),
                    key=lambda r: passing_order_key(r.paper_id, r.score, ranks))
    return FinalResult(
        query_id=question["query_id"], input_text=question["input_text"],
        query_plan=QueryPlan(question["input_text"]), papers=papers, reviews=reviews,
        retrieval_ranks=ranks,
        coverage_report={"total_aspects": 2, "covered_aspects": covered, "rounds": rounds,
                         "papers_reviewed": len(reviews), "papers_passed": len(papers),
                         "termination_reason": reason, "cache_hit": False},
        latency_by_stage={"understanding_ms": 3, "review_ms": 7, "total_ms": 20},
    )


class ReviewEvalTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.dataset = self.directory / "eval_set.jsonl"
        self.dataset.write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS), encoding="utf-8")
        self.results = self.directory / "runs"
        self.cfg = {
            "seed": 7, "paths": {"results": "original-results", "processed": "data/processed"},
            "retrieval": {"rerank": {"enabled": True, "top_k": 10}},
            "orchestration": {"loop": {"cache": True, "max_rounds": 3}, "review": {"backend": "llm"}},
            "llm": {"model": "fake-model", "api_key": "fake-api-key-secret",
                    "base_url": "https://user:fake-url-password@example.test/v1?api_key=fake-query-secret#token"},
            "kb": {"store": {"sqlite_file": "papers.sqlite"}, "fetch": {"password": "unused-fetch-secret"}},
        }
        self.original_cfg = copy.deepcopy(self.cfg)

    @contextmanager
    def harness(self, responder=None, *, code_hash="code-v1", indexes=None):
        client = FakeClient()

        def fake_query(text, *, query_id):
            self.assertFalse(self.cfg["orchestration"]["loop"]["cache"])
            self.assertEqual(Path(self.cfg["paths"]["results"]).parent, self.results.resolve())
            question = next(q for q in QUESTIONS if q["query_id"] == query_id)
            self.assertEqual(text, question["input_text"])
            client.spend()  # a failed query can already have consumed tokens
            return responder(question) if responder else result_for(question)

        with ExitStack() as stack:
            mocks = {}
            for name, kwargs in {
                "require_fresh_runtime": {}, "clear_runtime": {},
                "environment": {"return_value": {"python": "offline-test"}},
                "index_manifests": {"return_value": indexes or {"dense": {"manifest": {"version": 1}}}},
                "_input_file_metadata": {"return_value": {}},
                "_code_hash": {"return_value": code_hash}, "get_config": {"return_value": self.cfg},
                "get_client": {"return_value": client}, "run_query": {"side_effect": fake_query},
            }.items():
                mocks[name] = stack.enter_context(patch.object(review_eval, name, **kwargs))
            stack.enter_context(patch.object(review_eval.llm_client, "_CLIENT", None))
            yield SimpleNamespace(client=client, **mocks)

    def run_eval(self, **kwargs) -> dict:
        return review_eval.run_review_eval(self.dataset, self.results, kwargs.pop("tag", "run"), **kwargs)

    def checkpoint(self, number: int = 1, tag: str = "run") -> Path:
        return self.results / tag / "attempts" / f"{number:06d}.json"

    def test_positive_source_states_unknown_candidates_and_safe_provenance(self) -> None:
        with self.harness() as fake:
            report = self.run_eval()
            self.assertEqual([c.kwargs["query_id"] for c in fake.run_query.call_args_list], ["Q1", "Q2", "Q3", "Q4"])
            fake.require_fresh_runtime.assert_called_once_with()
            fake.clear_runtime.assert_called_once_with(release_models=True)
        self.assertEqual(self.cfg, self.original_cfg)
        self.assertEqual((report["status"], report["completed_queries"], report["selected_queries"]),
                         ("complete", 4, 4))
        self.assertEqual(report["usage"], {"calls": 4, "prompt_tokens": 44, "completion_tokens": 28, "total_tokens": 72})
        row = report["splits"]["eval"]
        self.assertEqual((row["known_source_papers"], row["source_reviewed"], row["source_passed"],
                          row["source_rejected"], row["source_not_reviewed"], row["unjudged_papers_reviewed"]),
                         (3, 2, 1, 1, 1, 3))
        self.assertAlmostEqual(row["source_review_rate"], 2 / 3)
        self.assertAlmostEqual(row["source_acceptance_rate"], 1 / 3)
        self.assertEqual(row["source_pass_rate_given_reviewed"], 0.5)
        by_query = row["per_query"]
        self.assertEqual(by_query["Q1"]["source_passed"], ["A"])
        self.assertEqual(by_query["Q2"]["source_rejected"], ["B"])
        self.assertEqual(by_query["Q3"]["source_not_reviewed"], ["C"])
        self.assertEqual((by_query["Q1"]["first_source_review_round"], by_query["Q1"]["first_source_pass_round"]), (2, 2))
        self.assertIsNone(by_query["Q3"]["first_source_review_round"])
        self.assertIsNone(by_query["Q2"]["first_source_pass_round"])
        self.assertEqual(by_query["Q1"]["first_gold_rank"], 2)
        ranking = row["known_source_ranking"]
        self.assertEqual(ranking["Hit@1"], 0.0)
        self.assertAlmostEqual(ranking["Hit@3"], 1 / 3)
        self.assertAlmostEqual(ranking["MRR@10"], 1 / 6)
        self.assertAlmostEqual(ranking["NDCG@10"], 1 / math.log2(3) / 3)
        self.assertFalse(any(key.startswith("P@") for key in ranking))
        for unavailable in ("accuracy", "precision", "recall", "f1"):
            self.assertNotIn(unavailable, row)  # unknown candidates never create false-positive labels
        self.assertEqual(row["coverage"]["coverage_met"], 1)
        self.assertAlmostEqual(row["coverage"]["coverage_met_rate"], 1 / 3)
        self.assertEqual(row["coverage"]["termination_reasons"]["min_papers"], 1)
        self.assertEqual(report["splits"]["dev"]["source_passed"], 1)
        self.assertEqual(report["cold_start_query_ids"], ["Q1"])
        manifest = json.loads((self.results / "run" / "manifest.json").read_text())
        self.assertEqual(manifest["effective_config"]["llm"]["api_key"], "<redacted>")
        self.assertEqual(manifest["effective_config"]["llm"]["base_url"], "https://example.test/v1")
        self.assertFalse(manifest["effective_config"]["orchestration"]["loop"]["cache"])
        self.assertNotIn("fetch", manifest["effective_config"]["kb"])
        self.assertEqual(json.loads((self.results / "run" / "report.json").read_text()), report)
        for path in (self.results / "run").rglob("*.json"):
            for secret in ("fake-api-key-secret", "fake-url-password", "fake-query-secret", "unused-fetch-secret"):
                self.assertNotIn(secret, path.read_text())

    def test_failure_stops_then_resume_skips_completed_and_accumulates_attempt_usage(self) -> None:
        def fail_second(question):
            if question["query_id"] == "Q2":
                raise RuntimeError("provider exposed fake-provider-secret")
            return result_for(question)

        with self.harness(fail_second) as first:
            failed = self.run_eval()
            self.assertEqual([c.kwargs["query_id"] for c in first.run_query.call_args_list], ["Q1", "Q2"])
            first.clear_runtime.assert_called_once_with(release_models=True)
        self.assertEqual(self.cfg, self.original_cfg)
        self.assertEqual((failed["status"], failed["completed_queries"], failed["failed_attempts"]), ("failed", 1, 1))
        self.assertEqual(failed["incomplete_query_ids"], ["Q2", "Q3", "Q4"])
        self.assertEqual(failed["splits"]["eval"]["n_queries"], 1)  # failure is not a retrieval miss
        self.assertEqual(failed["error"], {"query_id": "Q2", "type": "RuntimeError"})
        self.assertEqual(failed["usage"]["total_tokens"], 36)
        original_checkpoint = self.checkpoint().read_bytes()
        with self.harness() as second:
            resumed = self.run_eval(resume=True)
            self.assertEqual([c.kwargs["query_id"] for c in second.run_query.call_args_list], ["Q2", "Q3", "Q4"])
            second.clear_runtime.assert_called_once_with(release_models=True)
        self.assertEqual(self.cfg, self.original_cfg)
        self.assertEqual(self.checkpoint().read_bytes(), original_checkpoint)
        self.assertEqual((resumed["status"], resumed["completed_queries"], resumed["failed_attempts"]), ("complete", 4, 1))
        self.assertEqual(resumed["usage"], {"calls": 5, "prompt_tokens": 55, "completion_tokens": 35, "total_tokens": 90})
        self.assertEqual(resumed["cold_start_query_ids"], ["Q1", "Q2"])
        self.assertEqual(resumed["incomplete_query_ids"], [])
        for path in (self.results / "run").rglob("*.json"):
            self.assertNotIn("fake-provider-secret", path.read_text())

    def test_all_complete_resume_has_no_client_or_api_work(self) -> None:
        with self.harness():
            complete = self.run_eval()
        checkpoints = {p.name: p.read_bytes() for p in self.checkpoint().parent.glob("*.json")}
        with self.harness() as fake:
            resumed = self.run_eval(resume=True)
            fake.get_client.assert_not_called()
            fake.run_query.assert_not_called()
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(resumed["usage"], complete["usage"])
        self.assertEqual(resumed["splits"], complete["splits"])
        self.assertEqual({p.name: p.read_bytes() for p in self.checkpoint().parent.glob("*.json")}, checkpoints)
        self.assertEqual(self.cfg, self.original_cfg)

    def test_resume_fingerprint_changes_fail_before_client_and_restore_config(self) -> None:
        with self.harness():
            self.run_eval()
        original_data = self.dataset.read_bytes()
        for change in ("dataset", "config", "code", "indexes", "selection", "runtime", "input_files"):
            with self.subTest(change=change):
                self.dataset.write_bytes(original_data)
                self.cfg = copy.deepcopy(self.original_cfg)
                harness_kwargs, run_kwargs = {}, {"resume": True}
                if change == "dataset":
                    self.dataset.write_bytes(original_data + b"\n")
                elif change == "config":
                    self.cfg["retrieval"]["rerank"]["top_k"] = 11
                elif change == "code":
                    harness_kwargs["code_hash"] = "code-v2"
                elif change == "indexes":
                    harness_kwargs["indexes"] = {"dense": {"manifest": {"version": 2}}}
                elif change == "selection":
                    run_kwargs["limit"] = 1
                expected_cfg = copy.deepcopy(self.cfg)
                with self.harness(**harness_kwargs) as fake:
                    if change == "runtime":
                        fake.environment.return_value = {"python": "different-version"}
                    elif change == "input_files":
                        fake._input_file_metadata.return_value = {"index": {"bytes": 100, "mtime_ns": 200}}
                    with self.assertRaisesRegex(ValueError, "fingerprint"):
                        self.run_eval(**run_kwargs)
                    fake.get_client.assert_not_called()
                    fake.run_query.assert_not_called()
                self.assertEqual(self.cfg, expected_cfg)

    def test_invalid_live_trace_or_cached_result_is_failed_not_scored(self) -> None:
        mutations = {
            "missing_trace": lambda r: replace(r, reviews=[]),
            "cached": lambda r: replace(r, coverage_report={**r.coverage_report, "cache_hit": True}),
            "duplicate": lambda r: replace(r, reviews=[*r.reviews, r.reviews[0]]),
            "ranking": lambda r: replace(r, papers=list(reversed(r.papers))),
            "no_ranks": lambda r: replace(r, retrieval_ranks={}),
            "round": lambda r: replace(r, reviews=[replace(r.reviews[0], round=9), *r.reviews[1:]]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), self.harness(lambda q: mutate(result_for(q))) as fake:
                report = self.run_eval(tag=name)
                fake.run_query.assert_called_once()
                fake.clear_runtime.assert_called_once_with(release_models=True)
                self.assertEqual((report["status"], report["completed_queries"], report["failed_attempts"]), ("failed", 0, 1))
                self.assertEqual(report["error"]["type"], "ValueError")
                self.assertEqual(report["splits"]["eval"]["n_queries"], 0)
                self.assertIsNone(report["splits"]["eval"]["source_acceptance_rate"])
                self.assertEqual(report["usage"]["total_tokens"], 18)
                self.assertEqual(self.cfg, self.original_cfg)

    def test_corrupt_completed_checkpoint_is_rejected_before_any_api_work(self) -> None:
        with self.harness():
            self.run_eval()
        original = self.checkpoint().read_text()
        for corruption in ("cache", "trace", "identity", "duplicate_completion"):
            with self.subTest(corruption=corruption):
                self.checkpoint().write_text(original)
                duplicate = self.checkpoint(5)
                duplicate.unlink(missing_ok=True)
                row = json.loads(original)
                if corruption == "cache":
                    row["result"]["coverage_report"]["cache_hit"] = True
                elif corruption == "trace":
                    row["result"]["reviews"] = []
                elif corruption == "identity":
                    row["question"]["input_text"] = "changed input"
                else:
                    duplicate.write_text(original)
                self.checkpoint().write_text(json.dumps(row))
                with self.harness() as fake:
                    with self.assertRaises(ValueError):
                        self.run_eval(resume=True)
                    fake.get_client.assert_not_called()
                    fake.run_query.assert_not_called()
                self.assertEqual(self.cfg, self.original_cfg)

    def test_client_setup_failure_restores_configuration_and_releases_runtime(self) -> None:
        with self.harness() as fake:
            fake.get_client.side_effect = RuntimeError("client configuration failure")
            with self.assertRaisesRegex(RuntimeError, "client configuration"):
                self.run_eval()
            fake.run_query.assert_not_called()
            fake.clear_runtime.assert_called_once_with(release_models=True)
        self.assertEqual(self.cfg, self.original_cfg)
        # A reserved attempt is incomplete and remains resumable; it must not claim a result.
        if self.checkpoint().exists():
            checkpoint = json.loads(self.checkpoint().read_text())
            self.assertEqual(checkpoint["status"], "running")
            self.assertNotIn("result", checkpoint)

    def test_interrupted_attempt_is_retried_and_keeps_observed_usage(self) -> None:
        with self.harness(lambda q: (_ for _ in ()).throw(KeyboardInterrupt())) as first:
            with self.assertRaises(KeyboardInterrupt):
                self.run_eval(limit=1)
            first.clear_runtime.assert_called_once_with(release_models=True)
        self.assertEqual(json.loads(self.checkpoint().read_text())["status"], "running")
        self.assertEqual(self.cfg, self.original_cfg)
        with self.harness() as second:
            resumed = self.run_eval(limit=1, resume=True)
            second.run_query.assert_called_once()
        self.assertEqual((resumed["status"], resumed["completed_queries"], resumed["interrupted_attempts"]),
                         ("complete", 1, 1))
        self.assertEqual(resumed["usage"]["total_tokens"], 36)

    def test_split_and_limit_select_in_file_order_without_evaluating_other_queries(self) -> None:
        with self.harness() as fake:
            report = self.run_eval(split="dev", limit=1)
            fake.run_query.assert_called_once_with("dev question", query_id="Q4")
        self.assertEqual(report["provenance"]["selected_query_ids"], ["Q4"])
        self.assertEqual(report["splits"]["eval"]["n_queries"], 0)
        self.assertEqual(report["splits"]["dev"]["n_queries"], 1)
        self.assertEqual(report["selected_queries"], 1)


class PassingOrderValidationTest(unittest.TestCase):
    """Equal review scores keep the retrieval order (D47), not the paper-ID order."""

    question = {"query_id": "Q1", "input_text": "Find a source."}

    def result(self, order: list[str]) -> dict:
        reviews = [ReviewRecord("1904.01172", "pass", 5), ReviewRecord("1910.11473", "pass", 5),
                   ReviewRecord("1803.05457", "reject", 2)]
        by_id = {r.paper_id: r for r in reviews}
        return asdict(FinalResult(
            "Q1", "Find a source.", QueryPlan("Find a source."), [by_id[p] for p in order],
            {"total_aspects": 1, "covered_aspects": 1, "rounds": 1, "papers_reviewed": 3,
             "papers_passed": 2, "termination_reason": "coverage_met", "cache_hit": False},
            reviews=reviews, retrieval_ranks={"1904.01172": 4, "1910.11473": 1, "1803.05457": 2},
        ))

    def test_tie_follows_the_best_retrieval_rank(self) -> None:
        review_eval._validate_result(self.question, self.result(["1910.11473", "1904.01172"]))

    def test_paper_id_tie_order_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "inconsistent passing-paper ranking"):
            review_eval._validate_result(self.question, self.result(["1904.01172", "1910.11473"]))

    def test_ranks_must_cover_every_reviewed_paper(self) -> None:
        result = self.result(["1910.11473", "1904.01172"])
        del result["retrieval_ranks"]["1803.05457"]
        with self.assertRaisesRegex(ValueError, "retrieval ranks"):
            review_eval._validate_result(self.question, result)


if __name__ == "__main__":
    unittest.main()
