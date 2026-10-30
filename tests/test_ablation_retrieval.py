"""Offline retrieval-cell tests: effective switches, hand-computed ranks and resumability."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from eval import ablation_retrieval as cell
from retrieval.types import RetrieveResult, ScoredPassage


QUESTIONS = [
    {"query_id": "Q1", "input_text": "first question", "gold_paper_ids": ["A"], "split": "eval"},
    {"query_id": "Q2", "input_text": "second question", "gold_paper_ids": ["B"], "split": "eval"},
    {"query_id": "Q3", "input_text": "third question", "gold_paper_ids": ["C"], "split": "dev"},
]


def result_for(text: str, rerank: bool) -> RetrieveResult:
    # Q1 has two distinct passages from A: its paper rank is 2 without reranking, 1 with it.
    papers = {"first question": ["A", "A", "X"] if rerank else ["X", "A", "A"],
              "second question": ["Z"], "third question": []}[text]
    return RetrieveResult(
        hits=[ScoredPassage(f"{paper}_{rank}", paper, "evidence", 1 / rank, rank, "rerank" if rerank else "dense")
              for rank, paper in enumerate(papers, 1)],
        total_candidates=len(papers) + 4,
        latency_ms={"dense_ms": 3, "rerank_ms": 2 if rerank else 0, "total_ms": 8 if rerank else 6},
    )


class RetrievalCellTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.run_dir = self.directory / "cell"
        self.questions = copy.deepcopy(QUESTIONS)
        self.cfg = {"seed": 42, "paths": {"results": "unchanged-results"}, "retrieval": {
            "sparse": {"enabled": False}, "dense": {"enabled": True},
            "rerank": {"enabled": False, "top_k": 4}, "fusion": {"top_k": 17},
            "query_processing": {"enabled": True}, "parent_doc": {"enabled": False},
        }}
        self.original_cfg = copy.deepcopy(self.cfg)

    @contextmanager
    def harness(self, responder=None):
        observed = []

        def retrieve(text):
            observed.append(copy.deepcopy(self.cfg))
            return responder(text) if responder else result_for(text, self.cfg["retrieval"]["rerank"]["enabled"])

        with ExitStack() as stack:
            mocks = {}
            for name, kwargs in {"get_config": {"return_value": self.cfg}, "require_fresh_runtime": {},
                                 "clear_runtime": {}, "retrieve": {"side_effect": retrieve}}.items():
                mocks[name] = stack.enter_context(patch.object(cell, name, **kwargs))
            yield SimpleNamespace(observed=observed, **mocks)

    def run_cell(self, **kwargs) -> dict:
        return cell.run_retrieval_cell(kwargs.pop("questions", self.questions),
                                       kwargs.pop("run_dir", self.run_dir),
                                       kwargs.pop("fingerprint", "fingerprint-v1"), **kwargs)

    def attempt(self, number=1) -> Path:
        return self.run_dir / "attempts" / f"{number:06d}.json"

    def test_effective_switches_remain_untouched_and_ranks_match_hand_calculation(self) -> None:
        with self.harness() as fake:
            report = self.run_cell()
            self.assertEqual(fake.retrieve.call_args_list, [call(q["input_text"]) for q in QUESTIONS])
            self.assertEqual(fake.observed, [self.original_cfg] * 3)
            fake.require_fresh_runtime.assert_called_once_with()
            fake.clear_runtime.assert_called_once_with(release_models=True)
        self.assertEqual(self.cfg, self.original_cfg)
        self.assertEqual(self.questions, QUESTIONS)
        self.assertEqual((report["stage"], report["status"], report["completed_queries"], report["selected_queries"]),
                         ("retrieval", "complete", 3, 3))
        metrics = report["splits"]["eval"]
        # Eval paper ranks = [2, miss], despite two A passages: MRR=(1/2+0)/2.
        self.assertEqual(metrics["n_queries"], 2)
        self.assertEqual(metrics["Hit@1"], 0.0)
        self.assertEqual(metrics["Hit@3"], 0.5)
        self.assertEqual(metrics["MRR@10"], 0.25)
        self.assertAlmostEqual(metrics["NDCG@10"], 1 / math.log2(3) / 2)
        self.assertAlmostEqual(metrics["P@3"], 1 / 6)
        self.assertEqual(report["splits"]["dev"]["Hit@10"], 0.0)
        self.assertEqual(report["per_query"]["Q1"]["first_gold_rank"], 2)
        self.assertTrue(report["per_query"]["Q1"]["source_success"])
        self.assertIsNone(report["per_query"]["Q2"]["first_gold_rank"])
        self.assertFalse(report["per_query"]["Q2"]["source_success"])
        self.assertEqual([report["per_query"][qid]["cold_start"] for qid in ("Q1", "Q2", "Q3")], [True, False, False])
        self.assertEqual(report["per_query"]["Q1"]["latency_ms"], {"dense_ms": 3, "rerank_ms": 0, "total_ms": 6})
        self.assertGreaterEqual(report["per_query"]["Q1"]["wall_ms"], 0)
        self.assertEqual(report["usage"], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        checkpoint = json.loads(self.attempt().read_text())
        self.assertEqual(checkpoint["question"], QUESTIONS[0])
        self.assertEqual(checkpoint["record"]["top_k"], 4)
        self.assertEqual(checkpoint["record"]["stage"], "retrieval")
        self.assertEqual(checkpoint["record"]["hits"][1]["paper_id"], "A")
        self.assertEqual(json.loads((self.run_dir / "report.json").read_text()), report)
        self.assertFalse(list(self.run_dir.rglob("*.tmp")))

    def test_enabling_only_caller_rerank_switch_changes_results_without_wrapper_overrides(self) -> None:
        self.cfg["retrieval"]["rerank"]["enabled"] = True
        expected = copy.deepcopy(self.cfg)
        with self.harness() as fake:
            report = self.run_cell()
            self.assertEqual(fake.observed, [expected] * 3)
            self.assertEqual(fake.retrieve.call_count, 3)  # no warm-up request
        self.assertEqual(self.cfg, expected)
        self.assertEqual(report["per_query"]["Q1"]["first_gold_rank"], 1)
        self.assertEqual(report["splits"]["eval"]["MRR@10"], 0.5)
        self.assertEqual(report["per_query"]["Q1"]["latency_ms"]["rerank_ms"], 2)

    def test_failure_then_resume_skips_completed_queries_and_first_new_query_is_cold(self) -> None:
        def fail_second(text):
            if text == "second question":
                raise RuntimeError("do not persist secret provider details")
            return result_for(text, False)

        with self.harness(fail_second) as first:
            failed = self.run_cell()
            self.assertEqual(first.retrieve.call_args_list, [call("first question"), call("second question")])
            first.clear_runtime.assert_called_once_with(release_models=True)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["completed_queries"], 1)
        self.assertEqual(failed["failed_attempts"], 1)
        self.assertEqual(failed["incomplete_query_ids"], ["Q2", "Q3"])
        self.assertEqual(failed["error"], {"query_id": "Q2", "type": "RuntimeError"})
        self.assertEqual(failed["splits"]["eval"]["n_queries"], 1)  # failure is excluded, not a miss
        self.assertEqual(failed["splits"]["eval"]["MRR@10"], 0.5)
        original = self.attempt().read_bytes()
        with self.harness() as second:
            complete = self.run_cell(resume=True)
            self.assertEqual(second.retrieve.call_args_list, [call("second question"), call("third question")])
            second.clear_runtime.assert_called_once_with(release_models=True)
        self.assertEqual(self.attempt().read_bytes(), original)
        self.assertEqual((complete["status"], complete["completed_queries"], complete["failed_attempts"]), ("complete", 3, 1))
        self.assertTrue(complete["per_query"]["Q2"]["cold_start"])
        self.assertFalse(complete["per_query"]["Q3"]["cold_start"])
        self.assertEqual(complete["cold_start_query_ids"], ["Q1", "Q2"])
        self.assertEqual(self.cfg, self.original_cfg)
        for path in self.run_dir.rglob("*.json"):
            self.assertNotIn("secret provider details", path.read_text())

    def test_complete_resume_uses_no_retrieval_and_preserves_checkpoints(self) -> None:
        with self.harness():
            first = self.run_cell()
        original = {p.name: p.read_bytes() for p in self.attempt().parent.glob("*.json")}
        with self.harness() as fake:
            resumed = self.run_cell(resume=True)
            fake.retrieve.assert_not_called()
            fake.clear_runtime.assert_called_once_with(release_models=True)
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(resumed["splits"], first["splits"])
        self.assertEqual(resumed["per_query"], first["per_query"])
        self.assertEqual({p.name: p.read_bytes() for p in self.attempt().parent.glob("*.json")}, original)

    def test_active_cell_lock_refuses_a_second_writer_before_retrieval(self) -> None:
        self.run_dir.mkdir()
        with cell._lock(self.run_dir / ".lock"), self.harness() as fake:
            with self.assertRaisesRegex(RuntimeError, "already active"):
                self.run_cell()
            fake.retrieve.assert_not_called()
            fake.clear_runtime.assert_called_once_with(release_models=True)
        self.assertFalse((self.run_dir / "manifest.json").exists())

    def test_changed_fingerprint_questions_or_order_are_rejected_before_retrieval(self) -> None:
        with self.harness():
            self.run_cell()
        altered = copy.deepcopy(QUESTIONS)
        altered[0]["input_text"] = "changed question"
        for kwargs in ({"fingerprint": "fingerprint-v2"}, {"questions": altered},
                       {"questions": list(reversed(QUESTIONS))}, {"questions": QUESTIONS[:1]}):
            with self.subTest(kwargs=kwargs), self.harness() as fake:
                with self.assertRaisesRegex(ValueError, "fingerprint/questions"):
                    self.run_cell(resume=True, **kwargs)
                fake.retrieve.assert_not_called()
                fake.clear_runtime.assert_called_once_with(release_models=True)
        self.assertEqual(self.cfg, self.original_cfg)

    def test_corrupt_checkpoint_identity_hits_timing_or_sequence_stop_before_retrieval(self) -> None:
        with self.harness():
            self.run_cell()
        original = {p.name: p.read_bytes() for p in self.attempt().parent.glob("*.json")}
        for corruption in ("question", "fingerprint", "rank", "score", "latency", "status", "duplicate", "gap"):
            with self.subTest(corruption=corruption):
                for path in self.attempt().parent.glob("*.json"):
                    path.unlink()
                for name, content in original.items():
                    (self.attempt().parent / name).write_bytes(content)
                row = json.loads(original["000001.json"])
                if corruption == "question":
                    row["question"]["gold_paper_ids"] = ["wrong"]
                elif corruption == "fingerprint":
                    row["fingerprint"] = "other-cell"
                elif corruption == "rank":
                    row["record"]["hits"][0]["rank"] = 3
                elif corruption == "score":
                    row["record"]["hits"][0]["score"] = float("nan")
                elif corruption == "latency":
                    row["latency_ms"]["total_ms"] = -1
                elif corruption == "status":
                    row["status"] = "success"
                elif corruption == "duplicate":
                    self.attempt(4).write_bytes(original["000001.json"])
                else:
                    self.attempt(2).unlink()
                self.attempt().write_text(json.dumps(row))
                with self.harness() as fake:
                    with self.assertRaises(ValueError):
                        self.run_cell(resume=True)
                    fake.retrieve.assert_not_called()
                    fake.clear_runtime.assert_called_once_with(release_models=True)

    def test_invalid_live_result_is_checkpointed_as_failure(self) -> None:
        with self.harness(lambda text: RetrieveResult([], 0, {"total_ms": -1})) as fake:
            report = self.run_cell()
            fake.retrieve.assert_called_once_with("first question")
            fake.clear_runtime.assert_called_once_with(release_models=True)
        self.assertEqual((report["status"], report["completed_queries"]), ("failed", 0))
        self.assertEqual(report["error"], {"query_id": "Q1", "type": "ValueError"})
        self.assertEqual(report["splits"]["eval"]["n_queries"], 0)
        self.assertNotIn("record", json.loads(self.attempt().read_text()))

    def test_existing_missing_manifest_and_invalid_inputs_fail_without_retrieval(self) -> None:
        with self.harness() as fake:
            with self.assertRaises(FileNotFoundError):
                self.run_cell(resume=True)
            fake.retrieve.assert_not_called()
        with self.harness():
            self.run_cell()
        with self.harness() as fake:
            with self.assertRaises(FileExistsError):
                self.run_cell()
            fake.retrieve.assert_not_called()
        for questions in ([], [QUESTIONS[0], QUESTIONS[0]], [{**QUESTIONS[0], "gold_paper_ids": []}],
                          [{**QUESTIONS[0], "split": "test"}], [{**QUESTIONS[0], "input_text": " "}]):
            with self.subTest(questions=questions), self.harness() as fake:
                with self.assertRaises(ValueError):
                    self.run_cell(questions=questions)
                fake.retrieve.assert_not_called()
        self.assertEqual(self.cfg, self.original_cfg)

    def test_interruption_retains_running_attempt_and_resumes_only_incomplete_query(self) -> None:
        def interrupt(text):
            raise KeyboardInterrupt

        with self.harness(interrupt) as fake:
            with self.assertRaises(KeyboardInterrupt):
                self.run_cell()
            fake.clear_runtime.assert_called_once_with(release_models=True)
        self.assertEqual(json.loads(self.attempt().read_text())["status"], "running")
        with self.harness() as fake:
            report = self.run_cell(resume=True)
            self.assertEqual(fake.retrieve.call_count, 3)
        self.assertEqual((report["status"], report["interrupted_attempts"]), ("complete", 1))


if __name__ == "__main__":
    unittest.main()
