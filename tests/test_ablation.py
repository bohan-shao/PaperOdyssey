"""Offline matrix integration with real retrieval/review checkpoint runners and fake models."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from eval import ablation, ablation_retrieval, review_eval
from llm import client as llm_client
from orchestration.types import FinalResult, QueryPlan, ReviewRecord
from retrieval.types import RetrieveResult, ScoredPassage


QUESTIONS = [
    {"query_id": "D1", "input_text": "first dev", "gold_paper_ids": ["A"], "split": "dev"},
    {"query_id": "E1", "input_text": "first eval", "gold_paper_ids": ["B"], "split": "eval"},
    {"query_id": "D2", "input_text": "second dev", "gold_paper_ids": ["C"], "split": "dev"},
    {"query_id": "E2", "input_text": "second eval", "gold_paper_ids": ["D"], "split": "eval"},
]
STATES = {"baseline": (False, False), "a": (True, False), "b": (False, True), "ab": (True, True)}
RANKS = {
    (False, False): {"D1": None, "E1": 2, "D2": 1, "E2": None},
    (True, False): {"D1": 2, "E1": 1, "D2": 2, "E2": None},
    (False, True): {"D1": 3, "E1": None, "D2": 1, "E2": 2},
    (True, True): {"D1": 1, "E1": 1, "D2": 1, "E2": 1},
}


class UsageClient:
    def __init__(self) -> None:
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def spend(self) -> None:
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += 11
        self.usage["completion_tokens"] += 7
        llm_client._count_in_scope(11, 7)  # what LLMClient._record does (D48)

    def usage_summary(self) -> dict:
        return dict(self.usage)


class AblationIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.dataset = self.directory / "eval_set.jsonl"
        self.dataset.write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS), encoding="utf-8")
        self.results = self.directory / "matrices"
        self.cfg = {
            "seed": 7, "paths": {"results": "original-results", "processed": "data/processed"},
            "kb": {"store": {"sqlite_file": "papers.sqlite"}},
            "retrieval": {
                "sparse": {"enabled": True, "backend": "sqlite",
                           "field_weighting": {"enabled": False, "weights": [3, 2, 1]}},
                "dense": {"enabled": True}, "rerank": {"enabled": True, "top_k": 10},
                "diversity": {"enabled": True}, "query_processing": {"enabled": True},
                "parent_doc": {"enabled": False}, "fusion": {"top_k": 17},
            },
            "orchestration": {"review": {"backend": "llm"}, "hyde": {"enabled": False},
                              "loop": {"enabled": True, "max_rounds": 3, "cache": True},
                              "rewrite": {"max_queries_per_round": 2}},
            "llm": {"model": "fake", "api_key": "matrix-test-secret", "base_url": "https://example.test/v1"},
        }
        self.original_cfg = copy.deepcopy(self.cfg)

    @contextmanager
    def harness(self, retrieval_failure=None):
        events, clients = [], []

        def state():
            return (self.cfg["retrieval"]["rerank"]["enabled"], self.cfg["retrieval"]["diversity"]["enabled"])

        def fake_retrieve(text):
            question = next(q for q in QUESTIONS if q["input_text"] == text)
            flags = state()
            events.append(("retrieval", flags, question["query_id"], copy.deepcopy(self.cfg)))
            if retrieval_failure:
                retrieval_failure(question, flags)
            rank = RANKS[flags][question["query_id"]]
            papers = ([f"unknown-{i}" for i in range(1, rank)] + question["gold_paper_ids"]
                      if rank else ["unknown"])
            return RetrieveResult(
                [ScoredPassage(f"{paper}_{i}", paper, "evidence", 1 / i, i, "rerank")
                 for i, paper in enumerate(papers, 1)], len(papers), {"dense_ms": 2, "total_ms": 5})

        def fake_client():
            client = UsageClient()
            clients.append(client)
            return client

        def fake_search(text, *, query_id):
            question = next(q for q in QUESTIONS if q["query_id"] == query_id)
            self.assertEqual(text, question["input_text"])
            events.append(("review", state(), query_id, copy.deepcopy(self.cfg)))
            self.assertFalse(self.cfg["orchestration"]["loop"]["cache"])
            clients[-1].spend()
            looping = self.cfg["orchestration"]["loop"]["enabled"]
            reviews = [ReviewRecord("unknown", "reject", 1)]
            if looping:
                reviews.append(ReviewRecord(question["gold_paper_ids"][0], "pass", 5, round=2))
            papers = [r for r in reviews if r.verdict == "pass"]
            return FinalResult(
                query_id, text, QueryPlan(text), papers,
                {"total_aspects": 1, "covered_aspects": int(looping), "rounds": 2 if looping else 1,
                 "papers_reviewed": len(reviews), "papers_passed": len(papers),
                 "termination_reason": "coverage_met" if looping else "loop_disabled", "cache_hit": False},
                {"review_ms": 4, "total_ms": 9}, reviews=reviews,
                retrieval_ranks={r.paper_id: rank for rank, r in enumerate(reviews, 1)},
            )

        with ExitStack() as stack:
            mocked = {}
            for module in (ablation, ablation_retrieval, review_eval):
                stack.enter_context(patch.object(module, "get_config", return_value=self.cfg))
                mocked[f"{module.__name__.split('.')[-1]}_fresh"] = stack.enter_context(
                    patch.object(module, "require_fresh_runtime"))
            for module in (ablation, review_eval):
                for name, value in {
                    "environment": {"python": "offline", "versions": {"fake": "1"}},
                    "index_manifests": {"dense": {"manifest": {"version": 1}}},
                    "_input_file_metadata": {}, "_code_hash": "code-v1",
                }.items():
                    mocked[f"{module.__name__.split('.')[-1]}_{name}"] = stack.enter_context(
                        patch.object(module, name, return_value=copy.deepcopy(value)))
            mocked["seed"] = stack.enter_context(patch.object(ablation, "_seed_local"))
            mocked["retrieval_cell"] = stack.enter_context(
                patch.object(ablation, "run_retrieval_cell", wraps=ablation_retrieval.run_retrieval_cell))
            mocked["review_cell"] = stack.enter_context(
                patch.object(ablation, "run_review_eval", wraps=review_eval.run_review_eval))
            mocked["retrieve"] = stack.enter_context(patch.object(ablation_retrieval, "retrieve", side_effect=fake_retrieve))
            mocked["run_query"] = stack.enter_context(patch.object(review_eval, "run_query", side_effect=fake_search))
            mocked["get_client"] = stack.enter_context(patch.object(review_eval, "get_client", side_effect=fake_client))
            for module in (ablation_retrieval, review_eval):
                stack.enter_context(patch.object(module, "clear_runtime"))
            stack.enter_context(patch.object(ablation.llm_client, "_CLIENT", None))
            yield SimpleNamespace(events=events, clients=clients, **mocked)

    def run_matrix(self, **kwargs):
        return ablation.run_ablation(self.dataset, self.results, kwargs.pop("tag", "matrix"),
                                     factors=kwargs.pop("factors", ["rerank", "diversity"]),
                                     stage=kwargs.pop("stage", "retrieval"), **kwargs)

    def test_two_factor_four_rows_use_each_query_once_and_change_only_named_switches(self) -> None:
        with self.harness() as fake:
            report = self.run_matrix()
            self.assertEqual(fake.retrieve.call_count, 16)
            fake.run_query.assert_not_called()
            fake.get_client.assert_not_called()
            self.assertEqual(fake.seed.call_args_list, [call(7)] * 4)
            self.assertEqual([(event[1], event[2]) for event in fake.events],
                             [(flags, q["query_id"]) for flags in STATES.values() for q in QUESTIONS])
            for _, flags, _, actual in fake.events:
                expected = copy.deepcopy(self.original_cfg)
                expected["retrieval"]["rerank"]["enabled"], expected["retrieval"]["diversity"]["enabled"] = flags
                self.assertEqual(actual, expected)
        self.assertEqual(self.cfg, self.original_cfg)
        self.assertEqual((report["status"], report["completed_cells"], report["selected_cells"]), ("complete", 4, 4))
        self.assertEqual(report["query_executions"], 16)
        for row, flags in STATES.items():
            actual = report["rows"][row]["results"]["retrieval"]["per_query"]
            self.assertEqual({qid: values["first_gold_rank"] for qid, values in actual.items()}, RANKS[flags])
        edges = {(r["before"], r["after"]): r for r in report["comparisons"]["retrieval"]}
        self.assertEqual(set(edges), {("baseline", "a"), ("baseline", "b"), ("a", "ab"), ("b", "ab")})
        self.assertEqual(edges[("baseline", "a")]["splits"]["dev"]["fixed_query_ids"], ["D1"])
        # Dev MRR: baseline=.5, A=.5, B=(1/3+1)/2=2/3, AB=1; contrast=1/3.
        interaction = report["interaction"]["retrieval"]["splits"]
        self.assertAlmostEqual(interaction["dev"]["metrics"]["MRR@10"]["interaction"], 1 / 3)
        self.assertEqual(interaction["dev"]["metrics"]["Hit@10"]["interaction"], -0.5)
        # Eval MRR: baseline=.25, A=.5, B=.25, AB=1; contrast=.5.
        self.assertEqual(interaction["eval"]["metrics"]["MRR@10"]["interaction"], 0.5)
        self.assertEqual(report["usage"]["total_tokens"], 0)
        for path in (self.results / "matrix").rglob("*.json"):
            self.assertNotIn("matrix-test-secret", path.read_text())

    def test_split_filter_precedes_limit_and_selected_snapshot_is_stable(self) -> None:
        with self.harness() as fake:
            report = self.run_matrix(factors=["rerank"], split="eval", limit=1)
            self.assertEqual([e[2] for e in fake.events], ["E1", "E1"])
        self.assertEqual(report["selected_query_ids"], ["E1"])
        self.assertEqual(report["query_executions"], 2)
        snapshot = self.results / "matrix" / "selected_eval.jsonl"
        self.assertEqual([json.loads(line) for line in snapshot.read_text().splitlines()], [QUESTIONS[1]])
        self.assertEqual(report["comparisons"]["retrieval"][0]["splits"]["dev"]["n_selected"], 0)

    def test_failed_cell_stops_matrix_then_resume_never_reexecutes_completed_queries(self) -> None:
        def fail(question, flags):
            if flags == (True, False) and question["query_id"] == "E1":
                raise RuntimeError("fake failure with private-provider-detail")

        with self.harness(fail) as first:
            failed = self.run_matrix()
            self.assertEqual(len(first.events), 6)  # baseline all four, A D1 then failed E1
            self.assertEqual(first.retrieval_cell.call_count, 2)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["completed_cells"], 1)
        self.assertEqual((failed["error"]["row"], failed["error"]["stage"]), ("a", "retrieval"))
        self.assertEqual(failed["rows"]["a"]["results"]["retrieval"]["completed_queries"], 1)
        self.assertEqual(failed["rows"]["b"]["results"], {})
        paired = failed["comparisons"]["retrieval"][0]["splits"]
        self.assertEqual(paired["dev"]["paired_query_ids"], ["D1"])
        self.assertEqual(paired["eval"]["n_paired"], 0)
        base_dir = self.results / "matrix" / "rows" / "baseline" / "retrieval" / "attempts"
        old_checkpoints = {p.name: p.read_bytes() for p in base_dir.glob("*.json")}
        a_first = self.results / "matrix" / "rows" / "a" / "retrieval" / "attempts" / "000001.json"
        first_bytes = a_first.read_bytes()
        with self.harness() as resumed:
            complete = self.run_matrix(resume=True)
            self.assertEqual(Counter((e[1], e[2]) for e in resumed.events), Counter(
                [(STATES["a"], q["query_id"]) for q in QUESTIONS[1:]]
                + [(STATES[row], q["query_id"]) for row in ("b", "ab") for q in QUESTIONS]))
            self.assertEqual(len(resumed.events), 11)
        self.assertEqual(complete["status"], "complete")
        self.assertEqual(complete["completed_cells"], 4)
        self.assertEqual({p.name: p.read_bytes() for p in base_dir.glob("*.json")}, old_checkpoints)
        self.assertEqual(a_first.read_bytes(), first_bytes)
        self.assertEqual(self.cfg, self.original_cfg)
        for path in (self.results / "matrix").rglob("*.json"):
            self.assertNotIn("private-provider-detail", path.read_text())

    def test_complete_resume_has_zero_retrieve_and_preserves_cell_observations(self) -> None:
        with self.harness():
            first = self.run_matrix()
        with self.harness() as fake:
            resumed = self.run_matrix(resume=True)
            fake.retrieve.assert_not_called()
            fake.run_query.assert_not_called()
            fake.get_client.assert_not_called()
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(resumed["comparisons"], first["comparisons"])
        self.assertEqual(resumed["interaction"], first["interaction"])
        self.assertEqual(resumed["usage"], first["usage"])

    def test_fingerprint_and_snapshot_mismatches_refuse_before_any_cell(self) -> None:
        with self.harness():
            self.run_matrix()
        original_data = self.dataset.read_bytes()
        snapshot = self.results / "matrix" / "selected_eval.jsonl"
        original_snapshot = snapshot.read_bytes()
        for change in ("data", "config", "code", "indexes", "input_files", "runtime", "selection", "snapshot"):
            with self.subTest(change=change):
                self.dataset.write_bytes(original_data)
                snapshot.write_bytes(original_snapshot)
                self.cfg = copy.deepcopy(self.original_cfg)
                kwargs = {"resume": True}
                if change == "data":
                    self.dataset.write_bytes(original_data + b"\n")
                elif change == "config":
                    self.cfg["retrieval"]["fusion"]["top_k"] = 18
                elif change == "selection":
                    kwargs["limit"] = 1
                elif change == "snapshot":
                    snapshot.write_bytes(original_snapshot + b"\n")
                expected_cfg = copy.deepcopy(self.cfg)
                with self.harness() as fake:
                    if change == "code":
                        fake.ablation__code_hash.return_value = "code-v2"
                    elif change == "indexes":
                        fake.ablation_index_manifests.return_value = {"changed": True}
                    elif change == "input_files":
                        fake.ablation__input_file_metadata.return_value = {"changed": True}
                    elif change == "runtime":
                        fake.ablation_environment.return_value = {"python": "different"}
                    with self.assertRaisesRegex(ValueError, "fingerprint|snapshot"):
                        self.run_matrix(**kwargs)
                    fake.retrieval_cell.assert_not_called()
                    fake.review_cell.assert_not_called()
                    fake.retrieve.assert_not_called()
                    fake.run_query.assert_not_called()
                    fake.seed.assert_not_called()
                self.assertEqual(self.cfg, expected_cfg)

    def test_dry_run_has_no_runtime_index_model_or_filesystem_side_effects(self) -> None:
        with self.harness() as fake:
            plan = self.run_matrix(stage="all", dry_run=True)
            for name in ("ablation_fresh", "ablation_environment", "ablation_index_manifests", "ablation__code_hash",
                         "ablation__input_file_metadata", "seed", "retrieval_cell", "review_cell", "retrieve",
                         "run_query", "get_client"):
                getattr(fake, name).assert_not_called()
        self.assertEqual((plan["status"], plan["query_executions"], plan["review_query_executions"]), ("planned", 32, 16))
        self.assertEqual(plan["stages"], ["retrieval", "review"])
        self.assertFalse(self.results.exists())
        self.assertEqual(self.cfg, self.original_cfg)

    def test_factor_and_stage_preflight_failures_create_no_output(self) -> None:
        cases = [
            {"factors": []}, {"factors": "rerank"}, {"factors": ["rerank", "rerank"]},
            {"factors": ["rerank", "diversity", "query_views"]}, {"factors": ["unknown"]},
            {"factors": ["chunking"]}, {"factors": ["filters"]}, {"factors": ["jev"]},
            {"factors": ["hyde"], "stage": "retrieval"}, {"factors": ["loop"], "stage": "all"},
            {"factors": ["parent_doc"], "stage": "retrieval"}, {"stage": "generation"},
            {"tag": "../outside"}, {"split": "test"}, {"limit": 0}, {"limit": True},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.harness() as fake:
                with self.assertRaises(ValueError):
                    self.run_matrix(**kwargs)
                fake.ablation_fresh.assert_not_called()
                fake.retrieval_cell.assert_not_called()
                fake.review_cell.assert_not_called()
        self.assertFalse(self.results.exists())
        self.assertEqual(self.cfg, self.original_cfg)

    def test_backend_weight_loop_and_switch_guards_run_before_runtime(self) -> None:
        cases = [
            ("both channels disabled", ["rerank"], "retrieval", {"retrieval.sparse.enabled": False, "retrieval.dense.enabled": False}),
            ("Jev", ["rerank"], "review", {"orchestration.review.backend": "jev"}),
            ("non SQLite", ["field_weighting"], "retrieval", {"retrieval.sparse.backend": "rankbm25"}),
            ("disabled sparse", ["field_weighting"], "retrieval", {"retrieval.sparse.enabled": False}),
            ("equal weights", ["field_weighting"], "retrieval", {"retrieval.sparse.field_weighting.weights": [1, 1, 1]}),
            ("bad weights", ["field_weighting"], "retrieval", {"retrieval.sparse.field_weighting.weights": [3, 2, 0]}),
            ("nonfinite weights", ["field_weighting"], "retrieval", {"retrieval.sparse.field_weighting.weights": [3, 2, float("nan")]}),
            ("one round", ["loop"], "review", {"orchestration.loop.max_rounds": 1}),
            ("no rewrites", ["loop"], "review", {"orchestration.rewrite.max_queries_per_round": 0}),
            ("nonboolean switch", ["rerank"], "retrieval", {"retrieval.rerank.enabled": 1}),
            ("bad seed", ["rerank"], "retrieval", {"seed": True}),
        ]
        for name, factors, stage, overrides in cases:
            self.cfg = copy.deepcopy(self.original_cfg)
            for dotted, value in overrides.items():
                *parents, leaf = dotted.split(".")
                node = self.cfg
                for part in parents:
                    node = node[part]
                node[leaf] = value
            expected = copy.deepcopy(self.cfg)
            with self.subTest(name=name), self.harness() as fake:
                with self.assertRaises(ValueError):
                    self.run_matrix(factors=factors, stage=stage)
                fake.ablation_fresh.assert_not_called()
                fake.retrieval_cell.assert_not_called()
                fake.review_cell.assert_not_called()
            self.assertEqual(self.cfg, expected)
        self.assertFalse(self.results.exists())

    def test_review_loop_factor_uses_real_trace_runner_and_separates_rounds(self) -> None:
        with self.harness() as fake:
            report = self.run_matrix(factors=["loop"], stage="review", split="dev")
            fake.retrieve.assert_not_called()
            self.assertEqual(fake.run_query.call_count, 4)
            self.assertEqual(fake.get_client.call_count, 2)
            self.assertEqual([e[3]["orchestration"]["loop"]["enabled"] for e in fake.events], [False, False, True, True])
            self.assertEqual([e[2] for e in fake.events], ["D1", "D2", "D1", "D2"])
        self.assertEqual(self.cfg, self.original_cfg)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["usage"], {"calls": 4, "prompt_tokens": 44, "completion_tokens": 28, "total_tokens": 72})
        edge = report["comparisons"]["review"][0]["splits"]["dev"]
        self.assertEqual(edge["fixed_query_ids"], ["D1", "D2"])
        self.assertEqual(edge["metrics"]["mean_rounds"], {"n": 2, "before": 1.0, "after": 2.0, "delta": 1.0})
        self.assertEqual(edge["metrics"]["source_success_rate"]["delta"], 1.0)
        with self.harness() as fake:
            resumed = self.run_matrix(factors=["loop"], stage="review", split="dev", resume=True)
            fake.run_query.assert_not_called()
            fake.get_client.assert_not_called()
        self.assertEqual(resumed["usage"], report["usage"])

    def test_all_stage_runs_retrieval_then_review_with_separate_artifacts_per_row(self) -> None:
        with self.harness() as fake:
            report = self.run_matrix(factors=["rerank"], stage="all", limit=1)
            self.assertEqual([(e[0], e[1][0], e[2]) for e in fake.events],
                             [("retrieval", False, "D1"), ("review", False, "D1"),
                              ("retrieval", True, "D1"), ("review", True, "D1")])
            self.assertEqual(fake.seed.call_args_list, [call(7)] * 4)
        self.assertEqual((report["selected_cells"], report["completed_cells"]), (4, 4))
        self.assertEqual(report["usage"]["total_tokens"], 36)
        for row in ("baseline", "a"):
            for stage in ("retrieval", "review"):
                directory = self.results / "matrix" / "rows" / row / stage
                self.assertTrue((directory / "manifest.json").is_file())
                self.assertTrue((directory / "report.json").is_file())
                self.assertEqual(len(list((directory / "attempts").glob("*.json"))), 1)
        self.assertEqual(self.cfg, self.original_cfg)

    def test_corrupt_future_review_cell_blocks_resume_before_any_new_query(self) -> None:
        with self.harness():
            self.run_matrix(factors=["loop"], stage="review", split="dev")
        rows = self.results / "matrix" / "rows"
        # Make an earlier cell incomplete, while a later cell has invalid saved usage.
        (rows / "baseline" / "review" / "attempts" / "000002.json").unlink()
        path = rows / "a" / "review" / "attempts" / "000001.json"
        original = json.loads(path.read_text())
        for value in (-1, "invalid", float("nan")):
            with self.subTest(value=value):
                bad = copy.deepcopy(original)
                bad["usage"]["prompt_tokens"] = value
                path.write_text(json.dumps(bad))
                with self.harness() as fake:
                    with self.assertRaisesRegex(ValueError, "usage"):
                        self.run_matrix(factors=["loop"], stage="review", split="dev", resume=True)
                    fake.run_query.assert_not_called()
                    fake.get_client.assert_not_called()
                    fake.retrieval_cell.assert_not_called()
                    fake.review_cell.assert_not_called()

    def test_index_change_between_cells_stops_before_next_query(self) -> None:
        with self.harness() as fake:
            fake.ablation__input_file_metadata.side_effect = [{}, {}, {"changed": True}]
            report = self.run_matrix(factors=["rerank"], limit=1)
            self.assertEqual(fake.retrieve.call_count, 1)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["completed_cells"], 1)
        self.assertEqual(report["error"]["type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
