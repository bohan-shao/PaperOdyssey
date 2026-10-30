"""Offline tests for eval/retrieval_eval.py and scripts/run_eval.py (no models, no indexes).

Run: python -m unittest tests.test_retrieval_eval

retrieve() is a fake whose hits depend on the live sparse/dense/rerank switches; expected metric
values are worked out by hand in the comments.
"""

from __future__ import annotations

import contextlib
import copy
import datetime
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from config import PROJECT_ROOT
from eval import retrieval_eval
from eval.retrieval_metrics import compute_retrieval_metrics
from retrieval.types import RetrieveResult, ScoredPassage

QUESTIONS = [
    {"query_id": "Q1", "input_text": "first question", "gold_paper_ids": ["A"], "split": "eval",
     "question_type": "technical-detail", "question_form": "question"},
    {"query_id": "Q2", "input_text": "second question", "gold_paper_ids": ["B"], "split": "eval",
     "question_type": "paper-locating", "question_form": "request"},
    {"query_id": "Q3", "input_text": "third question", "gold_paper_ids": ["C"], "split": "eval",
     "question_type": "technical-detail", "question_form": "question"},
    {"query_id": "Q4", "input_text": "fourth question", "gold_paper_ids": ["D"], "split": "dev",
     "question_type": "survey", "question_form": "request"},
    {"query_id": "Q5", "input_text": "fifth question", "gold_paper_ids": ["E"], "split": "eval",
     "question_type": "paper-locating", "question_form": None},  # a rewritten question has no form
]
# The papers of each returned passage, best first; X, Y, Z are not gold. bm25 returns two passages
# of A for Q1, so its paper ranking is A, X, Y. Gold positions (eval split Q1 Q2 Q3 Q5 | dev Q4):
#   bm25 1 - 2 3 | 1    dense 2 1 - - | 2    hybrid_rrf 1 2 3 3 | 1    hybrid_rerank 1 1 - 1 | 1
RANKINGS = {
    "bm25": {"Q1": "AAXY", "Q2": "XYZ", "Q3": "XC", "Q4": "D", "Q5": "XYE"},
    "dense": {"Q1": "XA", "Q2": "B", "Q3": "", "Q4": "XD", "Q5": "X"},
    "hybrid_rrf": {"Q1": "A", "Q2": "XB", "Q3": "XYC", "Q4": "D", "Q5": "XYE"},
    "hybrid_rerank": {"Q1": "A", "Q2": "B", "Q3": "XYZ", "Q4": "D", "Q5": "E"},
}
ROWS = {(True, False, False): "bm25", (False, True, False): "dense",
        (True, True, False): "hybrid_rrf", (True, True, True): "hybrid_rerank"}


def retrieval_section() -> dict:
    # dense and rerank start off: every row must set all three switches itself
    return {"sparse": {"enabled": True, "top_k": 150}, "dense": {"enabled": False, "top_k": 150},
            "fusion": {"top_k": 50}, "rerank": {"enabled": False, "top_k": 10},
            "diversity": {"enabled": True}, "parent_doc": {"enabled": False}, "filters": {"enabled": True}}


class FakeRetrieve:
    """Stand-in for retrieve(): hits follow RANKINGS for the configuration the switches select."""

    def __init__(self, section: dict, fail_on: tuple[str, str] | None = None) -> None:
        self.section, self.fail_on = section, fail_on
        self.calls: list[tuple[str, str, dict]] = []

    def __call__(self, query: str, filters: object = None, top_k: int | None = None) -> RetrieveResult:
        switches = tuple(self.section[s]["enabled"] for s in ("sparse", "dense", "rerank"))
        row = ROWS[switches]
        self.calls.append((query, row, copy.deepcopy(self.section)))
        if filters is not None or top_k is not None:
            raise AssertionError("the eval retrieves with the configured defaults")
        if (row, query) == self.fail_on:
            raise RuntimeError("dense retrieval failed: boom")
        if query == "warm up":
            return RetrieveResult(hits=[], total_candidates=0, latency_ms={"total_ms": 900})
        qid = next(q["query_id"] for q in QUESTIONS if q["input_text"] == query)
        hits = [ScoredPassage(passage_id=f"{paper}_{i:04d}", paper_id=paper, text="t", score=1.0 / i, rank=i,
                              stage="rerank" if switches[2] else "hybrid")
                for i, paper in enumerate(RANKINGS[row][qid], start=1)]
        ms = {"sparse_ms": 4} if switches[0] else {}
        if switches[1]:
            ms["dense_ms"] = 12
        if switches[2]:
            ms["rerank_ms"] = 390
        ms["total_ms"] = sum(ms.values()) + 1
        return RetrieveResult(hits=hits, total_candidates=50, latency_ms=ms)


def write_eval_set(path: Path, questions: list[dict] = QUESTIONS, blank_lines: bool = False) -> Path:
    sep = "\n\n" if blank_lines else "\n"
    path.write_text(sep.join(json.dumps(q) for q in questions) + "\n", encoding="utf-8")
    return path


class EvalRunTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.eval_set = write_eval_set(self.dir / "eval_set.jsonl")
        self.results = self.dir / "retrieval_results"
        self.section = retrieval_section()
        self.cfg = {"seed": 7, "paths": {"eval": "e", "results": "r"}, "retrieval": self.section}

    def run_eval(self, fake: FakeRetrieve, **kwargs: object) -> tuple[dict, Mock, Mock]:
        with patch.object(retrieval_eval, "retrieve", fake), \
                patch.object(retrieval_eval, "get_config", return_value=self.cfg), \
                patch.object(retrieval_eval, "require_fresh_runtime") as fresh, \
                patch.object(retrieval_eval, "clear_runtime") as clear, \
                patch.object(retrieval_eval, "environment", return_value={"python": "test"}), \
                patch.object(retrieval_eval, "index_manifests", return_value={}):
            report = retrieval_eval.run_retrieval_eval(self.eval_set, self.results, "test", **kwargs)
        return report, fresh, clear

    def test_each_row_sets_its_switches_and_the_config_is_restored(self) -> None:
        fake = FakeRetrieve(self.section)
        report, fresh, clear = self.run_eval(fake)
        fresh.assert_called_once_with()
        clear.assert_called_once_with(release_models=True)
        self.assertEqual(self.section, retrieval_section())  # restored, also the switches set off before
        texts = ["warm up"] + [q["input_text"] for q in QUESTIONS]  # one warm-up request per row
        self.assertEqual([(q, row) for q, row, _ in fake.calls],
                         [(t, row) for row in RANKINGS for t in texts])
        self.assertTrue(all(live["diversity"] == {"enabled": True} and live["parent_doc"] == {"enabled": False}
                            for _, _, live in fake.calls))  # everything else stays as configured
        self.assertEqual(list(report["configs"]), list(RANKINGS))
        row = report["configs"]["hybrid_rerank"]
        self.assertEqual(row["overrides"], {"sparse.enabled": True, "dense.enabled": True, "rerank.enabled": True})
        self.assertTrue(row["effective_config"]["rerank"]["enabled"])
        self.assertEqual(report["config_snapshot"]["retrieval"], retrieval_section())  # as requested
        self.assertEqual(report["config_snapshot"]["seed"], 7)

    def test_hits_files_follow_schema_section_8(self) -> None:
        report, _, _ = self.run_eval(FakeRetrieve(self.section))
        self.assertEqual(sorted(p.name for p in self.results.iterdir()),
                         sorted(f"{name}_test.jsonl" for name in RANKINGS))  # no temporary files left
        lines = (self.results / "bm25_test.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 5)
        self.assertEqual(json.loads(lines[0]), {
            "query_id": "Q1", "sub_query": "first question", "stage": "bm25", "top_k": 10,
            "hits": [{"passage_id": "A_0001", "score": 1.0, "rank": 1, "paper_id": "A"},
                     {"passage_id": "A_0002", "score": 0.5, "rank": 2, "paper_id": "A"},
                     {"passage_id": "X_0003", "score": 1 / 3, "rank": 3, "paper_id": "X"},
                     {"passage_id": "Y_0004", "score": 0.25, "rank": 4, "paper_id": "Y"}],
            "latency_ms": 5})
        self.assertEqual(json.loads(lines[2])["hits"][1]["paper_id"], "C")
        dense_q3 = json.loads((self.results / "dense_test.jsonl").read_text(encoding="utf-8").splitlines()[2])
        self.assertEqual((dense_q3["hits"], dense_q3["latency_ms"]), ([], 13))
        self.assertTrue(report["configs"]["bm25"]["results_file"].endswith("bm25_test.jsonl"))

    def test_hand_computed_metrics_per_split(self) -> None:
        report, _, _ = self.run_eval(FakeRetrieve(self.section))
        bm25 = report["configs"]["bm25"]["metrics"]
        # eval split, bm25 gold positions 1, -, 2, 3
        self.assertEqual(bm25["eval"]["n_queries"], 4)
        self.assertEqual((bm25["eval"]["Hit@1"], bm25["eval"]["Hit@3"], bm25["eval"]["Hit@10"]), (0.25, 0.75, 0.75))
        self.assertAlmostEqual(bm25["eval"]["P@3"], 0.25)  # (1/3 + 0 + 1/3 + 1/3) / 4
        self.assertAlmostEqual(bm25["eval"]["MRR@10"], (1 + 0 + 1 / 2 + 1 / 3) / 4)
        # NDCG with one gold paper = 1 / log2(position + 1): (1 + 0 + 0.6309298 + 0.5) / 4
        self.assertAlmostEqual(bm25["eval"]["NDCG@10"], 0.5327324, places=6)
        self.assertEqual(bm25["eval"]["per_granularity"]["passage"]["n_queries"], 0)  # no passage gold
        self.assertNotIn("per_query", bm25["eval"])
        self.assertEqual((bm25["dev"]["n_queries"], bm25["dev"]["Hit@1"]), (1, 1.0))
        dense = report["configs"]["dense"]["metrics"]
        self.assertEqual((dense["dev"]["Hit@1"], dense["dev"]["Hit@3"], dense["dev"]["MRR@10"]), (0.0, 1.0, 0.5))
        # every split's numbers are compute_retrieval_metrics on that split's questions alone
        hits = {r["query_id"]: r["hits"] for r in map(json.loads, (self.results / "hybrid_rrf_test.jsonl")
                                                      .read_text(encoding="utf-8").splitlines())}
        expected = compute_retrieval_metrics(hits, {q["query_id"]: q for q in QUESTIONS if q["split"] == "eval"})
        expected.pop("per_query")
        self.assertEqual(report["configs"]["hybrid_rrf"]["metrics"]["eval"], expected)

    def test_breakdowns_ranks_and_run_statistics(self) -> None:
        report, _, _ = self.run_eval(FakeRetrieve(self.section))
        rerank = report["configs"]["hybrid_rerank"]
        by_type = rerank["by_question_type"]["eval"]
        self.assertEqual(list(by_type), ["paper-locating", "technical-detail"])
        self.assertEqual((by_type["paper-locating"]["n_queries"], by_type["paper-locating"]["Hit@1"]), (2, 1.0))
        self.assertEqual(by_type["technical-detail"]["Hit@1"], 0.5)  # Q1 found, Q3 missed
        self.assertEqual({k: v["n_queries"] for k, v in rerank["by_question_form"]["eval"].items()},
                         {"question": 2, "request": 1, "unknown": 1})
        self.assertEqual(list(rerank["by_question_type"]["dev"]), ["survey"])
        self.assertEqual(report["per_query"]["Q3"], {
            "split": "eval", "question_type": "technical-detail",
            "first_gold_rank": {"bm25": 2, "dense": None, "hybrid_rrf": 3, "hybrid_rerank": None}})
        bm25 = report["configs"]["bm25"]
        self.assertEqual((bm25["mean_hits"], bm25["mean_distinct_papers"]), (2.6, 2.4))  # 13 and 12 over 5
        self.assertEqual(report["configs"]["dense"]["questions_without_hits"], 1)
        self.assertEqual(bm25["latency_ms"]["total"]["p50"], 5)  # the warm-up request is not counted
        self.assertEqual(set(rerank["latency_ms"]), {"sparse", "dense", "rerank", "total"})
        self.assertEqual(report["eval_set"]["n_by_split"], {"dev": 1, "eval": 4})
        self.assertEqual(len(report["eval_set"]["sha256"]), 64)

    def test_comparisons_list_failed_to_succeeded_questions(self) -> None:
        report, _, _ = self.run_eval(FakeRetrieve(self.section))
        pairs = {(c["before"], c["after"]): c["splits"]["eval"] for c in report["comparisons"]}
        self.assertEqual(list(pairs), list(retrieval_eval.COMPARISONS))
        rerank = pairs[("hybrid_rrf", "hybrid_rerank")]  # rrf 1 2 3 3 -> rerank 1 1 - 1
        self.assertEqual({k: rerank[k] for k in ("n_queries", "fixed", "broken", "moved_up", "moved_down")},
                         {"n_queries": 4, "fixed": 0, "broken": 1, "moved_up": 2, "moved_down": 1})
        self.assertEqual(rerank["broken_examples"], [
            {"query_id": "Q3", "input_text": "third question", "rank_before": 3, "rank_after": None}])
        fusion = pairs[("dense", "hybrid_rrf")]  # dense 2 1 - - -> rrf 1 2 3 3
        self.assertEqual([e["query_id"] for e in fusion["fixed_examples"]], ["Q3", "Q5"])
        self.assertEqual((fusion["broken"], fusion["moved_up"], fusion["moved_down"]), (0, 3, 1))
        self.assertEqual(pairs[("bm25", "hybrid_rrf")]["fixed"], 1)  # Q2
        text = retrieval_eval.format_summary(report)
        for expected in ("5 questions (dev 1, eval 4), paper level", "split eval (n=4)", "NDCG@10",
                         "hybrid_rrf -> hybrid_rerank: fixed 0, broken 1; gold paper up 2, down 1 (of 4)"):
            self.assertIn(expected, text)

    def test_subset_of_configurations_and_limit(self) -> None:
        report, _, _ = self.run_eval(FakeRetrieve(self.section), configs=["hybrid_rerank", "hybrid_rrf"], limit=2)
        self.assertEqual(list(report["configs"]), ["hybrid_rerank", "hybrid_rrf"])
        self.assertEqual(len(report["comparisons"]), 1)  # only the pair that ran
        self.assertEqual((report["eval_set"]["n_questions"], report["eval_set"]["limit"]), (2, 2))
        self.assertEqual(report["configs"]["hybrid_rrf"]["metrics"]["dev"]["n_queries"], 0)
        self.assertIsNone(report["configs"]["hybrid_rrf"]["metrics"]["dev"]["Hit@1"])
        self.assertIn("first 2 only", retrieval_eval.format_summary(report))

    def test_a_retrieval_failure_stops_the_run_and_releases_components(self) -> None:
        fake = FakeRetrieve(self.section, fail_on=("dense", "third question"))
        with self.assertRaisesRegex(RuntimeError, "dense retrieval failed"):
            self.run_eval(fake)
        self.assertEqual(self.section, retrieval_section())
        self.assertEqual(sorted(p.name for p in self.results.iterdir()), ["bm25_test.jsonl"])  # finished rows kept

    def test_bad_arguments_are_rejected_before_retrieval(self) -> None:
        fake = FakeRetrieve(self.section)
        for kwargs in ({"configs": ["bm25", "bm25"]}, {"configs": ["rerank_only"]}, {"configs": []}, {"limit": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_eval(fake, **kwargs)
        with patch.object(retrieval_eval, "retrieve", fake), self.assertRaisesRegex(ValueError, "tag"):
            retrieval_eval.run_retrieval_eval(self.eval_set, self.results, "../escape")
        self.assertEqual(fake.calls, [])


class LoadEvalSetTest(unittest.TestCase):
    def test_reads_records_in_order_and_skips_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_eval_set(Path(tmp) / "eval_set.jsonl", blank_lines=True)
            self.assertEqual(retrieval_eval.load_eval_set(path), QUESTIONS)

    def test_broken_records_name_their_line(self) -> None:
        good = QUESTIONS[0]
        cases = {
            "not JSON": "{oops",
            "not a JSON object": "[1, 2]",
            "gold_paper_ids is missing": json.dumps({k: v for k, v in good.items() if k != "gold_paper_ids"}),
            "input_text is missing or not a str": json.dumps({**good, "input_text": 5}),
            "empty input_text": json.dumps({**good, "input_text": "  "}),
            "empty input_text or gold_paper_ids": json.dumps({**good, "gold_paper_ids": []}),
            "split must be one of": json.dumps({**good, "split": "test"}),
            "repeated query_id 'Q1'": json.dumps(good),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval_set.jsonl"
            for message, line in cases.items():
                with self.subTest(message=message):
                    path.write_text(json.dumps(good) + "\n" + line + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, f":2: .*{message}"):
                        retrieval_eval.load_eval_set(path)


def load_script() -> object:
    spec = importlib.util.spec_from_file_location("run_eval", PROJECT_ROOT / "scripts" / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunEvalScriptTest(unittest.TestCase):
    def test_retrieval_stage_writes_the_report_under_results_metrics(self) -> None:
        run_eval = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            paths = {"eval": Path(tmp) / "eval", "results": Path(tmp) / "results"}
            fake_run = Mock(return_value={"stage": "retrieval"})
            with patch.object(run_eval, "get_path", side_effect=lambda key: paths[key]), \
                    patch.object(run_eval, "run_retrieval_eval", fake_run), \
                    patch.object(run_eval, "run_review_eval") as review_run, \
                    patch.object(run_eval, "format_summary", return_value="summary"), \
                    patch.object(run_eval, "date") as fake_date, \
                    patch.object(run_eval.logging, "basicConfig"), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                fake_date.today.return_value = datetime.date(2026, 10, 30)
                run_eval.main(["--config", str(PROJECT_ROOT / "config.yaml"), "--stage", "retrieval",
                               "--configs", "bm25", "dense", "--limit", "3"])
                fake_run.assert_called_once_with(paths["eval"] / "eval_set.jsonl", paths["results"] / "retrieval_results",
                                                 "2026-10-30_first3", ["bm25", "dense"], 3)
                report = paths["results"] / "metrics" / "retrieval_2026-10-30_first3.json"
                self.assertEqual(json.loads(report.read_text(encoding="utf-8")), {"stage": "retrieval"})
                self.assertIn(f"summary\n\nreport -> {report}", out.getvalue())
                review_run.assert_not_called()

    def test_review_stage_passes_selection_and_resume_without_running_retrieval(self) -> None:
        run_eval = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            paths = {"eval": Path(tmp) / "eval", "results": Path(tmp) / "results"}
            report = {"status": "complete", "completed_queries": 3, "selected_queries": 3}
            with patch.object(run_eval, "get_path", side_effect=lambda key: paths[key]), \
                    patch.object(run_eval, "get_config", return_value={
                        "eval": {"annotation": {"eval_set_file": "fixed.jsonl"}}}), \
                    patch.object(run_eval, "run_retrieval_eval") as retrieval_run, \
                    patch.object(run_eval, "run_review_eval", return_value=report) as review_run, \
                    patch.object(run_eval, "format_review_summary", return_value="review complete 3/3") as summary, \
                    patch.object(run_eval.logging, "basicConfig"), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                run_eval.main(["--stage", "review", "--split", "eval", "--limit", "3", "--tag", "review-eval",
                               "--resume"])
            retrieval_run.assert_not_called()
            review_run.assert_called_once_with(paths["eval"] / "fixed.jsonl", paths["results"] / "review_runs",
                                               "review-eval", limit=3, split="eval", resume=True, workers=1)
            summary.assert_called_once_with(report)
            self.assertIn("review complete 3/3", out.getvalue())
            self.assertIn(str(paths["results"] / "review_runs" / "review-eval" / "report.json"), out.getvalue())
            self.assertFalse((paths["results"] / "metrics").exists())  # the runner owns its review artifacts

    def test_all_defaults_and_explicit_stage_run_retrieval_before_review(self) -> None:
        run_eval = load_script()
        for argv, split, resume in (([], "all", False),
                                    (["--stage", "all", "--split", "dev", "--resume"], "dev", True)):
            with self.subTest(argv=argv), tempfile.TemporaryDirectory() as tmp:
                paths = {"eval": Path(tmp) / "eval", "results": Path(tmp) / "results"}
                stages = []
                with patch.object(run_eval, "get_path", side_effect=lambda key: paths[key]), \
                        patch.object(run_eval, "get_config", return_value={
                            "eval": {"annotation": {"eval_set_file": "eval_set.jsonl"}}}), \
                        patch.object(run_eval, "date") as fake_date, \
                        patch.object(run_eval, "run_retrieval_eval",
                                     side_effect=lambda *args: stages.append("retrieval") or {}) as retrieval_run, \
                        patch.object(run_eval, "run_review_eval", side_effect=lambda *args, **kwargs:
                                     stages.append("review") or {"status": "complete"}) as review_run, \
                        patch.object(run_eval, "format_summary", return_value="retrieval summary"), \
                        patch.object(run_eval, "format_review_summary", return_value="review summary"), \
                        patch.object(run_eval.logging, "basicConfig"), \
                        contextlib.redirect_stdout(io.StringIO()) as out:
                    fake_date.today.return_value = datetime.date(2026, 10, 30)
                    run_eval.main(argv)
                self.assertEqual(stages, ["retrieval", "review"])
                retrieval_run.assert_called_once_with(paths["eval"] / "eval_set.jsonl",
                                                     paths["results"] / "retrieval_results", "2026-10-30",
                                                     list(run_eval.CONFIGS), None)
                review_run.assert_called_once_with(paths["eval"] / "eval_set.jsonl",
                                                   paths["results"] / "review_runs", "2026-10-30",
                                                   limit=None, split=split, resume=resume, workers=1)
                self.assertLess(out.getvalue().index("retrieval summary"), out.getvalue().index("review summary"))

    def test_incomplete_review_reports_summary_on_stderr_and_exits_one(self) -> None:
        run_eval = load_script()
        for status, completed in (("partial", 1), ("failed", 0)):
            report = {"status": status, "completed_queries": completed, "selected_queries": 3,
                      "error": "review service unavailable"}
            with self.subTest(status=status), \
                    patch.object(run_eval, "get_config", return_value={
                        "eval": {"annotation": {"eval_set_file": "eval_set.jsonl"}}}), \
                    patch.object(run_eval, "get_path", side_effect=lambda key: Path("/unused") / key), \
                    patch.object(run_eval, "run_retrieval_eval") as retrieval_run, \
                    patch.object(run_eval, "run_review_eval", return_value=report), \
                    patch.object(run_eval, "format_review_summary", return_value=f"{status}: {completed}/3"), \
                    patch.object(run_eval.logging, "basicConfig"), \
                    contextlib.redirect_stdout(io.StringIO()) as out, \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as caught:
                    run_eval.main(["--stage", "review", "--tag", "failed-run"])
                self.assertEqual(caught.exception.code, 1)
                self.assertEqual(out.getvalue(), "")
                self.assertIn(f"{status}: {completed}/3", err.getvalue())
                self.assertIn("review service unavailable", err.getvalue())
                self.assertIn("review_runs/failed-run/report.json", err.getvalue())
                retrieval_run.assert_not_called()

    def test_refused_resume_prints_error_without_claiming_a_report_exists(self) -> None:
        run_eval = load_script()
        with patch.object(run_eval, "get_config", return_value={
                "eval": {"annotation": {"eval_set_file": "eval_set.jsonl"}}}), \
                patch.object(run_eval, "get_path", side_effect=lambda key: Path("/unused") / key), \
                patch.object(run_eval, "run_review_eval", side_effect=ValueError("resume fingerprint differs")), \
                patch.object(run_eval, "format_review_summary") as summary, \
                patch.object(run_eval.logging, "basicConfig"), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as caught:
                run_eval.main(["--stage", "review", "--tag", "resume-run", "--resume"])
            self.assertEqual(caught.exception.code, 1)
            self.assertIn("resume fingerprint differs", err.getvalue())
            self.assertNotIn("report ->", err.getvalue())
            summary.assert_not_called()

    def test_workers_reach_review_runs_and_review_ablation_cells(self) -> None:
        run_eval = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            paths = {"eval": Path(tmp) / "eval", "results": Path(tmp) / "results"}
            with patch.object(run_eval, "get_path", side_effect=lambda key: paths[key]), \
                    patch.object(run_eval, "get_config", return_value={
                        "eval": {"annotation": {"eval_set_file": "eval_set.jsonl"}}}), \
                    patch.object(run_eval, "run_review_eval", return_value={"status": "complete"}) as review_run, \
                    patch.object(run_eval, "run_ablation", return_value={"status": "complete"}) as ablation_run, \
                    patch.object(run_eval, "format_review_summary", return_value="review"), \
                    patch.object(run_eval, "format_ablation_summary", return_value="ablation"), \
                    patch.object(run_eval.logging, "basicConfig"), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_eval.main(["--stage", "review", "--tag", "fast", "--workers", "8"])
                run_eval.main(["--stage", "review", "--ablation", "loop", "hyde", "--tag", "matrix", "--workers", "4"])
            self.assertEqual(review_run.call_args.kwargs["workers"], 8)
            self.assertEqual(ablation_run.call_args.kwargs["workers"], 4)

    def test_bad_flags_stop_before_any_stage(self) -> None:
        run_eval = load_script()
        for argv in (["--stage", "retrieval", "--ablation", "rerank_off"], ["--ablation", "loop"],
                     ["--stage", "retrieval", "--ablation", "rerank", "--configs", "hybrid_rerank"],
                     ["--stage", "retrieval", "--dry-run"], ["--stage", "review", "--ablation", "chunking"],
                     ["--stage", "retrieval", "--limit", "0"], ["--stage", "retrieval", "--configs", "colbert"],
                     ["--stage", "retrieval", "--split", "all"], ["--stage", "retrieval", "--resume"],
                     ["--stage", "review", "--configs", "bm25"], ["--stage", "review", "--split", "test"],
                     ["--stage", "retrieval", "--workers", "2"], ["--stage", "review", "--workers", "0"],
                     ["--stage", "review", "--workers", "33"],
                     ["--stage", "retrieval", "--ablation", "rerank", "--workers", "4"]):
            with self.subTest(argv=argv), patch.object(run_eval, "run_retrieval_eval") as retrieval_run, \
                    patch.object(run_eval, "run_review_eval") as review_run, \
                    patch.object(run_eval, "run_ablation") as ablation_run, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    run_eval.main(argv)
                self.assertEqual(caught.exception.code, 2)
                retrieval_run.assert_not_called()
                review_run.assert_not_called()
                ablation_run.assert_not_called()


class AblationScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run_eval = load_script()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.paths = {"eval": Path(temporary.name) / "eval", "results": Path(temporary.name) / "results"}
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(self.run_eval, "get_config", return_value={
            "eval": {"annotation": {"eval_set_file": "eval_set.jsonl"}}}))
        stack.enter_context(patch.object(self.run_eval, "get_path", side_effect=lambda key: self.paths[key]))
        stack.enter_context(patch.object(self.run_eval.logging, "basicConfig"))
        self.ablation = stack.enter_context(patch.object(self.run_eval, "run_ablation"))
        self.retrieval = stack.enter_context(patch.object(self.run_eval, "run_retrieval_eval"))
        self.review = stack.enter_context(patch.object(self.run_eval, "run_review_eval"))
        self.summary = stack.enter_context(patch.object(self.run_eval, "format_ablation_summary",
                                                        side_effect=lambda r: f"ablation {r['status']}"))

    def test_retrieval_ablation_forwards_selection_resume_and_publishes_each_status(self) -> None:
        out_path = self.paths["results"] / "metrics" / "ablation_query_views_diversity_comparison.json"
        for status in ("complete", "partial", "failed"):
            with self.subTest(status=status):
                report = {"stage": "ablation", "status": status, "factors": ["query_views", "diversity"],
                          "rows": {"baseline": {"status": "complete"}}}
                self.ablation.reset_mock(return_value=True)
                self.ablation.return_value = report
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text('{"previous": true}\n', encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()) as out, \
                        contextlib.redirect_stderr(io.StringIO()) as err:
                    argv = ["--stage", "retrieval", "--ablation", "query_views", "diversity", "--split", "dev",
                            "--limit", "3", "--tag", "comparison", "--resume"]
                    if status == "complete":
                        self.run_eval.main(argv)
                    else:
                        with self.assertRaises(SystemExit) as caught:
                            self.run_eval.main(argv)
                        self.assertEqual(caught.exception.code, 1)
                self.ablation.assert_called_once_with(
                    self.paths["eval"] / "eval_set.jsonl", self.paths["results"] / "ablation_runs", "comparison",
                    factors=["query_views", "diversity"], stage="retrieval", split="dev", limit=3,
                    resume=True, dry_run=False, workers=1)
                self.assertEqual(json.loads(out_path.read_text(encoding="utf-8")), report)
                self.assertEqual(list(out_path.parent.iterdir()), [out_path])  # no temporary output left
                output = out.getvalue() if status == "complete" else err.getvalue()
                self.assertIn(f"ablation {status}", output)
                self.assertIn(str(out_path), output)
        self.retrieval.assert_not_called()
        self.review.assert_not_called()

    def test_dry_run_only_prints_the_plan(self) -> None:
        self.ablation.return_value = {"stage": "ablation", "status": "planned",
                                      "factors": ["loop", "hyde"], "rows": {}}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.run_eval.main(["--stage", "review", "--ablation", "loop", "hyde", "--tag", "plan", "--dry-run"])
        self.ablation.assert_called_once_with(
            self.paths["eval"] / "eval_set.jsonl", self.paths["results"] / "ablation_runs", "plan",
            factors=["loop", "hyde"], stage="review", split="all", limit=None, resume=False, dry_run=True, workers=1)
        self.assertIn("ablation planned", out.getvalue())
        self.assertNotIn("report ->", out.getvalue())
        self.assertFalse(self.paths["results"].exists())
        self.retrieval.assert_not_called()
        self.review.assert_not_called()

    def test_factor_stage_preflight_failure_is_reported_without_an_output_claim(self) -> None:
        self.ablation.side_effect = ValueError("loop supports review only; cannot use stage=all")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as caught:
                self.run_eval.main(["--stage", "all", "--ablation", "loop", "--tag", "bad-plan"])
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("loop supports review only", err.getvalue())
        self.assertNotIn("report ->", err.getvalue())
        self.assertFalse(self.paths["results"].exists())
        self.summary.assert_not_called()
        self.retrieval.assert_not_called()
        self.review.assert_not_called()

    def test_failed_atomic_publish_retains_the_previous_report(self) -> None:
        self.ablation.return_value = {"stage": "ablation", "status": "complete", "factors": ["rerank"], "rows": {}}
        out_path = self.paths["results"] / "metrics" / "ablation_rerank_saved.json"
        out_path.parent.mkdir(parents=True)
        previous = '{"previous": true}\n'
        out_path.write_text(previous, encoding="utf-8")
        with patch.object(self.run_eval.os, "replace", side_effect=OSError("cannot publish report")), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as caught:
                self.run_eval.main(["--stage", "retrieval", "--ablation", "rerank", "--tag", "saved"])
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("cannot publish report", err.getvalue())
        self.assertEqual(out_path.read_text(encoding="utf-8"), previous)
        self.assertEqual(list(out_path.parent.iterdir()), [out_path])


if __name__ == "__main__":
    unittest.main()
