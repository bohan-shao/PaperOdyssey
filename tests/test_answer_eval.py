"""Offline answer evaluation: stored searches, a real lookup store, scripted generator and judge."""

from __future__ import annotations

import copy
import json
import tempfile
import threading
import unittest
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from downstream.types import EvidenceSource
from eval import answer_eval, answer_metrics
from orchestration.types import FinalResult, QueryPlan, ReviewRecord, passing_order_key
from retrieval.lookup import LookupStore, build_lookup

QUESTIONS = [
    {"query_id": "Q1", "input_text": "How does A work?", "gold_paper_ids": ["A"], "split": "eval"},
    {"query_id": "Q2", "input_text": "What does B find?", "gold_paper_ids": ["B"], "split": "eval"},
    {"query_id": "Q3", "input_text": "Where is C?", "gold_paper_ids": ["C"], "split": "eval"},
    {"query_id": "Q4", "input_text": "Does D work?", "gold_paper_ids": ["D"], "split": "dev"},
]
PAPERS = ("A", "X", "Y", "B", "D")
# Baseline searches: Q1 passes A and X, Q2 passes only the unrelated Y, Q3 passes nothing, Q4 passes D.
PASSING = {"Q1": ["A", "X"], "Q2": ["Y"], "Q3": [], "Q4": ["D"]}
ANSWERS = {
    "How does A work?": {"status": "answered", "answer": "A uses low-rank updates [1]. X adds noise [2]."},
    "What does B find?": {"status": "insufficient_evidence", "answer": "Only related work was found [1]."},
    "Does D work?": {"status": "answered", "answer": "D works [1]."},
}
CLAIMS = {"How does A work?": ["A uses low-rank updates.", "X adds noise."], "What does B find?": [],
          "Does D work?": ["D works."]}
SUPPORTED = {"A uses low-rank updates.": True, "X adds noise.": False, "D works.": True}
GENERATED = {"How does A work?": (["same", "other", "same"], False), "What does B find?": (["same"], True),
             "Does D work?": (["same"], False)}
RELEVANT = {"A": True, "X": False, "Y": False, "B": True, "D": True}


def search_for(question: dict, passing: list[str]) -> dict:
    reviews = [ReviewRecord(p, "pass", 5 - i, evidence={"top_passages": [f"P{p}"]}) for i, p in enumerate(passing)]
    reviews.append(ReviewRecord("Z", "reject", 1))
    ranks = {r.paper_id: rank for rank, r in enumerate(reviews, 1)}
    papers = sorted((r for r in reviews if r.verdict == "pass"),
                    key=lambda r: passing_order_key(r.paper_id, r.score, ranks))
    return asdict(FinalResult(
        query_id=question["query_id"], input_text=question["input_text"], query_plan=QueryPlan(question["input_text"]),
        papers=papers, reviews=reviews, retrieval_ranks=ranks,
        coverage_report={"total_aspects": 2, "covered_aspects": 1, "rounds": 1, "papers_reviewed": len(reviews),
                         "papers_passed": len(papers), "termination_reason": "min_papers", "cache_hit": False},
        latency_by_stage={"total_ms": 10}))


def write_review_run(directory: Path, passing: dict, fingerprint: str = "search-fp", skip=()) -> None:
    """A finished review run as eval/review_eval.py leaves it; `skip` questions failed in the search."""
    _write(directory / "manifest.json", {
        "tag": directory.name, "fingerprint": fingerprint, "dataset_sha256": "data-sha",
        "selected_query_ids": [q["query_id"] for q in QUESTIONS],
        "indexes": {"dense": {"version": 1}}, "input_files": {"lookup.sqlite": {"bytes": 1}}})
    for number, question in enumerate(QUESTIONS, 1):
        attempt = {"question": question, "status": "failed" if question["query_id"] in skip else "complete",
                   "cold_start": number == 1, "usage": {"calls": 1, "prompt_tokens": 1, "completion_tokens": 1},
                   "wall_ms": 1.0}
        if attempt["status"] == "complete":
            attempt["result"] = search_for(question, passing[question["query_id"]])
        _write(directory / "attempts" / f"{number:06d}.json", attempt)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class Spender:
    """Counts usage like LLMClient._record: per-question scopes and process totals."""

    def __init__(self, prompt: int, completion: int) -> None:
        self.cost = (prompt, completion)
        self.totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.lock = threading.Lock()
        self.calls: list[str] = []

    def spend(self, kind: str) -> None:
        with self.lock:
            self.calls.append(kind)
            self.totals["calls"] += 1
            self.totals["prompt_tokens"] += self.cost[0]
            self.totals["completion_tokens"] += self.cost[1]
        answer_eval.llm_client._count_in_scope(*self.cost)

    def usage_summary(self) -> dict:
        with self.lock:
            return dict(self.totals)


class FakeGenerator(Spender):
    def __init__(self) -> None:
        super().__init__(10, 3)
        self.fail_on: set[str] = set()

    def chat_json(self, messages, schema, max_tokens=None):
        payload = json.loads(messages[1]["content"])
        self.spend(payload["question"])
        if payload["question"] in self.fail_on:
            raise RuntimeError("provider said no")
        return dict(ANSWERS[payload["question"]])


class FakeJudge(Spender):
    """Answers each judge prompt from the scripted tables; `broken` gives a reply of the wrong shape."""

    def __init__(self) -> None:
        super().__init__(5, 2)
        self.broken: set[str] = set()
        self.question_by_answer = {}

    def chat_json(self, messages, schema, max_tokens=None):
        system, payload = messages[0]["content"], json.loads(messages[1]["content"])
        if system == answer_metrics.CLAIMS_PROMPT:
            self.spend("claims")
            self.question_by_answer[payload["answer"]] = payload["question"]
            if payload["question"] in self.broken:
                return {"claims": "not a list"}
            return {"claims": CLAIMS[payload["question"]]}
        if system == answer_metrics.FAITHFULNESS_PROMPT:
            self.spend("faithfulness")
            return {"verdicts": [{"claim": c["claim"], "supported": SUPPORTED[c["text"]], "reason": "r"}
                                 for c in payload["claims"]]}
        if system == answer_metrics.CONTEXT_PROMPT:
            self.spend("context")
            return {"verdicts": [{"source": e["number"], "relevant": RELEVANT[e["title"]], "reason": "r"}
                                 for e in payload["paper_excerpts"]]}
        self.spend("relevance")
        questions, noncommittal = GENERATED[self.question_by_answer[payload["answer"]]]
        return {"questions": questions, "noncommittal": noncommittal}


class FakeEncoder:
    """Questions and "same" map to one direction, "other" to an orthogonal one."""

    def encode(self, texts):
        return [[0.0, 1.0] if t == "other" else [1.0, 0.0] for t in texts]


class AnswerEvalTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        passages = [{"passage_id": f"P{p}", "paper_id": p, "text": f"Excerpt of {p}.", "arxiv_id": p,
                     "title": p, "section": "Method", "year": 2024} for p in PAPERS]
        (self.root / "passages.jsonl").write_text("".join(json.dumps(p) + "\n" for p in passages), encoding="utf-8")
        (self.root / "sections.jsonl").write_text("", encoding="utf-8")
        build_lookup(self.root / "passages.jsonl", self.root / "sections.jsonl", self.root / "lookup.sqlite")
        self.source = self.root / "review_runs" / "search"
        write_review_run(self.source, PASSING)
        self.results = self.root / "answer_runs"
        self.cfg = {
            "seed": 7, "paths": {"results": "original-results"},
            "retrieval": {"dense": {"model_name": "fake-dense"}}, "orchestration": {},
            "llm": {"model": "fake-model", "api_key": "fake-secret", "base_url": "https://example.test/v1"},
            "downstream": {"max_papers": 5, "passages_per_paper": 2, "max_chars_per_passage": 2400, "max_tokens": 512},
            "eval": {"answer": {"judge_model": None, "judge_temperature": 0.0, "judge_max_tokens": 256,
                                "relevance_questions": 3}},
        }
        self.original_cfg = copy.deepcopy(self.cfg)

    @contextmanager
    def harness(self, *, indexes=None, code_hash="code-v1"):
        generator, judge = FakeGenerator(), FakeJudge()
        stores = []

        def open_store():
            store = LookupStore(self.root / "lookup.sqlite")
            stores.append(store)
            return store

        with ExitStack() as stack:
            mocks = {}
            for name, kwargs in {
                "require_fresh_runtime": {}, "clear_runtime": {},
                "environment": {"return_value": {"python": "offline-test"}},
                "index_manifests": {"return_value": indexes or {"dense": {"version": 1}}},
                "_input_file_metadata": {"return_value": {"lookup.sqlite": {"bytes": 1}}},
                "_code_hash": {"return_value": code_hash}, "get_config": {"return_value": self.cfg},
                "get_client": {"return_value": generator}, "LLMClient": {"return_value": judge},
                "get_encoder": {"return_value": FakeEncoder()},
            }.items():
                mocks[name] = stack.enter_context(patch.object(answer_eval, name, **kwargs))
            stack.enter_context(patch.object(answer_eval.llm_client, "_CLIENT", None))
            stack.enter_context(patch("downstream.qa.get_config", return_value=self.cfg))
            stack.enter_context(patch("downstream.qa.get_client", return_value=generator))
            stack.enter_context(patch("downstream.qa.get_lookup_store", side_effect=open_store))
            stack.enter_context(patch("downstream.qa.run_query", side_effect=AssertionError("no new searches")))
            try:
                yield SimpleNamespace(generator=generator, judge=judge, **mocks)
            finally:
                for store in stores:
                    store.close()

    def run_eval(self, source=None, **kwargs) -> dict:
        return answer_eval.run_answer_eval(source or self.source, self.results, kwargs.pop("tag", "answers"), **kwargs)

    def test_hand_computed_scores_answers_file_and_usage(self) -> None:
        with self.harness() as fake:
            report = self.run_eval()
            fake.LLMClient.assert_called_once_with(model=None, temperature=0.0)
            fake.clear_runtime.assert_called_once_with(release_models=True)
        self.assertEqual(self.cfg, self.original_cfg)
        self.assertEqual((report["status"], report["source_kind"], report["selected_queries"]), ("complete", "review_run", 4))
        row = report["rows"]["run"]
        self.assertEqual((row["searched_queries"], row["completed_queries"], row["failed_query_ids"]), (4, 4, []))
        # Generation: Q1, Q2, Q4 (Q3 has no excerpts, so no call). Judge: 4 + 3 (no claims to verify) + 0 + 4.
        self.assertEqual(row["usage"], {"generation": {"calls": 3, "prompt_tokens": 30, "completion_tokens": 9},
                                        "judge": {"calls": 11, "prompt_tokens": 55, "completion_tokens": 22}})
        self.assertEqual(report["usage"], row["usage"])
        ev = row["splits"]["eval"]
        self.assertEqual((ev["n_queries"], ev["answered"], ev["insufficient_evidence"], ev["no_sources"]), (3, 1, 2, 1))
        self.assertAlmostEqual(ev["gold_in_sources"]["mean"], 1 / 3)
        self.assertAlmostEqual(ev["gold_cited"]["mean"], 1 / 3)
        self.assertEqual(ev["faithfulness"], {"n": 1, "mean": 0.5, "claims": 2, "supported_claims": 1, "pooled": 0.5})
        self.assertEqual((ev["answer_relevance"]["n"], ev["answer_relevance"]["noncommittal"]), (3, 2))
        self.assertAlmostEqual(ev["answer_relevance"]["mean"], (2 / 3) / 3)  # Q1 (1+0+1)/3; Q2, Q3 noncommittal
        self.assertEqual(ev["context_relevance"], {"n": 2, "mean": 0.25})  # Q1 1 of 2, Q2 0 of 1
        self.assertEqual(ev["context_precision"], {"n": 2, "mean": 0.5})  # Q1 AP 1.0, Q2 0.0
        self.assertEqual(ev["per_query"]["Q3"], {"status": "insufficient_evidence", "faithfulness": None,
                                                 "answer_relevance": 0.0, "context_relevance": None,
                                                 "context_precision": None, "gold_in_sources": 0.0, "gold_cited": 0.0})
        dev = row["splits"]["dev"]
        self.assertEqual((dev["faithfulness"]["mean"], dev["answer_relevance"]["mean"], dev["context_precision"]["mean"],
                          dev["gold_cited"]["mean"]), (1.0, 1.0, 1.0, 1.0))
        lines = [json.loads(line) for line in (self.results / "answers" / "rows" / "run" / "answers.jsonl")
                 .read_text(encoding="utf-8").splitlines()]
        self.assertEqual([line["query_id"] for line in lines], ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual(lines[0]["answer"], ANSWERS["How does A work?"]["answer"])
        self.assertEqual([s["paper_id"] for s in lines[0]["sources"]], ["A", "X"])
        self.assertNotIn("text", lines[0]["sources"][0])
        checkpoint = json.loads((self.results / "answers" / "rows" / "run" / "answers" / "Q1.json").read_text())
        self.assertEqual(checkpoint["answer"]["sources"][0]["text"], "Excerpt of A.")  # real store text
        self.assertNotIn("search_result", checkpoint["answer"])
        self.assertEqual(checkpoint["scores"]["claim_verdicts"][1]["supported"], False)
        manifest = json.loads((self.results / "answers" / "manifest.json").read_text())
        self.assertEqual(manifest["source"]["fingerprints"], {"run": "search-fp"})
        self.assertNotIn("fake-secret", json.dumps(manifest))

    def test_resume_skips_completed_and_requires_the_same_fingerprint(self) -> None:
        with self.harness():
            first = self.run_eval()
        with self.harness(), self.assertRaises(FileExistsError):
            self.run_eval()
        with self.harness() as fake:
            again = self.run_eval(resume=True)
            self.assertEqual(fake.generator.calls + fake.judge.calls, [])
        self.assertEqual(again["rows"]["run"]["splits"], first["rows"]["run"]["splits"])
        self.assertEqual(again["usage"], first["usage"])
        self.cfg["eval"]["answer"]["relevance_questions"] = 2
        with self.harness(), self.assertRaisesRegex(ValueError, "fingerprint"):
            self.run_eval(resume=True)
        self.cfg["eval"]["answer"]["relevance_questions"] = 3
        with self.harness(code_hash="code-v2"), self.assertRaisesRegex(ValueError, "fingerprint"):
            self.run_eval(resume=True)

    def test_failed_question_is_listed_then_retried_with_its_earlier_usage(self) -> None:
        with self.harness() as fake:
            fake.judge.broken.add("What does B find?")
            report = self.run_eval()
        self.assertEqual((report["status"], report["rows"]["run"]["failed_query_ids"]), ("partial", ["Q2"]))
        self.assertEqual(report["rows"]["run"]["splits"]["eval"]["n_queries"], 2)
        failed = json.loads((self.results / "answers" / "rows" / "run" / "answers" / "Q2.json").read_text())
        self.assertEqual(failed["error"], {"query_id": "Q2", "type": "ValueError"})
        # One generation call, then the claims prompt twice (one re-ask for the wrong shape).
        self.assertEqual(failed["usage"]["judge"]["calls"], 2)
        with self.harness() as fake:
            resumed = self.run_eval(resume=True)
            self.assertEqual(fake.generator.calls, ["What does B find?"])
        self.assertEqual((resumed["status"], resumed["rows"]["run"]["failed_query_ids"]), ("complete", []))
        retried = json.loads((self.results / "answers" / "rows" / "run" / "answers" / "Q2.json").read_text())
        self.assertEqual(retried["earlier_usage"]["judge"]["calls"], 2)
        self.assertEqual(resumed["usage"]["generation"]["calls"], 4)  # 3 first time + the retry
        self.assertEqual(resumed["usage"]["judge"]["calls"], 2 + 8 + 3)

    def test_failure_limit_stops_new_questions(self) -> None:
        with self.harness() as fake, patch.object(answer_eval, "MAX_FAILURES", 1):
            fake.generator.fail_on.add("How does A work?")
            report = self.run_eval()
            self.assertEqual(fake.generator.calls, ["How does A work?"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["rows"]["run"]["completed_queries"], 0)

    def test_source_checks_raise_before_any_call(self) -> None:
        with self.harness(indexes={"dense": {"version": 2}}) as fake, self.assertRaisesRegex(ValueError, "index"):
            self.run_eval()
        self.assertEqual(fake.generator.calls, [])
        with self.harness(), self.assertRaisesRegex(ValueError, "neither"):
            self.run_eval(self.root)
        with self.harness(), self.assertRaises(FileNotFoundError):
            self.run_eval(resume=True)

    def test_unsearched_questions_are_listed_and_selection_follows_split_and_limit(self) -> None:
        source = self.root / "review_runs" / "partial"
        write_review_run(source, PASSING, skip={"Q2"})
        with self.harness():
            report = self.run_eval(source, tag="partial", split="eval", limit=2)
        row = report["rows"]["run"]
        self.assertEqual(report["provenance"]["selection"]["query_ids"], ["Q1", "Q3"])
        self.assertEqual((row["completed_queries"], report["status"]), (2, "complete"))

    def test_workers_give_the_same_scores_and_exact_usage(self) -> None:
        with self.harness():
            serial = self.run_eval(tag="serial")
        with self.harness():
            parallel = self.run_eval(tag="parallel", workers=3)
        self.assertEqual(parallel["rows"]["run"]["splits"], serial["rows"]["run"]["splits"])
        self.assertEqual(parallel["usage"], serial["usage"])
        self.assertEqual(parallel["workers_used"], [3])

    def test_ablation_source_answers_every_row_and_compares_them(self) -> None:
        ablation = self.root / "ablation_runs" / "loop"
        _write(ablation / "manifest.json", {"plan": {
            "stages": ["review"], "edges": [["baseline", "a"]],
            "rows": {"baseline": {"enabled_factors": []}, "a": {"enabled_factors": ["loop"]}}}})
        write_review_run(ablation / "rows" / "baseline" / "review", PASSING, fingerprint="fp-base")
        write_review_run(ablation / "rows" / "a" / "review", {**PASSING, "Q2": ["B"]}, fingerprint="fp-a")
        ANSWERS_B = {"status": "answered", "answer": "B finds it [1]."}
        with self.harness(), patch.dict(ANSWERS, {"What does B find?": ANSWERS_B}), \
                patch.dict(CLAIMS, {"What does B find?": ["B finds it."]}), patch.dict(SUPPORTED, {"B finds it.": True}), \
                patch.dict(GENERATED, {"What does B find?": (["same"], False)}):
            report = self.run_eval(ablation, tag="ablation")
        self.assertEqual(report["source_kind"], "ablation")
        self.assertEqual(report["rows"]["a"]["enabled_factors"], ["loop"])
        edge = report["comparisons"][0]
        self.assertEqual((edge["before"], edge["after"]), ("baseline", "a"))
        cited = edge["splits"]["eval"]["metrics"]["gold_cited"]
        self.assertEqual((cited["n"], cited["before"], cited["after"]), (3, 1 / 3, 2 / 3))
        faith = edge["splits"]["eval"]["metrics"]["faithfulness"]
        self.assertEqual((faith["n"], faith["delta"]), (2, 0.0))  # Q1 and Q2 have claims in both rows; Q3 none
        self.assertEqual(edge["splits"]["dev"]["n_paired"], 1)


class JudgeAnswerTest(unittest.TestCase):
    def source(self, number: int) -> EvidenceSource:
        return EvidenceSource(number, "A", f"P{number}", "A", 2024, "Method", None, "text")

    def test_average_precision_by_hand(self) -> None:
        self.assertAlmostEqual(answer_metrics.average_precision([True, False, True]), (1 + 2 / 3) / 2)
        self.assertEqual(answer_metrics.average_precision([False, True]), 0.5)
        self.assertEqual(answer_metrics.average_precision([False]), 0.0)
        self.assertIsNone(answer_metrics.average_precision([]))

    def test_no_sources_makes_no_call(self) -> None:
        judge = FakeJudge()
        scores = answer_metrics.judge_answer(judge, FakeEncoder(), "q", "No evidence.", [])
        self.assertEqual(judge.calls, [])
        self.assertEqual((scores["faithfulness"], scores["answer_relevance"], scores["noncommittal"]), (None, 0.0, True))

    def test_wrong_verdict_shape_is_asked_again_once_then_raises(self) -> None:
        replies = iter([{"claims": ["c"]}, {"verdicts": [{"claim": True, "supported": True}]},
                        {"verdicts": [{"claim": 1, "supported": "yes"}]}])

        class Scripted:
            def chat_json(self, messages, schema, max_tokens=None):
                return next(replies)

        with self.assertRaisesRegex(ValueError, "wrong shape"):
            answer_metrics.judge_answer(Scripted(), FakeEncoder(), "q", "c [1]", [self.source(1)])

    def test_citations_are_removed_before_judging(self) -> None:
        judge = FakeJudge()
        with patch.dict(CLAIMS, {"q": ["D works."]}), patch.dict(GENERATED, {"q": (["same"], False)}):
            scores = answer_metrics.judge_answer(judge, FakeEncoder(), "q", "D works [1]. X[0] grows [1].",
                                                 [self.source(1)])
        self.assertEqual(list(judge.question_by_answer), ["D works. X[0] grows."])  # notation kept (D50)
        self.assertEqual(scores["claims"], ["D works."])
        self.assertEqual(scores["source_verdicts"], [{"source": 1, "relevant": True, "reason": "r"}])


if __name__ == "__main__":
    unittest.main()
