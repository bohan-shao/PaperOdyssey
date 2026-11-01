from __future__ import annotations

import json
import tempfile
import threading
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

from orchestration.errors import OrchestrationError
from orchestration.query_understanding import understand
from orchestration.review import batch_review
from orchestration.rewrite import rewrite
from orchestration.types import Criterion, FinalResult, QueryPlan, ReviewRecord, SubQuery
from retrieval.types import RetrieveResult, ScoredPassage


def make_config() -> dict:
    return {
        "kb": {"store": {"sqlite_file": "papers.sqlite"}},
        "orchestration": {
            "num_sub_queries": 3,
            "hyde": {"enabled": False},
            "review": {
                "batch_size": 2,
                "max_concurrency": 2,
                "max_tokens_per_paper": 200,
                "rubric": ["relevance", "coverage"],
            },
            "loop": {"enabled": True, "max_rounds": 3, "min_papers": 5, "cache": False},
            "rewrite": {"max_queries_per_round": 2},
        },
    }


def hit(paper_id: str, rank: int = 1, suffix: str = "1") -> ScoredPassage:
    return ScoredPassage(
        passage_id=f"{paper_id}_{suffix}",
        paper_id=paper_id,
        text=f"Evidence for {paper_id}",
        score=1.0 / rank,
        rank=rank,
        stage="hybrid",
        title=f"Title {paper_id}",
        year=2024,
        categories=["cs.CL"],
    )


class QueryUnderstandingTests(unittest.TestCase):
    def test_valid_plan_is_normalised(self) -> None:
        client = MagicMock()
        client.chat_json.return_value = {
            "sub_queries": [
                {
                    "query": "  efficient   fine tuning ",
                    "keywords": ["LoRA", "PEFT"],
                    "filters": {"year_min": 2023, "categories": ["cs.CL", "cs.CL"]},
                }
            ],
            "criteria": [{"criterion": " after 2023 ", "source": "extracted"}],
        }
        with (
            patch("orchestration.query_understanding.get_config", return_value=make_config()),
            patch("orchestration.query_understanding.get_client", return_value=client),
            patch("orchestration.query_understanding.build_understanding_prompt", return_value=[]),
        ):
            plan = understand("  compare   methods ")
        self.assertEqual(plan.input_text, "compare methods")
        self.assertEqual(plan.sub_queries[0].keywords, "LoRA PEFT")
        self.assertEqual(plan.sub_queries[0].filters, {"year_min": 2023, "categories": ["cs.CL"]})
        self.assertEqual(plan.criteria, [Criterion("after 2023", "extracted")])

    def test_two_invalid_outputs_fall_back(self) -> None:
        client = MagicMock()
        client.chat_json.side_effect = [ValueError("bad JSON"), {"sub_queries": []}]
        with (
            patch("orchestration.query_understanding.get_config", return_value=make_config()),
            patch("orchestration.query_understanding.get_client", return_value=client),
            patch("orchestration.query_understanding.build_understanding_prompt", return_value=[]),
        ):
            plan = understand("find robust RAG papers")
        self.assertEqual(plan.sub_queries, [SubQuery("find robust RAG papers")])
        self.assertEqual(plan.criteria[0].criterion, "find robust RAG papers")
        self.assertEqual(client.chat_json.call_count, 2)


class ReviewAndRewriteTests(unittest.TestCase):
    def test_review_batches_and_uses_real_evidence(self) -> None:
        cfg = make_config()
        cfg["orchestration"]["review"]["batch_size"] = 1
        client = MagicMock()
        client.chat_json.side_effect = [
            [{"paper_id": "P1", "verdict": "pass", "score": 5, "reasons": "yes", "covered": ["aspect"], "missing": []}],
            [{"paper_id": "P2", "verdict": "reject", "score": 2, "reasons": "no", "covered": [], "missing": ["aspect"]}],
        ]
        candidates = [
            {"paper_id": "P1", "title": "One", "abstract": "A", "top_passages": [{"passage_id": "P1_1", "text": "x"}]},
            {"paper_id": "P2", "title": "Two", "abstract": "B", "top_passages": [{"passage_id": "P2_1", "text": "y"}]},
        ]
        with (
            patch("orchestration.review.get_config", return_value=cfg),
            patch("orchestration.review.get_client", return_value=client),
            patch("orchestration.review.build_review_prompt", return_value=[]),
        ):
            records = batch_review(candidates, [Criterion("aspect")], "request", 1)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].evidence["top_passages"], ["P1_1"])
        self.assertEqual(client.chat_json.call_count, 2)
        self.assertIsNone(client.chat_json.call_args.kwargs["max_tokens"])  # unset: the client's llm.max_tokens

    def test_review_reply_limit_comes_from_config(self) -> None:
        cfg = make_config()
        cfg["orchestration"]["review"]["max_output_tokens"] = 2048
        client = MagicMock()
        client.chat_json.return_value = [
            {"paper_id": "P1", "verdict": "pass", "score": 5, "reasons": "yes", "covered": [], "missing": []}]
        candidate = {"paper_id": "P1", "title": "One", "abstract": "A", "top_passages": [{"passage_id": "P1_1", "text": "x"}]}
        with (
            patch("orchestration.review.get_config", return_value=cfg),
            patch("orchestration.review.get_client", return_value=client),
            patch("orchestration.review.build_review_prompt", return_value=[]),
        ):
            batch_review([candidate], [], "request", 1)
        self.assertEqual(client.chat_json.call_args.kwargs["max_tokens"], 2048)

    def test_invalid_review_retries_then_fails(self) -> None:
        client = MagicMock()
        client.chat_json.return_value = [
            {"paper_id": "invented", "verdict": "pass", "score": 7, "covered": [], "missing": []}
        ]
        with (
            patch("orchestration.review.get_config", return_value=make_config()),
            patch("orchestration.review.get_client", return_value=client),
            patch("orchestration.review.build_review_prompt", return_value=[]),
            self.assertRaises(OrchestrationError),
        ):
            batch_review([{"paper_id": "P1"}], [Criterion("aspect")], "request", 1)
        self.assertEqual(client.chat_json.call_count, 2)

    def test_rewrite_limits_and_removes_exact_duplicates(self) -> None:
        client = MagicMock()
        client.chat_json.return_value = {
            "sub_queries": [
                {"query": "existing query", "keywords": "same"},
                {"query": "new query one", "keywords": ["new", "one"]},
                {"query": "new query two", "keywords": "new two"},
                {"query": "new query three", "keywords": "new three"},
            ]
        }
        with (
            patch("orchestration.rewrite.get_config", return_value=make_config()),
            patch("orchestration.rewrite.get_client", return_value=client),
            patch("orchestration.rewrite.build_rewrite_prompt", return_value=[]),
        ):
            queries = rewrite(["missing"], ["Existing Query"])
        self.assertEqual([item.query for item in queries], ["new query one", "new query two"])

    def test_rewrite_failure_degrades_to_empty(self) -> None:
        client = MagicMock()
        client.chat_json.side_effect = RuntimeError("offline")
        with (
            patch("orchestration.rewrite.get_config", return_value=make_config()),
            patch("orchestration.rewrite.get_client", return_value=client),
            patch("orchestration.rewrite.build_rewrite_prompt", return_value=[]),
        ):
            self.assertEqual(rewrite(["missing"], []), [])


class LoopTests(unittest.TestCase):
    def _run_with_patches(self, cfg: dict, **patches: object):
        from orchestration import loop

        defaults = {
            "understand": lambda text: QueryPlan(text, [SubQuery("q1")], [Criterion("aspect")]),
            "retrieve": lambda query, filters=None: RetrieveResult([hit("P1")], 1),
            "batch_review": lambda candidates, criteria, text, round_no: [
                ReviewRecord(candidate["paper_id"], "pass", 5, covered=["aspect"], round=round_no)
                for candidate in candidates
            ],
            "load_paper": lambda paper_id, path: {"paper_id": paper_id, "abstract": "Abstract"},
        }
        defaults.update(patches)
        stack = [
            patch("orchestration.loop.get_config", return_value=cfg),
            patch("orchestration.loop._cache_path", side_effect=lambda digest: Path(tempfile.gettempdir()) / f"unused-{digest}.json"),
            patch("orchestration.loop._paper_database_path", return_value=Path("papers.sqlite")),
            patch("orchestration.loop.understand", side_effect=defaults["understand"]),
            patch("orchestration.loop.retrieve", side_effect=defaults["retrieve"]),
            patch("orchestration.loop.batch_review", side_effect=defaults["batch_review"]),
            patch("orchestration.loop.load_paper", side_effect=defaults["load_paper"]),
        ]
        for item in stack:
            item.start()
        self.addCleanup(lambda: [item.stop() for item in reversed(stack)])
        return loop.run_query("request")

    def test_deduplicates_across_paths_and_meets_coverage(self) -> None:
        cfg = make_config()
        plan = QueryPlan(
            "request",
            [SubQuery("q1"), SubQuery("q2")],
            [Criterion("aspect one"), Criterion("aspect two")],
        )
        reviewed_ids: list[str] = []

        def fake_retrieve(query: str, filters=None) -> RetrieveResult:
            hits = [hit("P1"), hit("P2")] if query == "q1" else [hit("P1", suffix="2"), hit("P3")]
            return RetrieveResult(hits, len(hits))

        def fake_review(candidates, criteria, text, round_no):
            reviewed_ids.extend(candidate["paper_id"] for candidate in candidates)
            return [
                ReviewRecord(
                    candidate["paper_id"],
                    "pass",
                    4,
                    covered=["aspect one", "aspect two"],
                    round=round_no,
                )
                for candidate in candidates
            ]

        result = self._run_with_patches(
            cfg,
            understand=lambda text: plan,
            retrieve=fake_retrieve,
            batch_review=fake_review,
        )
        self.assertEqual(sorted(reviewed_ids), ["P1", "P2", "P3"])
        self.assertEqual(result.coverage_report["termination_reason"], "coverage_met")
        self.assertEqual(result.coverage_report["papers_reviewed"], 3)

    def test_review_trace_keeps_rejects_without_changing_ranked_papers(self) -> None:
        records = {
            "P1": ReviewRecord("P1", "pass", 3, covered=["aspect"]),
            "P2": ReviewRecord("P2", "reject", 5, reasons="Does not answer the request",
                               missing=["aspect"], evidence={"top_passages": ["P2_1"]}),
            "P3": ReviewRecord("P3", "pass", 5, covered=["aspect"]),
            "P4": ReviewRecord("P4", "pass", 5, covered=["aspect"]),
        }
        result = self._run_with_patches(
            make_config(),
            retrieve=lambda query, filters=None: RetrieveResult([hit(pid) for pid in records], 4),
            batch_review=lambda candidates, criteria, text, round_no: [
                records[candidate["paper_id"]] for candidate in candidates
            ],
        )
        self.assertEqual(result.papers, [records[pid] for pid in ("P3", "P4", "P1")])
        self.assertEqual({record.paper_id: record for record in result.reviews}, records)
        self.assertEqual(result.coverage_report["papers_reviewed"], len(result.reviews))
        self.assertEqual(result.coverage_report["papers_passed"], len(result.papers))

    def test_equal_scores_follow_the_best_retrieval_rank_over_all_sub_queries(self) -> None:
        plan = QueryPlan("request", [SubQuery("q1"), SubQuery("q2")], [Criterion("aspect")])

        def fake_retrieve(query: str, filters=None) -> RetrieveResult:
            # P9 is a weak q1 hit but q2's top hit; by paper ID alone P1 would come first.
            hits = [hit("P2"), hit("P1", rank=2), hit("P9", rank=3)] if query == "q1" else [hit("P9", suffix="2")]
            return RetrieveResult(hits, len(hits))

        scores = {"P1": 5, "P2": 4, "P9": 5}
        result = self._run_with_patches(
            make_config(),
            understand=lambda text: plan,
            retrieve=fake_retrieve,
            batch_review=lambda candidates, criteria, text, round_no: [
                ReviewRecord(candidate["paper_id"], "pass", scores[candidate["paper_id"]],
                             covered=["aspect"], round=round_no)
                for candidate in candidates
            ],
        )
        self.assertEqual(result.retrieval_ranks, {"P1": 2, "P2": 1, "P9": 1})
        self.assertEqual([record.paper_id for record in result.papers], ["P9", "P1", "P2"])

    def test_review_trace_preserves_rounds_and_cache_roundtrip(self) -> None:
        from orchestration import loop

        cfg = make_config()
        cfg["orchestration"]["loop"]["max_rounds"] = 2

        def fake_review(candidates, criteria, text, round_no):
            return [ReviewRecord(
                candidate["paper_id"], "reject" if round_no == 1 else "pass", round_no + 2,
                reasons=f"Reviewed in round {round_no}", round=round_no,
                covered=[] if round_no == 1 else ["aspect"],
                missing=["aspect"] if round_no == 1 else [],
                evidence={"top_passages": [candidate["top_passages"][0]["passage_id"]]},
            ) for candidate in candidates]

        with patch("orchestration.loop.rewrite", return_value=[SubQuery("q2")]):
            result = self._run_with_patches(
                cfg,
                # P1 occurs again, but must retain its original reject record without re-review.
                retrieve=lambda query, filters=None: RetrieveResult(
                    [hit("P1")] if query == "q1" else [hit("P1"), hit("P2")], 2),
                batch_review=fake_review,
            )
        self.assertEqual([(r.paper_id, r.verdict, r.round) for r in result.reviews],
                         [("P1", "reject", 1), ("P2", "pass", 2)])
        self.assertEqual([record.paper_id for record in result.papers], ["P2"])
        self.assertEqual(result.coverage_report["rounds"], 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            loop._write_cache(path, result)
            restored = loop._read_cache(path, "request", "restored-id")
            self.assertIsNotNone(restored)
            self.assertEqual(restored.reviews, result.reviews)
            self.assertEqual(restored.papers, result.papers)
            self.assertEqual(restored.retrieval_ranks, {"P1": 1, "P2": 1})
            self.assertEqual(restored.query_id, "restored-id")
            self.assertTrue(restored.coverage_report["cache_hit"])
            self.assertEqual(restored.latency_by_stage, result.latency_by_stage)

    def test_a_later_round_reviews_a_paper_again_when_it_brings_new_passages(self) -> None:
        from eval.review_eval import _validate_result

        cfg = make_config()
        cfg["orchestration"]["loop"]["max_rounds"] = 3
        plan = QueryPlan("request", [SubQuery("q1")], [Criterion("aspect", "extracted")])
        seen_passages = []

        def fake_review(candidates, criteria, text, round_no):
            records = []
            for candidate in candidates:
                ids = [p["passage_id"] for p in candidate["top_passages"]]
                seen_passages.append((round_no, ids))
                ok = "P1_2" in ids  # only the passage found in round 2 covers the aspect
                records.append(ReviewRecord(
                    candidate["paper_id"], "pass" if ok else "reject", 5 if ok else 2, round=round_no,
                    covered=["aspect"] if ok else [], missing=[] if ok else ["aspect"],
                    evidence={"top_passages": ids}))
            return records

        def fake_retrieve(query, filters=None):
            # Round 2 finds the same paper again through a different passage; round 3 adds nothing new.
            return RetrieveResult([hit("P1")] if query == "q1" else [hit("P1", suffix="2"), hit("P1")], 1)

        with patch("orchestration.loop.rewrite", side_effect=[[SubQuery("q2")], [SubQuery("q3")]]):
            result = self._run_with_patches(cfg, understand=lambda text: plan,
                                            retrieve=fake_retrieve, batch_review=fake_review)
        self.assertEqual(seen_passages, [(1, ["P1_1"]), (2, ["P1_1", "P1_2"])])
        self.assertEqual([(r.paper_id, r.verdict, r.round) for r in result.reviews], [("P1", "pass", 2)])
        self.assertEqual([r.paper_id for r in result.papers], ["P1"])
        report = result.coverage_report
        self.assertEqual((report["rounds"], report["termination_reason"], report["covered_aspects"]),
                         (2, "coverage_met", 1))
        self.assertEqual((report["papers_reviewed"], report["papers_passed"], report["re_reviews"]), (1, 1, 1))
        _validate_result({"query_id": result.query_id, "input_text": "request"}, asdict(result))

    def test_re_review_off_keeps_the_pre_d53_behaviour(self) -> None:
        cfg = make_config()
        cfg["orchestration"]["loop"].update(max_rounds=2, re_review=False)
        reviewed = []

        def fake_review(candidates, criteria, text, round_no):
            reviewed.extend((round_no, c["paper_id"]) for c in candidates)
            return [ReviewRecord(c["paper_id"], "reject", 2, round=round_no, missing=["aspect"]) for c in candidates]

        with patch("orchestration.loop.rewrite", return_value=[SubQuery("q2")]):
            result = self._run_with_patches(
                cfg, batch_review=fake_review,
                retrieve=lambda query, filters=None: RetrieveResult(
                    [hit("P1")] if query == "q1" else [hit("P1", suffix="2")], 1))
        self.assertEqual(reviewed, [(1, "P1")])  # round 2's new passage of P1 is not reviewed
        self.assertEqual(result.coverage_report["re_reviews"], 0)

    def test_a_rejecting_re_review_keeps_the_earlier_pass_and_its_coverage(self) -> None:
        # Repro from the PR #28 review: aspect A's evidence is only in P_2, which the 3-slot
        # re-review no longer sees, so the re-review rejects. The run must not end worse than round 1.
        cfg = make_config()
        cfg["orchestration"]["loop"]["max_rounds"] = 2
        plan = QueryPlan("request", [SubQuery("q1")], [Criterion("A", "extracted"), Criterion("B", "extracted")])
        seen = []

        def fake_retrieve(query, filters=None):
            hits = [hit("P", 1, "0"), hit("P", 2, "1"), hit("P", 3, "2")] if query == "q1" else [hit("P", 1, "n0")]
            return RetrieveResult(hits, 1)

        def fake_review(candidates, criteria, text, round_no):
            out = []
            for c in candidates:
                ids = [p["passage_id"] for p in c["top_passages"]]
                seen.append((round_no, ids))
                ok = "P_2" in ids
                out.append(ReviewRecord("P", "pass" if ok else "reject", 4 if ok else 2, round=round_no,
                                        covered=["A"] if ok else [], missing=["B"] if ok else ["A", "B"],
                                        evidence={"top_passages": ids}))
            return out

        with patch("orchestration.loop.rewrite", return_value=[SubQuery("q2")]):
            result = self._run_with_patches(cfg, understand=lambda text: plan, retrieve=fake_retrieve,
                                            batch_review=fake_review)
        self.assertEqual(seen, [(1, ["P_0", "P_1", "P_2"]), (2, ["P_0", "P_n0", "P_1"])])
        self.assertEqual([r.paper_id for r in result.papers], ["P"])
        record = result.reviews[0]
        self.assertEqual((record.verdict, record.score, record.round, record.covered), ("pass", 4, 1, ["A"]))
        self.assertEqual(record.evidence, {"top_passages": ["P_0", "P_1", "P_2"], "review_rounds": [1, 2]})
        self.assertEqual(result.coverage_report["covered_criteria"], ["A"])
        self.assertEqual(result.coverage_report["re_reviews"], 1)

    def test_a_passing_re_review_adds_coverage_and_its_evidence(self) -> None:
        from eval.review_eval import _score, _validate_result

        cfg = make_config()
        cfg["orchestration"]["loop"]["max_rounds"] = 2
        plan = QueryPlan("request", [SubQuery("q1")], [Criterion("A", "extracted"), Criterion("B", "extracted")])

        def fake_review(candidates, criteria, text, round_no):
            ids = [p["passage_id"] for p in candidates[0]["top_passages"]]
            # Round 1 covers A only; the re-review with the new passage covers B but not A.
            covered, score = (["A"], 3) if round_no == 1 else (["B"], 5)
            return [ReviewRecord("P1", "pass", score, reasons=f"r{round_no}", round=round_no, covered=covered,
                                 missing=["B"] if round_no == 1 else ["A"], evidence={"top_passages": ids})]

        with patch("orchestration.loop.rewrite", return_value=[SubQuery("q2")]):
            result = self._run_with_patches(
                cfg, understand=lambda text: plan, batch_review=fake_review,
                retrieve=lambda query, filters=None: RetrieveResult(
                    [hit("P1")] if query == "q1" else [hit("P1", suffix="2")], 1))
        record = result.reviews[0]
        self.assertEqual((record.verdict, record.score, record.round), ("pass", 5, 1))
        self.assertEqual((record.covered, record.missing), (["A", "B"], []))
        self.assertEqual(record.reasons, "r1 Round 2 re-review: r2")
        # The answer reads the first two evidence passages, so the new one must be among them.
        self.assertEqual(record.evidence, {"top_passages": ["P1_1", "P1_2"], "review_rounds": [1, 2]})
        report = result.coverage_report
        self.assertEqual((report["termination_reason"], report["covered_aspects"], report["re_reviews"]),
                         ("coverage_met", 2, 1))
        row = {"question": {"query_id": result.query_id, "input_text": "request", "gold_paper_ids": ["P1"]},
               "result": asdict(result), "wall_ms": 1}
        _validate_result(row["question"], row["result"])
        rounds = _score([row])["per_query"][result.query_id]
        self.assertEqual((rounds["first_source_review_round"], rounds["first_source_pass_round"]), (1, 1))

    def test_a_pass_after_a_reject_reports_both_rounds(self) -> None:
        from eval.review_eval import _score

        cfg = make_config()
        cfg["orchestration"]["loop"]["max_rounds"] = 2

        def fake_review(candidates, criteria, text, round_no):
            ok = round_no == 2
            return [ReviewRecord("P1", "pass" if ok else "reject", 4 if ok else 2, round=round_no,
                                 covered=["aspect"] if ok else [], missing=[] if ok else ["aspect"])]

        with patch("orchestration.loop.rewrite", return_value=[SubQuery("q2")]):
            result = self._run_with_patches(
                cfg, batch_review=fake_review,
                retrieve=lambda query, filters=None: RetrieveResult(
                    [hit("P1")] if query == "q1" else [hit("P1", suffix="2")], 1))
        self.assertEqual([(r.verdict, r.round, r.evidence) for r in result.reviews],
                         [("pass", 2, {"review_rounds": [1, 2]})])
        row = {"question": {"query_id": result.query_id, "gold_paper_ids": ["P1"]}, "result": asdict(result), "wall_ms": 1}
        rounds = _score([row])["per_query"][result.query_id]
        self.assertEqual((rounds["first_source_review_round"], rounds["first_source_pass_round"]), (1, 2))

    def test_only_a_reject_or_a_pass_missing_an_uncovered_aspect_is_re_reviewed(self) -> None:
        from orchestration import loop

        def state(verdict, missing):
            return loop._Reviewed(ReviewRecord("P", verdict, 3, missing=missing),
                                  [{"passage_id": "p0", "text": "t"}], {"p0"})

        candidate = {"paper_id": "P", "top_passages": [{"passage_id": "n0", "text": "t"}]}
        for verdict, missing, expected in (("reject", [], True), ("pass", ["B"], True),
                                           ("pass", ["A"], False), ("pass", [], False)):
            claimed = loop._claim(candidate, {}, {"P": state(verdict, missing)}, {"B"})
            self.assertEqual(claimed is not None, expected, (verdict, missing))

    def test_only_the_passages_shown_count_as_seen(self) -> None:
        from orchestration import loop

        def passages(*ids):
            return [{"passage_id": pid, "text": pid} for pid in ids]

        reviewed = {"P": loop._Reviewed(ReviewRecord("P", "reject", 2), passages("p0", "p1", "p2"), {"p0", "p1", "p2"})}
        queued = {}
        claimed = loop._claim({"paper_id": "P", "top_passages": passages("n0", "n1")}, queued, reviewed, {"A"})
        self.assertEqual([p["passage_id"] for p in claimed["top_passages"]], ["p0", "n0", "p1"])
        self.assertEqual([p["passage_id"] for p in queued["P"]], ["p0", "n0", "p1"])
        # The loop then adds only the shown ids, so n1 found again later still triggers a re-review.
        reviewed["P"].seen |= {p["passage_id"] for p in queued["P"]}
        again = loop._claim({"paper_id": "P", "top_passages": passages("n1")}, {}, reviewed, {"A"})
        self.assertEqual([p["passage_id"] for p in again["top_passages"]], ["p0", "n1", "p1"])

    def test_plain_rag_keeps_the_retrieved_papers_in_rank_order_without_any_llm_call(self) -> None:
        from eval.review_eval import _validate_result

        cfg = make_config()
        cfg["orchestration"]["agentic"] = {"enabled": False}
        cfg["orchestration"]["hyde"]["enabled"] = True  # ignored: the plain baseline makes no LLM call
        queries = []

        def fake_retrieve(query, filters=None):
            queries.append((query, filters))
            hits = [hit("P2", 1), hit("P1", 2), hit("P2", 3, suffix="2"), hit("P3", 4), hit("P2", 5)]
            return RetrieveResult(hits, 50)

        def no_llm(*args, **kwargs):
            raise AssertionError("the plain RAG baseline must not call the LLM")

        with patch("orchestration.loop.get_client", side_effect=no_llm), \
                patch("orchestration.loop.rewrite", side_effect=no_llm):
            result = self._run_with_patches(cfg, understand=no_llm, retrieve=fake_retrieve, batch_review=no_llm)
        self.assertEqual(queries, [("request", None)])  # the question as typed, no filters
        self.assertEqual([p.paper_id for p in result.papers], ["P2", "P1", "P3"])
        self.assertEqual(result.papers[0].evidence["top_passages"], ["P2_1", "P2_2"])
        self.assertEqual({(p.verdict, p.score, p.round) for p in result.papers}, {("pass", 0, 1)})
        self.assertEqual(result.retrieval_ranks, {"P2": 1, "P1": 2, "P3": 4})
        self.assertEqual(result.reviews, result.papers)
        report = result.coverage_report
        self.assertEqual((report["termination_reason"], report["review_backend"], report["rounds"],
                          report["papers_passed"], report["total_aspects"]), ("plain_rag", "none", 1, 3, 0))
        self.assertEqual(result.query_plan.sub_queries, [SubQuery("request")])
        self.assertEqual((result.latency_by_stage["understanding_ms"], result.latency_by_stage["review_ms"]), (0, 0))
        _validate_result({"query_id": result.query_id, "input_text": "request"}, asdict(result))

    def test_legacy_cache_has_unavailable_trace_and_is_still_reused(self) -> None:
        from orchestration import loop

        cfg = make_config()
        cfg["orchestration"]["loop"]["cache"] = True
        passed = ReviewRecord("P1", "pass", 4, covered=["aspect"])
        previous = FinalResult("old-id", "request", QueryPlan("request"), [passed],
                               {"papers_reviewed": 2, "papers_passed": 1}, {"total_ms": 10})
        legacy = asdict(previous)
        legacy.pop("reviews")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            with (
                patch("orchestration.loop.get_config", return_value=cfg),
                patch("orchestration.loop._cache_path", return_value=path),
                patch("orchestration.loop._run_async") as workflow,
            ):
                result = loop.run_query("request", query_id="new-id")
            workflow.assert_not_called()
            self.assertEqual(result.reviews, [])
            self.assertEqual(result.papers, [passed])
            self.assertEqual(result.query_id, "new-id")
            self.assertEqual(result.coverage_report["papers_reviewed"], 2)
            self.assertTrue(result.coverage_report["cache_hit"])
            self.assertNotIn("reviews", json.loads(path.read_text(encoding="utf-8")))

    def test_review_starts_before_slow_retrieval_finishes(self) -> None:
        cfg = make_config()
        cfg["orchestration"]["loop"]["enabled"] = False
        cfg["orchestration"]["review"]["batch_size"] = 1
        cfg["orchestration"]["review"]["max_concurrency"] = 1
        review_started = threading.Event()
        observed_overlap: list[bool] = []

        def fake_retrieve(query: str, filters=None) -> RetrieveResult:
            if query == "slow":
                observed_overlap.append(review_started.wait(timeout=1.0))
                return RetrieveResult([hit("P2")], 1)
            return RetrieveResult([hit("P1")], 1)

        def fake_review(candidates, criteria, text, round_no):
            review_started.set()
            return [ReviewRecord(candidate["paper_id"], "reject", 2, round=round_no) for candidate in candidates]

        result = self._run_with_patches(
            cfg,
            understand=lambda text: QueryPlan(text, [SubQuery("fast"), SubQuery("slow")], [Criterion("aspect")]),
            retrieve=fake_retrieve,
            batch_review=fake_review,
        )
        self.assertEqual(observed_overlap, [True])
        self.assertEqual(result.coverage_report["termination_reason"], "loop_disabled")

    def test_hyde_adds_retrieval_path_and_failure_is_nonfatal(self) -> None:
        from orchestration import loop

        cfg = make_config()
        cfg["orchestration"]["hyde"]["enabled"] = True
        cfg["orchestration"]["loop"]["enabled"] = False
        queries: list[str] = []
        client = MagicMock()
        client.chat.return_value = "hypothetical abstract"

        with patch("orchestration.loop.get_client", return_value=client):
            result = self._run_with_patches(
                cfg,
                retrieve=lambda query, filters=None: (
                    queries.append(query) or RetrieveResult([hit(f"P{len(queries)}")], 1)
                ),
            )
        self.assertIn("hypothetical abstract", queries)
        self.assertEqual(len(result.query_plan.sub_queries), 2)

        client.chat.side_effect = RuntimeError("optional HyDE unavailable")
        queries.clear()
        with patch("orchestration.loop.get_client", return_value=client):
            result = self._run_with_patches(
                cfg,
                retrieve=lambda query, filters=None: (
                    queries.append(query) or RetrieveResult([hit("P1")], 1)
                ),
            )
        self.assertEqual(queries, ["q1"])
        self.assertEqual(len(result.query_plan.sub_queries), 1)

    def test_rewrite_exhausted_and_max_rounds(self) -> None:
        from orchestration import loop

        cfg = make_config()
        rejected = lambda candidates, criteria, text, round_no: [
            ReviewRecord(candidate["paper_id"], "reject", 1, missing=["aspect"], round=round_no)
            for candidate in candidates
        ]
        with patch("orchestration.loop.rewrite", return_value=[]):
            exhausted = self._run_with_patches(cfg, batch_review=rejected)
        self.assertEqual(exhausted.coverage_report["termination_reason"], "rewrite_exhausted")

        cfg = make_config()
        cfg["orchestration"]["loop"]["max_rounds"] = 2
        counter = {"value": 0}

        def different_hit(query: str, filters=None):
            counter["value"] += 1
            return RetrieveResult([hit(f"P{counter['value']}")], 1)

        with patch("orchestration.loop.rewrite", return_value=[SubQuery("next")]):
            capped = self._run_with_patches(cfg, batch_review=rejected, retrieve=different_hit)
        self.assertEqual(capped.coverage_report["termination_reason"], "max_rounds")
        self.assertEqual(capped.coverage_report["rounds"], 2)

    def test_min_papers_terminates_with_incomplete_coverage(self) -> None:
        cfg = make_config()
        cfg["orchestration"]["loop"]["min_papers"] = 2

        result = self._run_with_patches(
            cfg,
            retrieve=lambda query, filters=None: RetrieveResult([hit("P1"), hit("P2")], 2),
            batch_review=lambda candidates, criteria, text, round_no: [
                ReviewRecord(candidate["paper_id"], "pass", 4, covered=[], round=round_no)
                for candidate in candidates
            ],
        )
        self.assertEqual(result.coverage_report["termination_reason"], "min_papers")
        self.assertEqual(result.coverage_report["covered_aspects"], 0)

    def test_content_cache_normalises_input_and_overrides_query_id(self) -> None:
        from orchestration import loop

        cfg = make_config()
        cfg["orchestration"]["loop"]["cache"] = True
        understand_calls = {"count": 0}

        def fake_understand(text: str) -> QueryPlan:
            understand_calls["count"] += 1
            return QueryPlan(text, [SubQuery("q1")], [Criterion("aspect")])

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            cache_file = cache_dir / "result.json"
            cache_file.write_text("corrupt", encoding="utf-8")
            with (
                patch("orchestration.loop.get_config", return_value=cfg),
                patch("orchestration.loop._cache_path", return_value=cache_file),
                patch("orchestration.loop._paper_database_path", return_value=Path("papers.sqlite")),
                patch("orchestration.loop.understand", side_effect=fake_understand),
                patch("orchestration.loop.retrieve", return_value=RetrieveResult([hit("P1")], 1)),
                patch("orchestration.loop.load_paper", return_value={"abstract": "A"}),
                patch(
                    "orchestration.loop.batch_review",
                    side_effect=lambda candidates, criteria, text, round_no: [
                        ReviewRecord("P1", "pass", 5, covered=["aspect"], round=round_no)
                    ],
                ),
            ):
                first = loop.run_query("  cache   me ", query_id="first")
                second = loop.run_query("cache me", query_id="second")
            self.assertEqual(first.query_id, "first")
            self.assertEqual(second.query_id, "second")
            self.assertTrue(second.coverage_report["cache_hit"])
            self.assertEqual(understand_calls["count"], 1)
            self.assertEqual(json.loads(cache_file.read_text(encoding="utf-8"))["query_id"], "first")

    def test_cache_key_depends_on_settings_but_query_id_does_not(self) -> None:
        from orchestration import loop

        cfg = make_config()
        cfg["orchestration"]["loop"]["cache"] = True
        calls = {"understand": 0}

        def fake_understand(text: str) -> QueryPlan:
            calls["understand"] += 1
            return QueryPlan(text, [SubQuery("q1")], [Criterion("aspect")])

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("orchestration.loop.get_config", side_effect=lambda: cfg),
                patch("orchestration.loop._cache_path", side_effect=lambda digest: Path(directory) / f"{digest}.json"),
                patch("orchestration.loop._paper_database_path", return_value=Path("papers.sqlite")),
                patch("orchestration.loop.understand", side_effect=fake_understand),
                patch("orchestration.loop.retrieve", return_value=RetrieveResult([hit("P1")], 1)),
                patch("orchestration.loop.load_paper", return_value={"abstract": "A"}),
                patch(
                    "orchestration.loop.batch_review",
                    side_effect=lambda candidates, criteria, text, round_no: [
                        ReviewRecord("P1", "pass", 5, covered=["aspect"], round=round_no)
                    ],
                ),
            ):
                loop_on = loop.run_query("same request")
                cfg["orchestration"]["loop"]["enabled"] = False
                loop_off = loop.run_query("same request")
                loop_off_again = loop.run_query("same request")
            self.assertFalse(loop_off.coverage_report["cache_hit"])
            self.assertTrue(loop_off_again.coverage_report["cache_hit"])
            self.assertEqual(calls["understand"], 2)
            self.assertEqual(loop_on.query_id, loop_off.query_id)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)

    def test_cache_key_changes_when_the_corpus_is_rebuilt(self) -> None:
        import os

        from orchestration import loop

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = make_config()
            cfg["paths"] = {"indices": str(root / "indices"), "processed": str(root / "processed")}
            cfg["retrieval"] = {"lookup": {"db_file": "lookup.sqlite"}, "sparse": {"index_dir": "bm25"},
                                "dense": {"faiss_dir": "faiss"}}
            files = [root / "indices" / "lookup.sqlite", root / "indices" / "bm25" / "meta.json",
                     root / "indices" / "faiss" / "meta.json", root / "processed" / "papers.sqlite"]
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
                os.utime(path, ns=(1_000_000_000, 1_000_000_000))
            with patch("orchestration.loop._paper_database_path", return_value=files[3]):
                before = loop._cache_digest("request", cfg)
                self.assertEqual(loop._cache_digest("request", cfg), before)
                for path in files:  # every built file counts: rebuild one at a time
                    os.utime(path, ns=(2_000_000_000, 2_000_000_000))
                    after = loop._cache_digest("request", cfg)
                    self.assertNotEqual(after, before, path.name)
                    before = after
                files[1].write_text('{"num_docs": 2}', encoding="utf-8")  # same time, other content size
                os.utime(files[1], ns=(2_000_000_000, 2_000_000_000))
                self.assertNotEqual(loop._cache_digest("request", cfg), before)
        self.assertIsNone(loop._corpus_identity(make_config()))  # no corpus configured

    def test_corrupt_cache_is_rebuilt_and_failure_is_not_cached(self) -> None:
        from orchestration import loop

        cfg = make_config()
        cfg["orchestration"]["loop"]["cache"] = True
        with tempfile.TemporaryDirectory() as directory:
            cache_file = Path(directory) / "result.json"
            cache_file.write_text("not-json", encoding="utf-8")
            with (
                patch("orchestration.loop.get_config", return_value=cfg),
                patch("orchestration.loop._cache_path", return_value=cache_file),
                patch("orchestration.loop._paper_database_path", return_value=Path("papers.sqlite")),
                patch("orchestration.loop.understand", return_value=QueryPlan("request", [SubQuery("q1")], [Criterion("aspect")])),
                patch("orchestration.loop.retrieve", return_value=RetrieveResult([hit("P1")], 1)),
                patch("orchestration.loop.load_paper", return_value=None),
                patch("orchestration.loop.batch_review", side_effect=OrchestrationError("review down")),
                self.assertRaises(OrchestrationError),
            ):
                loop.run_query("request")
            self.assertEqual(cache_file.read_text(encoding="utf-8"), "not-json")


if __name__ == "__main__":
    unittest.main()
