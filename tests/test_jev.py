"""Offline tests for the Jev client (llm/jev.py) and the Jev review backend.

The client runs against an httpx.MockTransport that plays the TypeSafe /v1/systemone endpoint,
so no key or network is needed: python -m unittest tests.test_jev
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from tenacity import wait_none

from llm.jev import JevClient, JevError, choice, noul, score
from orchestration.errors import OrchestrationError
from orchestration.review import batch_review
from orchestration.types import Criterion

QUESTIONS = {
    "verdict": choice("Keep it?", {"pass": "yes", "reject": "no"}),
    "level": score("How good?", ["low", "mid", "high"]),
    "urgent": noul("Is it urgent?"),
}
ANSWERS = {
    "verdict": {"type": "choice", "choice": "pass", "confidence": 0.8, "probabilities": {"pass": 0.9, "reject": 0.1}},
    "level": {"type": "score", "score": 1.4, "confidence": 0.3, "probabilities": {"0": 0.0, "1": 0.6, "2": 0.4}},
    "urgent": {"type": "noul", "noul": 0.95},
}


class FakeJev:
    """Plays POST /v1/systemone: replies are consumed in order (a status code or a JSON body)."""

    def __init__(self, *replies: int | dict) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": f"status {reply}"})
        return httpx.Response(200, json=reply)


def make_client(server: FakeJev, usage_path: Path, **kwargs) -> JevClient:
    return JevClient(
        api_key="ts-test",
        usage_path=usage_path,
        transport=httpx.MockTransport(server),
        retry_wait=wait_none(),
        **kwargs,
    )


class JevClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.usage = Path(self.tmp.name) / "llm_usage.jsonl"

    def test_request_shape_answers_and_usage_log(self) -> None:
        usage = {"input_tokens": 300, "output_tokens": 20}
        server = FakeJev({"model": "jev-1.13.0", "answers": ANSWERS, "usage": usage})
        client = make_client(server, self.usage, model="jev-latest")
        answers = client.ask("Help, payouts failing", QUESTIONS)

        self.assertEqual(answers, ANSWERS)
        request = server.requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.url.path, "/v1/systemone")
        self.assertEqual(request.headers["authorization"], "Bearer ts-test")
        body = json.loads(request.content)
        self.assertEqual(body, {"state": "Help, payouts failing", "model": "jev-latest", "questions": QUESTIONS})
        record = json.loads(self.usage.read_text(encoding="utf-8").strip())
        self.assertEqual((record["model"], record["kind"]), ("jev-1.13.0", "jev"))
        self.assertEqual((record["prompt_tokens"], record["completion_tokens"]), (300, 20))
        self.assertEqual(client.usage_summary(), {"calls": 1, "prompt_tokens": 300, "completion_tokens": 20})

    def test_rate_limit_and_overload_are_retried(self) -> None:
        server = FakeJev(429, 529, {"answers": ANSWERS})
        answers = make_client(server, self.usage).ask("state", QUESTIONS)
        self.assertEqual(answers["urgent"]["noul"], 0.95)
        self.assertEqual(len(server.requests), 3)

    def test_retries_exhausted_raise(self) -> None:
        server = FakeJev(503, 503, 503)
        with self.assertRaisesRegex(JevError, "after 3 attempts"):
            make_client(server, self.usage, retries=2).ask("state", QUESTIONS)
        self.assertEqual(len(server.requests), 3)

    def test_bad_key_and_invalid_request_fail_immediately(self) -> None:
        for status, message in ((401, "API key"), (422, "HTTP 422")):
            server = FakeJev(status)
            with self.subTest(status=status), self.assertRaisesRegex(JevError, message):
                make_client(server, self.usage).ask("state", QUESTIONS)
            self.assertEqual(len(server.requests), 1)

    def test_malformed_replies_raise(self) -> None:
        missing = copy.deepcopy(ANSWERS)
        del missing["urgent"]
        unknown = copy.deepcopy(ANSWERS)
        unknown["verdict"]["choice"] = "maybe"
        not_number = copy.deepcopy(ANSWERS)
        not_number["level"]["score"] = "high"
        for name, body in (("missing", {"answers": missing}), ("unknown", {"answers": unknown}),
                           ("not_number", {"answers": not_number}), ("no_answers", {"usage": {}})):
            with self.subTest(name), self.assertRaises(JevError):
                make_client(FakeJev(body), self.usage).ask("state", QUESTIONS)

    def test_missing_key_raises_at_first_call_not_at_construction(self) -> None:
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}):
            client = JevClient(usage_path=self.usage, transport=httpx.MockTransport(FakeJev()))
            with self.assertRaisesRegex(JevError, "TYPESAFE_API_KEY"):
                client.ask("state", QUESTIONS)

    def test_question_builders_validate_limits(self) -> None:
        with self.assertRaises(ValueError):
            score("x", ["only one level"])
        with self.assertRaises(ValueError):
            choice("x", {str(i): None for i in range(256)})
        self.assertEqual(noul("x?", true="yes"), {"type": "noul", "instructions": "x?", "criteria": {"true": "yes"}})


def review_config(backend: str = "jev", **jev: float) -> dict:
    return {
        "orchestration": {
            "review": {
                "backend": backend,
                "jev": jev,
                "batch_size": 2,
                "max_tokens_per_paper": 200,
                "rubric": ["relevance", "technical_depth"],
            }
        }
    }


def jev_answers(p_pass: float, raw_score: float, aspects: list[float]) -> dict:
    answers = {
        "verdict": {
            "type": "choice",
            "choice": "pass" if p_pass >= 0.5 else "reject",
            "confidence": 0.5,
            "probabilities": {"pass": p_pass, "reject": 1 - p_pass},
        },
        "score": {"type": "score", "score": raw_score, "confidence": 0.4},
    }
    for i, prob in enumerate(aspects):
        answers[f"aspect_{i}"] = {"type": "noul", "noul": prob}
    return answers


CANDIDATES = [
    {"paper_id": "P1", "title": "LoRA", "abstract": "Low-rank adapters", "year": 2022,
     "categories": ["cs.CL"], "top_passages": [{"passage_id": "P1_1", "text": "We freeze the weights"}]},
    {"paper_id": "P2", "title": "Cooking", "abstract": "Recipes", "year": 2020,
     "categories": ["cs.AI"], "top_passages": []},
]
CRITERIA = [Criterion("parameter-efficient methods"), Criterion("after 2021"), Criterion("after 2021")]


class JevReviewTests(unittest.TestCase):
    def run_review(self, cfg: dict, jev_client: MagicMock, candidates=CANDIDATES):
        with (
            patch("orchestration.review.get_config", return_value=cfg),
            patch("orchestration.review.get_jev_client", return_value=jev_client),
            patch("orchestration.review.get_client", side_effect=AssertionError("LLM must not be called")),
        ):
            return batch_review(candidates, CRITERIA, "efficient fine-tuning after 2021", 2)

    def test_answers_become_review_records(self) -> None:
        jev = MagicMock()
        jev.ask.side_effect = [jev_answers(0.87, 3.4, [0.9, 0.7]), jev_answers(0.1, 0.2, [0.05, 0.3])]
        first, second = self.run_review(review_config(), jev)

        self.assertEqual((first.paper_id, first.verdict, first.score, first.round), ("P1", "pass", 4, 2))
        self.assertEqual(first.covered, ["parameter-efficient methods", "after 2021"])
        self.assertEqual(first.missing, [])
        self.assertIn("pass probability 0.87", first.reasons)
        self.assertEqual(first.evidence["top_passages"], ["P1_1"])
        self.assertEqual(first.evidence["jev"]["aspect_probabilities"]["after 2021"], 0.7)
        self.assertEqual((second.verdict, second.score, second.covered), ("reject", 1, []))
        self.assertEqual(second.missing, ["parameter-efficient methods", "after 2021"])
        self.assertIn("Not covered", second.reasons)

        state, questions = jev.ask.call_args_list[0].args
        self.assertEqual(state["research_requirement"], "efficient fine-tuning after 2021")
        self.assertEqual(state["paper"]["key_passages"], ["We freeze the weights"])
        # "after 2021" appears twice in CRITERIA but is asked only once
        self.assertEqual(sorted(questions), ["aspect_0", "aspect_1", "score", "verdict"])
        self.assertIn("technical depth", questions["verdict"]["instructions"])
        self.assertEqual(len(questions["score"]["criteria"]), 5)

    def test_thresholds_come_from_config(self) -> None:
        jev = MagicMock()
        jev.ask.return_value = jev_answers(0.8, 2.0, [0.6, 0.6])
        cfg = review_config(pass_threshold=0.9, covered_threshold=0.7)
        record = self.run_review(cfg, jev, CANDIDATES[:1])[0]
        self.assertEqual(record.verdict, "reject")
        self.assertEqual(record.covered, [])
        self.assertEqual(record.score, 3)

    def test_jev_failure_becomes_orchestration_error(self) -> None:
        jev = MagicMock()
        jev.ask.side_effect = JevError("HTTP 401")
        with self.assertRaisesRegex(OrchestrationError, "Jev review failed for P1"):
            self.run_review(review_config(), jev)

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(OrchestrationError, "unknown orchestration.review.backend"):
            self.run_review(review_config(backend="gpt"), MagicMock())


class JevLoopTests(unittest.TestCase):
    def test_run_query_end_to_end_with_jev_reviewer(self) -> None:
        from orchestration import loop
        from orchestration.types import QueryPlan, SubQuery
        from retrieval.types import RetrieveResult, ScoredPassage

        cfg = {
            "kb": {"store": {"sqlite_file": "papers.sqlite"}},
            "orchestration": {
                "num_sub_queries": 2,
                "hyde": {"enabled": False},
                "review": {**review_config()["orchestration"]["review"], "max_concurrency": 2},
                "loop": {"enabled": True, "max_rounds": 2, "min_papers": 5, "cache": False},
                "rewrite": {"max_queries_per_round": 2},
            },
        }

        def jev_server(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            relevant = body["state"]["paper"]["title"].startswith("LoRA")
            return httpx.Response(200, json={
                "model": "jev-1.13.0",
                "answers": jev_answers(0.9 if relevant else 0.2, 3.0 if relevant else 0.0, [0.8 if relevant else 0.1]),
                "usage": {"input_tokens": 250, "output_tokens": 30},
            })

        def passage(paper_id: str, title: str) -> ScoredPassage:
            return ScoredPassage(
                f"{paper_id}_0001", paper_id, f"text of {title}", 1.0, 1, "hybrid", title, 2023, ["cs.CL"]
            )

        with tempfile.TemporaryDirectory() as directory:
            client = JevClient(api_key="ts-test", usage_path=Path(directory) / "usage.jsonl",
                               transport=httpx.MockTransport(jev_server), retry_wait=wait_none())
            with (
                patch("orchestration.loop.get_config", return_value=cfg),
                patch("orchestration.review.get_config", return_value=cfg),
                patch("orchestration.review.get_jev_client", return_value=client),
                patch("orchestration.review.get_client", side_effect=AssertionError("LLM must not be called")),
                patch("orchestration.loop._paper_database_path", return_value=Path("papers.sqlite")),
                patch("orchestration.loop.understand", return_value=QueryPlan(
                    "efficient fine-tuning", [SubQuery("lora"), SubQuery("adapters")],
                    [Criterion("parameter-efficient methods")])),
                patch("orchestration.loop.retrieve", return_value=RetrieveResult(
                    [passage("P1", "LoRA: low-rank adaptation"), passage("P2", "A cooking dataset")], 2)),
                patch("orchestration.loop.load_paper", side_effect=lambda pid, path: {"abstract": f"abstract {pid}"}),
            ):
                result = loop.run_query("efficient fine-tuning")

            self.assertEqual([paper.paper_id for paper in result.papers], ["P1"])
            self.assertEqual(result.papers[0].score, 4)
            self.assertEqual(result.coverage_report["review_backend"], "jev")
            self.assertEqual(result.coverage_report["termination_reason"], "coverage_met")
            self.assertEqual(client.usage_summary()["calls"], 2)  # each paper reviewed once despite two paths


if __name__ == "__main__":
    unittest.main()
