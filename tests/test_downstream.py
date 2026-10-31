"""Offline paper-QA tests: a real lookup store, controlled search/model responses."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from config import get_config
from downstream import DownstreamError, run
from orchestration.types import FinalResult, QueryPlan, ReviewRecord
from retrieval.lookup import LookupStore, build_lookup


def passage(paper_id="2401.00001", suffix="0000", text="The model uses low-rank updates."):
    return {"passage_id": f"P{paper_id}_{suffix}", "paper_id": paper_id, "text": text,
            "arxiv_id": f"{paper_id}v2", "title": f"Paper {paper_id}", "section": "Method", "year": 2024}


def search_result(papers=None):
    if papers is None:
        papers = [ReviewRecord("2401.00001", "pass", 5, evidence={
            "title": "Paper 2401.00001", "top_passages": ["P2401.00001_0000"]})]
    return FinalResult("q-test", "How does it work?", QueryPlan("How does it work?"), papers, {})


def build_store(root, passages):
    """Write passages.jsonl and build the retrieval lookup store from it; returns the db path."""
    (root / "passages.jsonl").write_text("".join(json.dumps(p) + "\n" for p in passages), encoding="utf-8")
    (root / "sections.jsonl").write_text("", encoding="utf-8")
    build_lookup(root / "passages.jsonl", root / "sections.jsonl", root / "lookup.sqlite")
    return root / "lookup.sqlite"


class DownstreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stores = []
        self.cfg = copy.deepcopy(get_config())
        self.search = search_result()
        self.client = MagicMock()
        self.client.chat_json.return_value = {"status": "answered", "answer": "It uses low-rank updates [1]."}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("downstream.qa.get_config", return_value=self.cfg))
        self.store_factory = self.stack.enter_context(patch("downstream.qa.get_lookup_store", side_effect=self.open_store))
        self.client_factory = self.stack.enter_context(patch("downstream.qa.get_client", return_value=self.client))
        self.search_call = self.stack.enter_context(patch("downstream.qa.run_query", return_value=self.search))
        self.write([passage()])

    def write(self, passages):
        # Windows cannot replace a database while a previous subtest still has it open.
        for store in self.stores:
            store.close()
        self.stores.clear()
        self.db = build_store(self.root, passages)

    def open_store(self):
        store = LookupStore(self.db)
        self.stores.append(store)
        self.addCleanup(store.close)
        return store

    def test_answer_uses_real_text_and_one_search_and_generation(self):
        result = run("  How does it work?  ", query_id="custom")
        self.search_call.assert_called_once_with("How does it work?", query_id="custom")
        self.client.chat_json.assert_called_once()
        prompt = json.loads(self.client.chat_json.call_args.args[0][1]["content"])
        self.assertEqual(prompt["paper_excerpts"][0]["text"], passage()["text"])
        self.assertEqual(result.sources[0].url, "https://arxiv.org/abs/2401.00001v2")
        self.assertEqual(result.sources[0].section, "Method")
        self.assertEqual(result.cited_source_numbers, [1])
        self.assertEqual(result.status, "answered")
        self.assertIs(result.search_result, self.search)
        self.assertTrue(all(ms >= 0 for ms in result.latency_by_stage.values()))
        self.assertEqual(json.loads(json.dumps(asdict(result))), asdict(result))

    def test_paper_passage_and_context_limits_and_rejected_papers(self):
        self.cfg["downstream"].update(max_papers=2, passages_per_paper=2, max_chars_per_passage=12, max_tokens=300)
        docs = [passage("2401.00001", str(n)) for n in range(3)] + [passage("2401.00002")]
        self.write(docs)
        records = [
            ReviewRecord("2401.00003", "reject", 5, evidence={"top_passages": ["bad"]}),
            ReviewRecord("2401.00002", "pass", 3, evidence={"top_passages": [docs[-1]["passage_id"]]}),
            ReviewRecord("2401.00001", "pass", 5, evidence={"top_passages": [docs[0]["passage_id"]] + [d["passage_id"] for d in docs[:3]]}),
            ReviewRecord("2401.00001", "pass", 4, evidence={"top_passages": [docs[2]["passage_id"]]}),
            ReviewRecord("2401.00004", "pass", 1, evidence={"top_passages": ["excluded"]}),
        ]
        self.search_call.return_value = search_result(records)
        self.client.chat_json.return_value = {"status": "answered", "answer": "First [1][2], second [3], first again [1]."}
        result = run("Explain these papers")
        self.assertEqual([s.paper_id for s in result.sources], ["2401.00001", "2401.00001", "2401.00002"])
        self.assertEqual(result.cited_source_numbers, [1, 2, 3])
        self.assertTrue(all(len(s.text) == 12 for s in result.sources))
        self.assertEqual(self.client.chat_json.call_args.kwargs["max_tokens"], 300)

    def test_equal_scores_use_the_search_retrieval_ranks(self):
        """With max_papers=1, the better-ranked paper wins a tie even though its ID sorts later."""
        self.cfg["downstream"].update(max_papers=1)
        docs = [passage("2401.00001"), passage("2409.00002")]
        self.write(docs)
        records = [ReviewRecord(d["paper_id"], "pass", 5, evidence={"top_passages": [d["passage_id"]]}) for d in docs]
        search = replace(search_result(records), retrieval_ranks={"2401.00001": 3, "2409.00002": 1})
        self.search_call.return_value = search
        self.client.chat_json.return_value = {"status": "answered", "answer": "It uses low-rank updates [1]."}
        result = run("Explain these papers")
        self.assertEqual([s.paper_id for s in result.sources], ["2409.00002"])

    def test_empty_search_skips_model_and_lookup_store(self):
        self.search_call.return_value = search_result([])
        self.db.unlink()
        result = run("有什么证据？")
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIn("未找到", result.answer)
        self.assertEqual(result.sources, [])
        self.assertEqual(result.latency_by_stage["generation_ms"], 0)
        self.store_factory.assert_not_called()
        self.client_factory.assert_not_called()

    def test_missing_blank_or_wrong_paper_evidence_is_not_used(self):
        for records in [[], [passage(text="  ")], [dict(passage(), paper_id="different-paper")]]:
            with self.subTest(records=records):
                self.write(records)
                self.assertEqual(run("Question").status, "insufficient_evidence")
        self.client_factory.assert_not_called()

    def test_missing_lookup_store_is_an_operational_error(self):
        self.db.unlink()
        with self.assertRaisesRegex(DownstreamError, "lookup store") as caught:
            run("Question")
        self.assertIn("lookup store not found", str(caught.exception.__cause__))
        self.client_factory.assert_not_called()

    def test_invalid_or_uncited_answers_fail_after_one_reask(self):
        # Unparseable JSON is re-asked inside llm.client.chat_json; these parse but are invalid,
        # so they are re-asked once (D50) and fail when the second reply is invalid too.
        invalid = [[], {"status": "wrong", "answer": "text [1]"}, {"status": "answered", "answer": ""}]
        invalid += [{"status": "answered", "answer": answer}
                    for answer in ["No citations", "Unknown [2]", "Only a zero [0]", "Spaced unknown [ 2 ]",
                                   "Only an interval [0, 1]", "Only notation X[1]"]]
        for response in invalid:
            with self.subTest(response=response):
                self.client.chat_json.reset_mock()
                self.client.chat_json.return_value = response
                with self.assertRaisesRegex(DownstreamError, "valid cited answer"):
                    run("Question")
                self.assertEqual(self.client.chat_json.call_count, 2)

    def test_rejected_reply_is_reasked_once_with_the_reason(self):
        rejected = {"status": "answered", "answer": "It uses low-rank updates [2]."}
        self.client.chat_json.side_effect = [rejected, {"status": "answered", "answer": "It uses low-rank updates [1]."}]
        result = run("Question")
        self.assertEqual((result.answer, result.cited_source_numbers), ("It uses low-rank updates [1].", [1]))
        first, second = (call.args[0] for call in self.client.chat_json.call_args_list)
        self.assertEqual(second[:2], first)
        self.assertEqual((second[2]["role"], json.loads(second[2]["content"])), ("assistant", rejected))
        self.assertIn("cites [2], which is not a supplied source", second[3]["content"])
        self.assertIn("source numbers 1 to 1", second[3]["content"])

    def test_index_notation_and_zero_are_not_citations(self):
        """The D49 run's Q0452: answers quoting the paper's X[0] all failed as citing source 0."""
        self.cfg["downstream"]["passages_per_paper"] = 3
        docs = [passage("2401.00001", str(n)) for n in range(3)]
        self.write(docs)
        self.search_call.return_value = search_result([ReviewRecord("2401.00001", "pass", 5, evidence={
            "top_passages": [d["passage_id"] for d in docs]})])
        cases = [
            ("Start from X[0], then Y[l] [1].", [1]),
            ("X[1] grows [2], W_2[1, 2] and A[i][3] stay as written [1].", [2, 1]),
            ("x^{[2]} is defined [3]; a lone [0] is an index [1-2].", [3, 1, 2]),
            ("该方法使用低秩更新[1][3]。", [1, 3]),  # Chinese text attaches citations without a space
        ]
        for raw, cited in cases:
            with self.subTest(raw=raw):
                self.client.chat_json.reset_mock()
                self.client.chat_json.return_value = {"status": "answered", "answer": raw}
                result = run("Question")
                self.assertEqual(result.answer, raw.replace("[1-2]", "[1][2]"))
                self.assertEqual(result.cited_source_numbers, cited)
                self.client.chat_json.assert_called_once()

    def test_grouped_and_padded_citations_are_normalized(self):
        self.cfg["downstream"]["passages_per_paper"] = 3
        docs = [passage("2401.00001", str(n)) for n in range(3)]
        self.write(docs)
        self.search_call.return_value = search_result([ReviewRecord("2401.00001", "pass", 5, evidence={
            "top_passages": [d["passage_id"] for d in docs]})])
        cases = [
            ("A [1, 2] and B [2-3].", "A [1][2] and B [2][3].", [1, 2, 3]),
            ("Padded [01], spaced [ 2 ], repeated [3; 3].", "Padded [1], spaced [2], repeated [3].", [1, 2, 3]),
            # Brackets that are not all supplied source numbers stay text: intervals, year ranges.
            ("Scores lie in [0, 1] over [2019-2021] [1–2].", "Scores lie in [0, 1] over [2019-2021] [1][2].", [1, 2]),
        ]
        for raw, expected, cited in cases:
            with self.subTest(raw=raw):
                self.client.chat_json.return_value = {"status": "answered", "answer": raw}
                result = run("Question")
                self.assertEqual((result.answer, result.cited_source_numbers), (expected, cited))

    def test_model_can_report_insufficient_evidence(self):
        self.client.chat_json.return_value = {"status": "insufficient_evidence", "answer": "The excerpts do not report this result."}
        result = run("What was the accuracy?")
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.cited_source_numbers, [])
        self.assertEqual(len(result.sources), 1)

    def test_model_and_search_failures_are_not_unanswerable_results(self):
        self.client.chat_json.side_effect = RuntimeError("provider failure")
        with self.assertRaisesRegex(DownstreamError, "valid cited answer"):
            run("Question")
        self.search_call.side_effect = NotImplementedError("retrieval stub")
        with self.assertRaisesRegex(DownstreamError, "Paper search failed"):
            run("Question")

    def test_empty_input_does_not_search(self):
        for value in ["", "  ", None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                run(value)
        self.search_call.assert_not_called()

    def test_invalid_configuration(self):
        self.cfg["downstream"]["max_tokens"] = 0
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            run("Question")
        self.search_call.assert_not_called()

    def test_arxiv_links_are_derived_from_valid_ids_only(self):
        self.write([dict(passage(), arxiv_id="https://example.com/untrusted")])
        self.assertIsNone(run("Question").sources[0].url)


class PipelineIntegrationTests(unittest.TestCase):
    def test_real_orchestration_and_sdk_with_controlled_retrieval_and_transport(self):
        from kb.store import upsert_papers
        from llm.client import LLMClient
        from retrieval.types import RetrieveResult, ScoredPassage
        from tests.test_llm import Server, ok
        import httpx2

        cfg = copy.deepcopy(get_config())
        cfg["orchestration"]["loop"].update(cache=False, min_papers=1)
        cfg["orchestration"]["num_sub_queries"] = 1
        server = Server(
            ok('{"sub_queries":[{"query":"low rank","keywords":"low rank","filters":null}],"criteria":[{"criterion":"method","source":"extracted"}]}'),
            ok('[{"paper_id":"2401.00001","verdict":"pass","score":5,"reasons":"Method described","covered":["method"],"missing":[]}]'),
            ok('{"status":"answered","answer":"It uses low-rank updates [1]."}'),
        )
        doc = passage()
        hit = ScoredPassage(doc["passage_id"], doc["paper_id"], doc["text"], 0.9, 1, "rerank", title=doc["title"])
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            lookup = LookupStore(build_store(root, [doc]))
            stack.callback(lookup.close)
            upsert_papers([{"paper_id": doc["paper_id"], "title": doc["title"], "abstract": doc["text"]}], root / "papers.sqlite")
            http = httpx2.Client(transport=httpx2.MockTransport(server))
            stack.callback(http.close)
            client = LLMClient(model="test", base_url="http://mock.local/v1", api_key="sk-test",
                               retries=0, usage_path=root / "usage.jsonl", http_client=http)
            for module in ["downstream.qa", "orchestration.loop", "orchestration.query_understanding", "orchestration.review", "llm.prompt"]:
                stack.enter_context(patch(f"{module}.get_config", return_value=cfg))
            stack.enter_context(patch("orchestration.loop.get_path", return_value=root))
            stack.enter_context(patch("downstream.qa.get_lookup_store", return_value=lookup))
            for module in ["downstream.qa", "orchestration.query_understanding", "orchestration.review"]:
                stack.enter_context(patch(f"{module}.get_client", return_value=client))
            retrieve = stack.enter_context(patch("orchestration.loop.retrieve", return_value=RetrieveResult([hit], 1)))
            result = run("How does this method work?", query_id="integration")
            self.assertEqual(result.query_id, "integration")
            self.assertEqual(result.answer, "It uses low-rank updates [1].")
            self.assertEqual(result.sources[0].text, doc["text"])
            self.assertEqual(len(server.requests), 3)
            self.assertEqual(len((root / "usage.jsonl").read_text().splitlines()), 3)
            retrieve.assert_called_once()


if __name__ == "__main__":
    unittest.main()
