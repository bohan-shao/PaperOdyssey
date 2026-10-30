"""Offline regressions for corpus-aware retrieval, index compatibility and config selection."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import config
from retrieval import pipeline, sparse
from retrieval.lookup import parent_excerpt
from retrieval.query import query_views
from retrieval.selection import select_hits
from eval.retrieval_probe import source_metrics
from tests.test_retrieval import HAS_DENSE, HAS_RANKBM25, HashEncoder, _SparseCases, ranked, write_corpus


class SQLiteSparseTest(_SparseCases, unittest.TestCase):
    backend = "sqlite"

    def test_top_k_no_match_and_punctuation(self) -> None:
        self.assertEqual(len(self.ids("neural networks", top_k=1)), 1)
        self.assertEqual(self.ids("zzzz qqqq"), [])
        self.assertEqual(self.ids(""), [])
        # Porter conflates adapters/adaptation; ranking need not match rank-bm25.
        self.assertEqual(self.ids('LoRA: "low-rank" (adaptation)? AND OR*'),
                         self.ids("LoRA low rank adaptation AND OR"))
        self.assertIn("A", self.ids('LoRA: "low-rank" (adaptation)? AND OR*'))

    def test_filter_before_limit(self) -> None:
        self.assertEqual(self.ids("neural networks", pipeline.RetrievalFilters(year_max=2018), 1), ["C"])

    def test_nonpositive_k(self) -> None:
        self.assertEqual(self.ids("neural", top_k=-1), [])


class QueryAndSelectionTest(unittest.TestCase):
    def test_qwen_reranker_is_not_loaded_as_a_classification_model(self) -> None:
        from retrieval.rerank import Reranker

        with self.assertRaisesRegex(ValueError, "causal yes/no"):
            Reranker("Qwen/Qwen3-Reranker-0.6B")

    def test_normalization_preserves_semantic_query(self) -> None:
        q, lexical = query_views("  What is LoRA LoRA low‑rank adaptation? ",
                                 {"enabled": True, "stopwords": ["what", "is"]})
        self.assertEqual(q, "What is LoRA LoRA low-rank adaptation?")
        self.assertEqual(lexical, "lora low rank adaptation")
        self.assertEqual(query_views(q, {"enabled": False}), (q, q))

    def test_negation_numbers_and_technical_names_survive(self) -> None:
        _, q = query_views("not GPT‑4 but 8-bit QLoRA without training", config.get_config()["retrieval"]["query_processing"])
        self.assertEqual(q, "not gpt 4 but 8 bit qlora without training")

    def test_diversity_backfills_but_never_duplicates_body(self) -> None:
        hits = ranked("hybrid", "a_1", "a_2", "a_3", "b_1", "c_1")
        cfg = {"enabled": True, "candidate_per_paper": 1, "candidate_per_parent": 1}
        out = select_hits(hits, 3, cfg)
        self.assertEqual([h.paper_id for h in out], ["a", "b", "c"])
        self.assertEqual([h.rank for h in out], [1, 2, 3])
        self.assertEqual(select_hits(hits, 5, cfg), hits)
        dup = replace(hits[0], passage_id="a_duplicate", rank=6)
        self.assertEqual(len(select_hits(hits + [dup], 10, cfg)), 5)

    def test_full_fusion_pool_survives_until_selection(self) -> None:
        cfg = copy.deepcopy(config.get_config())
        cfg["retrieval"]["fusion"]["top_k"] = 3
        cfg["retrieval"]["diversity"].update(candidate_per_paper=1, candidate_per_parent=1)
        cfg["retrieval"]["rerank"]["enabled"] = False
        hits = ranked("bm25", "a_1", "a_2", "a_3", "b_1", "c_1")
        with patch.object(pipeline, "get_config", return_value=cfg), \
                patch.object(pipeline, "_ensure_loaded"), \
                patch.object(pipeline, "_sparse_search", return_value=(hits, 1)), \
                patch.object(pipeline, "_dense_search", return_value=(hits, 1)):
            result = pipeline.retrieve("LoRA", top_k=3)
        self.assertEqual([h.paper_id for h in result.hits], ["a", "b", "c"])
        self.assertIn("selection_ms", result.latency_ms)

    def test_empty_request_never_loads_models(self) -> None:
        with patch.object(pipeline, "_ensure_loaded") as loader:
            self.assertEqual(pipeline.retrieve("   ").hits, [])
            self.assertEqual(pipeline.retrieve("LoRA", top_k=-1).hits, [])
            loader.assert_not_called()

    def test_parent_window_contains_evidence_near_end(self) -> None:
        hit = replace(ranked("dense", "a")[0], title="Paper", text="[Paper] Method\nTarget evidence.")
        parent = "[Paper] Method\n" + "earlier material " * 1000 + "Target evidence. " + "later " * 100
        out = parent_excerpt(parent, hit, 150)
        self.assertLessEqual(len(out), 150)
        self.assertIn("Target evidence.", out)
        self.assertTrue(out.startswith("[Paper] Method\n"))
        self.assertEqual(parent_excerpt(parent, replace(hit, text="missing child"), 150), "missing child")
        self.assertEqual(parent_excerpt(parent, hit, 0), parent)


class ConfigSelectionTest(unittest.TestCase):
    def test_custom_index_root_and_legacy_relative_paths(self) -> None:
        cfg = copy.deepcopy(config.get_config())
        cfg["paths"]["indices"] = "data/experiment/indexes"
        for spelling in ("bm25", "indices/bm25"):
            cfg["retrieval"]["sparse"]["index_dir"] = spelling
            self.assertEqual(sparse.index_path(cfg), config.PROJECT_ROOT / "data/experiment/indexes/bm25")

    def test_explicit_path_stays_active_for_package_calls(self) -> None:
        import yaml

        saved, saved_path = config._CONFIG, config._CONFIG_PATH
        try:
            cfg = copy.deepcopy(config.get_config())
            cfg["paths"]["processed"] = "data/test_selection"
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "custom.yaml"
                path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
                explicit = config.get_config(path)
                self.assertIs(config.get_config(), explicit)
                self.assertEqual(config.get_path("processed"), config.PROJECT_ROOT / "data/test_selection")
        finally:
            config._CONFIG, config._CONFIG_PATH = saved, saved_path


@unittest.skipUnless(HAS_DENSE, "needs numpy and faiss")
class DenseManifestTest(unittest.TestCase):
    def test_instruction_formats(self) -> None:
        from retrieval.dense import QUERY_INSTRUCTION, query_instruction

        self.assertEqual(query_instruction("BAAI/bge-large-en-v1.5", {}), QUERY_INSTRUCTION)
        self.assertEqual(query_instruction("sentence-transformers/all-MiniLM-L6-v2", {}), "")
        self.assertEqual(query_instruction("BAAI/bge-m3", {}), "")
        self.assertEqual(query_instruction("Qwen/Qwen3-Embedding-0.6B", {"task_instruction": "Find papers"}),
                         "Instruct: Find papers\nQuery: ")
        self.assertEqual(query_instruction("local", {"query_instruction": "custom: "}), "custom: ")

    def test_manifest_accepts_query_changes_but_rejects_document_changes_and_mapping(self) -> None:
        from retrieval import dense

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            passages, _, lookup = write_corpus(root)
            encoder = HashEncoder()
            dense.build_index(passages, root / "dense", encoder)
            meta_path = root / "dense/meta.json"
            meta = json.loads(meta_path.read_text())
            try:
                meta_path.write_text(json.dumps({**meta, "query_instruction": "another query task"}))
                dense.DenseRetriever(root / "dense", lookup, encoder)
                for field, value in [("document_prompt", "wrong"), ("max_length", 123)]:
                    meta_path.write_text(json.dumps({**meta, field: value}))
                    with self.assertRaisesRegex(RuntimeError, "rebuild"):
                        dense.DenseRetriever(root / "dense", lookup, encoder)
                meta_path.write_text(json.dumps(meta))
                mapping = root / "dense/mapping.jsonl"
                lines = mapping.read_text().splitlines()
                mapping.write_text("\n".join(reversed(lines)) + "\n")
                with self.assertRaisesRegex(RuntimeError, "row order"):
                    dense.DenseRetriever(root / "dense", lookup, encoder)
            finally:
                lookup.close()


class ProbeMetricTest(unittest.TestCase):
    def test_known_source_rank_deduplicates_papers(self) -> None:
        hits = ranked("bm25", "a_1", "a_2", "b_1", "c_1")
        metrics = source_metrics(hits, "b")
        self.assertEqual(metrics["source_recall@10"], 1)
        self.assertEqual(metrics["source_mrr@10"], 0.5)
        self.assertAlmostEqual(metrics["source_ndcg@10"], 1 / 1.584962500721156)
        self.assertEqual(metrics["distinct_papers_in_top_10_passages"], 3)
        self.assertEqual(source_metrics(hits, "missing")["source_recall@10"], 0)


@unittest.skipUnless(HAS_RANKBM25, "needs rank-bm25")
class SparseParameterTest(unittest.TestCase):
    def test_existing_pickle_respects_new_k1_and_b(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            passages, _, lookup = write_corpus(root)
            try:
                sparse.build_index(passages, root / "sparse", backend="rankbm25")
                low = sparse.SparseRetriever(root / "sparse", lookup, backend="rankbm25", k1=0.1, b=0)
                high = sparse.SparseRetriever(root / "sparse", lookup, backend="rankbm25", k1=2.0, b=1)
                first, _ = low.search("low rank adaptation", None, 5)
                second, _ = high.search("low rank adaptation", None, 5)
                self.assertNotEqual([h.score for h in first], [h.score for h in second])
            finally:
                lookup.close()


if __name__ == "__main__":
    unittest.main()
