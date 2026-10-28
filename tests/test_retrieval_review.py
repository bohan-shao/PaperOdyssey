"""Offline regressions for the verified retrieval PR review findings."""

from __future__ import annotations

import copy
import json
import sqlite3
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import config
from eval import retrieval_probe as probe
from eval import latency
from retrieval import dense, lookup, pipeline, rerank, sparse
from retrieval.index_paths import validate_destination
from retrieval.query import query_views
from retrieval.sqlite_bm25 import SQLiteBM25
from retrieval.types import RetrievalFilters, RetrieveResult
from tests.test_retrieval import HAS_PYSERINI, ranked, write_corpus


class IndexSafetyTest(unittest.TestCase):
    def test_lookup_build_cannot_overwrite_its_source(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("passages.jsonl", "sections.jsonl", "lookup.sqlite.tmp"):
                (root / name).write_text("preserve input", encoding="utf-8")
            for source, target in ((root / "passages.jsonl", root / "passages.jsonl"),
                                   (root / "lookup.sqlite.tmp", root / "lookup.sqlite")):
                with self.assertRaisesRegex(ValueError, "overwrite KB inputs"):
                    lookup.build_lookup(source, root / "sections.jsonl", target)
                self.assertEqual(source.read_text(), "preserve input")

    def test_config_paths_reject_root_aliases_and_traversal(self) -> None:
        cfg = copy.deepcopy(config.get_config())
        for channel, key, resolve in (("sparse", "index_dir", sparse.index_path),
                                      ("dense", "faiss_dir", dense.index_path)):
            for spelling in ("indices", "indices/", ".", "", "../outside", "nested/..",
                             str(config.get_path("indices")), str(config.get_path("indices") / "..")):
                with self.subTest(channel=channel, spelling=spelling):
                    cfg["retrieval"][channel][key] = spelling
                    with self.assertRaises(ValueError):
                        resolve(cfg)

    def test_build_guards_run_before_model_loading_or_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = copy.deepcopy(config.get_config())
            cfg["paths"]["indices"] = str(root / "indices")
            shared = root / "indices"
            shared.mkdir()
            sentinel = shared / "lookup.sqlite"
            sentinel.write_bytes(b"keep shared lookup")
            source = root / "passages.jsonl"
            source.write_text("", encoding="utf-8")
            with patch("retrieval.index_paths.get_config", return_value=cfg), \
                    patch.object(dense, "get_encoder") as encoder:
                for build in (sparse.build_index, dense.build_index):
                    with self.subTest(build=build.__module__), self.assertRaises(ValueError):
                        build(source, shared)
                encoder.assert_not_called()
                self.assertEqual(sentinel.read_bytes(), b"keep shared lookup")
                unknown = root / "unknown"
                unknown.mkdir()
                (unknown / "notes.txt").write_text("keep", encoding="utf-8")
                for target in (root, unknown):
                    with self.assertRaises(ValueError):
                        validate_destination(target, source, "sparse")
                unowned = root / "unowned"
                unowned.mkdir()
                (unowned / "meta.json").write_text('{"notes": "preserve"}')
                with self.assertRaisesRegex(ValueError, "ownership manifest"):
                    validate_destination(unowned, source, "sparse")

    def test_dedicated_absolute_index_can_rebuild_and_preserves_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            passages, _, store = write_corpus(root)
            self.addCleanup(store.close)
            out = root / "sqlite"
            sparse.build_index(passages, out, backend="sqlite")
            sparse.build_index(passages, out, backend="sqlite")
            self.assertTrue(store.path.exists())
            self.assertEqual(sparse.SparseRetriever(out, store, "sqlite").search("neural", None, 2)[0][0].stage, "bm25")
            store.close()


class QueryBudgetTest(unittest.TestCase):
    def test_versions_and_dotted_categories_reach_the_backend(self) -> None:
        for query, expected in (("GPT-3.5", "gpt 3.5"), ("Llama 3.1", "llama 3.1"), ("cs.CL", "cs.cl")):
            self.assertEqual(query_views(query, {"enabled": True})[1], expected)

    @unittest.skipUnless(HAS_PYSERINI, "requires Pyserini and Java")
    def test_lucene_analyzes_original_and_processed_versions_equally(self) -> None:
        from pyserini.analysis import Analyzer, get_lucene_analyzer

        analyzer = Analyzer(get_lucene_analyzer())
        for query in ("GPT-3.5", "Llama 3.1", "cs.CL", "q_bio.NC"):
            self.assertEqual(analyzer.analyze(query_views(query, {"enabled": True})[1]), analyzer.analyze(query))
        self.assertIn("3.5", analyzer.analyze("GPT-3.5"))

    def test_budget_matrix_covers_channels_diversity_reranking_and_empty_pool(self) -> None:
        for channels, diversity, reranking, count in product(((True, False), (False, True), (True, True)),
                                                            (False, True), (False, True), (0, 150)):
            with self.subTest(channels=channels, diversity=diversity, reranking=reranking, count=count):
                cfg = copy.deepcopy(config.get_config())
                r = cfg["retrieval"]
                r["sparse"]["enabled"], r["dense"]["enabled"] = channels
                r["diversity"]["enabled"], r["rerank"]["enabled"] = diversity, reranking
                r["fusion"]["top_k"] = 50
                hits = ranked("bm25", *(f"paper{i}" for i in range(count)))
                with patch.object(pipeline, "get_config", return_value=cfg), \
                        patch.object(pipeline, "_ensure_loaded"), \
                        patch.object(pipeline, "_sparse_search", return_value=(hits, 1)), \
                        patch.object(pipeline, "_dense_search", return_value=(hits, 1)), \
                        patch.object(pipeline, "_rerank", side_effect=lambda q, hs, k: hs[:k]) as score:
                    result = pipeline.retrieve("neural networks", top_k=10)
                self.assertEqual(result.total_candidates, min(count, 50))
                self.assertEqual(len(result.hits), min(count, 10))
                if reranking:
                    self.assertEqual(len(score.call_args.args[1]), min(count, 50))
                else:
                    score.assert_not_called()


class SQLiteReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.passages, _, self.store = write_corpus(self.root)
        self.addCleanup(self.store.close)
        records = [json.loads(line) for line in self.passages.read_text().splitlines()]
        # Exercise normalization on the build side too, including duplicate aliases.
        records[0]["categories"] = ["cs CL", "cs_CL", "cs.CL"]
        self.passages.write_text("".join(json.dumps(r) + "\n" for r in records))
        sparse.build_index(self.passages, self.root / "index", backend="sqlite")
        self.searcher = sparse.SparseRetriever(self.root / "index", self.store, backend="sqlite")

    def test_category_aliases_have_equal_results(self) -> None:
        for query, aliases in (("low rank", ("cs.CL", "cs_CL", "cs CL", "CS.CL")),
                                ("spiking", ("q-bio.NC", "q_bio.NC", "q bio nc"))):
            results = [self.searcher.search(query, RetrievalFilters(categories=[c]), 5)[0] for c in aliases]
            self.assertTrue(results[0])
            self.assertTrue(all(hits == results[0] for hits in results))

    def test_legacy_sqlite_index_requests_rebuild(self) -> None:
        meta = self.root / "index/meta.json"
        doc = json.loads(meta.read_text())
        doc["format_version"] = 1
        meta.write_text(json.dumps(doc))
        with self.assertRaisesRegex(RuntimeError, "rebuild"):
            sparse.SparseRetriever(self.root / "index", self.store, backend="sqlite")

    def test_independent_read_connections_can_execute_concurrently(self) -> None:
        searcher = SQLiteBM25(self.root / "index", [1, 1, 1])
        expected = searcher.search("neural", None, 5)
        connect, barrier = sqlite3.connect, threading.Barrier(2, timeout=5)
        connections = []

        def concurrent_connect(*args, **kwargs):
            conn = connect(*args, **kwargs)
            connections.append(conn)

            def execute(*a):
                barrier.wait()
                return conn.execute(*a)

            return SimpleNamespace(execute=execute, close=conn.close)

        with patch("retrieval.sqlite_bm25.sqlite3.connect", side_effect=concurrent_connect), ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(searcher.search, "neural", None, 5) for _ in range(2)]
            self.assertEqual([f.result() for f in futures], [expected, expected])
        self.assertEqual(len(connections), 2)
        self.assertIsNot(connections[0], connections[1])


class EncoderConfigTest(unittest.TestCase):
    def make_encoder(self, cfg: dict, name: str):
        class Model:
            max_seq_length = 256
            tokenizer = SimpleNamespace(model_max_length=512)

            def __getitem__(self, key):
                return SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512)))

            def get_sentence_embedding_dimension(self):
                return 384

        factory = Mock(return_value=Model())
        modules = {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
                   "sentence_transformers": SimpleNamespace(SentenceTransformer=factory)}
        with patch.dict(sys.modules, modules), patch.object(dense, "get_config", return_value={"retrieval": {"dense": cfg}}):
            encoder = dense.Encoder(name, fp16=False)
        return encoder, factory

    def test_unrelated_model_keeps_its_defaults(self) -> None:
        cfg = {"model_name": "Qwen/Qwen3-Embedding-0.6B", "query_instruction": "Qwen task: ",
               "query_format": "qwen3", "max_length": 1024, "device": "cuda"}
        encoder, factory = self.make_encoder(cfg, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual((encoder.max_length, encoder.query_instruction), (256, ""))
        self.assertEqual(factory.call_args.kwargs["device"], "cpu")

    def test_configured_model_limit_is_capped_at_architecture(self) -> None:
        cfg = {"model_name": "local-MiniLM", "query_instruction": "task: ", "max_length": 1024}
        with self.assertLogs("retrieval.dense", level="WARNING"):
            encoder, _ = self.make_encoder(cfg, cfg["model_name"])
        self.assertEqual((encoder.max_length, encoder.query_instruction), (512, "task: "))
        cfg["max_length"] = 128
        self.assertEqual(self.make_encoder(cfg, cfg["model_name"])[0].max_length, 128)
        cfg["max_length"] = 0
        with self.assertRaises(ValueError):
            self.make_encoder(cfg, cfg["model_name"])


class ParentAndProbeTest(unittest.TestCase):
    def test_parent_benchmark_reports_actual_context_expansion(self) -> None:
        children = ranked("bm25", "a", "b", "c")
        hits = [replace(children[0], parent_section_id="parent", text="expanded context"),
                replace(children[1], parent_section_id="parent"), children[2]]
        fake_store = Mock()
        fake_store.get_passages.return_value = {h.passage_id: {"text": h.text} for h in children}
        with patch.object(latency, "retrieve", return_value=RetrieveResult(hits, 3, {"total_ms": 1})), \
                patch.object(lookup, "get_lookup_store", return_value=fake_store):
            row = latency.measure_config(copy.deepcopy(config.get_config()["retrieval"]), ["query"],
                                         {"parent_doc.enabled": True})
        self.assertEqual(row["parent_text_counts"], {"expanded": 1, "unchanged": 1, "no_parent": 1})

    def test_parent_whitespace_anchor_and_ambiguous_fallback_are_observable(self) -> None:
        child = replace(ranked("dense", "child")[0], title="Title", text="[Title] Method\nTarget evidence here.")
        parent = "[Title] Method\n" + "earlier " * 100 + "Target\nevidence  here." + " later" * 100
        text = lookup.parent_excerpt(parent, child, 100)
        self.assertIn("Target\nevidence  here.", text)
        self.assertLessEqual(len(text), 100)
        for tail, outcome in ((" Target evidence here.", "fallback_ambiguous"), ("", "fallback_unmatched")):
            source = parent + tail if tail else "unrelated " * 100
            with self.assertLogs("retrieval.lookup", level="WARNING") as records:
                self.assertEqual(lookup.parent_excerpt(source, child, 100), child.text)
            self.assertEqual(records.records[0].parent_excerpt_outcome, outcome)

    def test_probe_modes_and_artifact_names_do_not_collide(self) -> None:
        modes = [probe.probe_mode(h, r) for h, r in product((False, True), repeat=2)]
        self.assertEqual(set(modes), {"sparse", "sparse_rerank", "hybrid", "hybrid_rerank"})
        paths = [probe.report_path({"mode": m, "sample": {"run_id": "2026-10-24_run"}}) for m in modes]
        self.assertEqual(len(set(paths)), 4)
        self.assertNotEqual(probe.report_path({"mode": modes[0], "sample": {"run_id": "next"}}), paths[0])

    def test_probe_has_paired_queries_independent_of_tuned_stopwords(self) -> None:
        papers = [{"paper_id": "one", "title": "Find neural papers on learning", "abstract": "Adaptive networks learn"}]
        with patch.object(probe, "get_config", side_effect=AssertionError("must not read retrieval tuning")):
            queries = probe._queries(papers, 1, 42)
        self.assertEqual(queries[0]["template_text"], "Find papers on " + queries[0]["input_text"])
        self.assertFalse(queries[0]["input_text"].startswith("Find papers on "))

    def test_probe_reset_drops_retrievers_before_closing_all_stores(self) -> None:
        store, other = Mock(), Mock()
        with patch.object(lookup, "_STORE", store), \
                patch.object(sparse, "_RETRIEVER", SimpleNamespace(_lookup=store)), \
                patch.object(dense, "_RETRIEVER", SimpleNamespace(_lookup=other)), \
                patch.object(dense, "_ENCODER", object()), patch.object(rerank, "_RERANKER", object()), \
                patch.object(pipeline, "_LOADED", {"sparse", "dense", "rerank"}):
            encoder, scorer = dense._ENCODER, rerank._RERANKER
            probe._clear_runtime()
            store.close.assert_called_once()
            other.close.assert_called_once()
            self.assertIsNone(dense._RETRIEVER)
            self.assertIsNone(lookup._STORE)
            self.assertIs(dense._ENCODER, encoder)
            self.assertIs(rerank._RERANKER, scorer)
            self.assertFalse(pipeline._LOADED)
            probe._clear_runtime(release_models=True)
            self.assertIsNone(dense._ENCODER)
            self.assertIsNone(rerank._RERANKER)

    def test_probe_rejects_directly_loaded_components(self) -> None:
        for module, name in ((lookup, "_STORE"), (sparse, "_RETRIEVER"), (dense, "_ENCODER"),
                             (dense, "_RETRIEVER"), (rerank, "_RERANKER")):
            with self.subTest(name=f"{module.__name__}.{name}"), patch.object(module, name, object()), \
                    patch.object(probe, "prepare_sample") as prepare:
                with self.assertRaisesRegex(RuntimeError, "fresh dedicated process"):
                    probe.run_probe(1, 1)
                prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
