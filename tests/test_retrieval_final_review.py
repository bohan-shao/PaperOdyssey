"""Offline regressions for PR #16 final review: paths, experiment controls and metrics."""

from __future__ import annotations

import copy
import json
import tempfile
import threading
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import yaml

import config
from eval import latency, retrieval_metrics as metrics, retrieval_probe as probe, runtime
from retrieval import index_paths, pipeline, sparse
from retrieval.query import query_views
from retrieval.types import RetrieveResult
from tests.test_retrieval import HAS_RANKBM25, ranked, write_corpus


class ConfigurationTest(unittest.TestCase):
    def test_on_is_a_string_and_is_removed(self) -> None:
        cfg = config.get_config()["retrieval"]["query_processing"]
        self.assertTrue(all(isinstance(word, str) for word in cfg["stopwords"]))
        self.assertIn("on", cfg["stopwords"])
        self.assertEqual(query_views("Find papers on neural networks", cfg)[1], "neural networks")

    def test_invalid_custom_stopwords_fail_before_retrieval(self) -> None:
        cfg = copy.deepcopy(config.get_config())
        cfg["retrieval"]["query_processing"]["stopwords"] = [True]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.yaml"
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "quote YAML"):
                config._read_config(path)
        with self.assertRaisesRegex(ValueError, "strings"):
            query_views("neural", {"enabled": True, "stopwords": [False]})


class IndexPathTest(unittest.TestCase):
    def test_read_resolution_preserves_a_link_for_builder_validation(self) -> None:
        cfg = copy.deepcopy(config.get_config())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg["paths"]["indices"] = str(root)
            leaf = root / "bm25"
            with patch.object(index_paths, "_is_link", side_effect=lambda p: p == leaf):
                self.assertEqual(index_paths.index_directory("bm25", cfg), leaf)
                with self.assertRaisesRegex(ValueError, "linked index"):
                    index_paths.validate_destination(leaf, root / "passages.jsonl", "sparse")

    def test_system_ancestor_alias_does_not_block_sqlite_build(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            source = root / "passages.jsonl"
            source.write_text(json.dumps({"passage_id": "p", "paper_id": "p", "text": "neural networks"}) + "\n")
            # Model macOS /var being a link while preserving real build/file operations.
            real_is_link = Path.is_symlink
            with patch.object(Path, "is_symlink", lambda p: p == root.parent or real_is_link(p)):
                sparse.build_index(source, root / "index", backend="sqlite")
            self.assertTrue((root / "index/bm25.sqlite").is_file())

    def test_linked_index_is_readable_but_not_replaceable(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            passages, _, store = write_corpus(root)
            try:
                sparse.build_index(passages, root / "actual", backend="sqlite")
                alias = root / "linked"
                try:
                    alias.symlink_to(root / "actual", target_is_directory=True)
                except OSError as exc:
                    self.skipTest(f"native directory symlink unavailable: {exc}")
                try:
                    cfg = copy.deepcopy(config.get_config())
                    cfg["paths"]["indices"] = str(root)
                    cfg["retrieval"]["sparse"]["index_dir"] = "linked"
                    path = sparse.index_path(cfg)
                    self.assertEqual(path, alias)
                    self.assertTrue(sparse.SparseRetriever(path, store, "sqlite").search("neural", None, 5)[0])
                    before = (root / "actual/meta.json").read_bytes()
                    with self.assertRaisesRegex(ValueError, "linked index"):
                        sparse.build_index(passages, path, backend="sqlite")
                    self.assertEqual(before, (root / "actual/meta.json").read_bytes())
                finally:
                    alias.unlink()
            finally:
                store.close()

    def test_build_allows_ancestor_link_and_checks_canonical_shared_root(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            actual, alias = root / "actual", root / "alias"
            actual.mkdir()
            try:
                alias.symlink_to(actual, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"native directory symlink unavailable: {exc}")
            try:
                source = root / "passages.jsonl"
                source.write_text(json.dumps({"passage_id": "p", "paper_id": "p", "text": "neural"}) + "\n")
                cfg = copy.deepcopy(config.get_config())
                cfg["paths"]["indices"] = str(alias)
                with patch.object(index_paths, "get_config", return_value=cfg):
                    sparse.build_index(source, alias / "bm25", backend="sqlite")
                    self.assertTrue((actual / "bm25/bm25.sqlite").is_file())
                    with self.assertRaisesRegex(ValueError, "shared data"):
                        index_paths.validate_destination(actual, source, "sparse")
            finally:
                alias.unlink()

    def test_publish_failure_restores_old_index(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old, new = root / "index", root / "staged"
            for path, value in ((old, "old"), (new, "new")):
                path.mkdir()
                (path / "meta.json").write_text(value)
            rename = Path.rename

            def fail_publish(path, target):
                if path == new:
                    raise PermissionError("simulated publish failure")
                return rename(path, target)

            with patch.object(Path, "rename", fail_publish), self.assertRaises(PermissionError):
                index_paths.publish_index(new, old)
            self.assertEqual((old / "meta.json").read_text(), "old")
            self.assertEqual((new / "meta.json").read_text(), "new")
            self.assertFalse(list(root.glob("*.backup")))


class ExperimentTest(unittest.TestCase):
    def test_loaded_model_rejects_a_new_benchmark_before_mutating_config(self) -> None:
        with patch.object(runtime.dense, "_ENCODER", object()), \
                patch.object(latency, "_run_benchmark") as run, \
                patch.object(latency, "get_config") as read_config, \
                self.assertRaisesRegex(RuntimeError, "fresh dedicated process"):
            latency.run_benchmark()
        run.assert_not_called()
        read_config.assert_not_called()

    def test_latency_baseline_and_snapshot_do_not_inherit_new_switches(self) -> None:
        section = copy.deepcopy(config.get_config()["retrieval"])
        section["query_processing"]["enabled"] = section["diversity"]["enabled"] = True
        section["sparse"]["field_weighting"]["enabled"] = True
        section["sparse"]["top_k"] = section["fusion"]["top_k"] = 250
        original = copy.deepcopy(section)
        seen = []

        def retrieve(*args):
            seen.append(copy.deepcopy(section))
            return RetrieveResult([], 0, {"total_ms": 1})

        with patch.object(latency, "retrieve", side_effect=retrieve):
            row = latency.measure_config(section, ["neural"], {})
        for key, value in latency.BASE.items():
            node = row["effective_config"]
            for part in key.split("."):
                node = node[part]
            self.assertEqual(node, value, key)
        self.assertTrue(all(cfg == row["effective_config"] for cfg in seen))
        self.assertEqual(section, original)

    def test_benchmark_restores_config_and_models_on_failure(self) -> None:
        cfg = copy.deepcopy(config.get_config())
        original = copy.deepcopy(cfg)
        with patch.object(latency, "get_config", return_value=cfg), \
                patch.object(latency, "require_fresh_runtime"), \
                patch.object(latency, "clear_runtime") as clear, \
                patch.object(latency, "_run_benchmark", side_effect=RuntimeError("failure")), \
                self.assertRaisesRegex(RuntimeError, "failure"):
            latency.run_benchmark()
        self.assertEqual(cfg, original)
        clear.assert_called_once_with(release_models=True)

    def test_rerank_sweep_collects_the_requested_depth_and_reports_actual_pairs(self) -> None:
        section = copy.deepcopy(config.get_config()["retrieval"])
        original = copy.deepcopy(section)
        pool = ranked("hybrid", *(str(i) for i in range(150)))

        def retrieve(q, top_k):
            self.assertFalse(section["diversity"]["enabled"])
            size = min(section["sparse"]["top_k"], section["fusion"]["top_k"], top_k)
            return RetrieveResult(pool[:size], size, {})

        with patch.object(latency, "retrieve", side_effect=retrieve):
            candidates, cfg = latency._depth_candidates(section, ["query"], 150)
        self.assertEqual(len(candidates[0]), 150)
        self.assertEqual(cfg["fusion"]["top_k"], 150)
        self.assertEqual(section, original)
        rows = latency.sweep_rerank_depth(["query"], candidates, [50, 150], lambda q, hs, k: hs[:k])
        self.assertEqual([row["actual_pairs"]["mean"] for row in rows], [50, 150])

    def test_sparse_depth_has_a_valid_selection_comparison(self) -> None:
        rows, skipped = probe.ablation_rows(False)
        self.assertIn("depth_only", skipped)
        self.assertNotIn("depth_only", rows)
        self.assertIn("depth_only", probe.ablation_rows(True)[0])
        pool = ranked("bm25", *(f"a_{i}" for i in range(50)), *(f"b{i}_0" for i in range(100)))
        cfg = copy.deepcopy(config.get_config())
        cfg["retrieval"]["dense"]["enabled"] = cfg["retrieval"]["rerank"]["enabled"] = False
        results = []
        with patch.object(pipeline, "get_config", return_value=cfg), patch.object(pipeline, "_ensure_loaded"), \
                patch.object(pipeline, "_sparse_search", side_effect=lambda q, f, k: (pool[:k], 1)):
            for name in ("diversity_only", "depth_with_diversity"):
                with latency.config_overrides(cfg["retrieval"], {"sparse.top_k": 50, **rows[name]}):
                    results.append(pipeline.retrieve("query", top_k=10))
        self.assertEqual([r.total_candidates for r in results], [50, 50])
        self.assertGreater(len({h.paper_id for h in results[1].hits}), len({h.paper_id for h in results[0].hits}))

    def test_probe_rejects_absolute_inputs_before_opening_any_output(self) -> None:
        cfg = copy.deepcopy(config.get_config())
        paths = [str(Path("source.jsonl").resolve()), "../source.jsonl"]
        if Path("C:/").drive:
            paths.extend(["C:source.jsonl", "\\source.jsonl"])
        for path in paths:
            cfg["kb"]["store"]["passages_file"] = path
            with self.subTest(path=path), patch.object(probe, "get_config", return_value=cfg), \
                    patch.object(probe.store, "iter_papers") as reader, self.assertRaisesRegex(ValueError, "relative KB"):
                probe.prepare_sample(1, 1)
            reader.assert_not_called()


class MetricTest(unittest.TestCase):
    def test_probe_uses_canonical_rank_order_and_paper_cutoff(self) -> None:
        hits = ranked("bm25", *(f"a_{i}" for i in range(10)), "source_0", "c_0")
        hits.reverse()
        records = [asdict(h) for h in hits]
        actual = probe.source_metrics(hits, "source")
        self.assertEqual(actual["source_recall@10"], metrics.recall_at_k(records, {"source"}, 10))
        self.assertEqual(actual["source_mrr@10"], 0.5)
        self.assertEqual(actual["source_ndcg@10"], metrics.ndcg_at_k(records, {"source"}))
        self.assertEqual(actual["distinct_papers_in_top_10_passages"], 1)

    def test_invalid_cutoffs_do_not_produce_plausible_metrics(self) -> None:
        for k in (0, -1, True):
            for fn in (metrics.hit_at_k, metrics.precision_at_k, metrics.recall_at_k, metrics.mrr_at_k, metrics.ndcg_at_k):
                with self.subTest(k=k, fn=fn.__name__), self.assertRaises(ValueError):
                    fn([], set(), k)
            with self.assertRaises(ValueError):
                metrics.compute_retrieval_metrics({}, {}, [k])


class RetrievalFailureTest(unittest.TestCase):
    def test_dense_failure_drains_parallel_sparse_work(self) -> None:
        started, release, finished = threading.Event(), threading.Event(), threading.Event()

        def sparse_search(*args):
            started.set()
            self.assertTrue(release.wait(5))
            finished.set()
            raise RuntimeError("sparse failure")

        def dense_search(*args):
            self.assertTrue(started.wait(5))
            release.set()
            raise ValueError("original dense failure")

        cfg = copy.deepcopy(config.get_config())
        with patch.object(pipeline, "get_config", return_value=cfg), patch.object(pipeline, "_ensure_loaded"), \
                patch.object(pipeline, "_sparse_search", side_effect=sparse_search), \
                patch.object(pipeline, "_dense_search", side_effect=dense_search), \
                self.assertRaisesRegex(RuntimeError, "original dense failure"):
            pipeline.retrieve("query")
        self.assertTrue(finished.is_set())

    @unittest.skipUnless(HAS_RANKBM25, "requires rank-bm25")
    def test_rankbm25_keeps_matches_with_nonpositive_idf(self) -> None:
        from retrieval.lookup import LookupStore, build_lookup

        for documents, expected in ((["same", "same"], 2), (["same", "different"], 1)):
            with self.subTest(documents=documents), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source, sections = root / "passages.jsonl", root / "sections.jsonl"
                source.write_text("".join(json.dumps({"passage_id": str(i), "paper_id": str(i), "text": t}) + "\n"
                                          for i, t in enumerate(documents)))
                sections.write_text("")
                build_lookup(source, sections, root / "lookup.sqlite")
                sparse.build_index(source, root / "index", backend="rankbm25")
                with LookupStore(root / "lookup.sqlite") as store:
                    searcher = sparse.SparseRetriever(root / "index", store, "rankbm25")
                    self.assertEqual(len(searcher.search("same", None, 10)[0]), expected)


if __name__ == "__main__":
    unittest.main()
