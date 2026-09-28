"""Offline tests for retrieval/ (no models, no network). Run: python -m unittest tests.test_retrieval

The Pyserini tests need Java and the rank-bm25 tests need rank-bm25; each is skipped when its
backend is missing, so run them in Docker (./run_docker.sh python -m unittest tests.test_retrieval).
"""

from __future__ import annotations

import importlib.util
import shutil
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from kb import store
from retrieval import sparse
from retrieval.lookup import LookupStore, build_lookup
from retrieval.types import RetrievalFilters

META = {"arxiv_id": "2401.00001v2", "title": "A Test Paper", "year": 2024, "categories": ["cs.CL", "cs.LG"]}


def passage(paper_id: str, i: int, section_index: int) -> dict:
    return {"passage_id": f"P{paper_id}_{i:04d}", "paper_id": paper_id, **META, "section": "Method",
            "chunk_index": i, "text": f"[A Test Paper] Method\nchunk {i} of {paper_id}",
            "parent_section_id": f"{paper_id}_S{section_index:03d}"}


def section(paper_id: str, section_index: int) -> dict:
    return {"parent_section_id": f"{paper_id}_S{section_index:03d}", "paper_id": paper_id, **META,
            "section": "Method", "section_index": section_index,
            "text": f"[A Test Paper] Method\nwhole section {section_index} of {paper_id}"}


class LookupTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.passages, self.sections, self.db = tmp / "passages.jsonl", tmp / "sections.jsonl", tmp / "lookup.sqlite"
        store.write_passages([passage("2401.00001", i, i // 2) for i in range(4)], self.passages)
        store.write_parents([section("2401.00001", s) for s in range(2)], self.sections)
        self.info = build_lookup(self.passages, self.sections, self.db)
        self.lookup = LookupStore(self.db)

    def tearDown(self) -> None:
        self.lookup.close()
        store.reset_jsonl(self.passages)
        store.reset_jsonl(self.sections)
        self._tmp.cleanup()

    def test_build_counts_and_no_temp_file(self) -> None:
        self.assertEqual((self.info["num_passages"], self.info["num_sections"]), (4, 2))
        self.assertFalse(self.db.with_name(self.db.name + ".tmp").exists())

    def test_hydrate_keeps_order_ranks_and_metadata(self) -> None:
        hits = self.lookup.hydrate([("P2401.00001_0003", 9.5), ("P2401.00001_0000", 7)], stage="bm25")
        self.assertEqual([h.passage_id for h in hits], ["P2401.00001_0003", "P2401.00001_0000"])
        self.assertEqual([h.rank for h in hits], [1, 2])
        h = hits[0]
        self.assertEqual((h.paper_id, h.title, h.year, h.stage, h.score),
                         ("2401.00001", "A Test Paper", 2024, "bm25", 9.5))
        self.assertEqual(h.categories, ["cs.CL", "cs.LG"])
        self.assertEqual(h.parent_section_id, "2401.00001_S001")
        self.assertTrue(h.text.endswith("chunk 3 of 2401.00001"))
        self.assertEqual(self.lookup.hydrate([], stage="dense"), [])

    def test_unknown_id_raises(self) -> None:
        self.assertEqual(self.lookup.get_passages(["nope"]), {})
        with self.assertRaisesRegex(RuntimeError, "out of sync"):
            self.lookup.hydrate([("P2401.00001_0000", 1.0), ("nope", 0.5)], stage="dense")

    def test_parent_text_keeps_child_id(self) -> None:
        hits = self.lookup.hydrate([("P2401.00001_0002", 1.0)], stage="rerank")
        parent = self.lookup.with_parent_text(hits)[0]
        self.assertEqual(parent.passage_id, "P2401.00001_0002")
        self.assertTrue(parent.text.endswith("whole section 1 of 2401.00001"))

    def test_rebuild_replaces_store(self) -> None:
        store.write_passages([passage("2401.00002", 0, 0)], self.passages)
        store.write_parents([section("2401.00002", 0)], self.sections)
        self.assertEqual(build_lookup(self.passages, self.sections, self.db)["num_passages"], 5)
        with LookupStore(self.db) as fresh:
            self.assertIn("P2401.00002_0000", fresh.get_passages(["P2401.00002_0000"]))

    def test_duplicate_id_fails_and_keeps_old_store(self) -> None:
        dup = Path(self._tmp.name) / "dup.jsonl"
        dup.write_text(self.passages.read_text() * 2, encoding="utf-8")
        with self.assertRaises(sqlite3.IntegrityError):
            build_lookup(dup, self.sections, self.db)
        self.assertEqual(len(self.lookup.get_passages(["P2401.00001_0000"])), 1)
        self.assertFalse(self.db.with_name(self.db.name + ".tmp").exists())

    def test_read_only_and_threads(self) -> None:
        with self.assertRaises(sqlite3.OperationalError):
            self.lookup._conn().execute("DELETE FROM passages")
        results: list[int] = []
        threads = [threading.Thread(target=lambda: results.append(
            len(self.lookup.get_passages(f"P2401.00001_{i:04d}" for i in range(4))))) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results, [4, 4, 4, 4])

    def test_missing_store_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "build_indexes"):
            LookupStore(Path(self._tmp.name) / "missing.sqlite")


HAS_PYSERINI = importlib.util.find_spec("pyserini") is not None and shutil.which("java") is not None
HAS_RANKBM25 = importlib.util.find_spec("rank_bm25") is not None

# (passage_id suffix, year, categories, text): one passage per paper, distinct topics.
CORPUS = [
    ("A", 2019, ["cs.CL", "cs.LG"], "Low-rank adaptation fine-tunes large language models with few trainable parameters."),
    ("B", 2023, ["cs.CV"], "Vision transformers split images into patches for classification."),
    ("C", 2016, ["eess.AS", "cs.CL"], "Speech recognition with recurrent neural networks and attention."),
    ("D", 2021, ["q-bio.NC"], "Spiking neural networks simulate biological neurons."),
    ("E", 2024, ["cs.LG"], "Parameter-efficient fine-tuning surveys adapters, prefix tuning and low-rank methods."),
    ("F", 2015, ["stat.ML"], "Gaussian processes for regression and uncertainty estimation."),
]


class _SparseCases:
    """BM25 contract tests, run once per backend by the subclasses below."""

    backend = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.passages, cls.sections = tmp / "passages.jsonl", tmp / "sections.jsonl"
        for key, year, cats, text in CORPUS:
            pid = f"2401.0000{key}"
            store.write_passages([{
                "passage_id": f"P{pid}_0000", "paper_id": pid, "arxiv_id": pid + "v1", "title": f"Paper {key}",
                "section": "Abstract", "year": year, "categories": cats, "chunk_index": 0,
                "text": f"[Paper {key}] Abstract\n{text}", "parent_section_id": f"{pid}_S000"}], cls.passages)
            store.write_parents([{
                "parent_section_id": f"{pid}_S000", "paper_id": pid, "arxiv_id": pid + "v1", "title": f"Paper {key}",
                "section": "Abstract", "year": year, "categories": cats, "section_index": 0,
                "text": f"[Paper {key}] Abstract\n{text}"}], cls.sections)
        build_lookup(cls.passages, cls.sections, tmp / "lookup.sqlite")
        cls.lookup = LookupStore(tmp / "lookup.sqlite")
        cls.index_dir = tmp / "bm25"
        cls.info = sparse.build_index(cls.passages, cls.index_dir, backend=cls.backend, threads=1)
        cls.retriever = sparse.SparseRetriever(cls.index_dir, cls.lookup, backend=cls.backend)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.lookup.close()
        store.reset_jsonl(cls.passages)
        store.reset_jsonl(cls.sections)
        cls._tmp.cleanup()

    def ids(self, query: str, filters: RetrievalFilters | None = None, top_k: int = 10) -> list[str]:
        hits, _ = self.retriever.search(query, filters, top_k)
        return [h.paper_id[-1] for h in hits]  # "2401.0000A" -> "A"

    def test_build_replaces_and_counts(self) -> None:
        self.assertEqual((self.info["backend"], self.info["num_docs"]), (self.backend, len(CORPUS)))
        self.assertEqual(sorted(p.name for p in self.index_dir.parent.iterdir() if p.name.endswith(".tmp")), [])

    def test_best_match_first_and_hydrated(self) -> None:
        hits, latency_ms = self.retriever.search("low-rank adaptation of language models", None, 10)
        self.assertEqual(hits[0].passage_id, "P2401.0000A_0000")
        self.assertIn("P2401.0000E_0000", [h.passage_id for h in hits])  # shares "low-rank"
        self.assertEqual([h.rank for h in hits], list(range(1, len(hits) + 1)))
        self.assertEqual([h.score for h in hits], sorted((h.score for h in hits), reverse=True))
        self.assertEqual((hits[0].stage, hits[0].year, hits[0].categories), ("bm25", 2019, ["cs.CL", "cs.LG"]))
        self.assertIsInstance(latency_ms, int)

    def test_top_k_no_match_and_punctuation(self) -> None:
        self.assertEqual(len(self.ids("neural networks", top_k=1)), 1)
        self.assertEqual(self.ids("zzzz qqqq"), [])
        self.assertEqual(self.ids(""), [])
        self.assertEqual(self.ids('LoRA: "low-rank" (adaptation)? AND OR*')[0], "A")

    def test_year_filter_restricts_without_rescoring(self) -> None:
        self.assertEqual(sorted(self.ids("neural networks")), ["C", "D"])
        self.assertEqual(self.ids("neural networks", RetrievalFilters(year_min=2020)), ["D"])
        self.assertEqual(self.ids("neural networks", RetrievalFilters(year_max=2018)), ["C"])
        self.assertEqual(self.ids("neural networks", RetrievalFilters(year_min=2016, year_max=2016)), ["C"])
        all_hits, _ = self.retriever.search("neural networks", None, 10)
        only_d, _ = self.retriever.search("neural networks", RetrievalFilters(year_min=2020), 10)
        self.assertAlmostEqual(only_d[0].score, next(h.score for h in all_hits if h.passage_id == only_d[0].passage_id), places=4)

    def test_category_filter_handles_awkward_names(self) -> None:
        self.assertEqual(self.ids("neural networks", RetrievalFilters(categories=["q-bio.NC"])), ["D"])
        self.assertEqual(self.ids("neural networks", RetrievalFilters(categories=["eess.AS"])), ["C"])
        self.assertEqual(sorted(self.ids("neural networks", RetrievalFilters(categories=["q-bio.NC", "eess.AS"]))), ["C", "D"])
        self.assertEqual(self.ids("neural networks", RetrievalFilters(categories=["cs.XX"])), [])
        self.assertEqual(sorted(self.ids("neural networks", RetrievalFilters(categories=[]))), ["C", "D"])
        self.assertEqual(self.ids("neural networks", RetrievalFilters(year_min=2020, categories=["eess.AS"])), [])

    def test_open_errors(self) -> None:
        other = "rankbm25" if self.backend == "pyserini" else "pyserini"
        with self.assertRaisesRegex(RuntimeError, "rebuild"):
            sparse.SparseRetriever(self.index_dir, self.lookup, backend=other)
        with self.assertRaisesRegex(RuntimeError, "build_indexes"):
            sparse.SparseRetriever(self.index_dir / "missing", self.lookup, backend=self.backend)


@unittest.skipUnless(HAS_PYSERINI, "needs pyserini + Java (run in Docker)")
class PyseriniSparseTest(_SparseCases, unittest.TestCase):
    backend = "pyserini"


@unittest.skipUnless(HAS_RANKBM25, "needs rank-bm25 (run in Docker)")
class RankBM25SparseTest(_SparseCases, unittest.TestCase):
    backend = "rankbm25"


class CategoryTokenTest(unittest.TestCase):
    def test_one_token_per_category(self) -> None:
        self.assertEqual([sparse.category_token(c) for c in ["cs.CL", "q-bio.NC", "physics.soc-ph"]],
                         ["cs_cl", "q_bio_nc", "physics_soc_ph"])


if __name__ == "__main__":
    unittest.main()
