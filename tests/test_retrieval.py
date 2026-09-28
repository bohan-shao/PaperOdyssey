"""Offline tests for retrieval/ (no network). Run: python -m unittest tests.test_retrieval

The Pyserini tests need Java, the rank-bm25 tests rank-bm25, the dense tests numpy + faiss (with a
hashed bag-of-words encoder standing in for BGE), the real-encoder test the cached
all-MiniLM-L6-v2 model and the real-reranker test the cached bge-reranker-v2-m3. Each is skipped
when its dependency is missing, so run them in Docker
(./run_docker.sh python -m unittest tests.test_retrieval).
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
import zlib
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from kb import store
from retrieval import pipeline, rerank, sparse
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.lookup import LookupStore, build_lookup
from retrieval.types import RetrievalFilters, ScoredPassage

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

    def test_short_lived_threads_share_one_connection(self) -> None:
        conns: list[sqlite3.Connection] = []
        for _ in range(5):  # a fresh thread per query, as with asyncio.run or a Streamlit rerun
            t = threading.Thread(target=lambda: (self.lookup.get_passages(["P2401.00001_0000"]),
                                                 conns.append(self.lookup._conn())))
            t.start()
            t.join()
        self.assertTrue(all(c is conns[0] for c in conns))
        self.lookup.close()
        self.assertEqual(len(self.lookup.get_passages(["P2401.00001_0000"])), 1)  # reopens

    def test_missing_store_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "build_indexes"):
            LookupStore(Path(self._tmp.name) / "missing.sqlite")


HAS_PYSERINI = importlib.util.find_spec("pyserini") is not None and shutil.which("java") is not None
HAS_RANKBM25 = importlib.util.find_spec("rank_bm25") is not None
HAS_DENSE = all(importlib.util.find_spec(m) is not None for m in ("numpy", "faiss"))
HAS_MINILM = HAS_RERANKER = False
if HAS_DENSE:
    import numpy as np

    from retrieval import dense
if importlib.util.find_spec("sentence_transformers") is not None:
    from huggingface_hub import try_to_load_from_cache

    HAS_MINILM = isinstance(try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json"), str)
    HAS_RERANKER = isinstance(try_to_load_from_cache("BAAI/bge-reranker-v2-m3", "config.json"), str)

# (passage_id suffix, year, categories, text): one passage per paper, distinct topics.
CORPUS = [
    ("A", 2019, ["cs.CL", "cs.LG"], "Low-rank adaptation fine-tunes large language models with few trainable parameters."),
    ("B", 2023, ["cs.CV"], "Vision transformers split images into patches for classification."),
    ("C", 2016, ["eess.AS", "cs.CL"], "Speech recognition with recurrent neural networks and attention."),
    ("D", 2021, ["q-bio.NC"], "Spiking neural networks simulate biological neurons."),
    ("E", 2024, ["cs.LG"], "Parameter-efficient fine-tuning surveys adapters, prefix tuning and low-rank methods."),
    ("F", 2015, ["stat.ML"], "Gaussian processes for regression and uncertainty estimation."),
]


def write_corpus(tmp: Path) -> tuple[Path, Path, LookupStore]:
    """Write CORPUS as passages.jsonl + sections.jsonl under tmp and open a lookup store over it."""
    passages, sections = tmp / "passages.jsonl", tmp / "sections.jsonl"
    for key, year, cats, text in CORPUS:
        pid = f"2401.0000{key}"
        meta = {"paper_id": pid, "arxiv_id": pid + "v1", "title": f"Paper {key}", "section": "Abstract",
                "year": year, "categories": cats, "text": f"[Paper {key}] Abstract\n{text}"}
        store.write_passages([{**meta, "passage_id": f"P{pid}_0000", "chunk_index": 0,
                               "parent_section_id": f"{pid}_S000"}], passages)
        store.write_parents([{**meta, "parent_section_id": f"{pid}_S000", "section_index": 0}], sections)
    build_lookup(passages, sections, tmp / "lookup.sqlite")
    return passages, sections, LookupStore(tmp / "lookup.sqlite")


class _SparseCases:
    """BM25 contract tests, run once per backend by the subclasses below."""

    backend = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.passages, cls.sections, cls.lookup = write_corpus(tmp)
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


class HashEncoder:
    """Offline stand-in for BGE: hashed bag-of-words unit vectors, so shared words raise the cosine."""

    def __init__(self, name: str = "hash-bow", dim: int = 256) -> None:
        self.name, self.dim, self.batch_size = name, dim, 64
        self.seen: list[str] = []

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        self.seen.extend(texts)
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                vecs[row, zlib.crc32(token.encode()) % self.dim] += 1.0
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)

    def encode_query(self, query: str) -> np.ndarray:
        return dense.Encoder.encode_query(self, query)  # the real prefix logic


@unittest.skipUnless(HAS_DENSE, "needs numpy + faiss (run in Docker)")
class DenseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.passages, cls.sections, cls.lookup = write_corpus(tmp)
        cls.index_dir = tmp / "faiss"
        cls.encoder = HashEncoder()
        cls.info = dense.build_index(cls.passages, cls.index_dir, encoder=cls.encoder)
        # exact_max=1000: every filtered search is exact; exact_max=0: HNSW with the ID selector.
        cls.exact = dense.DenseRetriever(cls.index_dir, cls.lookup, encoder=cls.encoder, exact_max=1000)
        cls.hnsw = dense.DenseRetriever(cls.index_dir, cls.lookup, encoder=cls.encoder, exact_max=0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.lookup.close()
        store.reset_jsonl(cls.passages)
        store.reset_jsonl(cls.sections)
        cls._tmp.cleanup()

    @staticmethod
    def ids(retriever: dense.DenseRetriever, query: str, filters: RetrievalFilters | None = None,
            top_k: int = 10) -> list[str]:
        hits, _ = retriever.search(query, filters, top_k)
        return [h.paper_id[-1] for h in hits]  # "2401.0000A" -> "A"

    def test_build_layout_mapping_and_rebuild(self) -> None:
        self.assertEqual((self.info["model_name"], self.info["dim"], self.info["num_vectors"]),
                         ("hash-bow", 256, len(CORPUS)))
        self.assertEqual(sorted(p.name for p in self.index_dir.iterdir()), ["index.faiss", "mapping.jsonl", "meta.json"])
        rows = [json.loads(line) for line in (self.index_dir / "mapping.jsonl").read_text().splitlines()]
        self.assertEqual([(r["row"], r["passage_id"]) for r in rows],
                         [(i, f"P2401.0000{key}_0000") for i, (key, *_) in enumerate(CORPUS)])
        self.assertEqual(rows[3]["categories"], ["q-bio.NC"])
        meta = json.loads((self.index_dir / "meta.json").read_text())
        self.assertEqual((meta["hnsw"]["M"], meta["query_instruction"]), (32, dense.QUERY_INSTRUCTION))
        again = dense.build_index(self.passages, self.index_dir, encoder=HashEncoder())
        self.assertEqual(again["num_vectors"], len(CORPUS))
        self.assertEqual([p.name for p in self.index_dir.parent.iterdir() if p.name.endswith(".tmp")], [])

    def test_best_match_first_and_hydrated(self) -> None:
        hits, latency_ms = self.exact.search("low-rank adaptation of language models", None, 10)
        self.assertEqual(hits[0].passage_id, "P2401.0000A_0000")
        self.assertEqual(len(hits), len(CORPUS))  # dense always returns the nearest neighbours
        self.assertEqual([h.rank for h in hits], list(range(1, len(hits) + 1)))
        self.assertEqual([h.score for h in hits], sorted((h.score for h in hits), reverse=True))
        self.assertLessEqual(hits[0].score, 1.0 + 1e-5)  # cosine of unit vectors
        self.assertEqual((hits[0].stage, hits[0].year, hits[0].categories), ("dense", 2019, ["cs.CL", "cs.LG"]))
        self.assertIsInstance(latency_ms, int)

    def test_query_gets_bge_instruction(self) -> None:
        self.exact.search("spiking neurons", None, 3)
        self.assertEqual(self.encoder.seen[-1], dense.QUERY_INSTRUCTION + "spiking neurons")

    def test_top_k_and_empty_query(self) -> None:
        self.assertEqual(len(self.ids(self.exact, "neural networks", top_k=2)), 2)
        self.assertEqual(self.ids(self.exact, ""), [])
        self.assertEqual(self.ids(self.exact, "   "), [])

    def test_filters_on_exact_and_hnsw_paths(self) -> None:
        for name, r in (("exact", self.exact), ("hnsw", self.hnsw)):
            with self.subTest(path=name):
                q = "spiking neural networks"
                self.assertEqual(self.ids(r, q)[0], "D")
                recent = self.ids(r, q, RetrievalFilters(year_min=2020))
                self.assertEqual((recent[0], sorted(recent)), ("D", ["B", "D", "E"]))
                self.assertEqual(sorted(self.ids(r, q, RetrievalFilters(year_max=2018))), ["C", "F"])
                self.assertEqual(self.ids(r, q, RetrievalFilters(categories=["q-bio.NC"])), ["D"])
                self.assertEqual(self.ids(r, q, RetrievalFilters(categories=["eess.AS"])), ["C"])
                self.assertEqual(sorted(self.ids(r, q, RetrievalFilters(categories=["q-bio.NC", "eess.AS"]))), ["C", "D"])
                self.assertEqual(self.ids(r, q, RetrievalFilters(categories=["cs.XX"])), [])
                self.assertEqual(len(self.ids(r, q, RetrievalFilters(categories=[]))), len(CORPUS))
                self.assertEqual(self.ids(r, q, RetrievalFilters(year_min=2020, categories=["eess.AS"])), [])
                all_hits, _ = r.search(q, None, 10)
                only_d, _ = r.search(q, RetrievalFilters(categories=["q-bio.NC"]), 10)
                self.assertAlmostEqual(only_d[0].score, all_hits[0].score, places=5)

    def test_open_errors(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "rebuild"):
            dense.DenseRetriever(self.index_dir, self.lookup, encoder=HashEncoder(name="other-model"))
        with self.assertRaisesRegex(RuntimeError, "dim"):
            dense.DenseRetriever(self.index_dir, self.lookup, encoder=HashEncoder(dim=128))
        with self.assertRaisesRegex(RuntimeError, "build_indexes"):
            dense.DenseRetriever(self.index_dir / "missing", self.lookup, encoder=self.encoder)


@unittest.skipUnless(HAS_DENSE and HAS_MINILM, "needs sentence-transformers + cached all-MiniLM-L6-v2 (run in Docker)")
class RealEncoderTest(unittest.TestCase):
    def test_unit_vectors_similarity_and_query_prefix(self) -> None:
        enc = dense.Encoder("sentence-transformers/all-MiniLM-L6-v2", batch_size=8)
        vecs = enc.encode(["A cat sits on the mat.", "A kitten is sitting on a rug.", "Stock markets fell sharply today."])
        self.assertEqual((vecs.shape, vecs.dtype), ((3, enc.dim), np.float32))
        np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3)
        self.assertGreater(vecs[0] @ vecs[1], vecs[0] @ vecs[2])
        np.testing.assert_allclose(enc.encode_query("cat on a mat"),
                                   enc.encode([dense.QUERY_INSTRUCTION + "cat on a mat"]), atol=1e-3)


def ranked(stage: str, *pids: str, scores: list[float] | None = None) -> list[ScoredPassage]:
    """Hits in the given order, ranks 1..n (scores default to a descending 10, 9, ...)."""
    scores = scores or [10.0 - i for i in range(len(pids))]
    return [ScoredPassage(passage_id=pid, paper_id=pid.split("_")[0], text=f"text of {pid}", score=score,
                          rank=i, stage=stage, year=2020) for i, (pid, score) in enumerate(zip(pids, scores), 1)]


class FusionTest(unittest.TestCase):
    def test_scores_follow_the_rrf_formula(self) -> None:
        fused = reciprocal_rank_fusion(ranked("bm25", "a", "b", "c"), ranked("dense", "c", "d"))
        by_id = {h.passage_id: h.score for h in fused}
        self.assertAlmostEqual(by_id["c"], 1 / 63 + 1 / 61)  # rank 3 in sparse, rank 1 in dense
        self.assertAlmostEqual(by_id["a"], 1 / 61)
        self.assertAlmostEqual(by_id["d"], 1 / 62)

    def test_output_is_deduplicated_sorted_and_hybrid(self) -> None:
        fused = reciprocal_rank_fusion(ranked("bm25", "a", "b", "c"), ranked("dense", "c", "b", "d"))
        # c = 1/63 + 1/61 = 0.032266 edges out b = 2/62 = 0.032258; both beat single-list a and d
        self.assertEqual([h.passage_id for h in fused], ["c", "b", "a", "d"])
        self.assertEqual([h.rank for h in fused], [1, 2, 3, 4])
        self.assertEqual({h.stage for h in fused}, {"hybrid"})
        self.assertEqual([h.score for h in fused], sorted((h.score for h in fused), reverse=True))
        self.assertEqual((fused[0].text, fused[0].year, fused[0].paper_id), ("text of c", 2020, "c"))

    def test_only_ranks_matter_not_score_scale(self) -> None:
        sparse_hits = ranked("bm25", "a", "b", "c", scores=[25.1, 11.7, 3.2])
        dense_hits = ranked("dense", "b", "d", "a", scores=[0.83, 0.79, 0.41])
        rescaled = ranked("bm25", "a", "b", "c", scores=[0.3, 0.2, 0.1])
        self.assertEqual(reciprocal_rank_fusion(sparse_hits, dense_hits),
                         reciprocal_rank_fusion(rescaled, dense_hits))

    def test_k_controls_how_much_the_top_rank_dominates(self) -> None:
        sparse_hits, dense_hits = ranked("bm25", "top", "x", "both"), ranked("dense", "y", "z", "both")
        self.assertEqual(reciprocal_rank_fusion(sparse_hits, dense_hits, k=0)[0].passage_id, "top")  # 1/1 > 2/3
        self.assertEqual(reciprocal_rank_fusion(sparse_hits, dense_hits, k=60)[0].passage_id, "both")  # 2/63 > 1/61

    def test_weights(self) -> None:
        sparse_hits, dense_hits = ranked("bm25", "s1", "s2"), ranked("dense", "d1", "d2")
        self.assertEqual([h.passage_id for h in reciprocal_rank_fusion(sparse_hits, dense_hits, weights=(2.0, 1.0))][:2],
                         ["s1", "s2"])  # 2/61, 2/62 > 1/61
        self.assertEqual([h.passage_id for h in reciprocal_rank_fusion(sparse_hits, dense_hits, weights=[0.0, 1.0])],
                         ["d1", "d2"])  # a weight-0 list adds no candidates

    def test_ties_are_deterministic(self) -> None:
        fused = reciprocal_rank_fusion(ranked("bm25", "zeta", "b"), ranked("dense", "alpha", "c"))
        self.assertEqual([h.passage_id for h in fused], ["alpha", "zeta", "b", "c"])  # equal 1/61: by id

    def test_duplicate_within_one_list_counts_once_at_best_rank(self) -> None:
        sparse_hits = ranked("bm25", "a", "b") + [replace(ranked("bm25", "a")[0], rank=5)]
        self.assertAlmostEqual(reciprocal_rank_fusion(sparse_hits, [])[0].score, 1 / 61)

    def test_top_k_and_empty_inputs(self) -> None:
        self.assertEqual(len(reciprocal_rank_fusion(ranked("bm25", "a", "b", "c"), ranked("dense", "d"), top_k=2)), 2)
        self.assertEqual(reciprocal_rank_fusion([], []), [])
        self.assertEqual([h.passage_id for h in reciprocal_rank_fusion([], ranked("dense", "d", "e"))], ["d", "e"])

    def test_inputs_not_modified(self) -> None:
        sparse_hits = ranked("bm25", "a", "b")
        before = list(sparse_hits)
        reciprocal_rank_fusion(sparse_hits, ranked("dense", "b"))
        self.assertEqual(sparse_hits, before)
        self.assertEqual(sparse_hits[0].stage, "bm25")

    def test_invalid_arguments(self) -> None:
        hits = ranked("bm25", "a")
        for kwargs in ({"k": -1}, {"weights": (-1.0, 1.0)}, {"weights": (0.0, 0.0)}, {"weights": (1.0,)}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                reciprocal_rank_fusion(hits, [], **kwargs)
        with self.assertRaisesRegex(ValueError, "1-based"):
            reciprocal_rank_fusion([replace(hits[0], rank=0)], [])


class RankByScoresTest(unittest.TestCase):
    def test_orders_by_new_score_and_keeps_passage_fields(self) -> None:
        fused = reciprocal_rank_fusion(ranked("bm25", "a", "b", "c"), ranked("dense", "c", "d"))
        out = rerank.rank_by_scores(fused, [0.2 if h.passage_id != "d" else 0.9 for h in fused], top_k=2)
        self.assertEqual([h.passage_id for h in out], ["d", "c"])  # d: 0.9; then the 0.2 tie goes to fused rank
        self.assertEqual([(h.rank, h.stage) for h in out], [(1, "rerank"), (2, "rerank")])
        self.assertEqual([h.score for h in out], [0.9, 0.2])
        self.assertEqual((out[0].text, out[0].paper_id, out[0].year), ("text of d", "d", 2020))

    def test_top_k_and_length_mismatch(self) -> None:
        hits = ranked("hybrid", "a", "b", "c")
        self.assertEqual(len(rerank.rank_by_scores(hits, [0.1, 0.2, 0.3], top_k=10)), 3)
        self.assertEqual(rerank.rank_by_scores(hits, [0.1, 0.2, 0.3], top_k=0), [])
        with self.assertRaises(ValueError):
            rerank.rank_by_scores(hits, [0.1, 0.2], top_k=2)


class OverlapScorer:
    """Stand-in for the cross-encoder: fraction of query words found in the passage."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        words = set(re.findall(r"[a-z]+", query.lower()))
        return [len(words & set(re.findall(r"[a-z]+", t.lower()))) / len(words) for t in texts]

    def rerank(self, query: str, candidates: list[ScoredPassage], top_k: int = 10) -> list[ScoredPassage]:
        return rerank.Reranker.rerank(self, query, candidates, top_k)  # the real rerank logic


class RerankerTest(unittest.TestCase):
    def test_scores_query_with_child_text_and_reorders(self) -> None:
        candidates = [replace(h, text=t) for h, t in zip(ranked("hybrid", "a", "b", "c"), [
            "Gaussian processes for regression.",
            "Speech recognition with recurrent networks.",
            "Low-rank adaptation of large language models."])]
        scorer = OverlapScorer()
        out = scorer.rerank("low-rank adaptation for language models", candidates, top_k=2)
        self.assertEqual(scorer.calls, [("low-rank adaptation for language models", [c.text for c in candidates])])
        self.assertEqual([h.passage_id for h in out], ["c", "a"])  # c overlaps most; a's "for" beats b
        self.assertEqual([h.rank for h in out], [1, 2])

    def test_nothing_to_rerank(self) -> None:
        scorer = OverlapScorer()
        self.assertEqual(scorer.rerank("query", [], top_k=10), [])
        self.assertEqual(scorer.rerank("query", ranked("hybrid", "a"), top_k=0), [])
        self.assertEqual(scorer.calls, [])  # the model is not called at all


@unittest.skipUnless(HAS_RERANKER, "needs sentence-transformers + cached bge-reranker-v2-m3 (run in Docker)")
class RealRerankerTest(unittest.TestCase):
    def test_relevant_passage_first_with_scores_in_unit_range(self) -> None:
        texts = ["Photosynthesis converts light into chemical energy in plants.",
                 "The Eiffel Tower is a landmark in Paris.",
                 "Paris is the capital and largest city of France."]
        candidates = [replace(h, text=t) for h, t in zip(ranked("hybrid", "bio", "tower", "capital"), texts)]
        out = rerank.Reranker("BAAI/bge-reranker-v2-m3").rerank("What is the capital of France?", candidates, top_k=3)
        self.assertEqual([h.passage_id for h in out][0], "capital")
        self.assertEqual([h.passage_id for h in out][-1], "bio")
        self.assertTrue(all(0.0 <= h.score <= 1.0 for h in out))
        self.assertGreater(out[0].score, 0.5)
        self.assertLess(out[-1].score, 0.1)


def pipeline_cfg(overrides: dict[str, object]) -> dict:
    """The retrieval config section with every stage on; overrides like {"rerank.enabled": False}."""
    cfg = {
        "sparse": {"enabled": True, "top_k": 50}, "dense": {"enabled": True, "top_k": 50},
        "fusion": {"rrf_k": 60, "weights": [1.0, 1.0], "top_k": 50},
        "rerank": {"enabled": True, "top_k": 10}, "parent_doc": {"enabled": False},
        "filters": {"enabled": True}, "parallel": True,
    }
    for key, value in overrides.items():
        section, _, name = key.rpartition(".")
        (cfg[section] if section else cfg)[name] = value
    return {"retrieval": cfg}


class FakeStages:
    """Records every stage call. sparse -> a, b, c; dense -> c, d; the reranker puts d first."""

    def __init__(self, fail: dict[str, Exception] | None = None) -> None:
        self.calls: list[dict] = []
        self.fail = fail or {}

    def _record(self, stage: str, **info: object) -> None:
        self.calls.append({"stage": stage, "thread": threading.get_ident(), **info})
        if stage in self.fail:
            raise self.fail[stage]

    def stages(self) -> list[str]:
        return [c["stage"] for c in self.calls]

    def call(self, stage: str) -> dict:
        return next(c for c in self.calls if c["stage"] == stage)

    def sparse(self, query: str, filters: RetrievalFilters | None, top_k: int) -> tuple[list[ScoredPassage], int]:
        self._record("sparse", filters=filters, top_k=top_k)
        return ranked("bm25", "a", "b", "c"), 3

    def dense(self, query: str, filters: RetrievalFilters | None, top_k: int) -> tuple[list[ScoredPassage], int]:
        self._record("dense", filters=filters, top_k=top_k)
        return ranked("dense", "c", "d"), 7

    def rerank(self, query: str, candidates: list[ScoredPassage], top_k: int) -> list[ScoredPassage]:
        self._record("rerank", ids=[c.passage_id for c in candidates], texts=[c.text for c in candidates], top_k=top_k)
        return rerank.rank_by_scores(candidates, [0.9 if c.passage_id == "d" else 0.1 for c in candidates], top_k)

    def with_parent_text(self, hits: list[ScoredPassage]) -> list[ScoredPassage]:
        self._record("parent")
        return [replace(h, text=f"parent of {h.passage_id}") for h in hits]


class PipelineTest(unittest.TestCase):
    def run_pipeline(self, overrides: dict[str, object] | None = None, filters: RetrievalFilters | None = None,
                     top_k: int | None = None, stages: FakeStages | None = None) -> tuple:
        stages = stages or FakeStages()
        with patch.object(pipeline, "get_config", return_value=pipeline_cfg(overrides or {})), \
                patch.object(pipeline, "_ensure_loaded", lambda cfg: None), \
                patch.object(pipeline, "_sparse_search", stages.sparse), \
                patch.object(pipeline, "_dense_search", stages.dense), \
                patch.object(pipeline, "_rerank", stages.rerank), \
                patch.object(pipeline, "get_lookup_store", return_value=stages):
            return pipeline.retrieve("low-rank adaptation", filters, top_k), stages

    @staticmethod
    def ids(result: object) -> list[str]:
        return [h.passage_id for h in result.hits]

    def test_default_config_is_hybrid_plus_rerank(self) -> None:
        result, stages = self.run_pipeline()
        self.assertEqual(stages.call("rerank")["ids"], ["c", "a", "b", "d"])  # fused order goes in
        self.assertEqual(self.ids(result), ["d", "c", "a", "b"])  # reranked order comes out
        self.assertEqual({h.stage for h in result.hits}, {"rerank"})
        self.assertEqual(result.total_candidates, 4)
        self.assertEqual((stages.call("sparse")["top_k"], stages.call("dense")["top_k"]), (50, 50))
        self.assertEqual(set(result.latency_ms), {"sparse_ms", "dense_ms", "fusion_ms", "rerank_ms", "total_ms"})
        self.assertEqual((result.latency_ms["sparse_ms"], result.latency_ms["dense_ms"]), (3, 7))

    def test_ablation_matrix_is_config_only(self) -> None:
        rows = {  # EVALUATION.md section 2
            "bm25": ({"dense.enabled": False, "rerank.enabled": False}, ["a", "b", "c"], ["sparse"]),
            "dense": ({"sparse.enabled": False, "rerank.enabled": False}, ["c", "d"], ["dense"]),
            "hybrid": ({"rerank.enabled": False}, ["c", "a", "b", "d"], ["sparse", "dense"]),
            "rerank": ({}, ["d", "c", "a", "b"], ["sparse", "dense", "rerank"]),
        }
        for stage, (overrides, expected_ids, expected_stages) in rows.items():
            with self.subTest(row=stage):
                result, stages = self.run_pipeline(overrides)
                self.assertEqual(self.ids(result), expected_ids)
                self.assertEqual({h.stage for h in result.hits}, {stage})
                self.assertEqual(sorted(stages.stages()), sorted(expected_stages))
                self.assertEqual("fusion_ms" in result.latency_ms, stage in ("hybrid", "rerank"))

    def test_single_retriever_goes_to_rerank_without_fusion(self) -> None:
        result, stages = self.run_pipeline({"dense.enabled": False})
        self.assertEqual(stages.call("rerank")["ids"], ["a", "b", "c"])
        self.assertNotIn("fusion_ms", result.latency_ms)

    def test_top_k_default_and_override(self) -> None:
        result, stages = self.run_pipeline({"rerank.top_k": 2})
        self.assertEqual((len(result.hits), stages.call("rerank")["top_k"]), (2, 2))
        self.assertEqual(len(self.run_pipeline(top_k=1)[0].hits), 1)
        self.assertEqual(self.ids(self.run_pipeline({"rerank.enabled": False}, top_k=3)[0]), ["c", "a", "b"])

    def test_filters_switch(self) -> None:
        recent = RetrievalFilters(year_min=2020)
        _, stages = self.run_pipeline(filters=recent)
        self.assertEqual((stages.call("sparse")["filters"], stages.call("dense")["filters"]), (recent, recent))
        _, stages = self.run_pipeline({"filters.enabled": False}, filters=recent)
        self.assertEqual((stages.call("sparse")["filters"], stages.call("dense")["filters"]), (None, None))

    def test_parent_doc_after_rerank_keeps_child_ids(self) -> None:
        result, stages = self.run_pipeline({"parent_doc.enabled": True})
        self.assertTrue(all(t.startswith("text of") for t in stages.call("rerank")["texts"]))  # child text scored
        self.assertEqual([h.text for h in result.hits][:2], ["parent of d", "parent of c"])
        self.assertEqual(self.ids(result), ["d", "c", "a", "b"])
        self.assertIn("parent_ms", result.latency_ms)
        _, stages = self.run_pipeline()
        self.assertNotIn("parent", stages.stages())

    def test_parallel_switch(self) -> None:
        caller = threading.get_ident()
        _, stages = self.run_pipeline()
        self.assertNotEqual(stages.call("sparse")["thread"], caller)  # worker thread
        self.assertEqual(stages.call("dense")["thread"], caller)
        _, stages = self.run_pipeline({"parallel": False})
        self.assertEqual(stages.call("sparse")["thread"], caller)

    def test_fusion_settings_are_used(self) -> None:
        result, stages = self.run_pipeline({"fusion.top_k": 2})
        self.assertEqual((result.total_candidates, stages.call("rerank")["ids"]), (2, ["c", "a"]))
        result, _ = self.run_pipeline({"fusion.weights": [1.0, 0.0], "rerank.enabled": False})
        self.assertEqual(self.ids(result), ["a", "b", "c"])  # dense-only d gets no weight

    def test_failures_name_the_stage(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "both disabled"):
            self.run_pipeline({"sparse.enabled": False, "dense.enabled": False})
        cases = {
            "sparse": (ValueError("JVM crashed"), "sparse retrieval failed: JVM crashed"),
            "dense": (RuntimeError("FAISS index not found"), "dense retrieval failed: FAISS index not found"),
            "rerank": (OSError("model missing"), "rerank failed: model missing"),
        }
        for stage, (exc, message) in cases.items():
            with self.subTest(stage=stage), self.assertRaisesRegex(RuntimeError, message):
                self.run_pipeline(stages=FakeStages(fail={stage: exc}))

    def test_components_load_once_one_at_a_time(self) -> None:
        active, max_active, loads = [0], [0], []
        lock = threading.Lock()

        def loader(name: str):
            def load() -> None:
                with lock:
                    active[0] += 1
                    max_active[0] = max(max_active[0], active[0])
                    loads.append(name)
                time.sleep(0.02)  # long enough for the other threads to pile up
                with lock:
                    active[0] -= 1
            return load

        fakes = {name: loader(name) for name, _ in pipeline._COMPONENTS}
        cfg = pipeline_cfg({})["retrieval"]
        with patch.object(pipeline, "_loaders", return_value=fakes), patch.object(pipeline, "_LOADED", set()):
            threads = [threading.Thread(target=pipeline._ensure_loaded, args=(cfg,)) for _ in range(4)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            self.assertEqual(loads, ["sparse", "dense", "rerank"])  # enabled ones, each once; parent_doc is off
            self.assertEqual(max_active[0], 1)  # never two at the same time
            pipeline._ensure_loaded(pipeline_cfg({"parent_doc.enabled": True})["retrieval"])
            self.assertEqual(loads[-1], "parent_doc")  # switched on later: loaded then

    def test_load_failure_names_the_component(self) -> None:
        def missing() -> None:
            raise RuntimeError("BM25 index not found")

        fakes = {name: (missing if name == "sparse" else (lambda: None)) for name, _ in pipeline._COMPONENTS}
        with patch.object(pipeline, "_loaders", return_value=fakes), patch.object(pipeline, "_LOADED", set()):
            with self.assertRaisesRegex(RuntimeError, "loading sparse failed: BM25 index not found"):
                pipeline._ensure_loaded(pipeline_cfg({})["retrieval"])
            self.assertNotIn("sparse", pipeline._LOADED)  # retried on the next call


class CategoryTokenTest(unittest.TestCase):
    def test_one_token_per_category(self) -> None:
        self.assertEqual([sparse.category_token(c) for c in ["cs.CL", "q-bio.NC", "physics.soc-ph"]],
                         ["cs_cl", "q_bio_nc", "physics_soc_ph"])


if __name__ == "__main__":
    unittest.main()
