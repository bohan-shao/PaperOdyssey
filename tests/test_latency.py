"""Offline tests for eval/latency.py (no models, no indexes). Run: python -m unittest tests.test_latency

retrieve() is replaced by fakes, so everything runs on the host except the efSearch sweep test,
which needs numpy + faiss (skipped when missing; run in Docker).
"""

from __future__ import annotations

import importlib.util
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from eval import latency
from kb import store
from retrieval.types import RetrievalFilters, RetrieveResult, ScoredPassage

HAS_FAISS = all(importlib.util.find_spec(m) is not None for m in ("numpy", "faiss"))


def hit(pid: str, rank: int = 1) -> ScoredPassage:
    return ScoredPassage(passage_id=pid, paper_id=pid, text=f"text of {pid}", score=1.0 / rank,
                         rank=rank, stage="hybrid", title=f"Title {pid}", year=2020)


def retrieval_section() -> dict:
    return {"sparse": {"enabled": True, "top_k": 50}, "dense": {"enabled": True, "top_k": 50},
            "rerank": {"enabled": True, "top_k": 10}, "parent_doc": {"enabled": False},
            "filters": {"enabled": True}, "parallel": True}


class StatisticsTest(unittest.TestCase):
    def test_percentile_matches_linear_interpolation(self) -> None:
        values = list(range(1, 101))  # 1..100
        self.assertAlmostEqual(latency.percentile(values, 50), 50.5)
        self.assertAlmostEqual(latency.percentile(values, 95), 95.05)
        self.assertEqual(latency.percentile([7], 95), 7)
        self.assertEqual(latency.percentile([30, 10, 20], 50), 20)  # unsorted input
        with self.assertRaises(ValueError):
            latency.percentile([], 50)

    def test_summarize(self) -> None:
        self.assertEqual(latency.summarize([1, 2, 3, 4]),
                         {"n": 4, "mean": 2.5, "p50": 2.5, "p95": 3.85, "max": 4})
        self.assertEqual(latency.summarize([]), {"n": 0})

    def test_recall_and_overlap(self) -> None:
        self.assertEqual(latency.recall_at_k([3, 1, -1], [1, 2, 3, 4]), 0.5)  # -1 = empty slot
        self.assertEqual(latency.recall_at_k([-1], [-1]), 1.0)
        self.assertEqual(latency.overlap_at_k(["a", "x"], ["a", "b"]), 0.5)
        self.assertEqual(latency.overlap_at_k([], []), 1.0)


class QueriesAndOverridesTest(unittest.TestCase):
    def test_title_queries_are_distinct_normalized_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "papers.sqlite"
            titles = ["Attention Is All You Need", "BERT:\n  Pre-training", "", "   ", "Deep Residual Learning",
                      "Attention Is All You Need", "Adam: A Method"]
            store.upsert_papers([{"paper_id": f"p{i}", "title": t} for i, t in enumerate(titles)], db)
            first = latency.sample_title_queries(db, 3, seed=42)
            self.assertEqual(first, latency.sample_title_queries(db, 3, seed=42))
            self.assertEqual(len(set(first)), 3)
            everything = latency.sample_title_queries(db, 100, seed=1)
            self.assertEqual(sorted(everything), ["Adam: A Method", "Attention Is All You Need",
                                                  "BERT: Pre-training", "Deep Residual Learning"])

    def test_overrides_apply_and_restore(self) -> None:
        section = retrieval_section()
        original = retrieval_section()
        with latency.config_overrides(section, {"rerank.enabled": False, "fusion.top_k": 20}) as live:
            self.assertIs(live, section)  # changed in place: retrieve() reads this dict
            self.assertEqual((section["rerank"]["enabled"], section["fusion"]["top_k"]), (False, 20))
        self.assertEqual(section, original)  # new keys removed too
        with self.assertRaises(KeyError), latency.config_overrides(section, {"dense.enabled": False}):
            raise KeyError("boom")
        self.assertEqual(section, original)  # restored even when the block raises


class FakeRetrieve:
    """Stand-in for retrieve(): latency depends on the live config; optionally sleeps."""

    def __init__(self, section: dict, sleep_s: float = 0.0) -> None:
        self.section, self.sleep_s = section, sleep_s
        self.calls: list[tuple[str, RetrievalFilters | None, dict]] = []
        self.lock = threading.Lock()

    def __call__(self, query: str, filters: RetrievalFilters | None = None, top_k: int | None = None) -> RetrieveResult:
        time.sleep(self.sleep_s)
        enabled = {s: self.section[s]["enabled"] for s in ("sparse", "dense", "rerank", "parent_doc")}
        with self.lock:
            self.calls.append((query, filters, enabled))
        ms = {}
        if enabled["sparse"]:
            ms["sparse_ms"] = 4
        if enabled["dense"]:
            ms["dense_ms"] = 12
        if enabled["rerank"]:
            ms["rerank_ms"] = 390
        ms["total_ms"] = sum(ms.values()) + 1
        hits = [hit(f"{query}-{i}", i) for i in range(1, (top_k or 10) + 1)]
        return RetrieveResult(hits=hits, total_candidates=50, latency_ms=ms)


class MeasurementTest(unittest.TestCase):
    def test_measure_config_uses_the_overrides_and_summarizes_stages(self) -> None:
        section = retrieval_section()
        fake = FakeRetrieve(section)
        recent = RetrievalFilters(year_min=2020)
        with patch.object(latency, "retrieve", fake):
            row = latency.measure_config(section, ["q1", "q2", "q3", "q4"], {"dense.enabled": False,
                                                                           "rerank.enabled": False}, recent)
        measured = fake.calls[latency._UNMEASURED:]  # the first calls warm up and are not counted
        self.assertEqual([q for q, _, _ in measured], ["q1", "q2", "q3", "q4"])
        self.assertTrue(all(f is recent and e == {"sparse": True, "dense": False, "rerank": False, "parent_doc": False}
                            for _, f, e in fake.calls))
        self.assertEqual(set(row["latency_ms"]), {"sparse", "total"})
        self.assertEqual(row["latency_ms"]["total"], {"n": 4, "mean": 5, "p50": 5, "p95": 5, "max": 5})
        self.assertEqual((row["filters"]["year_min"], row["mean_hits"], row["mean_candidates"]), (2020, 10, 50))
        self.assertEqual(section, retrieval_section())  # config restored

    def test_every_row_sets_all_switches(self) -> None:
        section = retrieval_section()
        section["parent_doc"]["enabled"] = True  # whatever config.yaml says...
        fake = FakeRetrieve(section)
        with patch.object(latency, "retrieve", fake):
            latency.measure_config(section, ["q"], {})
        self.assertFalse(fake.calls[-1][2]["parent_doc"])  # ...rows use BASE plus their own overrides

    def test_throughput_scales_with_concurrent_callers(self) -> None:
        section = retrieval_section()
        with patch.object(latency, "retrieve", FakeRetrieve(section, sleep_s=0.02)):
            one, four = latency.measure_throughput(section, [f"q{i}" for i in range(8)], [1, 4])
        self.assertEqual((one["concurrency"], four["concurrency"], four["latency_ms"]["n"]), (1, 4, 8))
        self.assertGreater(four["qps"], 2.5 * one["qps"])  # 8 x 20 ms: ~160 ms alone, ~40 ms with 4 callers

    def test_rerank_depth_sweep(self) -> None:
        cands = [hit(f"p{i}", i) for i in range(1, 51)]
        best = {"p45", "p2", "p30"}  # the "relevant" ones, some deep in the fused list

        def fake_rerank(query: str, candidates: list[ScoredPassage], top_k: int) -> list[ScoredPassage]:
            time.sleep(0.0005 * len(candidates))  # cost grows with the number of pairs
            ordered = sorted(candidates, key=lambda h: (h.passage_id not in best, h.rank))
            return [replace(h, rank=i) for i, h in enumerate(ordered[:top_k], 1)]

        rows = latency.sweep_rerank_depth(["q1", "q2"], [cands, cands], [10, 30, 50], fake_rerank, top_k=5)
        self.assertEqual([r["depth"] for r in rows], [10, 30, 50])
        self.assertLess(rows[0]["rerank_ms"]["p50"], rows[2]["rerank_ms"]["p50"])
        self.assertEqual(rows[-1]["top_k_agreement"], 1.0)  # the deepest run is the reference
        self.assertEqual(rows[0]["top_k_agreement"], 0.6)  # depth 10 misses p30 and p45 -> 3 of 5 agree

    def test_example_results(self) -> None:
        section = retrieval_section()
        with patch.object(latency, "retrieve", FakeRetrieve(section)):
            rows = latency.example_results(section, ["first example", "second example"], top_n=2)
        self.assertEqual([r["query"] for r in rows], ["first example", "second example"])
        self.assertEqual([t["passage_id"] for t in rows[0]["top"]], ["first example-1", "first example-2"])
        self.assertIn("total_ms", rows[0]["latency_ms"])


@unittest.skipUnless(HAS_FAISS, "needs numpy + faiss (run in Docker)")
class EfSearchSweepTest(unittest.TestCase):
    def test_recall_rises_to_exact_with_ef(self) -> None:
        import faiss
        import numpy as np

        rng = np.random.default_rng(0)
        vecs = rng.normal(size=(3000, 32)).astype("float32")
        faiss.normalize_L2(vecs)
        index = faiss.IndexHNSWFlat(32, 8, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 40
        index.add(vecs)
        queries = rng.normal(size=(30, 32)).astype("float32")
        faiss.normalize_L2(queries)
        rows = latency.sweep_ef_search(queries, index, [10, 400], k=20)
        self.assertEqual([r["ef_search"] for r in rows], [10, 400])
        self.assertLessEqual(rows[0]["recall_at_k"], rows[1]["recall_at_k"])
        self.assertGreaterEqual(rows[1]["recall_at_k"], 0.98)
        self.assertEqual(rows[1]["search_ms"]["n"], 30)


class FormatSummaryTest(unittest.TestCase):
    def test_headline_tables(self) -> None:
        stats = {"n": 5, "mean": 9, "p50": 8, "p95": 12, "max": 13}
        report = {
            "baseline": "production", "queries": {"n": 5}, "corpus": {"passages": 1997}, "cold_start_s": 28.4,
            "configs": {"hybrid_rerank": {"latency_ms": {"sparse": stats, "rerank": stats, "total": stats}}},
            "budget": {"config": "hybrid_rerank", "p95_ms": 12, "limit_ms": 500, "within": True},
            "throughput": [{"concurrency": 4, "qps": 2.5, "latency_ms": stats}],
            "ef_search_sweep": {"k": 50, "rows": [{"ef_search": 64, "search_ms": stats, "recall_at_k": 0.99}]},
            "rerank_depth_sweep": {"top_k": 10, "rows": [{"depth": 30, "rerank_ms": stats, "top_k_agreement": 0.9}]},
        }
        text = latency.format_summary(report)
        for expected in ("(production): 5 queries, 1997 passages, cold start 28.4 s", "hybrid_rerank", "sparse 8, rerank 8",
                         "-> within", "recall@50", "top-10 agree", "0.99", "0.9"):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
