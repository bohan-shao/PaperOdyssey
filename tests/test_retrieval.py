"""Offline tests for retrieval/ (no models, no indexes). Run: python -m unittest tests.test_retrieval"""

from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from kb import store
from retrieval.lookup import LookupStore, build_lookup

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


if __name__ == "__main__":
    unittest.main()
