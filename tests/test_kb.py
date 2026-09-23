"""Offline tests for kb/ (no network, no models). Run: python -m unittest tests.test_kb"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pymupdf

from kb import chunk, fetch, stats, store
from kb.parse import ParseOptions, PDFParseError, parse_pdf, quality_check

META = {"arxiv_id": "2401.00001v2", "title": "A Test Paper", "year": 2024, "categories": ["cs.CL"]}
BODY = ("Retrieval augmented generation combines a retriever with a generator. "
        "The retriever selects passages from a corpus. The generator conditions on them. ") * 6


def make_pdf(path: Path) -> None:
    """Two-page PDF: title, headings (bold), body, running header, page numbers, references."""
    doc = pymupdf.open()
    for pno in range(2):
        page = doc.new_page()
        page.insert_text((72, 30), "Preprint. Under review.", fontsize=8)  # running header
        page.insert_text((300, 820), str(pno + 1), fontsize=9)  # page number
        y = 80
        if pno == 0:
            page.insert_text((72, y), "A Test Paper", fontsize=16, fontname="hebo")
            y += 40
            blocks = [("1 Introduction", BODY), ("2 Method", BODY)]
        else:
            blocks = [("3 Experiments", BODY), ("References", "[1] Some cited work. 2020.")]
        for heading, text in blocks:
            page.insert_text((72, y), heading, fontsize=12, fontname="hebo")
            y += 22
            rect = pymupdf.Rect(72, y, 520, y + 260)
            page.insert_textbox(rect, text, fontsize=10, fontname="helv")
            y += 280
    doc.save(str(path))


class ParseTest(unittest.TestCase):
    def test_sections_headers_and_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "p.pdf"
            make_pdf(pdf)
            secs = parse_pdf(pdf, ParseOptions(min_words=50))
            names = [s["section"] for s in secs]
            self.assertEqual(names, ["Introduction", "Method", "Experiments"])
            text = " ".join(s["text"] for s in secs)
            self.assertNotIn("Preprint. Under review", text)  # R2 running header
            self.assertNotIn("Some cited work", text)  # R1 reference truncation
            self.assertNotIn("A Test Paper", text)  # R5 front matter
            q = quality_check(pdf, secs, ParseOptions(min_words=50))
            self.assertTrue(q["ok"])
            self.assertEqual(q["num_sections"], 3)

    def test_heading_and_noise_rules(self) -> None:
        from kb.parse import _Line, _heading_name, _is_table_block, _normalize_heading, _run_in_heading

        opts = ParseOptions()

        def line(text: str, bold: bool = True, size: float = 10.0, **kw) -> _Line:
            return _Line(text, size, bold, 0, 100, 110, 72, 1, **kw)

        self.assertEqual(_heading_name(line("3.2 Model Architecture"), 10.0, opts), ("Model Architecture", 2))
        self.assertEqual(_heading_name(line("II. RELATED WORK", bold=False), 10.0, opts), ("Related Work", 1))
        self.assertIsNone(_heading_name(line("7 else"), 10.0, opts))  # numbered pseudo-code line
        self.assertIsNone(_heading_name(line("Training"), 10.0, opts, numbered_mode=True))  # bold table cell
        self.assertIsNone(_heading_name(line("2 Method", size=8.0), 10.0, opts))  # smaller than body
        self.assertIsNone(_heading_name(line("A Multi-task Framework for Emotion"), 10.0, opts))  # title, not "A."
        self.assertEqual(_heading_name(line("Bibliographical References", bold=False), 10.0, opts)[0],
                         "Bibliographical References")
        self.assertEqual(_normalize_heading("BERT"), "BERT")
        self.assertEqual(_normalize_heading("AREN'T EXISTING SOLUTIONS GOOD"), "Aren't Existing Solutions Good")
        run_in = line("2.1. Complexity analysis. In this subsection we prove", bold=False,
                      bold_prefix="2.1. Complexity analysis.")
        self.assertEqual(_run_in_heading(run_in, 10.0), ("Complexity analysis", 2, "In this subsection we prove"))
        self.assertTrue(_is_table_block(["Model", "MRR HITS@1", "HITS@3 HITS@10"]))
        self.assertFalse(_is_table_block(["Table 6: Confusion", "matrix for", "emotions."]))
        self.assertFalse(_is_table_block(["We train the", "model with a", "learning rate of", "0.1 for ten epochs."]))

    def test_corrupt_pdf_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.pdf"
            bad.write_bytes(b"%PDF-1.4 not really a pdf")
            with self.assertRaises(PDFParseError):
                parse_pdf(bad)


class ChunkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = chunk.get_tokenizer()
        self.sections = [{"section": "Introduction", "text": BODY * 4}, {"section": "Method", "text": BODY}]

    def test_split_sentences_keeps_abbreviations(self) -> None:
        sents = chunk.split_sentences(
            "As shown by Vaswani et al. (2017), attention works. See Fig. 3 for details. It is e.g. fast."
        )
        self.assertEqual(sents, [
            "As shown by Vaswani et al. (2017), attention works.",
            "See Fig. 3 for details.",
            "It is e.g. fast.",
        ])

    def test_fixed_chunking_contract(self) -> None:
        ps = chunk.chunk_section_fixed("2401.00001", self.sections, 120, 40, meta=META, tokenizer=self.tok)
        self.assertGreater(len(ps), 3)
        self.assertEqual([p["chunk_index"] for p in ps], list(range(len(ps))))
        self.assertEqual(ps[0]["passage_id"], "P2401.00001_0000")
        for p in ps:
            prefix, body = p["text"].split("\n", 1)
            self.assertTrue(prefix.startswith("[A Test Paper] "))
            self.assertLessEqual(self.tok.count(body), 120 + 50)  # budget (+ merged tail allowance)
            self.assertEqual(p["year"], 2024)
            self.assertEqual(p["categories"], ["cs.CL"])
        # chunks never cross sections; parent ids follow section index
        self.assertEqual({p["parent_section_id"] for p in ps if p["section"] == "Method"}, {"2401.00001_S001"})
        # adjacent chunks of one section overlap by whole sentences
        intro = [p["text"].split("\n", 1)[1] for p in ps if p["section"] == "Introduction"]
        first_sentence_of_second = chunk.split_sentences(intro[1])[0]
        self.assertIn(first_sentence_of_second, intro[0])

    def test_overlap_must_be_smaller(self) -> None:
        with self.assertRaises(ValueError):
            chunk.chunk_section_fixed("x", self.sections, 100, 100)

    def test_semantic_chunking_with_stub_embeddings(self) -> None:
        import numpy as np

        def embed(sents: list[str]) -> np.ndarray:
            # two "topics": sentences mentioning 'retriever' vs the rest
            vecs = np.array([[1.0, 0.0] if "retriever" in s else [0.0, 1.0] for s in sents])
            return vecs

        ps = chunk.chunk_semantic("2401.00001", self.sections, 0.5, 10, 200, meta=META,
                                  tokenizer=self.tok, embed_fn=embed)
        self.assertGreater(len(ps), 2)
        self.assertEqual([p["chunk_index"] for p in ps], list(range(len(ps))))

    def test_parent_records_and_map(self) -> None:
        ps = chunk.chunk_section_fixed("2401.00001", self.sections, 120, 40, meta=META)
        parents = chunk.build_parent_records("2401.00001", self.sections, meta=META)
        self.assertEqual([p["parent_section_id"] for p in parents], ["2401.00001_S000", "2401.00001_S001"])
        stripped = [{k: v for k, v in p.items() if k != "parent_section_id"} for p in ps]
        mapped = chunk.build_parent_map(stripped, self.sections)
        self.assertEqual([p["parent_section_id"] for p in mapped], [p["parent_section_id"] for p in ps])


class StoreTest(unittest.TestCase):
    def test_jsonl_idempotent_and_torn_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "passages.jsonl"
            a = [{"paper_id": "A", "passage_id": "PA_0000", "text": "old"}]
            b = [{"paper_id": "B", "passage_id": "PB_0000", "text": "b"}]
            store.write_passages(a, path)
            store.write_passages(b, path)
            store.write_passages([{"paper_id": "A", "passage_id": "PA_0000", "text": "new"}], path)
            rows = list(store.iter_passages(path))
            self.assertEqual(sorted((r["paper_id"], r["text"]) for r in rows), [("A", "new"), ("B", "b")])
            # simulate a crash mid-write, then a fresh process resuming
            with path.open("a") as fh:
                fh.write('{"paper_id": "C", "pass')
            store._paper_ids_cache.clear()
            self.assertEqual(store.stored_paper_ids(path), {"A", "B"})
            self.assertEqual(len(list(store.iter_passages(path))), 2)

    def test_sqlite_upsert_preserves_other_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "papers.sqlite"
            store.upsert_papers([{"paper_id": "A", "title": "T", "categories": ["cs.CL"], "year": 2024}], db)
            store.set_num_chunks({"A": 7}, db)
            store.upsert_papers([{"paper_id": "A", "title": "T2", "abstract": "abs"}], db)
            p = store.load_paper("A", db)
            self.assertEqual((p["title"], p["num_chunks"], p["categories"], p["abstract"]), ("T2", 7, ["cs.CL"], "abs"))
            self.assertIsNone(store.load_paper("missing", db))
            cols = [r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(papers)")]
            self.assertEqual(cols, store.PAPER_COLUMNS)


class StatsTest(unittest.TestCase):
    def test_counts_use_sections_without_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ps_path, sec_path = Path(tmp) / "p.jsonl", Path(tmp) / "s.jsonl"
            a = {"paper_id": "A", "section": "Intro", "year": 2020, "categories": ["cs.CL"]}
            b = {"paper_id": "B", "section": "", "year": 2021, "categories": ["cs.LG"]}
            store.write_passages([
                {**a, "text": "[T] Intro\nHello world. Foo"},  # chunks overlap on "Foo"
                {**a, "text": "[T] Intro\nFoo bar!"},
                {**b, "text": "[U] Body\nhello"},
            ], ps_path)
            store.write_parents([
                {"paper_id": "A", "parent_section_id": "A_S000", "text": "[T] Intro\nHello world. Foo bar!"},
                {"paper_id": "B", "parent_section_id": "B_S000", "text": "[U] Body\nhello"},
            ], sec_path)
            s = stats.compute_corpus_stats(ps_path, sec_path)
            self.assertEqual((s["n_documents"], s["n_chunks"], s["n_words"], s["n_types"]), (2, 3, 5, 4))
            self.assertEqual(s["by_year"], {"2020": 1, "2021": 1})
            stats.write_stats_report(s, Path(tmp) / "out.json")
            self.assertEqual(json.loads((Path(tmp) / "out.json").read_text())["n_words"], 5)


class FetchTest(unittest.TestCase):
    ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/1810.04805v2</id>
    <title>BERT:  Pre-training
      of Deep Transformers</title>
    <summary> We introduce BERT. </summary>
    <published>2018-10-11T00:00:00Z</published>
    <category term="cs.LG"/><category term="cs.CL"/>
    <arxiv:primary_category term="cs.CL"/>
  </entry>
</feed>"""

    def test_parse_atom(self) -> None:
        total, recs = fetch._parse_atom_feed(self.ATOM)
        self.assertEqual(total, 1)
        self.assertEqual(recs[0], {
            "paper_id": "1810.04805", "arxiv_id": "1810.04805v2", "title": "BERT: Pre-training of Deep Transformers",
            "abstract": "We introduce BERT.", "year": 2018, "categories": ["cs.CL", "cs.LG"],
        })

    def test_select_papers_is_year_stratified(self) -> None:
        rows = [{"paper_id": f"{y}-{i}", "year": y, "categories": ["cs.CL"], "citation_count": c}
                for y, cites in ((2018, [500, 400, 300, 200]), (2019, [5, 4, 3, 2])) for i, c in enumerate(cites)]
        rows.append({"paper_id": "x", "year": 2019, "categories": ["cs.CV"], "citation_count": 999})
        df = pd.DataFrame(rows)
        sel = fetch.select_papers(df, citation_min=0, target_papers=4, year_min=2018, year_max=2019,
                                  categories=["cs.CL"])
        self.assertEqual(len(sel), 4)
        self.assertNotIn("x", set(sel["paper_id"]))  # category filter
        self.assertEqual(sorted(sel["year"]), [2018, 2018, 2019, 2019])  # low-citation years keep their quota

    def test_metadata_csv_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame([{"paper_id": "0801.0001", "arxiv_id": "0801.0001v1", "title": "t", "abstract": "a",
                                "year": 2015, "categories": ["cs.CL"], "citation_count": 3, "pdf_path": ""}])
            fetch.write_metadata_csv(df, Path(tmp) / "m.csv")
            back = fetch.read_metadata_csv(Path(tmp) / "m.csv")
            self.assertEqual(back.loc[0, "paper_id"], "0801.0001")  # leading zero kept
            self.assertEqual(back.loc[0, "categories"], ["cs.CL"])
            self.assertEqual(back.loc[0, "pdf_path"], "")

    def test_backoff_honours_retry_after(self) -> None:
        from unittest.mock import MagicMock

        wait = fetch._backoff(multiplier=1, max_s=10)
        state = MagicMock(attempt_number=1)
        state.outcome.exception.return_value = fetch._RetryableHTTPError("429", retry_after=42.0)
        self.assertEqual(wait(state), 42.0)
        state.outcome.exception.return_value = fetch._RetryableHTTPError("503")
        self.assertLessEqual(wait(state), 10)

    def test_openalex_quota_stops_and_keeps_cache(self) -> None:
        from unittest.mock import patch

        import httpx

        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                work = {"cited_by_count": 7, "locations": [{"landing_page_url": "http://arxiv.org/abs/2401.00001"}]}
                return httpx.Response(200, json={"results": [work]})
            return httpx.Response(429, headers={"x-ratelimit-remaining": "0"})

        real_client = httpx.Client
        df = pd.DataFrame({"paper_id": ["2401.00001", "2401.00002"]})
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            fetch.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)
        ):
            cache = Path(tmp) / "cit.jsonl"
            with self.assertRaises(fetch.OpenAlexQuotaError):
                fetch.enrich_citations(df, batch_size=1, cache_path=cache, request_interval_s=0)
            self.assertEqual(json.loads(cache.read_text()), {"paper_id": "2401.00001", "citation_count": 7})

    def test_clean_setting(self) -> None:
        self.assertEqual(fetch.clean_setting("${OPENALEX_API_KEY}"), "")
        self.assertEqual(fetch.clean_setting(" key "), "key")


if __name__ == "__main__":
    unittest.main()
