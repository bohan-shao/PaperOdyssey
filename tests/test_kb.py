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
        self.assertEqual(_heading_name(line("INTRODUCTION", size=24.8), 10.0, opts, numbered_mode=True),
                         ("Introduction", 1))  # a thesis chapter title
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

    def test_heading_false_positives_and_acm_style(self) -> None:
        from kb.parse import _heading_name, _is_mojibake, _Line, _numbered_run_in, _repair

        opts = ParseOptions()

        def line(text: str, bold: bool = True, size: float = 10.0, **kw) -> _Line:
            return _Line(text, size, bold, 0, 100, 110, 72, 1, **kw)

        # ACM journals: numbered bold headings set at ~0.9x body size
        self.assertEqual(_heading_name(line("1. INTRODUCTION", size=9.0), 10.0, opts), ("Introduction", 1))
        self.assertEqual(_heading_name(line("2.1. Grouping-based methods", size=9.0), 10.0, opts),
                         ("Grouping-based methods", 2))
        # numbered list items and math lines followed by a sentence are not headings
        self.assertIsNone(_heading_name(line("1. We identify and describe the problem of in-", bold=False, size=11.0),
                                        10.0, opts))
        self.assertIsNone(_heading_name(line("0. The hyperparameters are set as follows for both LSTM and",
                                             bold=False), 10.0, opts))
        self.assertIsNone(_heading_name(line("2. Then taking", bold=False), 10.0, opts))
        self.assertEqual(_heading_name(line("A. The Proposed Model", bold=False), 10.0, opts), ("The Proposed Model", 1))
        self.assertEqual(_heading_name(line("3.2 BERT vs. GPT"), 10.0, opts), ("BERT vs. GPT", 2))
        # numbered run-in heading in one style: split into heading + body
        run_in = line("3.1 Attacks. Perturbing data adversarially to exploit weaknesses")
        self.assertIsNone(_heading_name(run_in, 10.0, opts))
        self.assertEqual(_numbered_run_in(run_in, 10.0, opts, True),
                         ("Attacks", 2, "Perturbing data adversarially to exploit weaknesses"))
        self.assertEqual(_repair("\x07The algorithmic condition"), "The algorithmic condition")
        self.assertEqual(_repair("located at \x04\x05\x02 of layer l\uf020"), "located at  of layer l")
        self.assertTrue(_is_mojibake("Ë@ úÍ@ ø Xñ"))  # Arabic in a font without a Unicode map
        self.assertTrue(_is_mojibake("CDFðL 1Þ 1⁄4 N: ð4Þ"))  # Springer math font
        self.assertFalse(_is_mojibake("[Küçük 2017a]."))
        self.assertFalse(_is_mojibake("xi ∈ Rdx×D×H×W, and yi ∈ Rdy×D×H×W."))
        self.assertIsNone(_heading_name(line("9 CCQCP ←HashmapInit()", bold=False), 10.0, opts))  # pseudo-code
        self.assertIsNone(_heading_name(line("4 For each state s do", bold=False), 10.0, opts))
        self.assertIsNone(_heading_name(line("1 Corresponding author: a.b@gmail.com", bold=False), 10.0, opts))
        self.assertIsNone(_heading_name(line("6 # Define the model and SGD optimizer", bold=False), 10.0, opts))
        self.assertIsNone(_heading_name(line("3 Hence,", bold=False), 10.0, opts))
        self.assertIsNone(_heading_name(line("1. Tania Fleitas Kanonnikoff, Ana Miralles Marco, Manuel Cabeza"),
                                        10.0, opts))  # a list of names
        self.assertEqual(_heading_name(line("4 Datasets, Baselines, and Implementation Details"), 10.0, opts),
                         ("Datasets, Baselines, and Implementation Details", 1))
        # statements are headings even in numbered papers, so that R6 can drop them
        self.assertEqual(_heading_name(line("Data Availability"), 10.0, opts, numbered_mode=True),
                         ("Data Availability", 1))

    def test_bibliography_detection(self) -> None:
        from kb.parse import _bib_line, _is_references_line

        for text in ("References", "7 REFERENCES", "References:", "Bibliography", "Reference"):
            self.assertTrue(_is_references_line(text), text)
        self.assertFalse(_is_references_line("References 218"))  # a contents entry with its page number
        self.assertTrue(_is_references_line("Re F E R E N C E S"))  # letter-spaced
        self.assertTrue(_bib_line("[1] K. Simonyan and A. Zisserman, “Very deep convolutional networks,” 2014."))
        self.assertTrue(_bib_line("Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT:"))
        self.assertTrue(_bib_line("Case of Facebook. Webology, 2019. 16(1)."))
        self.assertTrue(_bib_line("Smith, J. (2019). Learning to rank with noisy labels. Journal of AI, 3."))
        self.assertFalse(_bib_line("as shown by Devlin et al. (2019), pre-training helps on many tasks"))
        self.assertFalse(_bib_line("(Smith et al., 2019; Lee and Kim, 2020) propose methods for ranking"))
        # two-column entries wrap over short lines; each line carries its own feature
        for text in ("Shlomo Argamon-Engelson and Ido Dagan.", "search (JAIR), 11:335–360.",
                     "[Angeli and Manning2013] Gabor Angeli and Christo-", "[1] https://developer.nvidia.com/jetson-tx2.",
                     "Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and", "tional Linguistics, 34(4):555–596."):
            self.assertTrue(_bib_line(text), text)
        self.assertFalse(_bib_line("Table 3: The average number of words in translations of beginning sentences"))
        self.assertFalse(_bib_line("The statistical results are shown in Table 3. Without using the cache model"))
        self.assertFalse(_is_references_line("reference."))  # a wrapped sentence, not a title
        self.assertTrue(_bib_line("1. Simon, H. A. The sciences of the artificial. Cambridge"))
        self.assertFalse(_bib_line("2. Supportive, whether the psychologist expresses support"))  # a numbered list
        for text in ("1 Suerbaum, S. & Michetti, P. Helicobacter pylori", "infection. N Engl J Med 347, 1175-1186,",
                     "doi:10.1056/NEJMra020542 (2002).", "incidence analysis. The Lancet Global Health 8, e180-e190"):
            self.assertTrue(_bib_line(text), text)  # Nature / medical style

    def test_reference_list_rules(self) -> None:
        """Contents pages are dropped, a "Reference" table header does not cut the paper, a small
        bold "References" title and a reference list without any title do."""
        entries = [f"[{k}] A. Author and B. Writer. A study of topic {k}. In Proceedings of the Conference "
                   f"on Learning, pages 1-10, 20{10 + k}." for k in range(1, 7)]
        with tempfile.TemporaryDirectory() as tmp:
            for variant in ("titled", "untitled"):
                pdf = Path(tmp) / f"{variant}.pdf"
                doc = pymupdf.open()
                toc = doc.new_page()
                toc.insert_text((72, 80), "Contents", fontsize=12, fontname="hebo")
                toc.insert_textbox(pymupdf.Rect(72, 100, 520, 400),
                                   "1 Introduction\n2\n2 Method\n3\nReferences\n4\n5\n6", fontsize=10, fontname="hebo")
                toc.insert_textbox(pymupdf.Rect(72, 420, 520, 520), "1.1 Background . . . . . . . . . 2\n"
                                   "1.2 Scope . . . . . . . . . 2\n2.1 Model . . . . . . . . . 3", fontsize=10, fontname="helv")
                page = doc.new_page()
                page.insert_text((72, 80), "1 Introduction", fontsize=12, fontname="hebo")
                page.insert_textbox(pymupdf.Rect(72, 100, 520, 300), BODY, fontsize=10, fontname="helv")
                page.insert_text((72, 320), "Reference", fontsize=10, fontname="helv")  # table column header
                page.insert_text((72, 335), "[22]", fontsize=10, fontname="helv")
                page.insert_text((72, 360), "2 Method", fontsize=12, fontname="hebo")
                page.insert_textbox(pymupdf.Rect(72, 380, 520, 600), BODY, fontsize=10, fontname="helv")
                refs = doc.new_page()
                y = 80
                if variant == "titled":
                    refs.insert_text((72, y), "References", fontsize=8, fontname="hebo")  # smaller than body
                    y += 20
                for entry in entries:
                    refs.insert_textbox(pymupdf.Rect(72, y, 520, y + 40), entry, fontsize=10, fontname="helv")
                    y += 45
                doc.save(str(pdf))
                secs = parse_pdf(pdf, ParseOptions(min_words=50))
                self.assertEqual([s["section"] for s in secs], ["Introduction", "Method"], variant)
                text = " ".join(s["text"] for s in secs)
                self.assertNotIn("Proceedings", text, variant)
                self.assertEqual(text.count("The retriever selects passages"), 12, variant)  # both sections kept

    def test_dropped_section_overflow_and_letter_appendix(self) -> None:
        """An acknowledgments paragraph is dropped, but pages of text after it (a missed heading)
        are kept; "B Proofs" counts as a heading because "B.1 ..." follows."""
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "p.pdf"
            doc = pymupdf.open()
            page = doc.new_page()
            page.insert_text((72, 80), "1 Introduction", fontsize=12, fontname="hebo")
            page.insert_textbox(pymupdf.Rect(72, 100, 520, 330), BODY, fontsize=10, fontname="helv")
            page.insert_text((72, 350), "Acknowledgments", fontsize=12, fontname="hebo")
            page.insert_textbox(pymupdf.Rect(72, 370, 520, 400), "We thank the anonymous reviewers.",
                                fontsize=10, fontname="helv")
            page = doc.new_page()  # a missed appendix heading: pages of text after the acknowledgments
            self.assertGreater(page.insert_textbox(pymupdf.Rect(72, 60, 540, 800), (BODY + " ") * 3,
                                                   fontsize=10, fontname="helv"), 0)
            page = doc.new_page()
            page.insert_text((72, 80), "B Proofs", fontsize=12, fontname="hebo")
            page.insert_textbox(pymupdf.Rect(72, 100, 520, 330), BODY, fontsize=10, fontname="helv")
            page.insert_text((72, 350), "B.1 Proof of Lemma 1", fontsize=12, fontname="hebo")
            page.insert_textbox(pymupdf.Rect(72, 370, 520, 600), BODY, fontsize=10, fontname="helv")
            doc.save(str(pdf))
            secs = parse_pdf(pdf, ParseOptions(min_words=50))
            self.assertEqual([s["section"] for s in secs], ["Introduction", "", "Proofs", "Proofs / Proof of Lemma 1"])
            text = " ".join(s["text"] for s in secs)
            self.assertNotIn("anonymous reviewers", text)
            self.assertEqual(len(secs[1]["text"].split()), 3 * len(BODY.split()))

    def test_accents_code_lines_and_numbered_lists(self) -> None:
        from kb.parse import _attach_accents, _attach_drop_caps, _bib_line, _heading_name, _Line, _list_item, _repair

        def chars(*spec: tuple[str, float, float]) -> list[dict]:
            return [{"c": c, "bbox": (x0, 0.0, x1, 10.0)} for c, x0, x1 in spec]

        def fixed(*spec: tuple[str, float, float]) -> str:
            return _repair("".join(_attach_accents(chars(*spec))))

        # TeX draws an accent as its own glyph over the letter: "Man´e" -> "Mané", "Mart´ın" -> "Martín"
        self.assertEqual(fixed(("M", 0, 9), ("a", 9, 13), ("n", 13, 18), ("´", 18.5, 21.8), ("e", 18, 22.4)), "Mané")
        self.assertEqual(fixed(("t", 0, 3), ("´", 3.2, 6), ("ı", 3, 6), ("n", 6, 11)), "tín")
        self.assertEqual(fixed(("c", 0, 4), ("¸", 1, 3), ("o", 4, 9)), "ço")  # cedilla after its letter
        self.assertEqual(fixed(("l", 0, 3), ("̸", 1, 4), (" ", 4, 6), ("=", 6, 12), (" ", 12, 14), ("m", 14, 22)),
                         "l ≠ m")  # TeX \not
        self.assertEqual(fixed(("n", 0, 5), (" ", 5, 8), ("´", 8, 11), (" ", 11, 14), ("k", 14, 19)),
                         "n ´ k")  # over no letter: kept as printed, not split into " ́"
        # a decorative initial set as its own large glyph goes back in front of its paragraph
        cap = _Line("A ", 27.5, False, 0, 462, 492, 313, 2, x1=336)
        first = _Line("ccording to the World Health Organization", 10.0, False, 0, 467, 478, 333, 1, block_lines=3)
        other = _Line("B", 27.5, False, 0, 600, 630, 313, 3, x1=330)  # nothing next to it
        self.assertEqual([ln.text for ln in _attach_drop_caps([first, cap, other])],
                         ["According to the World Health Organization", "B"])
        from kb.parse import _is_bold
        for font, bold in (("ABCDEF+CMBX10", True), ("CMBX12", True), ("SFBX1000", True), ("FranklinGothic-Demi", True),
                           ("NimbusRomNo9L-Medi", True), ("CMR10", False), ("CMTI10", False), ("bbding10", False)):
            self.assertEqual(_is_bold({"font": font, "flags": 4}), bold, font)  # TeX bold fonts carry no bold flag
        self.assertEqual(_repair("the inverse \ue000\u0302 of"), "the inverse ˆ of")  # base glyph unmapped
        self.assertEqual(_repair("the trade-offbetween speed and a puffin"), "the trade-off between speed and a puffin")
        self.assertTrue(_bib_line("Abadi, M.; Agarwal, A.; Barham, P.; Brevdo, E.; Chen, Z.;"))  # AAAI author list
        self.assertFalse(_bib_line("In Paris, France, and Rome, Italy, the models were evaluated"))

        opts = ParseOptions()

        def line(text: str, bold: bool = False, size: float = 10.0) -> _Line:
            return _Line(text, size, bold, 0, 100, 110, 72, 1)

        for text in ("1 Initialize K = 0", "4 HIDDEN_SIZE = 20", "8 W_xh = model.add_parameters((H, 2))",
                     "3. For each citation x = 1", "5. Supplementary Figure 2. Flowchart of the method",
                     "44 Towards a principled approach", "2. The results are below.",
                     "0.01. The precision will decrease quickly if we continue",
                     "2. Brown clustering features benefit from two additional features:"):
            self.assertIsNone(_heading_name(line(text), 10.0, opts), text)
        for text in ("2 U2QΓ1VT",):  # ALL-CAPS formula inside a paragraph block
            self.assertIsNone(_heading_name(_Line(text, 10.0, False, 0, 100, 110, 72, 1, block_lines=2), 10.0, opts), text)
        for text in ():
            self.assertIsNone(_heading_name(line(text), 10.0, opts), text)
        self.assertEqual(_heading_name(line("A. Feature Extraction:"), 10.0, opts), ("Feature Extraction", 1))  # IEEE
        self.assertEqual(_heading_name(line("4.2 Regret analysis of RELEAF for Drel = 1"), 10.0, opts),
                         ("Regret analysis of RELEAF for Drel = 1", 2))  # "=" in a title, not an assignment
        # a numbered line in body style is a list item after "...:", before a lowercase run-on, or
        # next to the adjacent number (a few short formula lines between)
        item = line("3. Obtain the outputs of the network")
        self.assertTrue(_list_item(item, [line("the method has the following steps:")], [], 10.0, opts))
        self.assertTrue(_list_item(item, [line("x = 1"), line("2. Train a simple network")], [], 10.0, opts))
        self.assertTrue(_list_item(line("2. Medieval texts, with Mishneh Torah"), [line("Some text.")],
                                   [line("being of notable significance, were obtained")], 10.0, opts))
        self.assertFalse(_list_item(line("2 RESULTS"), [line("much attention before."), line("3. Our templates are causal")],
                                    [line("2.1 Performance evaluation")], 10.0, opts))  # after contribution 3: a section
        heading = line("3 Method")
        self.assertFalse(_list_item(heading, [line("Prior work studies retrieval for many tasks."), line("2 Related Work")],
                                    [line("We describe the model.")], 10.0, opts))  # prose between: a section
        self.assertFalse(_list_item(line("3 Method", bold=True), [line("x:")], [], 10.0, opts))
        self.assertFalse(_list_item(line("6.3.1 Content-based Features"), [line("the features are as follows:")],
                                    [line("cGAN features are extracted from each post")], 10.0, opts))  # a subsection
        self.assertFalse(_list_item(line("3.1 Matrix Formulation"), [line("3.2."), line("described in Section")],
                                    [line("The original model uses matrices.")], 10.0, opts))  # a bare "3.2." ends a sentence

    def test_hyphenated_line_breaks(self) -> None:
        from collections import Counter

        from kb.parse import _join_hyphenated, _join_lines, _spellings

        paper = _spellings(["We study real-world data and the representation of", "manually labelled text written by",
                            "annotators with TensorFlow.", "The x-axis shows the state-of-the-art.", "out of ing allow", "an answer"])
        self.assertEqual(paper["real-world"], 1)
        self.assertEqual(paper["real"], 1)  # the parts of a compound count as words
        cases = [
            ("of real-", "world data", "of real-world data"),  # the paper spells it with a hyphen
            ("the represen-", "tation of", "the representation of"),  # a syllable break
            ("the state-of-the-", "art model", "the state-of-the-art model"),  # already hyphenated
            ("the q-", "values of", "the q-values of"),  # one letter before the break
            ("on CIFAR-", "10 images", "on CIFAR-10 images"),  # before a digit
            ("with Tensor-", "Flow and", "with TensorFlow and"),  # the paper writes "TensorFlow"
            ("manually-", "written text", "manually-written text"),  # two words of the paper
            ("both pre-", "and post-training", "both pre- and post-training"),  # suspended hyphen
            ("with-", "out data", "without data"),  # "with" and "out" are words, "without" one closed word
            ("allow-", "ing us", "allowing us"),  # a suffix
            ("a question-an-", "swer pair", "a question-answer pair"),  # a syllable break after a hyphen
            ("semi-", "written text", "semi-written text"),  # "semi" is no word on its own but keeps its hyphen
        ]
        for left, right, want in cases:
            self.assertEqual(_join_hyphenated(left, right, paper), want)
        self.assertEqual(_join_lines(["the represen-", "tation of real-", "world data"], paper),
                         "the representation of real-world data")
        self.assertEqual(_join_lines(["a well-", "known result"], Counter()), "a wellknown result")  # no evidence

    def test_drop_leading_abstract(self) -> None:
        from kb.parse import drop_leading_abstract

        abstract = ("Retrieval augmented generation combines a retriever with a generator so that answers are "
                    "grounded in passages. We study how chunk size and overlap change answer quality on three "
                    "question answering benchmarks and find that sentence aligned chunks work best.")
        intro = "Large language models hallucinate. Retrieval is a common remedy for this problem."
        front = f"A Study of Chunking\nAda Author, Bob Writer\nUniversity of Somewhere\nAbstract—{abstract}\n"
        secs = [{"section": "", "text": front + "Keywords. retrieval, chunking. " + intro},
                {"section": "Method", "text": "We use BM25."}]
        self.assertEqual(drop_leading_abstract(secs, abstract),
                         [{"section": "", "text": intro}, {"section": "Method", "text": "We use BM25."}])
        # a named first section, or text without the abstract, is left alone
        self.assertEqual(drop_leading_abstract(secs[1:], abstract), secs[1:])
        other = [{"section": "", "text": intro * 20}]
        self.assertEqual(drop_leading_abstract(other, abstract), other)

    def test_running_bibliography_header_is_one_list(self) -> None:
        """A "References" header repeated on each page of one list is not a chapter bibliography:
        the paper is cut at the first title and the appendix after the list is dropped."""
        entries = [f"[{k}] A. Author and B. Writer. A study of topic {k}. In Proceedings of the Conference "
                   f"on Learning, pages 1-10, 20{10 + k % 10}." for k in range(1, 19)]
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "p.pdf"
            doc = pymupdf.open()
            page = doc.new_page()
            page.insert_text((72, 80), "1 Introduction", fontsize=12, fontname="hebo")
            page.insert_textbox(pymupdf.Rect(72, 100, 520, 300), BODY, fontsize=10, fontname="helv")
            page.insert_text((72, 320), "2 Method", fontsize=12, fontname="hebo")
            page.insert_textbox(pymupdf.Rect(72, 340, 520, 600), BODY, fontsize=10, fontname="helv")
            for part in range(3):
                refs = doc.new_page()
                refs.insert_text((72, 60), "References", fontsize=10, fontname="helv")  # running header
                y = 90
                for entry in entries[6 * part:6 * part + 6]:
                    refs.insert_textbox(pymupdf.Rect(72, y, 520, y + 40), entry, fontsize=10, fontname="helv")
                    y += 45
            page = doc.new_page()
            page.insert_text((72, 80), "A Proofs", fontsize=12, fontname="hebo")
            page.insert_textbox(pymupdf.Rect(72, 100, 520, 330), BODY, fontsize=10, fontname="helv")
            page.insert_text((72, 350), "A.1 Proof of Lemma 1", fontsize=12, fontname="hebo")
            page.insert_textbox(pymupdf.Rect(72, 370, 520, 600), BODY, fontsize=10, fontname="helv")
            doc.save(str(pdf))
            secs = parse_pdf(pdf, ParseOptions(min_words=50))
            self.assertEqual([s["section"] for s in secs], ["Introduction", "Method"])
            self.assertNotIn("Proceedings", " ".join(s["text"] for s in secs))

            # a compilation of short reports, each ending with its own list: all reports are kept
            pdf = Path(tmp) / "c.pdf"
            doc = pymupdf.open()
            for k, name in enumerate(("Retrieval", "Generation", "Evaluation")):
                page = doc.new_page()
                page.insert_text((72, 80), f"{k + 1} {name}", fontsize=12, fontname="hebo")
                page.insert_textbox(pymupdf.Rect(72, 100, 520, 330), BODY, fontsize=10, fontname="helv")
                page.insert_text((72, 350), "References", fontsize=12, fontname="hebo")
                y = 370
                for entry in entries[:6]:
                    page.insert_textbox(pymupdf.Rect(72, y, 520, y + 40), entry, fontsize=10, fontname="helv")
                    y += 45
            doc.save(str(pdf))
            secs = parse_pdf(pdf, ParseOptions(min_words=50))
            self.assertEqual([s["section"] for s in secs], ["Retrieval", "Generation", "Evaluation"])
            self.assertNotIn("Proceedings", " ".join(s["text"] for s in secs))

    def test_qa_review_heading_rules(self) -> None:
        """Heading cases found in the 50-paper parse review (see kb/README.md)."""
        from kb.parse import (_continues_heading, _heading_name, _Line, _list_item, _normalize_heading,
                              _paper_casings, _restore_casing, _roman_value)
        opts = ParseOptions()

        def line(text: str, bold: bool = True, size: float = 10.0, **kw) -> _Line:
            return _Line(text, size, bold, 0, 100, 110, 72, 1, **kw)

        self.assertEqual(_heading_name(line("1.2 Our Contributions"), 10.0, opts), ("Our Contributions", 2))
        self.assertIsNone(_heading_name(line("1. We identify and describe the problem", bold=False), 10.0, opts))
        self.assertIsNone(_heading_name(line("X. Assume σ|s| ≥", bold=False), 10.0, opts))  # a math line
        self.assertEqual(_heading_name(line("4.2.1 Intra-dataset Evaluation on FF++"), 10.0, opts)[0],
                         "Intra-dataset Evaluation on FF++")
        self.assertEqual(_heading_name(line("APPENDIX A", bold=False, size=12.0), 10.0, opts), ("Appendix A", 1))
        self.assertEqual(_heading_name(line("APPENDIX B. THE LIMIT OF STANDARD CONTINUOUS HOPFIELD NETWORKS.",
                                            bold=False, size=12.0), 10.0, opts),
                         ("The Limit of Standard Continuous Hopfield Networks", 1))
        self.assertIsNone(_heading_name(line("Appendix A shows the details"), 10.0, opts))  # a sentence
        self.assertEqual(_heading_name(line("2.2 word2vec and GloVe"), 10.0, opts), ("word2vec and GloVe", 2))
        self.assertIsNone(_heading_name(line("7. apply behavior: b", bold=False, size=11.1), 10.0, opts))
        self.assertEqual(_heading_name(line("3.2 MODEL B. MODERN HOPFIELD NETWORKS LIMIT AND ATTENTION MECHANISM.",
                                            bold=False), 10.0, opts)[1], 2)
        self.assertEqual(_heading_name(line("4. Part II: Minimizing the impact on the form of the original text"),
                                       10.0, opts)[0], "Part II: Minimizing the impact on the form of the original text")
        self.assertIsNone(_heading_name(line("2. Test the model against one of the smaller, human-curated trial "
                                             "outcome label"), 10.0, opts))  # a bold list item
        self.assertEqual(_heading_name(line("I. INTRODUCTION", bold=False, size=8.0), 10.0, opts),
                         ("Introduction", 1))  # IEEE small capitals
        self.assertEqual(_normalize_heading("DISCUSSION AND CONCLUSIONS"), "Discussion and Conclusions")
        self.assertEqual(_roman_value("IV"), 4)
        self.assertEqual(_roman_value("IX"), 9)

        # casing from the paper's own text: "Dnn" -> "DNN", one-word titles title-cased unless written so
        body = [line("We train a DNN and SpectralFormer on vulnerabilities; SpectralFormer beats the DNN baselines.",
                     bold=False), line("Known vulnerabilities and one vulnerabiliTies typo in the text body.", bold=False)]
        casings = _paper_casings(body)
        self.assertEqual(_restore_casing("Analysis of Dnn Inference", "ANALYSIS OF DNN INFERENCE", casings),
                         "Analysis of DNN Inference")
        self.assertEqual(_restore_casing("Spectralformer", "SPECTRALFORMER", casings), "SpectralFormer")
        self.assertEqual(_restore_casing("Surface Vulnerabilities", "SURFACE VULNERABILITIES", casings),
                         "Surface Vulnerabilities")  # the paper's usual spelling, not a one-off
        self.assertEqual(_restore_casing("IMPLEMENTATION", "IMPLEMENTATION", casings), "Implementation")

        # a heading wrapped onto a line ending in "?" continues
        head = _Line("4 How Much Does Valid Reasoning", 12.0, True, 0, 100, 112, 72, 1, block_lines=2)
        self.assertTrue(_continues_heading(head, _Line("Matter?", 12.0, True, 0, 113, 125, 72, 1, pos_in_block=1),
                                           10.0, opts))

        # list items: the previous item's indented continuation and a row of numbers do not hide the neighbour
        item = _Line("2. Decompose the regret in each episode", 10.0, False, 0, 200, 210, 72, 7)
        before = [_Line("such that M is in the set for all k with probability", 10.0, False, 0, 188, 198, 84, 6),
                  _Line("1. Design confidence sets (via concentration inequality)", 10.0, False, 0, 176, 186, 72, 6)]
        self.assertTrue(_list_item(item, before, [], 10.0, opts))
        row = _Line("19 MVCNN (-HLBL)", 10.0, False, 0, 200, 210, 72, 7)
        self.assertTrue(_list_item(row, [], [_Line("88.5 48.7", 10.0, False, 0, 212, 222, 200, 7),
                                             _Line("20 MVCNN (-Huang)", 10.0, False, 0, 224, 234, 72, 7)], 10.0, opts))

    def test_qa_review_paragraph_and_appendix_rules(self) -> None:
        from kb.parse import _letter_headings, _Line, _merge_paragraphs, _merge_split_numbers

        # a complete caption between the halves of a broken paragraph moves after the joined paragraph
        self.assertEqual(_merge_paragraphs(["Considering the sensitivity, the most significant",
                                            "Table 6: Change in BLEU for random perturbation.",
                                            "change is in the softmax component."], floats=[False, True, False]),
                         ["Considering the sensitivity, the most significant change is in the softmax component.",
                          "Table 6: Change in BLEU for random perturbation."])
        # an unfinished "caption" (body text that starts a line with "Table 6.") is continued as before
        self.assertEqual(_merge_paragraphs(["Some sentences are given in", "Table 6. Overall, the samples are",
                                            "unsatisfactory."], floats=[False, True, False]),
                         ["Some sentences are given in", "Table 6. Overall, the samples are unsatisfactory."])

        # a caption never continues body text; the body resumes after it ("them-" + "selves")
        self.assertEqual(_merge_paragraphs(["we evaluate the primitives, not the operations them-",
                                            "Table 4. Comparison of the networks on 3D data.",
                                            "selves). This allows a direct comparison."], floats=[False, True, False]),
                         ["we evaluate the primitives, not the operations themselves). This allows a direct comparison.",
                          "Table 4. Comparison of the networks on 3D data."])
        # a paragraph ending on "in"/"of"/... runs on even before a capital, but not into a new sentence
        self.assertEqual(_merge_paragraphs(["As shown in", "Figure 2, there is no recurrence."]),
                         ["As shown in Figure 2, there is no recurrence."])
        self.assertEqual(len(_merge_paragraphs(["It suffices to show that", "Let e > 0. Then the bound holds."])), 2)
        self.assertEqual(len(_merge_paragraphs(["is lower bounded as", "Remarks: First, the bound is tight."])), 2)

        # a bare appendix letter on the title's baseline is merged; a superscript digit is not
        merged = _merge_split_numbers([_Line("A", 12.0, True, 0, 100, 112, 72, 1),
                                       _Line("Brief Review of Imitation", 12.0, True, 0, 100, 112, 90, 1)])
        self.assertEqual([x.text for x in merged], ["A Brief Review of Imitation"])
        merged = _merge_split_numbers([_Line("1", 5.0, False, 0, 100, 106, 72, 1),
                                       _Line("2 )T XT XBD", 10.0, False, 0, 100, 110, 80, 1)])
        self.assertEqual(len(merged), 2)
        # consecutive bold letters in one size are appendix titles even without "A.1" subsections
        heads = [_Line("A Brief Review of Imitation", 12.0, True, 0, 100, 112, 72, 1),
                 _Line("Some text of the appendix.", 10.0, False, 0, 120, 130, 72, 2),
                 _Line("B Decoding Latent States", 12.0, True, 1, 100, 112, 72, 3)]
        self.assertEqual(_letter_headings(heads), {id(heads[0]), id(heads[2])})

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "p.pdf"
            doc = pymupdf.open()
            page = doc.new_page()
            page.insert_text((72, 80), "1 Introduction", fontsize=12, fontname="hebo")
            page.insert_textbox(pymupdf.Rect(72, 100, 520, 330), BODY, fontsize=10, fontname="helv")
            page.insert_text((72, 350), "2 Ethics", fontsize=12, fontname="hebo")
            page.insert_textbox(pymupdf.Rect(72, 370, 520, 600), BODY, fontsize=10, fontname="helv")
            page = doc.new_page()
            page.insert_text((72, 60), "2.1 Ethical Considerations", fontsize=11, fontname="hebo")
            self.assertGreater(page.insert_textbox(pymupdf.Rect(72, 80, 540, 800), (BODY + " ") * 4,
                                                   fontsize=10, fontname="helv"), 0)  # > 400 words: content
            page = doc.new_page()
            page.insert_text((72, 80), "Acknowledgments", fontsize=12, fontname="hebo")
            page.insert_textbox(pymupdf.Rect(72, 100, 520, 130), "We thank the anonymous reviewers.",
                                fontsize=10, fontname="helv")
            page.insert_text((72, 160), "APPENDIX A", fontsize=12, fontname="helv")
            page.insert_textbox(pymupdf.Rect(72, 180, 520, 420), BODY, fontsize=10, fontname="helv")
            doc.save(str(pdf))
            secs = parse_pdf(pdf, ParseOptions(min_words=50))
            self.assertEqual([s["section"] for s in secs],
                             ["Introduction", "Ethics", "Ethics / Ethical Considerations", "Appendix A"])
            self.assertNotIn("anonymous reviewers", " ".join(s["text"] for s in secs))
            self.assertEqual(len(secs[3]["text"].split()), len(BODY.split()))  # the appendix is not dropped with R6

    def test_caption_runs_running_headers_and_footnote_marks(self) -> None:
        from kb.parse import _header_footer_keys, _Line, _merge_paragraphs
        # "Fig. 2." captions in body-size text between the halves of a sentence are passed over
        self.assertEqual(_merge_paragraphs(["It takes input symbols and", "Fig. 2. A toy example of the WFST.",
                                            "emits output symbols."]),
                         ["It takes input symbols and emits output symbols.", "Fig. 2. A toy example of the WFST."])
        self.assertEqual(_merge_paragraphs(["positional information to", "Figure 3: Encoding the robot pose.",
                                            "Table 2. Adding it improves performance.", "enable the model to run."],
                                           None, [False, True, False, False])[0],
                         "positional information to enable the model to run.")
        # body text starting with "Table 3." is not a caption: unfinished, or not followed by a lowercase half
        self.assertEqual(_merge_paragraphs(["The results are given in", "Table 3. Overall, the model wins.",
                                            "We then compare."]),
                         ["The results are given in", "Table 3. Overall, the model wins.", "We then compare."])
        self.assertEqual(_merge_paragraphs(["The results are presented in", "Table 3. Overall, for any ratio applied to the video",
                                            "Table 3: Intra-dataset evaluation.", "dataset, our method wins."],
                                           None, [False, False, True, False])[1],
                         "Table 3. Overall, for any ratio applied to the video dataset, our method wins.")
        self.assertEqual(len(_merge_paragraphs(["ImageNet-118K-LT [33] OpenImages-118K [82]",  # figure labels, not text
                                                "Figure 7.1. Example images.", "benchmark [127] is used."])), 3)
        # a two-word line on every even page of a 5-page paper is a running header; a table cell is not
        lines = [_Line(t, 9.0, False, p, 30, 40, 72, p * 10 + k) for p in (1, 3) for k, t in enumerate(["Happe and Cito", "Hindi"])]
        keys = _header_footer_keys(lines, [800.0] * 5, ParseOptions())
        self.assertIn("happe and cito", keys)
        self.assertNotIn("hindi", keys)
        with tempfile.TemporaryDirectory() as tmp:  # a footnote mark set as a superscript leaves the name; R² keeps its 2
            path = Path(tmp) / "sup.pdf"
            doc = pymupdf.open()
            page = doc.new_page()
            y = 72
            for head in ("1 Introduction", "2 Related Work</b><sup>3</sup><b>", "3 Fit of R</b><sup>2</sup><b>"):
                page.insert_htmlbox(pymupdf.Rect(72, y, 520, y + 30), f'<p style="font-size:14pt"><b>{head}</b></p>')
                page.insert_textbox(pymupdf.Rect(72, y + 30, 520, y + 230), BODY, fontsize=10, fontname="helv")
                y += 240
            doc.save(str(path))
            names = [s["section"] for s in parse_pdf(path)]
        self.assertEqual(names, ["Introduction", "Related Work", "Fit of R2"])

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


class BuildTest(unittest.TestCase):
    def test_rebuild_resets_num_chunks_of_a_paper_that_now_fails(self) -> None:
        import contextlib, copy, importlib.util, io, sys  # noqa: E401
        from concurrent.futures import ThreadPoolExecutor
        from unittest import mock

        from config import PROJECT_ROOT, get_config
        spec = importlib.util.spec_from_file_location("build_kb", PROJECT_ROOT / "scripts" / "build_kb.py")
        build_kb = importlib.util.module_from_spec(spec)
        sys.modules["build_kb"] = build_kb
        spec.loader.exec_module(build_kb)
        failing: set[str] = set()

        def fake_process(row: dict, raw_dir: str, cfg: dict, mode: str) -> dict:
            pid = row["paper_id"]
            quality = {"paper_id": pid, "pdf_path": row["pdf_path"], "word_count": 40, "num_sections": 1,
                       "ok": pid not in failing}
            if pid in failing:
                return {"paper_id": pid, "error": "quality check failed", "quality": quality}
            text = f"[{row['title']}] Intro\n" + "word " * 40
            passages = [{"paper_id": pid, "passage_id": f"{pid}_{i:04d}", "parent_section_id": f"{pid}_s00",
                         "section": "Intro", "year": 2024, "categories": ["cs.CL"], "text": text} for i in range(3)]
            parents = [{"paper_id": pid, "parent_section_id": f"{pid}_s00", "section": "Intro", "text": text}]
            return {"paper_id": pid, "passages": passages, "parents": parents, "quality": quality, "sections": []}

        with tempfile.TemporaryDirectory() as tmp:
            cfg = copy.deepcopy(get_config())
            cfg["paths"] = {k: str(Path(tmp) / k) for k in ("raw", "processed", "results")}
            (Path(tmp) / "raw").mkdir()
            fetch.write_metadata_csv(pd.DataFrame([
                {"paper_id": p, "arxiv_id": p, "title": "T", "abstract": "", "year": 2024, "categories": ["cs.CL"],
                 "citation_count": 0, "pdf_path": f"pdf/{p}.pdf"} for p in ("A", "B")]), Path(tmp) / "raw" / "metadata.csv")

            def run(*flags: str) -> None:
                with mock.patch.object(build_kb, "get_config", return_value=cfg), \
                        mock.patch.object(build_kb, "process_paper", fake_process), \
                        mock.patch.object(build_kb, "ProcessPoolExecutor", ThreadPoolExecutor), \
                        mock.patch.object(build_kb.logging, "basicConfig"), \
                        mock.patch.object(sys, "argv", ["build_kb.py", "--workers", "1", *flags]), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    build_kb.main()

            db = Path(tmp) / "processed" / cfg["kb"]["store"]["sqlite_file"]
            run()
            self.assertEqual((store.load_paper("A", db)["num_chunks"], store.load_paper("B", db)["num_chunks"]), (3, 3))
            failing.add("B")  # B now fails the quality check: a rebuild must not keep its old count
            run("--rebuild")
            self.assertEqual((store.load_paper("A", db)["num_chunks"], store.load_paper("B", db)["num_chunks"]), (3, 0))


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
