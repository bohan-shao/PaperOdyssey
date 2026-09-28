"""PDF full-text extraction and cleaning.

`parse_pdf(pdf_path, options)` reads a PDF with pymupdf (span-level `get_text("dict")`, so
font size/weight are available) and returns `[{"section": str, "text": str}, ...]`.

Cleaning rules (enumerable; all thresholds live in config `kb.parse.*` and are reported):
  R0  Contents pages: pages with >= 3 dot-leader entry lines ("Introduction . . . 3"), or a
      "Contents" title and >= 5 bare page numbers, are dropped (their entries would otherwise
      become headings such as "Introduction 3").
  R1  Reference truncation: stop at the first "References"/"Bibliography" title (incl. variants
      such as "Bibliographical References", "7 REFERENCES", "References:"; a bare title line in any
      style, even set smaller than body text or inside a block) that is followed by bibliography
      entries (see _bibliography_follows: entry labels, "2008." years, page ranges, venues, author
      lists), or at a reference list without a title ("[1] ...",
      "[2] ...", "[3] ..." entries). Everything after it (incl. appendices) is
      discarded. A "Reference(s)" line not followed by entries (a table column header) is dropped.
      Documents with >= 3 reference lists (a bibliography per chapter, or per report in a
      compilation) skip each list up to the next heading instead; a title right after entries (a
      running "Bibliography" header on the next page of the same list) does not start a new list.
  R2  Header/footer removal: lines starting/ending in the top/bottom `margin_ratio` of the page whose
      digit-normalized text repeats on >= `header_footer_min_ratio` of pages (on >= 3 pages for a
      chapter header: numbered, with a page number, or ALL CAPS; in a short paper, on every odd or
      every even page after the first for a line of >= 2 words, "Happe and Cito"); bare page
      numbers in the margins; the rotated arXiv sidebar stamp (non-horizontal text) and
      "arXiv:XXXX.XXXXXvN [cs.XX] date" lines.
  R3  Formula/figure/table noise: lines whose letter ratio (letters / non-space chars) is below
      `min_alpha_ratio` and that do not end a sentence; lines set in a font smaller than
      `min_font_ratio` x body size (footnotes, figure labels, small tables); mojibake lines (>= 4
      and >= 20% accented Latin-1 letters, from fonts without a Unicode map); fragments of < 4
      words; table-like blocks (>= 3 lines averaging < 3.5 words, no real sentence, not a caption);
      table header cells set in a caption's block before its label ("Method NN Table 4. ...").
  R4  URLs/DOIs: URL-only and DOI-only lines are dropped; inline URLs are removed.
  R5  Front matter: text before the first detected heading (title/authors/affiliations) is
      dropped when at least one heading exists, unless it exceeds `front_matter_max_words`
      (a missed first heading), in which case it is kept as an unnamed section. Likewise an
      "Abstract" of > 600 words keeps its first paragraph(s) (>= 100 words, <= 3 paragraphs) and
      the rest becomes an unnamed section. When the metadata abstract is inserted (no Abstract
      heading found), drop_leading_abstract removes the title, authors and the paper's own copy
      of the abstract from such a leading unnamed section.
  R6  Dropped sections: headings listed in `drop_sections` (e.g. acknowledgments) and their
      text (the first paragraph, and a second one of <= 250 words if both total <= 400) are skipped; >= 300 more words before
      the next heading belong to a missed heading (e.g. an appendix) and are kept, unnamed. A
      dropped heading does not name the subsections that follow it. An ethics/impact/
      reproducibility title set as a numbered subsection with > 400 words (a survey's "6.1.2
      Ethical Considerations") is content and kept.
  R7  Captions: "Figure N:"/"Table N:" blocks (and a block that continues an unfinished caption)
      are kept or dropped per `keep_captions`.
  R8  Repair: accents that TeX draws as separate glyphs are attached to the letter they sit on
      ("Man´e" -> "Mané", "l̸ =" -> "l ≠"), and a large decorative initial goes back in front of
      its paragraph ("A" + "ccording" -> "According"); NFKC normalization (the ligature "ﬁ" becomes "fi");
      "(cid:N)" glyph garbage, control characters, zero-width spaces and private-use glyphs
      removed; the space lost after an "ff" ligature restored ("trade-offbetween"); hyphenated line/column breaks joined, keeping the hyphen of compounds (TeX's
      hyphenation limits and the paper's own spelling decide, see _join_hyphenated); paragraphs
      split across columns/pages merged when the first does not end a sentence and the next
      starts lowercase, or the first ends in a hyphenated word, also across the complete captions
      and ALL-CAPS table titles set between the two halves (they follow the joined paragraph), and
      across complete "Fig. 2. ..." captions in body-size text when the next half starts lowercase; a
      caption never continues body text. A paragraph ending on a word no sentence ends on ("... as
      shown in") also runs on into a block starting with a capital ("Figure 2, there is ..."),
      unless that block starts a new sentence ("Let", "Here", ...), a list item, a caption or a
      label ("Remarks:", "Proof.").

Section-title detection: a line is a heading when it
- is short (a numbered bold title up to 14 words when it has a subtitle colon or is in title
  case), does not end with a period (except numbered ALL-CAPS titles) or a relation sign
  ("X. Assume σ|s| ≥" is math), and starts with a capital letter (a bold subsection may start
  with a lowercase name: "2.2 word2vec and GloVe");
- is visually emphasized: bold (incl. TeX bold fonts named only by convention, "CMBX10"), or font
  >= body + `heading_size_delta`, or (IEEE/Springer styles)
  a numbered line that is alone in its block or written in ALL CAPS;
- is not set smaller than body text (except ACM-style numbered bold headings at >= 0.85x body) and
  does not sit in the middle of a multi-line block (that would be a bold paragraph);
- is numbered ("3", "3.2", "A.1", "IV.", "APPENDIX A", "Appendix B. Title") or is a well-known heading word ("Introduction",
  "Related Work", ...); in papers without numbered sections, after the first heading, also any
  emphasized short title followed by a body-text paragraph (_unnumbered_heading);
- is not a numbered list item ("1. We propose ..."; a bold "1.2 Our Contributions" is a title) and, when set in plain body style, does not
  read as a sentence fragment (section number 0, "Then ...", "... and"), is not a code or
  algorithm line ("1 Initialize K = 0") and is not an item of a numbered list (after a line ending
  in ":", before a lowercase run-on sentence, or next to the adjacent number, see _list_item);
- is not a caption ("Supplementary Figure 2. ..."), holds no URL, and a top-level number is <= 40.
IEEE papers (Roman "I." / "II." section lines) also have italic "A. Title" subsections, found even
when they share a block with the section title or wrap onto a second italic line, and small-caps
Roman titles set below body size; a Roman number not above the current section's ("I." after
"H.") is a lettered subsection. Outside IEEE papers a body-style "A. ..." line is a lettered list
option ("A. Absolutely must be false"), not a heading.
Run-in headings are split into heading + body: SIAM style (bold "2.1. Title." followed by body
text on the same line) and numbered headings followed by a sentence in the same emphasized style
("7.2.2 Method. Common human evaluation methods ..."); a lettered run-in ("A. Zachary's karate
club. The ...") nests below the current section. Headings are capped at 14 words when
rejoined from wrapped lines (20 for a numbered bold title wrapped onto a bold line), and a page
number never continues a heading. A footnote mark set as a superscript after the title
("Comparison to Mayr et al." + "12") is not part of the name; an exponent after a variable is ("R2").

Section numbers set on their own line at the same baseline as the title are merged. Numbers are
stripped; subsections are named "Parent / Child" (top two levels) so repeated subsection
titles stay distinguishable; ALL-CAPS/small-caps names are title-cased, with small words in
lowercase and words the paper writes with inner capitals in its text spelled that way ("DNN",
"SpectralFormer"); headings wrapped onto a second line (also ending in "?") are rejoined. In papers with >=3 numbered headings (numbered list items do not count), unnumbered headings are only
accepted for UNNUMBERED_OK names (rejects bold table cells such as "Training"). Appendix headings
numbered by a bare letter ("B Proofs") count when their "B.1" subsection follows or when two
consecutive letters are set bold in the same size ("A Brief Review ...", "B Decoding ..."); a
bare letter set apart on the title's baseline is merged with it (a much smaller digit is a
superscript, not a section number).

`quality_check(pdf_path, sections)` returns a heuristic score for sampled manual review.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# ---------------------------------------------------------------- options


@dataclass
class ParseOptions:
    """Cleaning thresholds (mirrors config `kb.parse.*`)."""

    keep_captions: bool = True
    min_alpha_ratio: float = 0.55
    min_font_ratio: float = 0.9
    heading_size_delta: float = 0.8
    header_footer_min_ratio: float = 0.3
    margin_ratio: float = 0.12
    drop_sections: list[str] = field(default_factory=lambda: list(STATEMENT_SECTIONS))
    front_matter_max_words: int = 400
    min_words: int = 500
    max_bad_char_ratio: float = 0.02

    @classmethod
    def from_config(cls, parse_cfg: dict | None) -> ParseOptions:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (parse_cfg or {}).items() if k in known})


class PDFParseError(Exception):
    """The PDF is corrupted, encrypted, image-only, or yields no usable text."""


# ---------------------------------------------------------------- patterns

# Statements rather than paper content (R6 default drop list; also accepted as unnumbered headings,
# so their text does not run into the preceding section).
STATEMENT_SECTIONS = (
    "acknowledgments", "acknowledgements", "acknowledgment", "acknowledgement", "acknowledge",
    "acknowledgments and disclosure of funding", "acknowledgements and disclosure of funding",
    "author contributions", "funding", "ethics statement", "ethical statement", "ethical considerations",
    "ethical consideration", "broader impact", "broader impacts", "impact statement",
    "reproducibility statement", "declarations", "conflict of interest", "conflicts of interest",
    "competing interests", "declaration of competing interest", "declaration of interests",
    "data availability", "data availability statement", "code availability",
)

KNOWN_HEADINGS = {
    "abstract", "introduction", "related work", "related works", "background",
    "preliminaries", "preliminary", "problem formulation", "problem statement", "problem setup",
    "method", "methods", "methodology", "approach", "our approach", "proposed method",
    "model", "models", "framework", "architecture", "model architecture",
    "experiments", "experiment", "experimental setup", "experimental settings", "experimental results",
    "setup", "evaluation", "results", "results and discussion", "analysis", "ablation study",
    "ablation studies", "discussion", "discussions", "conclusion", "conclusions",
    "conclusion and future work", "conclusions and future work", "future work", "limitations",
    "summary", "overview", "datasets", "data", "implementation details", "training",
    "training details", "case study", "error analysis",
    "acknowledgments", "acknowledgements", "acknowledgment", "acknowledgement",
    "author contributions", "funding", "ethics statement", "broader impact",
    "broader impacts", "reproducibility statement", "impact statement",
    "references", "bibliography", "appendix", "appendices", "supplementary material",
    *STATEMENT_SECTIONS,
}
REFERENCE_HEADINGS = {"references", "bibliography", "reference", "literature cited"}
# Chapter titles of theses and books, set large and without a number even when sections are numbered.
CHAPTER_HEADINGS = {"introduction", "background", "related work", "related works", "literature review", "preliminaries",
                    "methodology", "discussion", "summary", "conclusion", "conclusions", "conclusion and future work",
                    "conclusions and future work", "future work"}
# Reference-list headings incl. template variants ("Bibliographical References" in LREC).
_REFERENCES_RE = re.compile(
    r"^(?:bibliographical |cited |selected )?references?(?: cited)?$|^bibliography$|^(?:literature|works) cited$"
)
# Headings that stay unnumbered even in papers with numbered sections.
UNNUMBERED_OK = REFERENCE_HEADINGS | {
    "abstract", "acknowledgments", "acknowledgements", "acknowledgment", "acknowledgement",
    "author contributions", "funding", "ethics statement", "broader impact", "broader impacts",
    "reproducibility statement", "impact statement", "limitations", "appendix", "appendices",
    "supplementary material", *STATEMENT_SECTIONS,
}

# Section numbers such as "3", "3.2.", "A.", "A.1", "IV.". A lone letter needs a period, so a title
# like "A Survey of ..." is not mistaken for a numbered heading.
_SECNUM = r"(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[A-H](?:\.\d{1,2}){1,3}\.?|[A-H]\.|[IVX]{1,5}\.)"
_NUMBERED_RE = re.compile(rf"^{_SECNUM}\s+(?P<title>\S.*)$")
_NUMBER_ONLY_RE = re.compile(rf"^{_SECNUM}$")
# Appendix headings numbered by a bare letter ("B Proofs of Main Results"); accepted only when
# their first subsection ("B.1 ...") follows, so titles such as "A Survey of ..." stay excluded.
_LETTER_HEAD_RE = re.compile(r"^(?P<num>[A-H])\s+(?P<title>[A-Z].*)$")
_ARXIV_STAMP_RE = re.compile(r"arXiv:\s*\d{4}\.\d{4,5}(v\d+)?\s*\[[a-z\-]+(\.[A-Za-z\-]+)?\]", re.I)
_URL_RE = re.compile(r"(https?://|www\.)\S+", re.I)
_DOI_LINE_RE = re.compile(r"^\s*(doi:|https?://doi\.org/)\S+\s*$", re.I)
_URL_LINE_RE = re.compile(r"^\s*((https?://|www\.)\S+\s*)+$", re.I)
_CID_RE = re.compile(r"\(cid:\d+\)")
_PAGE_NUM_RE = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.I)
_CAPTION_RE = re.compile(
    r"^\s*(?:supplementary\s+|extended\s+data\s+)?(figure|fig\.|table|tab\.)\s*[A-Z]?\d+[a-z]?\s*[:.|]", re.I)
_CAPTION_IN_RE = re.compile(r"\b(?:figure|fig\.|table|tab\.)\s*[A-Z]?\d+[a-z]?\s*[:.|]", re.I)
# Code and pseudo-code: arrows, identifiers with "_", method calls; an assignment ("K = 0") after one
# token or in a line starting with an algorithm verb ("Set root= 0"), not "Case p = 1".
_CODE_RE = re.compile(r"←|\w_\w|\w\.\w+\(|\(\(")
_ALGO_VERB_RE = re.compile(
    r"^(?:initiali[sz]e|set|repeat|for|while|until|if|else|return|update|compute|let|sample|obtain|choose|"
    r"select|draw|input|output|end)\b", re.I)

_SENT_END_RE = re.compile(r"[.!?:;]['\")\]]?$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u200b\u2060\ufeff]")  # incl. zero-width space, BOM
_PRIVATE_USE_RE = re.compile("[\ue000-\uf8ff\U000f0000-\U0010ffff]")  # unmapped symbol glyphs
# Accents TeX draws as a glyph of their own over or under a letter ("Man\u00b4e" is "Man\u00e9", R8): spacing
# forms map to their combining mark, combining marks to themselves.
_SPACING_ACCENTS = {
    "\u00b4": "\u0301", "\u02ca": "\u0301", "`": "\u0300", "\u02cb": "\u0300", "\u02c6": "\u0302",
    "\u02dc": "\u0303", "\u00af": "\u0304", "\u02c9": "\u0304", "\u02d8": "\u0306", "\u02d9": "\u0307",
    "\u00a8": "\u0308", "\u02da": "\u030a", "\u02dd": "\u030b", "\u02c7": "\u030c", "\u00b8": "\u0327",
    "\u02db": "\u0328",
}
_ACCENT_MARKS = {**_SPACING_ACCENTS, **{chr(c): chr(c) for c in range(0x300, 0x370)}}
_ACCENT_CHAR_RE = re.compile("[" + "".join(re.escape(c) for c in _ACCENT_MARKS) + "]")
_NOT_MARK = "\u0338"  # TeX \not: "\u0338 =" is "\u2260"
_DOTLESS = {"\u0131": "i", "\u0237": "j"}  # TeX accents i and j on their dotless forms
# NFKC turns an accent left on its own ("n \u00b4 k" in a font without a proper map) into a space and a
# combining mark; _repair puts the spacing form back (and drops a mark that has none).
_LOOSE_MARK_RE = re.compile("[ \u00a0]([\u0300-\u036f])")
_ORPHAN_MARK_RE = re.compile("(^|[ \u00a0])([\u0300-\u036f])")
_LOOSE_MARKS = {"\u0301": "\u00b4", "\u0300": "`", "\u0302": "\u02c6", "\u0303": "\u02dc", "\u0304": "\u00af", "\u0306": "\u02d8",
                "\u0307": "\u02d9", "\u0308": "\u00a8", "\u030a": "\u02da", "\u030b": "\u02dd", "\u0327": "\u00b8",
                "\u0328": "\u02db"}
# Mojibake from fonts without a Unicode map (legacy Arabic fonts, Springer math: "ð x Þ 1⁄4"):
# accented Latin-1 letters (not × ÷) making up a large share of a line.
_MOJIBAKE_RE = re.compile("[\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff]")
# Table of contents / list of figures: dot leaders ("Introduction . . . . . 3") and their title.
_LEADER_RE = re.compile(r"(?:\.\s?){6,}|…{2,}|(?:·\s?){6,}")
_TOC_TITLE_RE = re.compile(r"^(?:table of )?contents$|^list of (?:figures|tables)$", re.I)
_PAGE_REF_RE = re.compile(r"^(?:\d{1,4}|[ivxlc]{1,7})$", re.I)
# Bibliography entries (R1 validation): numbered starts, years, author names, venues.
_YEAR_RE = re.compile(r"\b(?:19|20)\d\d[a-z]?\b")
# Bibliography entry features (R1 validation). Entries wrap over several short lines in two-column
# layouts, so each line is tested on its own for: an entry label ("[12] ", "[Angeli and Manning2013] ",
# "3. Smith, J.", "3 Smith, J."), a year followed by punctuation ("2008.", "2016,", "(2002)."), page
# ranges ("pages 854-864", "34(4):555-596", "347, 1175-1186") or a DOI, venue phrases, a list of author names, or a short all-names line ending in ".".
_BIB_LABEL_RE = re.compile(
    r"^(?:\[\d{1,3}\]\.?\s|\[[A-Z][^\[\]]{1,40}\d{2,4}[a-z]?\]\s|\d{1,3}\.?\s+[A-Z][\w'’\-]+,\s+(?:[A-Z]\.[\s-]?)+)")
# "1. Simon, H. A." / "1 Suerbaum, S." need the initials: "1. First, we propose ..." is a numbered list.
_BIB_FIRST_RE = re.compile(r"^(?:\[1\]\s+\S|1\.?\s+[A-Z][\w'’\-]+,\s+(?:[A-Z]\.[\s-]?)+)")
_BIB_YEAR_RE = re.compile(r"\b(?:19[5-9]\d|20[0-4]\d)[a-z]?[.,):]")
_BIB_PAGES_RE = re.compile(
    r"(?:pages?|pp\.?)\s*\d+\s*[–-]\s*\d+|\d+\s*\(\d+\)\s*:\s*\d+|:\s*\d+\s*[–-]\s*\d+"
    r"|\b\d{1,4},\s*e?\d{1,6}\s*[–-]\s*e?\d{1,6}\b|\bdoi\s*:|doi\.org/"  # "347, 1175-1186", "8, e180-e190"
)
_BIB_VENUE_RE = re.compile(
    r"\bIn Proc|\bProceedings of\b|\bConference on\b|\bJournal of\b|\bTransactions on\b|\barXiv preprint\b|"
    r"\barXiv:\s?\d|\bWorkshop on\b|\bSymposium on\b|\bAdvances in Neural\b|\bAssociation for Computational\b|"
    r"\bUniversity Press\b|\bSpringer\b|\bIn [A-Z]{3,}[\w-]*[ ,]"
)
_NAME_LIST_RE = re.compile(r"(?:[A-Z][\w'’\-]*\.? ){1,3}[A-Z][\w'’\-]+,")
_INITIALS_LIST_RE = re.compile(r"\b[A-Z][\w'’\-]+,\s(?:[A-Z]\.\s?-?){1,3}[;,]")  # AAAI "Abadi, M.; Agarwal, A.;"
# In-text citations: "(Smith et al., 2019)", "[Kim 2018]", "Lee et al. (2020)". A bare "(2019)" is
# left alone: it is how APA-style reference entries give the year.
_INTEXT_CITE_RE = re.compile(
    r"\([^()]*[A-Za-z][^()]*\b(?:19|20)\d\d[a-z]?[^()]*\)|\[[^\[\]]*[A-Za-z][^\[\]]*\b(?:19|20)\d\d[a-z]?[^\[\]]*\]"
    r"|\b[A-Z][a-z'’\-]+(?: et al\.?| (?:and|&) [A-Z][a-z'’\-]+)? \((?:19|20)\d\d[a-z]?\)"
)
# Numbered list items ("1. We propose ...") are not section headings.
_PRONOUN_STARTS = {"we", "our", "it", "they", "here", "i", "there", "this"}
_ABBREVIATIONS = {"vs", "e.g", "i.e", "al", "fig", "figs", "cf", "etc", "eq", "eqs", "no", "dr", "mr", "st", "sec"}
# Words that start or end a sentence fragment but not a heading ("2. Then taking", "... LSTM and").
_FRAGMENT_STARTS = {"then", "thus", "hence", "where", "which", "and", "but", "so", "if", "when", "let",
                    "since", "therefore", "because", "while", "whereas", "note", "namely"}
_FRAGMENT_ENDS = {"and", "or", "the", "of", "for", "to", "with", "a", "an", "in", "on", "by", "as",
                  "from", "that", "is", "are", "be", "at"}
_MAX_HEADING_WORDS = 14
_SENTENCE_VERBS = {"is", "are", "was", "were", "be", "been", "has", "have", "can", "will", "we", "shows", "show"}
# A heading does not end in a relation or operator: "X. Assume σ|s| ≥" is a line of math whose
# variable "X." looks like a Roman section number.
_MATH_END_RE = re.compile(r"[=≤≥≈∼∝±×÷∈∉⊂⊆⊃⊇∪∩→⇒⇔−≠≡≺≻⪯⪰]$")  # not "+" ("FF++", "C++")
# Pseudo-code keywords, set in bold by algorithm packages ("7 end for"): not lowercase headings.
_PSEUDO_CODE_WORDS = {"end", "for", "if", "else", "elif", "return", "while", "do", "repeat", "until", "function",
                      "procedure", "input", "output", "require", "ensure", "then", "break", "continue", "loop"}
# Appendix headings: "APPENDIX A", "Appendix B. The limit of ...", "Appendix C: Case Study".
_APPENDIX_RE = re.compile(r"^appendix\s+(?P<num>[A-Z]|\d{1,2})(?:\s*[.:\-–—]\s*|\s+|$)(?P<title>.*)$", re.I)
# Statements that are content when they are a numbered subsection (a survey's "6.1.2 Ethical Considerations").
_CONTENT_LIKE_STATEMENTS = {"ethics statement", "ethical statement", "ethical considerations", "ethical consideration",
                            "broader impact", "broader impacts", "impact statement", "reproducibility statement"}
# Words kept lowercase when an ALL-CAPS heading is title-cased ("Discussion and Conclusions").
_SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "nor", "of", "on", "or",
                "over", "per", "the", "to", "via", "vs", "vs.", "with", "without"}


# ---------------------------------------------------------------- extraction


@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    page: int
    y0: float
    y1: float
    x0: float
    block: int
    bold_prefix: str = ""       # leading bold text when only part of the line is bold (run-in headings)
    pos_in_block: int = 0       # line position within its pymupdf block
    block_lines: int = 1        # number of lines in its pymupdf block
    heading: str | None = None  # emitted section name ("Parent / Child") when the line is a heading
    heading_key: str = ""       # lowercased own heading title (for drop_sections matching)
    heading_level: int = 1      # heading level (1 = section, 2 = subsection, ...)
    x1: float = 0.0             # right edge
    italic: bool = False        # all text italic (IEEE subsection titles "A. Datasets")
    sup_tail: str = ""          # footnote mark set as a superscript at the end ("... evaluation" + "12")


# Bold fonts whose names say so only by convention: TeX Computer Modern / EC ("CMBX10", "SFBX1000"),
# "...-Demi", Adobe "AdvOT1234.B".
_BOLD_FONT_RE = re.compile(r"(?:^|\+)(?:cmbx|cmb\d|cmbsy|ecbx|ecbi|sfbx|sfbi)|demi|\.b$", re.I)


def _is_bold(span: dict) -> bool:
    font = span.get("font", "").lower()
    return (bool(span.get("flags", 0) & 16) or any(t in font for t in ("bold", "medi", "black", "heavy", "semibold"))
            or bool(_BOLD_FONT_RE.search(font)))


def _is_italic(span: dict) -> bool:
    font = span.get("font", "").lower()
    return bool(span.get("flags", 0) & 2) or any(t in font for t in ("ital", "oblique", "cmti"))


def _attach_accents(chars: list[dict]) -> list[str]:
    """A line's characters with each accent glyph moved onto the letter it is drawn on (R8).

    TeX sets an accent as a glyph of its own, usually before its letter ("Man´e", "ˆc") and a
    cedilla before or after it ("Franc¸ois", "Ha¸sim"). The accent joins, as a combining mark, the
    neighbouring character whose box contains the accent's centre, and NFKC then composes "é".
    "̸" (TeX \\not) negates the next symbol ("l̸ = m" -> "l ≠ m"). Accents drawn over no neighbour
    (symbols of a font without a proper Unicode map) are left alone.
    """
    out = [c["c"] for c in chars]
    for k, c in enumerate(chars):
        mark = _ACCENT_MARKS.get(c["c"])
        if mark is None:
            continue
        if mark == _NOT_MARK:
            j = next((j for j in range(k + 1, len(chars)) if out[j].strip()), None)
        else:
            cx = (c["bbox"][0] + c["bbox"][2]) / 2
            order = (k + 1, k - 1) if c["c"] in _SPACING_ACCENTS else (k - 1, k + 1)  # combining: previous first
            j = next((j for j in order if 0 <= j < len(chars) and out[j].strip()
                      and chars[j]["c"] not in _ACCENT_MARKS
                      and chars[j]["bbox"][0] - 0.5 <= cx <= chars[j]["bbox"][2] + 0.5), None)
        if j is None:
            continue
        out[j] = _DOTLESS.get(out[j][0], out[j][0]) + out[j][1:] + mark
        out[k] = ""
    return out


def _attach_drop_caps(lines: list[_Line]) -> list[_Line]:
    """Put a decorative initial back in front of its paragraph (R8): IEEE and other templates set
    the first letter as a large glyph of its own ("A" + "ccording to the World Health ...").

    The initial is a single capital >= 1.8x the size of the first line of a paragraph that starts
    right of its middle (its box may include a trailing space) and within 8 pt of it, at its top.
    """
    out: list[_Line] = []
    moved: set[int] = set()
    for ln in lines:
        letter = ln.text.strip()
        if re.fullmatch("[A-Z]", letter):
            target = next((t for t in lines if t is not ln and t.page == ln.page and t.pos_in_block == 0
                           and (ln.x0 + ln.x1) / 2 <= t.x0 <= ln.x1 + 8 and ln.y0 - 3 <= t.y0 <= (ln.y0 + ln.y1) / 2
                           and t.size * 1.8 <= ln.size and t.text[:1].isalpha()), None)
            if target is not None and id(target) not in moved:
                target.text = letter + target.text.lstrip()
                moved.add(id(target))
                continue
        out.append(ln)
    return out


def _page_blocks(page: pymupdf.Page) -> list[dict]:
    """The page's "dict" blocks, with accents attached to their letters (see _attach_accents)."""
    tp = page.get_textpage(flags=pymupdf.TEXTFLAGS_DICT)
    blocks = page.get_text("dict", textpage=tp)["blocks"]
    if not any(_ACCENT_CHAR_RE.search(s["text"]) for b in blocks for ln in b.get("lines", []) for s in ln["spans"]):
        return blocks
    raw = page.get_text("rawdict", textpage=tp)["blocks"]
    if len(raw) != len(blocks):
        return blocks
    for block, rblock in zip(blocks, raw):
        for line, rline in zip(block.get("lines", []), rblock.get("lines", [])):
            if len(line["spans"]) != len(rline["spans"]) or not any(
                    _ACCENT_CHAR_RE.search(s["text"]) for s in line["spans"]):
                continue
            chars = [c for s in rline["spans"] for c in s["chars"]]
            owner = [k for k, s in enumerate(rline["spans"]) for _ in s["chars"]]
            texts = _attach_accents(chars)
            for k, span in enumerate(line["spans"]):
                span["text"] = "".join(t for t, o in zip(texts, owner) if o == k)
    return blocks


def _extract_lines(doc: pymupdf.Document) -> tuple[list[_Line], list[float]]:
    """Horizontal text lines with style info, in content-stream (reading) order."""
    lines: list[_Line] = []
    heights: list[float] = []
    block_id = 0
    for pno, page in enumerate(doc):
        heights.append(page.rect.height)
        for block in _page_blocks(page):
            if block.get("type") != 0:
                continue
            block_id += 1
            start = len(lines)
            for line in block.get("lines", []):
                dx, dy = line.get("dir", (1.0, 0.0))
                if abs(dy) > 0.1 or dx < 0.9:  # rotated text: arXiv sidebar stamp, axis labels
                    continue
                spans = [s for s in line.get("spans", []) if s.get("text")]
                text = "".join(s["text"] for s in spans)
                if not text.strip():
                    continue
                visible = [s for s in spans if s["text"].strip()]
                size = max(s["size"] for s in visible)
                bold = all(_is_bold(s) for s in visible)
                prefix = ""
                if not bold:
                    for s in spans:
                        if s["text"].strip() and not _is_bold(s):
                            break
                        prefix += s["text"]
                sup_tail = ""
                for s in reversed(visible):  # superscript flag, smaller than the line's text
                    if not (s["flags"] & 1 and s["size"] < 0.85 * size and re.fullmatch(r"[\d*†‡§¶,]+", s["text"].strip())):
                        break
                    sup_tail = s["text"].strip() + sup_tail
                if sup_tail and not re.search(r"(?:[^\W\d_]{2}|[.)])\s*$", text.rstrip()[:-len(sup_tail)]):
                    sup_tail = ""  # an exponent after a variable ("R" + "2" is R²), not a footnote mark
                x0, y0, x1, y1 = line["bbox"]
                lines.append(_Line(text, size, bold, pno, y0, y1, x0, block_id, prefix.strip(), x1=x1,
                                   italic=all(_is_italic(s) for s in visible), sup_tail=sup_tail))
            for k, ln in enumerate(lines[start:]):
                ln.pos_in_block, ln.block_lines = k, len(lines) - start
    return lines, heights


def _body_font_size(lines: list[_Line]) -> float:
    weights: Counter[float] = Counter()
    for ln in lines:
        weights[round(ln.size, 1)] += len(ln.text)
    return weights.most_common(1)[0][0] if weights else 10.0


def _merge_split_numbers(lines: list[_Line]) -> list[_Line]:
    """Merge a lone section number with the title set at the same baseline ("2" + "Related Work").

    A bare appendix letter ("A" + "Brief Review of ...") is merged only with a title in the same
    style (both bold, same size): a letter next to a line of text is usually a figure label.
    """
    out: list[_Line] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        number = re.sub("[\u200b\u2060\ufeff]", "", cur.text).strip()  # "1.\u200b" (zero-width space) in Google Docs PDFs
        letter = (nxt is not None and re.fullmatch(r"[A-H]", number) is not None and cur.bold and nxt.bold
                  and abs(cur.size - nxt.size) < 0.5 and nxt.text.strip()[:1].isupper())
        if (nxt is not None and (_NUMBER_ONLY_RE.match(number) or letter) and cur.page == nxt.page
                and abs(cur.y0 - nxt.y0) < 3 and nxt.x0 > cur.x0 and cur.size >= 0.6 * nxt.size):  # not a superscript
            out.append(_Line(f"{number} {nxt.text.strip()}", max(cur.size, nxt.size),
                             cur.bold and nxt.bold, cur.page, cur.y0, nxt.y1, cur.x0, nxt.block,
                             "", nxt.pos_in_block, nxt.block_lines, x1=nxt.x1, italic=cur.italic and nxt.italic,
                             sup_tail=nxt.sup_tail))
            i += 2
            continue
        out.append(cur)
        i += 1
    return out


def _normalize_heading(title: str) -> str:
    """Collapse whitespace; title-case ALL-CAPS/small-caps names (single-word acronyms like BERT stay).

    Articles, conjunctions and short prepositions stay lowercase except at the start or after a
    colon ("Discussion and Conclusions"); see _restore_casing for "DNN".
    """
    title = " ".join(title.split()).strip(" .:")
    if _mostly_caps(title) and (len(title.split()) > 1 or title.lower() in KNOWN_HEADINGS):
        title = " ".join(_title_case(title.split()))
    return title


def _mostly_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8


def _title_case(words: list[str], at_start: bool = True) -> list[str]:
    """Title-case words, keeping small words lowercase except first (at_start) or after a colon."""
    out: list[str] = []
    for w in words:
        after_colon = bool(out) and out[-1].endswith((":", "—", "–"))
        if w.lower() in _SMALL_WORDS and not after_colon and (out or not at_start):
            out.append(w.lower())
        else:
            out.append(w[:1].upper() + w[1:].lower())
    return out


_WORD_EDGE = "()[],:;\"'“”‘’."


def _paper_casings(lines: list[_Line]) -> dict[str, str]:
    """How the paper writes words with inner capitals in ordinary (mixed-case) lines, keyed by the
    upper-case form: "DNN", "LSTMs", "SpectralFormer", "rEGGression" (the most frequent spelling)."""
    counts: Counter[str] = Counter()
    for ln in lines:
        letters = [c for c in ln.text if c.isalpha()]
        if len(letters) < 20 or sum(c.isupper() for c in letters) > 0.5 * len(letters):
            continue  # a heading or an ALL-CAPS line says nothing about a word's usual case
        for w in ln.text.split():
            core = w.strip(_WORD_EDGE)
            if any(c.isalpha() for c in core) and core.lower() not in _SMALL_WORDS:
                counts[core] += 1
    best: dict[str, tuple[int, str]] = {}  # upper-case form -> (count, spelling) of the most frequent spelling
    plain: Counter[str] = Counter()  # uses in lowercase or with only a first capital ("vulnerabilities")
    for word, n in counts.items():
        key = word.upper()
        if any(c.isupper() for c in word[1:]):
            if n > best.get(key, (0, ""))[0]:
                best[key] = (n, word)
        else:
            plain[key] += n
    out: dict[str, str] = {}
    for key, (n, word) in best.items():
        if n >= plain[key]:  # the paper's usual spelling, not an emphasis or a glyph artifact ("vulnerabiliTies")
            if word[:1].islower() and "-" in word and word.split("-")[0].islower() and len(word.split("-")[0]) >= 3:
                word = word[:1].upper() + word[1:]  # "robot-IoT" inside a sentence, "Robot-IoT" in a title
            out[key] = word
    return out


def _restore_casing(name: str, raw: str, casings: dict[str, str]) -> str:
    """Give the words that title-casing an ALL-CAPS heading changed ("Dnn", "Spectralformer") the
    spelling the paper uses in its text ("DNN", "SpectralFormer"); a one-word title the paper does
    not spell that way is title-cased ("IMPLEMENTATION" -> "Implementation")."""
    raw_words = {w.strip(_WORD_EDGE) for w in raw.split()}
    out = []
    for w in name.split():
        core = w.strip(_WORD_EDGE)
        if core and core.upper() in raw_words:
            if core.upper() in casings:
                w = w.replace(core, casings[core.upper()])
            elif "-" in core:  # "Ai-text" -> "AI-text" from "AI"
                w = w.replace(core, "-".join(casings.get(p.upper(), p) for p in core.split("-")))
        out.append(w)
    name = " ".join(out)
    if re.fullmatch(r"[A-Z][A-Z\-]{3,}", name) and name not in casings:
        name = name.capitalize()
    return name


def _split_sentence(title: str) -> tuple[str, str] | None:
    """Split "Method. Common human evaluation methods include ..." into (head, rest).

    Returns None when the title holds no sentence break followed by at least 3 words, or the
    period belongs to an abbreviation ("BERT vs. GPT", "e.g. Transformers").
    """
    for m in re.finditer(r"(?<=[A-Za-z0-9)])\.\s+(?=[A-Z(\[])", title):
        head, rest = title[:m.start()].strip(), title[m.end():].strip()
        last = head.split()[-1].lower().rstrip(".") if head.split() else ""
        if last in _ABBREVIATIONS or len(last) == 1:  # abbreviation or initial ("J. Smith")
            continue
        if len(rest.split()) >= 3:
            return head, rest
    return None


def _code_line(title: str) -> bool:
    """A code or algorithm line ("1 Initialize K = 0", "4 HIDDEN_SIZE = 20"), not a heading ("Case p = 1")."""
    if _CODE_RE.search(title):
        return True
    m = re.search(r"[=≠∈]", title)
    return bool(m) and (len(title[:m.start()].split()) <= 1 or bool(_ALGO_VERB_RE.match(title)))


def _appendix_heading(ln: _Line, text: str, styled: bool, body_size: float) -> tuple[str, int] | None:
    """An appendix title ("APPENDIX A", "Appendix B. The limit of ...", "Appendix C: Case Study").

    Set bold, larger than body text, or in capitals, at the start of its block. Level 1, named by
    its title ("Appendix A" when it has none). A sentence ("Appendix A shows ...") is not a title.
    """
    m = _APPENDIX_RE.match(text)
    if m is None or ln.size < body_size - 0.3 or (ln.block_lines > 1 and ln.pos_in_block > 0):
        return None
    title = m.group("title").strip(" .:")
    if len(title.split()) > _MAX_HEADING_WORDS or (title and not (title[0].isupper() or title[0] in "\"'“‘(")):
        return None
    if not (styled or text.split()[0].isupper()):
        return None
    return (_normalize_heading(title) if title else f"Appendix {m.group('num').upper()}"), 1


def _heading_name(
    ln: _Line, body_size: float, opts: ParseOptions, numbered_mode: bool = False, letter: bool = False,
) -> tuple[str, int] | None:
    """Return (normalized section name, level) if the line is a section heading, else None.

    level = 1 + number of dots in the section number ("3" -> 1, "3.2" -> 2); unnumbered -> 1.
    In numbered_mode (the paper numbers its sections), unnumbered headings are accepted only
    for UNNUMBERED_OK names. This rejects bold table cells such as "Training". letter=True admits a
    bare-letter appendix number ("B Proofs"), see _letter_headings.
    """
    text = " ".join(ln.text.split())
    if not text or len(text) > 100:
        return None
    styled = ln.bold or ln.size >= body_size + opts.heading_size_delta
    appendix = _appendix_heading(ln, text, styled, body_size)
    if appendix is not None:
        return appendix
    m = _NUMBERED_RE.match(text) or (_LETTER_HEAD_RE.match(text) if letter else None)
    n_title = len(m.group("title").split()) if m else len(text.split())
    # a long bold title is a subtitle or in title case ("4. Part II: Minimizing the impact on the form of the
    # original text", "Faces of the Family Variable Polytope defined by Orders and by Sinks"), not a list item
    content_words = [w for w in (m.group("title").split() if m else []) if w.lower() not in _SMALL_WORDS]
    long_ok = (m is not None and styled and "," not in text and text.split()[-1].lower() not in _FRAGMENT_ENDS
               and (":" in text or sum(w[:1].isupper() for w in content_words) >= 0.6 * max(1, len(content_words))))
    if (n_title > _MAX_HEADING_WORDS) if long_ok else len(text.split()) > 12:
        return None  # a long numbered bold line is a list item ("2. Test the model against one of the smaller, ...")
    caps_title = m is not None and bool(re.search(r"[^\W\d_]{3}", m.group("title"))) and all(
        c.isupper() for c in m.group("title") if c.isalpha())
    if text.endswith(".") and not (m and (len(text.split()) <= 8 or caps_title)):
        return None  # a sentence; "2.1. Complexity analysis." / "3.2 MODEL B. MODERN HOPFIELD ..." are headings
    # Table cells / captions / footnotes are set smaller than body text; ACM journals (and some
    # revtex preprints) set numbered bold headings ("1. INTRODUCTION", "2.1. Grouping-based
    # methods") at ~0.9x body size, at the top of their block (position 1 when the section
    # number was set separately and merged by _merge_split_numbers).
    acm_heading = m and ln.bold and ln.pos_in_block <= 1 and ln.size >= body_size * 0.85
    # IEEE section titles in small capitals can come out smaller than body text ("I. INTRODUCTION" at 8pt)
    ieee_caps = (m is not None and re.fullmatch(r"[IVX]{1,5}\.", text.split()[0]) is not None and caps_title
                 and ln.block_lines == 1 and ln.size >= body_size * 0.75)
    if ln.size < body_size - 0.3 and not acm_heading and not ieee_caps:
        return None
    if ln.block_lines > 4 and ln.pos_in_block > 1:
        return None  # inside a (bold) paragraph, not a heading
    title = m.group("title") if m else text
    if _split_sentence(title):
        return None  # run-in heading + sentence ("3.1 Attacks. Perturbing data ..."): see _numbered_run_in
    first = title.split()[0].lower().strip(",:")
    if m and first in _PRONOUN_STARTS and not (styled and first in ("our", "this") and len(title.split()) <= 6):
        return None  # numbered list item ("1. We identify and describe ..."), not "1.2 Our Contributions"
    if _MATH_END_RE.search(title):
        return None  # a formula line ("X. Assume σ|s| ≥")
    if m and (re.search(r"←|:=|[{};]|\w\(\)", title) or title.split()[-1] in ("do", "then")):
        return None  # numbered pseudo-code line ("9 CCQCP ←HashmapInit()", "4 for s in S do")
    if "@" in title or _FRONT_MATTER_RE.match(title):
        return None  # a footnote ("1 Corresponding author: ...@...")
    if _CAPTION_RE.match(title) or _URL_RE.search(title):
        return None  # "5. Supplementary Figure 2. Flowchart ...", "Номер статті <http://...>"
    if m and text.split()[0].rstrip(".").isdigit() and int(text.split()[0].rstrip(".")) > 40:
        return None  # a numbered entry of a long list ("44. Euzenat. Towards ..."), not a section
    items = [x.split() for x in title.split(",")]
    if len(items) >= 3 and sum(1 for x in items if len(x) >= 2 and all(w[:1].isupper() for w in x)) >= 2:
        return None  # a list of names ("1. Tania Fleitas Kanonnikoff, Ana Miralles Marco, ..."), not a title
    words = re.findall(r"[A-Za-z]{2,}", title)
    if not words or sum(c.isalpha() for c in title) / max(1, len(title.replace(" ", ""))) < 0.7:
        return None
    if not next(c for c in title if c.isalpha()).isupper():
        # titles start with a capital: rejects numbered pseudo-code ("7 else") and math lines; a bold
        # subsection may start with a lowercase name ("2.2 word2vec and GloVe", "2.1.1 logistic")
        if not (styled and m and title[0].isalpha() and first not in _PSEUDO_CODE_WORDS
                and "." in text.split()[0].rstrip(".")):
            return None  # (a subsection number: "7. apply behavior: b" is an algorithm step)
    if not (title[0].isalnum() or title[0] in "\"'“‘(") or title.endswith(","):
        return None  # code/math lines ("# Define the model", "+ Lmaxη2M") and fragments ("Hence,")
    name = _normalize_heading(title)
    key = name.lower()
    if _REFERENCES_RE.match(key) and len(text.split()) <= 3:
        return name, 1  # bare "References" line, any style (R1)
    letters = [c for c in title if c.isalpha()]
    all_caps = len(letters) >= 4 and all(c.isupper() for c in letters)
    # IEEE/Springer-style headings ("II. RELATED WORK", "A. Datasets", "1 INTRODUCTION") are set at
    # body size without bold; a numbered line alone in its block, or numbered ALL CAPS, counts as emphasized.
    all_caps = all_caps and bool(re.search(r"[^\W\d_]{3}", title))  # a word, not a formula ("U2QΓ1VT")
    emphasized = styled or (m is not None and (ln.block_lines == 1 or all_caps))
    if not emphasized:
        return None
    if not styled and m and text.endswith(":") and len(title.split()) > 6:
        return None  # a numbered sentence introducing a list ("2. Brown clustering features benefit from ...:")
    if not styled and _code_line(title):
        return None  # a numbered code or algorithm line ("1 Initialize K = 0", "4 HIDDEN_SIZE = 20")
    if not styled and not all_caps:
        # body-style numbered line alone in its block (IEEE "A. Datasets"): also what a math line
        # ending in a number followed by a sentence looks like ("0. The hyperparameters are ...")
        words_lc = title.lower().split()
        if (text.split()[0].split(".")[0] == "0" or words_lc[0] in _FRAGMENT_STARTS
                or words_lc[-1] in _FRAGMENT_ENDS):
            return None
        if text.endswith(".") and _SENTENCE_VERBS & set(words_lc):
            return None  # a sentence after a split-off superscript ("2. The results are below.")
    if m:
        return name, text.split()[0].rstrip(".").count(".") + 1
    if key in UNNUMBERED_OK or (key in KNOWN_HEADINGS and not numbered_mode) or (
            key in CHAPTER_HEADINGS and ln.size >= 1.5 * body_size and ln.block_lines == 1):
        return name, 1  # a chapter title ("INTRODUCTION" at 2.5x body size) even in a numbered thesis
    return None


def _roman_value(numeral: str) -> int:
    values = {"I": 1, "V": 5, "X": 10}
    total = 0
    for k, c in enumerate(numeral):
        v = values[c]
        total += -v if k + 1 < len(numeral) and values[numeral[k + 1]] > v else v
    return total


def _section_number(text: str) -> tuple[int, ...] | None:
    m = re.match(r"(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s", text.strip() + " ")
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def _list_item(ln: _Line, before: list[_Line], after: list[_Line], body_size: float, opts: ParseOptions) -> bool:
    """A one-level numbered line in body style that is an item of a numbered list, not a section heading.

    `before`/`after` are the raw lines around it (nearest first; formula lines included, so a colon
    that introduces a display equation is not taken for one that introduces a list). It is a list item when the text
    above introduces a list ("... the following steps:"), when its sentence runs on in lowercase on
    the next line, or when the item numbered one lower is just above it or one higher just below it
    ("3. Obtain ..." after "2. Train ...", with at most 3 short lines between; formula lines, rows of
    numbers and the indented continuation lines of an item are skipped): a section heading is
    followed by its text, never directly by the next section.
    """
    number = _section_number(ln.text)
    if number is None or len(number) > 1 or ln.bold or ln.size >= body_size + opts.heading_size_delta:
        return False  # list items, steps and algorithm lines have one-level numbers; "3.2.1" is a subsection
    before = [x for x in before if not _PAGE_NUM_RE.match(x.text)]  # a page number between
    after = [x for x in after if not _PAGE_NUM_RE.match(x.text)]
    if before and before[0].heading is None and before[0].text.rstrip().endswith(":"):
        return True
    if after and after[0].text[:1].islower() and not after[0].bold and len(after[0].text.split()) >= 3 \
            and abs(after[0].size - ln.size) < 0.5:
        return True  # "2. Medieval texts, with" + "being of notable significance, ..." (not a table cell "sMAPE")
    for side, step in ((before, -1), (after, 1)):
        seen = 0
        for other in side:
            numbered = _NUMBERED_RE.match(other.text)
            if numbered is None and other.page == ln.page and ln.x0 + 4 < other.x0 < ln.x0 + 40:
                continue  # the indented continuation of an item ("1. Design confidence sets" / "such that ...")
            if (numbered is None and _alpha_ratio(other.text) < 0.5) or (
                    numbered is not None and not re.search(r"[^\W\d_]", numbered.group("title"))):
                continue  # a formula, or a table row of numbers ("88.5 48.7") between items
            seen += 1
            if seen > 4:
                break
            n = _section_number(other.text) if _NUMBERED_RE.match(other.text) else None  # not a bare "3.2."
            if n is not None:
                if len(n) == len(number) and n[-1] == number[-1] + step:
                    return True  # "2." above "3.", or "4." below it (a list counts up)
                break
            if len(other.text.split()) >= 6:
                break  # prose between: the neighbouring number starts a section of its own
    return False


def _continues_heading(head: _Line, nxt: _Line, body_size: float, opts: ParseOptions) -> bool:
    """A heading wrapped onto a second line: same block and style, not itself numbered.

    Headings set in body style (IEEE ALL-CAPS, not bold) share their block with the paragraph
    that follows, so for them only an ALL-CAPS next line (or, for an italic IEEE subsection title,
    an italic one) counts as a continuation. A question mark may end a heading ("... Matter?").
    """
    text = nxt.text.strip()
    if (nxt.block != head.block or nxt.bold != head.bold or abs(nxt.size - head.size) >= 0.5
            or _NUMBERED_RE.match(text) or len(text.split()) > 8
            or (_SENT_END_RE.search(text) and not text.endswith("?"))):
        return False
    if _PAGE_REF_RE.match(text) or _is_references_line(text):
        return False  # a page number ("Introduction" + "3" in a contents list) or its own heading
    if head.bold or head.size >= body_size + opts.heading_size_delta or (head.italic and nxt.italic):
        return True
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _in_margin(ln: _Line, page_height: float, opts: ParseOptions) -> bool:
    """Line starts in the top zone or ends in the bottom zone of the page."""
    return ln.y0 < page_height * opts.margin_ratio or ln.y1 > page_height * (1 - opts.margin_ratio)


def _run_in_heading(ln: _Line, body_size: float) -> tuple[str, int, str] | None:
    """SIAM-style run-in heading: bold "2.1. Complexity analysis." followed by body text on the line.

    Only numbered prefixes ending with a period qualify (unnumbered bold run-ins are paragraph
    labels, not sections). Returns (name, level, remaining body text) or None.
    """
    prefix = " ".join(_repair(ln.bold_prefix).split())  # same normalization as ln.text
    text = " ".join(ln.text.split())
    if (not prefix.endswith(".") or len(prefix) > 100 or ln.size < body_size - 0.3
            or not text.startswith(prefix)):
        return None
    m = _NUMBERED_RE.match(prefix)
    if not m:
        return None
    title = m.group("title").rstrip(".").strip()
    first = next((c for c in title if c.isalpha()), "")
    if not first.isupper() or len(re.findall(r"[A-Za-z]{2,}", title)) == 0:
        return None
    rest = text[len(prefix):].strip()
    return _normalize_heading(title), prefix.split()[0].rstrip(".").count(".") + 1, rest


def _numbered_run_in(
    ln: _Line, body_size: float, opts: ParseOptions, numbered_mode: bool,
) -> tuple[str, int, str] | None:
    """Numbered heading with the first sentence on the same line, set in one style.

    "7.2.2 Method. Common human evaluation methods include ..." -> ("Method", 3, "Common ...").
    The heading must be visually emphasized (bold or larger than body) and pass _heading_name.
    """
    text = " ".join(ln.text.split())
    m = _NUMBERED_RE.match(text)
    if not m or not (ln.bold or ln.size >= body_size + opts.heading_size_delta):
        return None
    split = _split_sentence(m.group("title"))
    if split is None or len(split[0].split()) > 8:
        return None
    head, rest = split
    probe = _Line(f"{text.split()[0]} {head}", ln.size, ln.bold, ln.page, ln.y0, ln.y1, ln.x0,
                  ln.block, "", ln.pos_in_block, ln.block_lines)
    found = _heading_name(probe, body_size, opts, numbered_mode)
    return (found[0], found[1], rest) if found else None


_FRONT_MATTER_RE = re.compile(
    r"^(?:corresponding|correspondence|e-?mail|received|accepted|published|copyright|citation|affiliations?)\b",
    re.I)
# Numbered affiliations ("1 Indian Institute of Science, India") can look like a first heading.
_AFFILIATION_RE = re.compile(
    r"\b(?:universit|institut|laborator|college|school of|department|dept\.|faculty|academy|hospital|"
    r"research (?:center|centre|lab)|inc\.|corporation|gmbh|ltd)", re.I)
_LABEL_START_RE = re.compile(
    r"^(?:algorithm|theorem|lemma|definition|proof|corollary|proposition|remark|example|figure|fig\.|table|"
    r"tab\.|keywords|index terms|equation|eq\.|step|input|output|corresponding|correspondence|e-?mail|"
    r"received|accepted|published|copyright|citation|affiliations?)\b", re.I)


def _unnumbered_heading(ln: _Line, following: list[_Line], body_size: float, opts: ParseOptions) -> str | None:
    """A custom unnumbered heading in a paper without numbered sections ("Historical Background").

    Emphasized (bold or larger; bold down to 0.85x body size when alone in its block, as in revtex),
    short, capitalized, no end punctuation, not a caption/theorem/front-matter label, alone in its
    block or first in it, and followed (possibly after one sub-heading line) by >= 15 words of
    body-style text: table header cells are followed by other short cells.
    """
    text = re.sub(r"^\d{1,2}\s*[-–)]\s*", "", " ".join(ln.text.split()))  # "1- Introduction", "2) Data"
    words = text.split()
    if not words or len(words) > 10 or len(text) > 90 or text[-1] in ".:;,!?" or "@" in text:
        return None
    if not (ln.bold or ln.size >= body_size + opts.heading_size_delta) or ln.size < body_size * 0.85:
        return None
    if ln.size < body_size - 0.3 and not (ln.bold and ln.block_lines == 1):
        return None
    if ln.block_lines > 1 and ln.pos_in_block != 0:
        return None
    if not text[0].isupper() or _alpha_ratio(text) < 0.7 or _LABEL_START_RE.match(text):
        return None
    if re.match(r"(?:fig(?:ure)?|tab(?:le)?|algorithm)\.?\d", text.replace(" ", ""), re.I):
        return None  # small-caps caption label "FIGUR E 1"
    if (sum(c.isalpha() for c in text) < 4 or "|" in text  # "Σ", "MU", "Red: Removed | Green: Added"
            or sum(1 for w in words if len(w) == 1 and w.isalpha()) >= 2  # letter-spaced "Figur E 1"
            or sum(1 for x in text.split(",") if len(x.split()) >= 2) >= 2):  # a list of names
        return None
    rest = following[:7]
    if rest and (rest[0].bold or rest[0].size >= body_size + opts.heading_size_delta) \
            and len(rest[0].text.split()) <= 10 and not _SENT_END_RE.search(rest[0].text):
        rest = rest[1:]  # a sub-heading right below: the prose follows it
    body_words = 0
    for nxt in rest[:6]:
        if nxt.bold or abs(nxt.size - body_size) > 0.6:
            break
        body_words += len(nxt.text.split())
    return _normalize_heading(text) if body_words >= 15 else None


def _is_references_line(text: str) -> bool:
    """A line that is only a reference-list title ("References", "7 REFERENCES", "Bibliography:")."""
    text = " ".join(text.split())
    m = _NUMBERED_RE.match(text)
    title = m.group("title") if m else text
    if len(title) > 8 and sum(1 for w in title.split() if len(w) <= 2) >= 0.6 * len(title.split()):
        title = title.replace(" ", "")  # letter-spaced "R E F E R E N C E S"
    return (len(title.split()) <= 3 and title[:1].isupper()  # "reference." in a wrapped sentence is prose
            and bool(_REFERENCES_RE.match(_normalize_heading(title).lower())))


def _bib_line(text: str) -> bool:
    """One line of a bibliography entry (see the _BIB_* features), not prose that cites papers."""
    text = " ".join(text.split())
    words = text.split()
    if len(words) < 2:
        return False
    if _BIB_LABEL_RE.match(text):
        return True
    stripped = _INTEXT_CITE_RE.sub(" ", text)
    if _BIB_YEAR_RE.search(stripped) or _BIB_PAGES_RE.search(stripped) or _BIB_VENUE_RE.search(stripped):
        return True
    if len(_NAME_LIST_RE.findall(stripped)) >= 2 or len(_INITIALS_LIST_RE.findall(stripped)) >= 2:
        return True  # "Tom Brown, Benjamin Mann, Nick Ryder, ...", "Abadi, M.; Agarwal, A.; ..."
    capitalized = sum(1 for w in words if w[:1].isupper())  # "Shlomo Argamon-Engelson and Ido Dagan."
    return 3 <= len(words) <= 10 and stripped.rstrip().endswith(".") and capitalized / len(words) >= 0.7


def _bib_hits(lines: list[_Line], start: int) -> list[bool]:
    return [_bib_line(ln.text) for ln in lines[start:start + 40] if len(ln.text.split()) >= 2]


def _bibliography_follows(lines: list[_Line], i: int) -> bool:
    """R1 validation: the lines from i on read as a reference list, not a table under a
    "Reference(s)" column header.

    Accepted when >= 2 of the next 8 content lines look like entries and >= 10% of the next 40 do,
    or (a table or figure interleaved in reading order) >= 1 of the next 8 and >= 30% of the 40
    lines after those; at the end of a document (< 8 content lines left), when half of them do.
    """
    hits = _bib_hits(lines, i)
    if not hits:
        return False
    if len(hits) < 8:  # a short list on the last page
        return sum(hits) >= max(1, len(hits) / 2)
    if sum(hits[:8]) >= 2 and sum(hits) >= 0.1 * len(hits):
        return True
    later = _bib_hits(lines, i + 40)
    return sum(hits[:8]) >= 1 and bool(later) and sum(later) >= 0.3 * len(later)


def _reference_lists(lines: list[_Line], titles: list[int]) -> int:
    """How many separate reference lists the validated reference titles at `titles` start.

    A title right after bibliography entries (>= 2 of the 5 lines before it) continues the list
    above it: a running "Bibliography" header repeated on each page of one list. A title after
    prose starts a list of its own (the end of a chapter, or of one report in a compilation).
    """
    count = 0
    for k in titles:
        before = [ln.text for ln in lines[max(0, k - 12):k] if len(ln.text.split()) >= 2][-5:]
        if sum(_bib_line(t) for t in before) < 2:
            count += 1
    return count


def _headingless_bibliography(lines: list[_Line], i: int) -> bool:
    """A reference list without a title: "[1] K. Simonyan and ...", then [2], [3] and entries."""
    if not _BIB_FIRST_RE.match(lines[i].text.strip()):
        return False
    bracket = lines[i].text.strip().startswith("[")
    starts = {"[2]", "[3]"} if bracket else {"2.", "3."}
    seen = {ln.text.strip().split()[0] for ln in lines[i + 1:i + 40] if ln.text.strip()}
    return starts <= seen and _bibliography_follows(lines, i)


def _letter_headings(lines: list[_Line]) -> set[int]:
    """ids of bare-letter appendix titles ("B Proofs of Main Results").

    A line qualifies when its "B.1 ..." subsection follows within 150 lines, or when it is one of
    two consecutive letters set bold in the same size ("A Brief Review ..." then "B Decoding ...").
    """
    out: set[int] = set()
    bold_heads: list[tuple[int, int, _Line]] = []
    for k, ln in enumerate(lines):
        m = _LETTER_HEAD_RE.match(" ".join(ln.text.split()))
        if not m:
            continue
        first_sub = re.compile(rf"^{m.group('num')}\.1\.?\s+[A-Z]")
        if any(first_sub.match(x.text.strip()) for x in lines[k + 1:k + 150]):
            out.add(id(ln))
        if ln.bold and ln.pos_in_block <= 1 and len(ln.text.split()) <= _MAX_HEADING_WORDS:
            bold_heads.append((k, ord(m.group("num")), ln))
    for (_, a, head_a), (_, b, head_b) in zip(bold_heads, bold_heads[1:]):
        if b == a + 1 and abs(head_a.size - head_b.size) < 0.5:
            out.update((id(head_a), id(head_b)))
    return out


def _toc_pages(lines: list[_Line]) -> set[int]:
    """Pages holding a table of contents / list of figures (their entries look like headings).

    A page qualifies with >= 3 entries (text, dot leader, page number at the line end or on the
    next line; not "......" in a diagram or an elided prompt), or with a "Contents" title and >= 5 bare page numbers; a page right after a
    contents page qualifies with >= 8 bare page numbers and >= 1 dot-leader line (a table alone
    also has many bare numbers).
    """
    leaders: Counter[int] = Counter()
    numbers: Counter[int] = Counter()
    titled: set[int] = set()
    for k, ln in enumerate(lines):
        text = " ".join(ln.text.split())
        nxt = " ".join(lines[k + 1].text.split()) if k + 1 < len(lines) else ""
        page_no = (re.search(r"(?:[.…·]\s?)\s*(?:\d{1,4}|[ivxlc]{1,7})$", text, re.I)
                   or (re.search(r"[.…·]$", text) and _PAGE_REF_RE.match(nxt)))
        if _LEADER_RE.search(text) and len(re.findall(r"[A-Za-z]", text)) >= 3 and page_no:
            leaders[ln.page] += 1  # an entry: title, dot leader, page number
        elif _PAGE_REF_RE.match(text):
            numbers[ln.page] += 1
        if _TOC_TITLE_RE.match(text.strip(" :")):
            titled.add(ln.page)
    pages = {p for p, n in leaders.items() if n >= 3}
    pages |= {p for p in titled if numbers[p] >= 5 or leaders[p] >= 1}
    for p in sorted(pages):
        nxt = p + 1
        while nxt not in pages and numbers[nxt] >= 8 and leaders[nxt] >= 1:
            pages.add(nxt)
            nxt += 1
    return pages


def _header_footer_keys(lines: list[_Line], heights: list[float], opts: ParseOptions) -> dict[str, float]:
    """Digit-normalized margin lines repeated on >= header_footer_min_ratio of pages -> their usual size.

    The default ratio (0.3) also catches running headers that alternate between odd and even
    pages (title / authors), which each appear on ~half the pages. A margin line that starts with a
    number, ends with a page number or is in ALL CAPS (a thesis chapter header: "12 1. INTRODUCTION",
    "5. EXPLANATORY RECTIFICATIONS") counts from 3 pages. The size tells a heading at the top of a
    page ("4.1. Quality assessment", 14 pt) from the running header that repeats it (9 pt). In a
    short paper, a line of at least two words on every odd or every even page after the first
    counts too ("Happe and Cito" on pages 2 and 4 of 5).
    """
    n_pages = len(heights)
    if n_pages < 3:
        return {}
    pages_with: dict[str, set[int]] = {}
    sizes: dict[str, Counter] = {}
    chapterish: set[str] = set()
    alternating: set[str] = set()
    for ln in lines:
        if _in_margin(ln, heights[ln.page], opts):
            key = re.sub(r"\d+", "#", " ".join(ln.text.split()).lower())
            pages_with.setdefault(key, set()).add(ln.page)
            sizes.setdefault(key, Counter())[round(ln.size, 1)] += 1
            if len(re.findall(r"[^\W\d_]{2,}", ln.text)) >= 2:
                alternating.add(key)
            letters = [c for c in ln.text if c.isalpha()]
            if key.startswith("#") or key.endswith("#") or (len(letters) >= 4 and all(c.isupper() for c in letters)):
                chapterish.add(key)
    threshold = max(3, opts.header_footer_min_ratio * n_pages)
    # a chapter's running header ("12 1. INTRODUCTION", "5. EXPLANATORY RECTIFICATIONS") repeats only
    # over that chapter's pages
    return {k: sizes[k].most_common(1)[0][0] for k, pages in pages_with.items()
            if len(pages) >= threshold or (k in chapterish and len(pages) >= 3)
            or (k in alternating and len(pages) >= 2 and pages in (set(range(1, n_pages, 2)), set(range(2, n_pages, 2))))}


def _is_mojibake(text: str) -> bool:
    chars = [c for c in text if not c.isspace()]
    n = len(_MOJIBAKE_RE.findall(text))
    return n >= 4 and n / max(1, len(chars)) >= 0.2  # >= 4: a name like "Küçük" alone is not mojibake


def _alpha_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return sum(c.isalpha() for c in chars) / len(chars) if chars else 0.0


def _repair(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _LOOSE_MARK_RE.sub(lambda m: _LOOSE_MARKS.get(m.group(1), m.group(0)), text)  # NFKC's " ́" -> "´"
    text = _CID_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)  # e.g. "\x07" before headings, math symbols in custom encodings
    text = _PRIVATE_USE_RE.sub("", text)
    text = _FF_SPACE_RE.sub(r"\1 \2", text)
    # an accent whose base glyph was unmapped and removed ("Ĥ" in an equation font): spacing form or nothing
    text = _ORPHAN_MARK_RE.sub(lambda m: m.group(1) + _LOOSE_MARKS.get(m.group(2), ""), text)
    return text.replace("\u00ad", "").replace("\ufffd", "")  # soft hyphen, replacement char


_WORD_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*")
_BREAK_LEFT_RE = re.compile(r"([^\W_]+(?:-[^\W_]+)*)-$")
_BREAK_RIGHT_RE = re.compile(r"([^\W_]+)(-[^\W_])?")
_SUSPENDED_RE = re.compile(r"^(?:and|or|nor)\s+[^\W_]+-[^\W_]")  # "pre-" + "and post-training"
# Line-break halves that are words of the paper, yet form one closed word: function words and prefixes
# TeX splits off ("al-most", "with-out", "per-form", "more-over"), and suffixes ("allow-ing").
_CLOSED_LEFT = {"al", "an", "any", "at", "be", "by", "can", "con", "de", "dis", "do", "else", "en", "every", "ex", "for",
                "further", "here", "how", "im", "in", "into", "mean", "more", "never", "no", "not", "of", "on", "other",
                "out", "over", "per", "re", "some", "sub", "there", "through", "to", "un", "under", "up", "what", "when",
                "where", "which", "who", "with", "it", "them", "him", "her", "our", "your", "my", "auto", "eigen", "hyper"}
_SUFFIXES = {"ing", "ings", "ed", "ly", "fully", "tion", "tions", "sion", "sions", "ation", "ations", "ment", "ments",
             "ness", "able", "ably", "ible", "ability", "abilities", "ity", "ities", "ies", "ers", "est", "ful", "ous",
             "ive", "ives", "ence", "ences", "ance", "ances", "ize", "ized", "izes", "izing", "ise", "ised", "ical",
             "ically", "ship", "less", "most", "work", "works"}
# Prefixes that are not words on their own but keep their hyphen before one ("semi-" + "supervised").
_OPEN_LEFT = {"self", "semi", "cross"}
# The space after an "ff" ligature is often missing from the text layer ("trade-offbetween", "offthe").
_FF_SPACE_RE = re.compile(r"\b((?:[Tt]rade-?|[Pp]ay|[Cc]ut|[Hh]and|[Tt]ake|[Kk]ick|[Ff]all|[Rr]oll|[Dd]rop|[Ss]how|[Bb]ack)?"
                          r"off)(the|in|be|between|of|to|and|is|for|at|on|with|as|by|we|it|this|that|are|was|from|or)\b")


def _spellings(lines: list[str]) -> Counter:
    """How often the paper writes each word and hyphen compound (lowercased) away from line ends.

    A compound also counts its parts ("real-world" counts "real" and "world"). The words on both
    sides of a hyphenated line break are left out: they are what _join_hyphenated decides.
    """
    counts: Counter = Counter()
    broken = False
    for text in lines:
        words = text.split()
        breaks = bool(_BREAK_LEFT_RE.search(text))
        for word in words[1 if broken else 0:len(words) - 1 if breaks else len(words)]:
            for tok in _WORD_RE.findall(word.lower()):
                counts[tok] += 1
                if "-" in tok:
                    counts.update(tok.split("-"))
        broken = breaks
    return counts


def _join_hyphenated(left: str, right: str, spellings: Counter) -> str:
    """Join text that breaks after a hyphen ("represen-" + "tation ...") at a line/column end (R8).

    The hyphen stays where TeX would not have hyphenated: the word already holds a hyphen
    ("state-of-the-" + "art", "out-" + "of-distribution") or one letter precedes the break ("q-" +
    "values"). A suspended hyphen keeps its space ("pre-" + "and post-training"). Otherwise the
    paper's own spelling elsewhere decides ("real-world" / "frequency"), then <= 2 letters after the
    break keep the hyphen; failing that the hyphen stays before a capital or digit ("CIFAR-" + "10")
    and between two words the paper uses on their own ("manually-" + "written"), unless the first
    is a function word or prefix that forms closed words ("with-" + "out", "al-" + "most") or the
    second a suffix ("allow-" + "ing"); a syllable break ("represen-" + "tation") is joined.
    """
    m, n = _BREAK_LEFT_RE.search(left), _BREAK_RIGHT_RE.match(right)
    if not m or not n or _SUSPENDED_RE.match(right):
        return f"{left} {right}"
    x, y = m.group(1), n.group(1)
    last = x.split("-")[-1]
    keep, join = left + right, left[:-1] + right
    if "-" in x and spellings[f"{last}{y}".lower()] and not spellings[f"{last}-{y}".lower()]:
        return join  # a syllable break inside the last part ("question-an-" + "swer" -> "question-answer")
    if "-" in x or n.group(2) or len(last) == 1:
        return keep
    hyphenated, joined = f"{x}-{y}".lower(), f"{x}{y}".lower()
    if spellings[hyphenated] != spellings[joined]:
        return keep if spellings[hyphenated] > spellings[joined] else join
    if len(y) <= 2 and y.isalpha():
        return keep  # TeX leaves >= 3 letters after a hyphenation point ("frequen-" + "cy" only if the paper says so)
    if not y[0].islower():
        return keep
    left, right = last.lower(), y.lower()
    both_words = ((spellings[left] or left in _OPEN_LEFT) and spellings[right] and left not in _CLOSED_LEFT
                  and right not in _SUFFIXES)
    return keep if both_words else join


def _join_lines(lines: list[str], spellings: Counter | None = None) -> str:
    """Join a block's lines, repairing end-of-line hyphenation (R8)."""
    out = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if out.endswith("-") and len(out) > 1 and out[-2].isalnum() and line[:1].isalnum():
            out = _join_hyphenated(out, line, spellings or Counter())
        elif out:
            out += " " + line
        else:
            out = line
    return out


_PROSE_WORDS = {"the", "a", "an", "of", "is", "are", "which", "that", "we", "this", "these", "our"}


def _is_table_block(lines: list[str]) -> bool:
    """A block of >= 3 short lines (< 3.5 words on average) that are not sentences = table column.

    Captions and blocks containing a real sentence (>= 8 words ending in . ! ?) are prose set in a
    narrow column, not tables, and are kept; so is a sentence that inline math broke into one-word
    lines ("Alg. 1 provides the pseudocode of BTIL, which ..." / "independent" / "variational"):
    >= 4 lowercase one-word lines and articles or relative words, which table cells lack.
    """
    if len(lines) < 3:
        return False
    mean_words = sum(len(t.split()) for t in lines) / len(lines)
    sentence_ends = sum(1 for t in lines if _SENT_END_RE.search(t.strip()))
    if mean_words >= 3.5 or sentence_ends > 1:
        return False
    joined = " ".join(t.strip() for t in lines)
    if _CAPTION_RE.match(joined):
        return False
    words = [w.lower().strip(",.;:()") for w in joined.split()]
    one_word = sum(1 for t in lines if len(t.split()) == 1 and t.strip()[:1].islower())
    if len(words) >= 12 and sum(w in _PROSE_WORDS for w in words) >= 2 and one_word >= 4:
        return False
    return not any(len(piece.split()) >= 8 and piece.rstrip()[-1:] in ".!?"
                   for piece in re.split(r"(?<=[.!?])\s+", joined))


# Words no sentence ends on: a paragraph ending in one runs on in the next block even when that
# block starts with a capital ("As shown in" + "Figure 2, there is no ...").
_OPEN_END_WORDS = {"the", "a", "an", "of", "in", "to", "for", "with", "by", "from", "and", "or", "at", "on",
                   "as", "that", "into", "than", "via"}


# Capitalized words that start a new sentence rather than continue one: after "... it suffices to
# show that" + a dropped display equation, "Let e > 0." is a new sentence; "Raspberry Pi ...",
# "GPT-4o-mini, ...", "Figure 2, there is ..." continue it.
_SENTENCE_STARTS = _SMALL_WORDS | _FRAGMENT_STARTS | _PRONOUN_STARTS | {
    "let", "here", "therefore", "using", "all", "note", "finally", "given", "consider", "suppose", "assume",
    "recall", "next", "first", "second", "third", "now", "however", "moreover", "furthermore", "also",
    "similarly", "specifically", "these", "those", "their", "its", "each", "every", "both", "such", "other",
    "another", "some", "many", "most", "no", "not", "one", "two", "although", "though", "after", "before",
    "during", "further", "additionally", "overall", "instead", "otherwise", "indeed", "notably",
    "importantly", "unlike", "like", "following", "based", "what", "how", "why", "see", "yet", "in",
    "to", "our", "you", "he", "she", "his", "her", "any", "only", "to", "do", "does", "is", "are"}
# A list item, a caption ("Algorithm 1 Pseudocode ..." but not "Algorithm 1 can ..."), a run-in label
# ("Remarks:", "Proof.") or a theorem-like statement: not the rest of a sentence.
_NEW_UNIT_RE = re.compile(
    r"^(?:\d{1,2}[.)]\s|[•●▪◦*–-]\s|(?i:figure|fig\.|table|tab\.|algorithm|listing)\s*\d+\s*(?:[:.|]|\s[A-Z])"
    r"|[A-Z][a-z]+:\s|(?i:proof|remark|note|example|definition|lemma|theorem|corollary|proposition|claim)\b)")


def _open_end(text: str) -> bool:
    """Ends mid-sentence for sure: on a hyphen, a comma or a word such as "of", "the", "are"."""
    words = text.split()
    return bool(words) and (text.endswith(("-", ",")) or words[-1].lower() in _OPEN_END_WORDS | _FRAGMENT_ENDS)


def _continues(prev: str, para: str, open_end: bool = True) -> tuple[bool, bool]:
    """(hyphenated break, `para` continues `prev`): `prev` ends in a hyphenated word, or does not
    end a sentence and `para` starts in lowercase, or (open_end) `prev` ends on a word no sentence
    ends on ("... as shown in")."""
    hyphen_break = prev.endswith("-") and len(prev) > 1 and prev[-2].isalnum() and para[:1].isalnum()
    if hyphen_break:
        return True, True
    if _SENT_END_RE.search(prev):
        return False, False
    words = prev.split()
    first = para.split()[0].lower().strip(",.:;") if para.split() else ""
    if para.split() and para.split()[0].endswith("ly,"):
        first = "however"  # a sentence adverb: "Typically, when training ..."
    head = [c for c in " ".join(para.split()[:5]) if c.isalpha()]
    title = bool(head) and sum(c.isupper() for c in head) > 0.6 * len(head)  # "VERIFICATION W/ FIXED ENCODER: ..."
    return False, para[:1].islower() or (open_end and bool(words) and words[-1].lower() in _OPEN_END_WORDS
                                          and para[:1].isalnum() and first not in _SENTENCE_STARTS
                                          and not title and not _NEW_UNIT_RE.match(para))


def _merge_paragraphs(paragraphs: list[str], spellings: Counter | None = None,
                      floats: list[bool] | None = None) -> list[str]:
    """Merge paragraphs broken by a column/page break (R8).

    `floats` marks captions. A caption never continues body text, and complete captions set
    between the two halves of a broken paragraph ("... the most significant" / "Table 6: ..." /
    "change in the network ...") stay after the joined paragraph instead of splitting it. A caption
    left open ("... results of the model on", "... inferior to the") is not passed over: the next
    paragraph may continue it. A "Fig. 2. ..." caption in body-size text is not marked as one (a
    column break can leave "Table 3. Overall, ..." at the start of a block of body text); complete
    ones count as captions in a run of captions that a paragraph of text ending mid-sentence comes
    before and a paragraph starting lowercase after, which can only be its other half ("... input
    symbols and" / "Fig. 2. A toy example ..." / "emits a sequence ..."). An unfinished one may be
    that body text ("... presented in" / "Table 3. Overall, ... applied to the video" / "Table 3:
    ..." / "dataset, our method ...") and ends the run.
    """
    floats = list(floats or [False] * len(paragraphs))

    def caption(k: int) -> bool:
        return floats[k] or bool(_CAPTION_RE.match(paragraphs[k]) and _SENT_END_RE.search(paragraphs[k]))

    for i in range(1, len(paragraphs)):
        prev = paragraphs[i - 1]
        if (not caption(i) or caption(i - 1) or _CAPTION_RE.match(prev) or _SENT_END_RE.search(prev)
                or len(re.findall(r"\b[a-z]{2,}\b", prev)) < 3):
            continue  # (text, not figure labels such as "ImageNet-118K-LT [33] OpenImages-118K [82]")
        j = i
        while j < len(paragraphs) and caption(j):
            j += 1
        if j < len(paragraphs) and _SENT_END_RE.search(paragraphs[j - 1]) and paragraphs[j][:1].islower():
            floats[i:j] = [True] * (j - i)
    merged: list[str] = []
    kinds: list[bool] = []
    for para, is_float in zip(paragraphs, floats):
        target = len(merged) - 1
        if not is_float:
            body = target
            while body >= 0 and kinds[body] and not _open_end(merged[body]):
                body -= 1
            if 0 <= body < target and not kinds[body] and _continues(merged[body], para)[1]:
                target = body  # the paragraph resumes after the captions that interrupted it
        if target >= 0 and not (is_float and not kinds[target]):  # a caption does not continue body text
            hyphen_break, cont = _continues(merged[target], para, open_end=not kinds[target] and not is_float)
            if cont:
                merged[target] = (_join_hyphenated(merged[target], para, spellings or Counter()) if hyphen_break
                                  else f"{merged[target]} {para}")
                continue
        merged.append(para)
        kinds.append(is_float)
    return merged


# ---------------------------------------------------------------- public API


def parse_pdf(pdf_path: str | Path, options: ParseOptions | None = None) -> list[dict[str, str]]:
    """Parse a single PDF into section segments.

    Args:
        pdf_path: path to a PDF under raw/pdf.
        options: cleaning thresholds (defaults = ParseOptions()).

    Returns:
        [{"section": "Introduction", "text": "..."}, ...] in document order; paragraphs within a
        section are separated by "\\n". Section "" holds text when no heading was detected.

    Raises:
        PDFParseError: when the file is corrupted/encrypted or no valid text can be extracted.
    """
    opts = options or ParseOptions()
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:  # noqa: BLE001 (pymupdf raises several exception types)
        raise PDFParseError(f"cannot open {pdf_path}: {exc}") from exc
    try:
        if doc.needs_pass:
            raise PDFParseError(f"{pdf_path} is encrypted")
        lines, heights = _extract_lines(doc)
    except PDFParseError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PDFParseError(f"text extraction failed for {pdf_path}: {exc}") from exc
    finally:
        doc.close()
    if not lines:
        raise PDFParseError(f"{pdf_path} has no extractable text (image-only PDF?)")

    lines = _attach_drop_caps(_merge_split_numbers(lines))
    for ln in lines:
        ln.text = _repair(ln.text).strip()
    lines = [ln for ln in lines if ln.text]
    toc = _toc_pages(lines)
    lines = [ln for ln in lines if ln.page not in toc]  # contents entries are not headings
    spellings = _spellings([ln.text for ln in lines])
    # Body font size from the paper itself: long supplementary material after the reference list
    # (tables, a different template) can outweigh the main text and make it look like footnotes.
    ref_titles = [k for k, ln in enumerate(lines)
                  if _is_references_line(ln.text) and _bibliography_follows(lines, k + 1)]
    ref_at = ref_titles[0] if ref_titles else None
    main = lines[:ref_at] if ref_at is not None and ref_at >= 0.2 * len(lines) else lines
    body_size = _body_font_size(main)
    hf_keys = _header_footer_keys(lines, heights, opts)  # on repaired text, as looked up below
    drop = {s.lower() for s in opts.drop_sections}
    # The paper numbers its sections if >= 3 lines of its main text (before the reference list, so
    # numbered member lists or appendices do not count) qualify as numbered headings.
    numbered_mode = sum(
        1 for k, ln in enumerate(main) if _NUMBERED_RE.match(ln.text) and _heading_name(ln, body_size, opts)
        and not _list_item(ln, main[max(0, k - 10):k][::-1], main[k + 1:k + 11], body_size, opts)
    ) >= 3
    letter_heads = _letter_headings(lines)
    # IEEE numbering: "II. Related Work" sections with "A. Datasets" subsections set in body style.
    # Elsewhere a body-style "A. ..." line is a lettered list option, not a heading.
    # (the Roman headings themselves may be set small: "I. INTRODUCTION" at 8pt in a 10pt paper)
    ieee = len({ln.text.split()[0] for ln in main if len(ln.text.split()) <= 12
                and re.match(r"(?:I|II|III|IV|V|VI)\.\s+[A-Z]", ln.text)}) >= 2
    casings = _paper_casings(main)
    # Books, theses and lecture notes end every chapter with its own reference list: with >= 3
    # validated lists, each list is skipped up to the next heading instead of ending the paper.
    chapter_bibs = _reference_lists(lines, ref_titles) >= 3
    in_bibliography = False

    # Pass 1: filter noise lines, mark headings (with wrapped continuations), cut at references.
    kept: list[_Line] = []
    caption_blocks: set[int] = set()
    parents: dict[int, str] = {}  # heading level -> title, for "Parent / Child" section names
    roman_sections = False  # IEEE numbering seen: "A." headings are subsections of the Roman ones
    roman_current = 0  # value of the current Roman section number
    letter_level = 0  # level of the current lettered run-in heading ("A. Zachary's karate club. The ...")
    i = 0
    while i < len(lines):
        ln = lines[i]
        i += 1
        text = ln.text
        in_margin = _in_margin(ln, heights[ln.page], opts)
        hf_size = hf_keys.get(re.sub(r"\d+", "#", " ".join(text.split()).lower())) if in_margin else None
        if in_margin and (_PAGE_NUM_RE.match(text) or (hf_size is not None and abs(ln.size - hf_size) < 1.0)):
            continue  # R2
        if _ARXIV_STAMP_RE.search(text):
            continue  # R2
        if _URL_LINE_RE.match(text) or _DOI_LINE_RE.match(text):
            continue  # R4
        if _is_references_line(text) and _bibliography_follows(lines, i):
            if not chapter_bibs:
                break  # R1: title in any style (small bold ACM, inside a block, "References:")
            in_bibliography = True
            continue
        if not chapter_bibs and i > len(lines) * 0.15 and _headingless_bibliography(lines, i - 1):
            break  # R1: reference list without a title ("[1] K. Simonyan and ...")
        found = _heading_name(ln, body_size, opts, numbered_mode, letter=id(ln) in letter_heads)
        styled = ln.bold or ln.size >= body_size + opts.heading_size_delta
        if found is not None and not styled and not ieee and re.match(r"[A-H]\.\s", text):
            found = None  # a lettered option in body text ("A. Absolutely must be false"), not IEEE "A. Datasets"
        if (found is None and ieee and ln.italic and re.match(r"[A-H]\.\s+[A-Z]", text)
                and len(text.split()) <= _MAX_HEADING_WORDS and not text.endswith(".")
                and _alpha_ratio(text) >= 0.7 and ln.size >= body_size - 0.3):
            found = (_normalize_heading(text.split(None, 1)[1]), 2)  # IEEE italic subsection sharing its block
        if in_bibliography:
            prose = [x.text for x in lines[i - 1:i + 5] if len(x.text.split()) >= 6]
            if found is None and not (len(prose) >= 4 and not any(_bib_line(t) for t in prose[:4])):
                continue  # a chapter's reference list, up to the next heading or until prose resumes
            if found is not None and _REFERENCES_RE.match(found[0].lower()):
                continue
            in_bibliography = False
        if found is not None and _list_item(ln, lines[max(0, i - 11):i - 1][::-1], lines[i:i + 10], body_size, opts):
            found = None
        run_in = None
        if found is None and ln.bold_prefix:
            run_in = _run_in_heading(ln, body_size)
        if found is None and run_in is None:
            run_in = _numbered_run_in(ln, body_size, opts, numbered_mode)
        if found is None and run_in is None and not numbered_mode:
            custom = _unnumbered_heading(ln, lines[i:i + 7], body_size, opts)
            if custom and (parents or custom.lower() in KNOWN_HEADINGS):  # not front matter
                found = (custom, 1)
        if run_in is not None:
            name, level, rest = run_in
            if not roman_sections and re.fullmatch(r"[A-H]\.", ln.text.split()[0]):
                # a lettered run-in paragraph heading sits below the current section
                level = letter_level if letter_level in parents else max(parents, default=0) + 1
                letter_level = level
            found = (name, level)
            if rest:  # the rest of the line is body text: re-queue it as its own line
                lines.insert(i, _Line(rest, ln.size, False, ln.page, ln.y0, ln.y1, ln.x0, ln.block,
                                      "", ln.pos_in_block, ln.block_lines))
        if found is not None:
            name, level = found
            number = ln.text.split()[0]
            roman = _roman_value(number[:-1]) if re.fullmatch(r"[IVX]{1,5}\.", number) else 0
            if roman and roman_sections and roman <= roman_current:
                level = 2  # subsection "I." after "H." under section IV, not section I
            elif roman and (roman_sections or roman <= 3):
                roman_sections, roman_current = True, roman
            elif roman_sections and re.fullmatch(r"[A-H]\.", number):
                level = 2
            raw = [ln.text]
            sup_tail = ln.sup_tail
            while i < len(lines) and _continues_heading(ln, lines[i], body_size, opts):
                cont = lines[i].text.strip()
                limit = (20 if ln.bold and lines[i].bold and _NUMBERED_RE.match(" ".join(ln.text.split()))
                         and not cont.endswith(",") else _MAX_HEADING_WORDS)  # a bold 16-word "7.3 ..." on two lines
                if len(name.split()) + len(cont.split()) > limit:
                    break  # a heading does not run on for sentences: the rest is body text
                raw.append(cont)
                sup_tail = lines[i].sup_tail
                if _mostly_caps(ln.text) and _mostly_caps(cont):  # "III. MODELING AND ANALYSIS OF" + "DNN ..."
                    cont = " ".join(_title_case(cont.split(), at_start=name.endswith(":")))
                name = name[:-1] + cont if name.endswith("-") else f"{name} {cont}"
                i += 1
            if name.endswith("-") and i < len(lines) and lines[i].block == ln.block:
                # hyphenated heading whose remainder starts the next line ("Minimax Prob-" / "lems. In ...")
                head, _, rest = lines[i].text.strip().partition(" ")
                name = name[:-1] + head.rstrip(".:")
                lines[i].text = rest
                if not rest:
                    i += 1
            name = _restore_casing(_normalize_heading(name), " ".join(raw), casings)
            if sup_tail and name.endswith(sup_tail) and name[:-len(sup_tail)].strip():
                name = name[:-len(sup_tail)].rstrip()  # footnote mark ("Comparison to Mayr et al." + "12")
            titled = _APPENDIX_RE.match(name)
            if titled and len(raw) > 1 and titled.group("title"):
                name = titled.group("title")  # "APPENDIX A" + "EXPRESSIONS FOR THE GRADIENTS"
            if "@" in name or (not parents and _AFFILIATION_RE.search(name)):
                continue  # an affiliation ("1 Microsoft, India" + "name@microsoft.com"), not a heading
            if not parents and name.lower() not in KNOWN_HEADINGS and any(
                    re.sub(rf"^{_SECNUM}\s+", "", " ".join(x.text.split())).lower().strip(" .:")
                    in ("abstract", "introduction") for x in lines[i:i + 40]):
                continue  # front matter (an address, a numbered author line) right above the Abstract
            if _REFERENCES_RE.match(name.lower()):
                if _bibliography_follows(lines, i) and not chapter_bibs:
                    break  # R1
                in_bibliography = chapter_bibs and _bibliography_follows(lines, i)
                continue  # (or a table column header named "Reference(s)", not the reference list)
            parents = {lv: t for lv, t in parents.items() if lv < level}
            parents[level] = name
            # a dropped parent ("Acknowledgments") or the Abstract does not name the sections that follow it
            named = [parents[lv] for lv in sorted(parents)
                     if ((parents[lv].lower() not in drop or (lv > 1 and parents[lv].lower() in _CONTENT_LIKE_STATEMENTS))
                         and parents[lv].lower() != "abstract") or lv == level]
            ln.heading = " / ".join(named[-2:])
            # R6: a statement is a top-level section; a numbered subsection "6.1.2 Ethical Considerations"
            # in a survey is content (acknowledgments and funding notes are dropped at any level)
            ln.heading_key = name.lower()
            ln.heading_level = level
            kept.append(ln)
            continue
        if _CAPTION_RE.match(text):
            caption_blocks.add(ln.block)
        elif ln.pos_in_block == 0 and i >= 2 and lines[i - 2].block in caption_blocks and lines[i - 2].page == ln.page \
                and abs(lines[i - 2].size - ln.size) < 0.5 and not _SENT_END_RE.search(lines[i - 2].text) \
                and text[:1].islower():
            caption_blocks.add(ln.block)  # the caption runs on in the next block ("... added for each" + "language and ...")
        is_caption = ln.block in caption_blocks
        if is_caption and not opts.keep_captions:
            continue  # R7
        if not is_caption and ln.size < body_size * opts.min_font_ratio:
            continue  # R3 (small font)
        if _alpha_ratio(text) < opts.min_alpha_ratio and not _SENT_END_RE.search(text):
            continue  # R3 (formula/table noise)
        if _is_mojibake(text):
            continue  # R3 (text in a font without a Unicode map)
        kept.append(ln)

    # Pass 2: group into sections -> paragraphs (one paragraph per pymupdf block).
    sections: list[tuple[str, list[str]]] = []
    current_name: str | None = None
    current_key = ""
    current_level = 1
    current_paras: list[str] = []
    current_floats: list[bool] = []
    block_lines: list[str] = []
    block_sizes: list[float] = []
    last_block = -1

    def flush_block() -> None:
        nonlocal block_lines, block_sizes
        para = _URL_RE.sub("", _join_lines(block_lines, spellings)).strip()  # R4 inline URLs
        para = re.sub(r"\s{2,}", " ", para)
        label = (_CAPTION_RE.match(para) or _CAPTION_IN_RE.search(para)) if last_block in caption_blocks else None
        prefix = para[:label.start()].split() if label else []
        if label and prefix and len(prefix) <= 8 and not any(w[:1].islower() for w in prefix):
            para, prefix = para[label.start():], []  # R3: table header cells in the caption's block ("Method NN Table 4. ...")
            label = _CAPTION_IN_RE.match(para)
        # A caption for paragraph merging: "Table 3:" style, or set smaller than body text. A line of
        # body text may start with "Table 3." ("... are given in" / "Table 3. Overall, ...").
        label_size = next((sz for t, sz in zip(block_lines, block_sizes) if _CAPTION_RE.match(t)), body_size)
        is_float = bool(label) and not prefix and (label.group(0).rstrip()[-1] in ":|"
                                                   or label_size < body_size - 0.3)
        if (label is None and last_block in caption_blocks and para[:1].islower()
                and current_floats and current_floats[-1]):
            is_float = True  # the caption runs on in the next block ("... the ConceptNet vector" + "ensemble from ...")
        caps = [c for c in para if c.isalpha()]
        if len(para.split()) <= 15 and caps and sum(c.isupper() for c in caps) > 0.6 * len(caps):
            is_float = True  # an ALL-CAPS table title ("VERIFICATION W/ FIXED ENCODER: TEST SET") in the text flow
        if (len(para.split()) < 4 and not _SENT_END_RE.search(para)) or not re.search(r"[^\W\d_]{2}", para):
            para = ""  # R3 (figure labels / stray fragments; "." or "B.," with no word)
        elif _is_table_block(block_lines):
            para = ""  # R3 (table cells flattened into a column of short lines)
        if para:
            current_paras.append(para)
            current_floats.append(is_float)
        block_lines, block_sizes = [], []

    def flush_section() -> None:
        flush_block()
        if not current_paras:
            return
        paras = _merge_paragraphs(current_paras, spellings, current_floats)
        content = (current_level > 1 and current_key in _CONTENT_LIKE_STATEMENTS
                   and sum(len(p.split()) for p in paras) > 400)  # a survey's "6.1.2 Ethical Considerations"
        if current_key in drop and not content:
            # R6: the statement itself is a paragraph or two (<= 400 words); a long run of text
            # after it belongs to a section whose heading was missed (e.g. an appendix): keep it
            n_drop, words = 1, len(paras[0].split())
            second = len(paras[1].split()) if len(paras) > 1 else 0
            if 0 < second <= 250 and words + second <= 400:
                n_drop = 2
            rest = paras[n_drop:]
            if sum(len(p.split()) for p in rest) >= 300:
                sections.append(("", rest))
            return
        sections.append((current_name or "", paras))

    for ln in kept:
        if ln.heading is not None:
            flush_section()
            current_name, current_key, current_paras, current_floats = ln.heading, ln.heading_key, [], []
            current_level = ln.heading_level
            last_block = -1
            continue
        if ln.block != last_block:
            flush_block()
            last_block = ln.block
        block_lines.append(ln.text)
        block_sizes.append(ln.size)
    flush_section()

    # R5: text before the first heading is title/authors/affiliations. Drop it unless it is long
    # (the first real heading was missed), in which case keep it as an unnamed section.
    if len(sections) > 1 and sections[0][0] == "":
        front_words = sum(len(p.split()) for p in sections[0][1])
        if front_words <= opts.front_matter_max_words:
            sections = sections[1:]
    # An "Abstract" running on for pages means the headings after it were missed: the abstract is
    # its first paragraph(s) (>= 100 words, at most 3 paragraphs); the rest is kept, unnamed.
    for k, (name, paras) in enumerate(sections):
        if name.lower() == "abstract" and sum(len(p.split()) for p in paras) > 600 and len(paras) > 1:
            n = 1
            while n < min(3, len(paras)) and sum(len(p.split()) for p in paras[:n]) < 100:
                n += 1
            sections[k:k + 1] = [(name, paras[:n]), ("", paras[n:])]
            break
    out = [{"section": name, "text": "\n".join(paras)} for name, paras in sections if paras]
    if not out:
        raise PDFParseError(f"{pdf_path}: no usable text after cleaning")
    return out


def drop_leading_abstract(sections: list[dict[str, str]], abstract: str) -> list[dict[str, str]]:
    """R5 when the metadata abstract is inserted as its own section: drop the paper's own copy.

    If the first heading was found late, the front matter is kept as a leading unnamed section
    (> `front_matter_max_words`) that starts with the title, authors and the abstract (unlabelled
    or run in: "Abstract—..."). That part is dropped, up to the end of the abstract's last sentence
    and a keywords / index terms line after it. The abstract is the densest run (gaps of <= 25
    words, or <= 150 words where the copy continues further into the abstract, within the first
    400 + 2 x (abstract length) words, at most 1.25 x the abstract's length from its start) of 6-word sequences shared with
    `abstract`, and must hold at least half of the abstract's sequences; otherwise nothing is
    dropped. Later sentences that reuse the abstract's wording (an introduction) stay.
    """
    if not sections or sections[0]["section"] or not abstract:
        return sections
    a = [w for w in (re.sub(r"\W", "", x.lower()) for x in abstract.split()) if w]
    grams = {tuple(a[k:k + 6]) for k in range(len(a) - 5)}
    if len(grams) < 15:
        return sections
    paras = sections[0]["text"].split("\n")
    toks = [(pi, wi, re.sub(r"\W", "", w.lower())) for pi, p in enumerate(paras) for wi, w in enumerate(p.split())]
    toks = [t for t in toks if t[2]][:400 + 2 * len(a)]
    seq = [t[2] for t in toks]
    where = {}
    for k in range(len(a) - 5):
        where.setdefault(tuple(a[k:k + 6]), k)
    hits = [k for k in range(len(seq) - 5) if tuple(seq[k:k + 6]) in grams]
    runs: list[list[int]] = []
    for k in hits:
        if runs and k - runs[-1][-1] <= 25:
            runs[-1].append(k)
        elif (runs and k - runs[-1][-1] <= 150
              and where[tuple(seq[k:k + 6])] >= where[tuple(seq[runs[-1][-1]:runs[-1][-1] + 6])] - 10):
            runs[-1].append(k)  # the copy goes on further into the abstract after an interruption (a figure)
        else:
            runs.append([k])
    best = max(runs, key=lambda r: len({tuple(seq[k:k + 6]) for k in r}), default=[])
    gaps = sum(max(0, b - a_ - 6) for a_, b in zip(best, best[1:]) if b - a_ > 25)
    best = [k for k in best if k <= best[0] + 1.25 * len(a) + 20 + gaps]  # the copy is about as long as the abstract
    if len({tuple(seq[k:k + 6]) for k in best}) < 0.5 * len(grams):
        return sections
    end_pi, end_wi, _ = toks[best[-1] + 5]
    words = paras[end_pi].split()
    while end_wi + 1 < len(words) and not re.search(r"[.!?][\"')\]]?$", words[end_wi]):
        end_wi += 1  # to the end of the sentence
    rest = [" ".join(words[end_wi + 1:]), *paras[end_pi + 1:]]
    rest = [p for p in rest if p.strip()]
    if rest and re.match(r"(?:key ?words|index terms)\b", rest[0], re.I):
        head = rest[0].split()  # the keywords, up to their first full stop (the introduction may follow)
        stop = next((k for k, w in enumerate(head[:40]) if k and w.endswith(".")), len(head) - 1 if len(head) <= 40 else None)
        if stop is not None:
            rest = [" ".join(head[stop + 1:]), *rest[1:]] if head[stop + 1:] else rest[1:]
    return ([{"section": "", "text": "\n".join(rest)}] if rest else []) + sections[1:]


def quality_check(
    pdf_path: str | Path,
    sections: list[dict[str, str]],
    options: ParseOptions | None = None,
) -> dict:
    """Heuristic score for parse quality.

    bad_char_ratio counts characters that indicate broken extraction: private-use glyphs,
    control characters, and runs of isolated single letters ("t h e" from letter-spaced text).

    Returns:
        {"pdf_path", "word_count": int, "bad_char_ratio": float, "num_sections": int,
         "mean_word_len": float, "ok": bool} (ok=False triggers manual review/removal).
    """
    opts = options or ParseOptions()
    text = "\n".join(s["text"] for s in sections)
    words = text.split()
    n_chars = max(1, len(text))
    bad = sum(1 for c in text if unicodedata.category(c) in ("Co", "Cc", "Cs") and c not in "\n\t")
    bad += sum(len(m.group(0)) for m in re.finditer(r"(?:\b[A-Za-z] ){4,}", text))
    mean_len = statistics.fmean(len(w) for w in words) if words else 0.0
    ratio = bad / n_chars
    return {
        "pdf_path": str(pdf_path),
        "word_count": len(words),
        "bad_char_ratio": round(ratio, 4),
        "num_sections": sum(1 for s in sections if s["section"]),
        "mean_word_len": round(mean_len, 2),
        "ok": len(words) >= opts.min_words and ratio <= opts.max_bad_char_ratio and 3.0 <= mean_len <= 9.0,
    }
