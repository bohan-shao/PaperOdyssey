"""PDF full-text extraction and cleaning.

`parse_pdf(pdf_path, options)` reads a PDF with pymupdf (span-level `get_text("dict")`, so
font size/weight are available) and returns `[{"section": str, "text": str}, ...]`.

Cleaning rules (enumerable; all thresholds live in config `kb.parse.*` and are reported):
  R1  Reference truncation: stop at the first "References"/"Bibliography" heading (incl. variants
      such as "Bibliographical References"; or a bare "References" line in any style); everything
      after it (incl. appendices) is discarded.
  R2  Header/footer removal: lines starting/ending in the top/bottom `margin_ratio` of the page whose
      digit-normalized text repeats on >= `header_footer_min_ratio` of pages; bare page
      numbers in the margins; the rotated arXiv sidebar stamp (non-horizontal text) and
      "arXiv:XXXX.XXXXXvN [cs.XX] date" lines.
  R3  Formula/figure/table noise: lines whose letter ratio (letters / non-space chars) is below
      `min_alpha_ratio` and that do not end a sentence; lines set in a font smaller than
      `min_font_ratio` x body size (footnotes, figure labels, small tables); fragments of < 4
      words; table-like blocks (>= 3 lines averaging < 3.5 words, no real sentence, not a caption).
  R4  URLs/DOIs: URL-only and DOI-only lines are dropped; inline URLs are removed.
  R5  Front matter: text before the first detected heading (title/authors/affiliations) is
      dropped when at least one heading exists, unless it exceeds `front_matter_max_words`
      (a missed first heading), in which case it is kept as an unnamed section.
  R6  Dropped sections: headings listed in `drop_sections` (e.g. acknowledgments) and
      their text are skipped.
  R7  Captions: "Figure N:"/"Table N:" blocks are kept or dropped per `keep_captions`.
  R8  Repair: NFKC normalization (the ligature "ﬁ" becomes "fi"), "(cid:N)" glyph garbage removed,
      end-of-line hyphenation joined, paragraphs split across columns/pages merged when the
      first does not end a sentence and the next starts lowercase.

Section-title detection: a line is a heading when it
- is short, does not end with a period and starts with a capital letter;
- is visually emphasized: bold, or font >= body + `heading_size_delta`, or (IEEE/Springer styles)
  a numbered line that is alone in its block or written in ALL CAPS;
- is not set smaller than body text and does not sit in the middle of a multi-line block
  (that would be a bold paragraph);
- is numbered ("3", "3.2", "A.1", "IV.") or is a well-known heading word ("Introduction",
  "Related Work", ...).
SIAM-style run-in headings (bold "2.1. Title." followed by body text on the same line) are split
into heading + body.

Section numbers set on their own line at the same baseline as the title are merged. Numbers are
stripped; subsections are named "Parent / Child" (top two levels) so repeated subsection
titles stay distinguishable; ALL-CAPS/small-caps names are title-cased; headings wrapped onto
a second line are rejoined. In papers with >=3 numbered headings, unnumbered headings are only
accepted for UNNUMBERED_OK names (rejects bold table cells such as "Training").

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
    margin_ratio: float = 0.1
    drop_sections: list[str] = field(default_factory=lambda: [
        "acknowledgments", "acknowledgements", "acknowledgment", "acknowledgement",
        "author contributions", "funding", "ethics statement", "broader impact",
        "reproducibility statement",
    ])
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
}
REFERENCE_HEADINGS = {"references", "bibliography", "reference", "literature cited"}
# Reference-list headings incl. template variants ("Bibliographical References" in LREC).
_REFERENCES_RE = re.compile(
    r"^(?:bibliographical |cited |selected )?references?(?: cited)?$|^bibliography$|^(?:literature|works) cited$"
)
# Headings that stay unnumbered even in papers with numbered sections.
UNNUMBERED_OK = REFERENCE_HEADINGS | {
    "abstract", "acknowledgments", "acknowledgements", "acknowledgment", "acknowledgement",
    "author contributions", "funding", "ethics statement", "broader impact", "broader impacts",
    "reproducibility statement", "impact statement", "limitations", "appendix", "appendices",
    "supplementary material",
}

# Section numbers such as "3", "3.2.", "A.", "A.1", "IV.". A lone letter needs a period, so a title
# like "A Survey of ..." is not mistaken for a numbered heading.
_SECNUM = r"(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[A-H](?:\.\d{1,2}){1,3}\.?|[A-H]\.|[IVX]{1,5}\.)"
_NUMBERED_RE = re.compile(rf"^{_SECNUM}\s+(?P<title>\S.*)$")
_NUMBER_ONLY_RE = re.compile(rf"^{_SECNUM}$")
_ARXIV_STAMP_RE = re.compile(r"arXiv:\s*\d{4}\.\d{4,5}(v\d+)?\s*\[[a-z\-]+(\.[A-Za-z\-]+)?\]", re.I)
_URL_RE = re.compile(r"(https?://|www\.)\S+", re.I)
_DOI_LINE_RE = re.compile(r"^\s*(doi:|https?://doi\.org/)\S+\s*$", re.I)
_URL_LINE_RE = re.compile(r"^\s*((https?://|www\.)\S+\s*)+$", re.I)
_CID_RE = re.compile(r"\(cid:\d+\)")
_PAGE_NUM_RE = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.I)
_CAPTION_RE = re.compile(r"^\s*(figure|fig\.|table|tab\.)\s*[A-Z]?\d+[a-z]?\s*[:.|]", re.I)
_SENT_END_RE = re.compile(r"[.!?:;]['\")\]]?$")


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


def _is_bold(span: dict) -> bool:
    font = span.get("font", "").lower()
    return bool(span.get("flags", 0) & 16) or any(t in font for t in ("bold", "medi", "black", "heavy", "semibold"))


def _extract_lines(doc: pymupdf.Document) -> tuple[list[_Line], list[float]]:
    """Horizontal text lines with style info, in content-stream (reading) order."""
    lines: list[_Line] = []
    heights: list[float] = []
    block_id = 0
    for pno, page in enumerate(doc):
        heights.append(page.rect.height)
        for block in page.get_text("dict")["blocks"]:
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
                x0, y0, _, y1 = line["bbox"]
                lines.append(_Line(text, size, bold, pno, y0, y1, x0, block_id, prefix.strip()))
            for k, ln in enumerate(lines[start:]):
                ln.pos_in_block, ln.block_lines = k, len(lines) - start
    return lines, heights


def _body_font_size(lines: list[_Line]) -> float:
    weights: Counter[float] = Counter()
    for ln in lines:
        weights[round(ln.size, 1)] += len(ln.text)
    return weights.most_common(1)[0][0] if weights else 10.0


def _merge_split_numbers(lines: list[_Line]) -> list[_Line]:
    """Merge a lone section number with the title set at the same baseline ("2" + "Related Work")."""
    out: list[_Line] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if (nxt is not None and _NUMBER_ONLY_RE.match(cur.text.strip()) and cur.page == nxt.page
                and abs(cur.y0 - nxt.y0) < 3 and nxt.x0 > cur.x0):
            out.append(_Line(f"{cur.text.strip()} {nxt.text.strip()}", max(cur.size, nxt.size),
                             cur.bold and nxt.bold, cur.page, cur.y0, nxt.y1, cur.x0, nxt.block,
                             "", nxt.pos_in_block, nxt.block_lines))
            i += 2
            continue
        out.append(cur)
        i += 1
    return out


def _normalize_heading(title: str) -> str:
    """Collapse whitespace; title-case ALL-CAPS/small-caps names (single-word acronyms like BERT stay)."""
    title = " ".join(title.split()).strip(" .:")
    letters = [c for c in title if c.isalpha()]
    all_caps = bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8
    if all_caps and (len(title.split()) > 1 or title.lower() in KNOWN_HEADINGS):
        title = " ".join(w[:1].upper() + w[1:].lower() for w in title.split())
    return title


def _heading_name(
    ln: _Line, body_size: float, opts: ParseOptions, numbered_mode: bool = False,
) -> tuple[str, int] | None:
    """Return (normalized section name, level) if the line is a section heading, else None.

    level = 1 + number of dots in the section number ("3" -> 1, "3.2" -> 2); unnumbered -> 1.
    In numbered_mode (the paper numbers its sections), unnumbered headings are accepted only
    for UNNUMBERED_OK names. This rejects bold table cells such as "Training".
    """
    text = " ".join(ln.text.split())
    if not text or len(text) > 100 or len(text.split()) > 12:
        return None
    if text.endswith(".") and not _NUMBER_ONLY_RE.match(text.split()[0]):
        return None
    if ln.size < body_size - 0.3:
        return None  # table cells / captions / footnotes are set smaller than body text
    if ln.block_lines > 4 and ln.pos_in_block > 1:
        return None  # inside a (bold) paragraph, not a heading
    m = _NUMBERED_RE.match(text)
    title = m.group("title") if m else text
    words = re.findall(r"[A-Za-z]{2,}", title)
    if not words or sum(c.isalpha() for c in title) / max(1, len(title.replace(" ", ""))) < 0.7:
        return None
    if not next(c for c in title if c.isalpha()).isupper():
        return None  # titles start with a capital: rejects numbered pseudo-code ("7 else") and math lines
    name = _normalize_heading(title)
    key = name.lower()
    if _REFERENCES_RE.match(key) and len(text.split()) <= 3:
        return name, 1  # bare "References" line, any style (R1)
    letters = [c for c in title if c.isalpha()]
    all_caps = len(letters) >= 4 and all(c.isupper() for c in letters)
    # IEEE/Springer-style headings ("II. RELATED WORK", "A. Datasets", "1 INTRODUCTION") are set at
    # body size without bold; a numbered line alone in its block, or numbered ALL CAPS, counts as emphasized.
    emphasized = (ln.bold or ln.size >= body_size + opts.heading_size_delta
                  or (m is not None and (ln.block_lines == 1 or all_caps)))
    if not emphasized:
        return None
    if m:
        return name, text.split()[0].rstrip(".").count(".") + 1
    if key in (UNNUMBERED_OK if numbered_mode else KNOWN_HEADINGS):
        return name, 1
    return None


def _continues_heading(head: _Line, nxt: _Line, body_size: float, opts: ParseOptions) -> bool:
    """A heading wrapped onto a second line: same block and style, not itself numbered.

    Headings set in body style (IEEE ALL-CAPS, not bold) share their block with the paragraph
    that follows, so for them only an ALL-CAPS next line counts as a continuation.
    """
    text = nxt.text.strip()
    if (nxt.block != head.block or nxt.bold != head.bold or abs(nxt.size - head.size) >= 0.5
            or _NUMBERED_RE.match(text) or len(text.split()) > 8 or _SENT_END_RE.search(text)):
        return False
    if head.bold or head.size >= body_size + opts.heading_size_delta:
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


def _header_footer_keys(lines: list[_Line], heights: list[float], opts: ParseOptions) -> set[str]:
    """Digit-normalized margin lines repeated on >= header_footer_min_ratio of pages.

    The default ratio (0.3) also catches running headers that alternate between odd and even
    pages (title / authors), which each appear on ~half the pages.
    """
    n_pages = len(heights)
    if n_pages < 3:
        return set()
    pages_with: dict[str, set[int]] = {}
    for ln in lines:
        if _in_margin(ln, heights[ln.page], opts):
            key = re.sub(r"\d+", "#", " ".join(ln.text.split()).lower())
            pages_with.setdefault(key, set()).add(ln.page)
    threshold = max(3, opts.header_footer_min_ratio * n_pages)
    return {k for k, pages in pages_with.items() if len(pages) >= threshold}


def _alpha_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return sum(c.isalpha() for c in chars) / len(chars) if chars else 0.0


def _repair(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CID_RE.sub("", text)
    return text.replace("\u00ad", "").replace("\ufffd", "")  # soft hyphen, replacement char


def _join_lines(lines: list[str]) -> str:
    """Join a block's lines, repairing end-of-line hyphenation (R8)."""
    out = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and line[:1].islower():
            out = out[:-1] + line
        elif out:
            out += " " + line
        else:
            out = line
    return out


def _is_table_block(lines: list[str]) -> bool:
    """A block of >= 3 short lines (< 3.5 words on average) that are not sentences = table column.

    Captions and blocks containing a real sentence (>= 8 words ending in . ! ?) are prose set in a
    narrow column, not tables, and are kept.
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
    return not any(len(piece.split()) >= 8 and piece.rstrip()[-1:] in ".!?"
                   for piece in re.split(r"(?<=[.!?])\s+", joined))


def _merge_paragraphs(paragraphs: list[str]) -> list[str]:
    """Merge paragraphs broken by a column/page break (R8)."""
    merged: list[str] = []
    for para in paragraphs:
        if merged and not _SENT_END_RE.search(merged[-1]) and para[:1].islower():
            prev = merged[-1]
            merged[-1] = prev[:-1] + para if prev.endswith("-") else f"{prev} {para}"
        else:
            merged.append(para)
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

    body_size = _body_font_size(lines)
    lines = _merge_split_numbers(lines)
    for ln in lines:
        ln.text = _repair(ln.text).strip()
    lines = [ln for ln in lines if ln.text]
    hf_keys = _header_footer_keys(lines, heights, opts)  # on repaired text, as looked up below
    drop = {s.lower() for s in opts.drop_sections}
    # The paper numbers its sections if >= 3 lines qualify as numbered headings.
    numbered_mode = sum(
        1 for ln in lines if _NUMBERED_RE.match(ln.text) and _heading_name(ln, body_size, opts)
    ) >= 3

    # Pass 1: filter noise lines, mark headings (with wrapped continuations), cut at references.
    kept: list[_Line] = []
    caption_blocks: set[int] = set()
    parents: dict[int, str] = {}  # heading level -> title, for "Parent / Child" section names
    i = 0
    while i < len(lines):
        ln = lines[i]
        i += 1
        text = ln.text
        in_margin = _in_margin(ln, heights[ln.page], opts)
        if in_margin and (_PAGE_NUM_RE.match(text) or re.sub(r"\d+", "#", " ".join(text.split()).lower()) in hf_keys):
            continue  # R2
        if _ARXIV_STAMP_RE.search(text):
            continue  # R2
        if _URL_LINE_RE.match(text) or _DOI_LINE_RE.match(text):
            continue  # R4
        found = _heading_name(ln, body_size, opts, numbered_mode)
        if found is None and ln.bold_prefix and (run_in := _run_in_heading(ln, body_size)):
            name, level, rest = run_in
            found = (name, level)
            if rest:  # the rest of the line is body text: re-queue it as its own line
                lines.insert(i, _Line(rest, ln.size, False, ln.page, ln.y0, ln.y1, ln.x0, ln.block,
                                      "", ln.pos_in_block, ln.block_lines))
        if found is not None:
            name, level = found
            while i < len(lines) and _continues_heading(ln, lines[i], body_size, opts):
                cont = lines[i].text.strip()
                name = name[:-1] + cont if name.endswith("-") else f"{name} {cont}"
                i += 1
            if name.endswith("-") and i < len(lines) and lines[i].block == ln.block:
                # hyphenated heading whose remainder starts the next line ("Minimax Prob-" / "lems. In ...")
                head, _, rest = lines[i].text.strip().partition(" ")
                name = name[:-1] + head.rstrip(".:")
                lines[i].text = rest
                if not rest:
                    i += 1
            name = _normalize_heading(name)
            if _REFERENCES_RE.match(name.lower()):
                break  # R1
            parents = {lv: t for lv, t in parents.items() if lv < level}
            parents[level] = name
            ln.heading = " / ".join(parents[lv] for lv in sorted(parents)[-2:])
            ln.heading_key = name.lower()
            kept.append(ln)
            continue
        if _CAPTION_RE.match(text):
            caption_blocks.add(ln.block)
        is_caption = ln.block in caption_blocks
        if is_caption and not opts.keep_captions:
            continue  # R7
        if not is_caption and ln.size < body_size * opts.min_font_ratio:
            continue  # R3 (small font)
        if _alpha_ratio(text) < opts.min_alpha_ratio and not _SENT_END_RE.search(text):
            continue  # R3 (formula/table noise)
        kept.append(ln)

    # Pass 2: group into sections -> paragraphs (one paragraph per pymupdf block).
    sections: list[tuple[str, list[str]]] = []
    current_name: str | None = None
    current_key = ""
    current_paras: list[str] = []
    block_lines: list[str] = []
    last_block = -1

    def flush_block() -> None:
        nonlocal block_lines
        para = _URL_RE.sub("", _join_lines(block_lines)).strip()  # R4 inline URLs
        para = re.sub(r"\s{2,}", " ", para)
        if len(para.split()) < 4 and not _SENT_END_RE.search(para):
            para = ""  # R3 (figure labels / stray fragments)
        elif _is_table_block(block_lines):
            para = ""  # R3 (table cells flattened into a column of short lines)
        if para:
            current_paras.append(para)
        block_lines = []

    def flush_section() -> None:
        flush_block()
        if current_paras and current_key not in drop:
            sections.append((current_name or "", _merge_paragraphs(current_paras)))

    for ln in kept:
        if ln.heading is not None:
            flush_section()
            current_name, current_key, current_paras = ln.heading, ln.heading_key, []
            last_block = -1
            continue
        if current_key in drop:
            continue  # R6
        if ln.block != last_block:
            flush_block()
            last_block = ln.block
        block_lines.append(ln.text)
    flush_section()

    # R5: text before the first heading is title/authors/affiliations. Drop it unless it is long
    # (the first real heading was missed), in which case keep it as an unnamed section.
    if len(sections) > 1 and sections[0][0] == "":
        front_words = sum(len(p.split()) for p in sections[0][1])
        if front_words <= opts.front_matter_max_words:
            sections = sections[1:]
    out = [{"section": name, "text": "\n".join(paras)} for name, paras in sections if paras]
    if not out:
        raise PDFParseError(f"{pdf_path}: no usable text after cleaning")
    return out


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
