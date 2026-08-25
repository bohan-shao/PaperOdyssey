"""PDF full-text extraction and cleaning. TODO.

Implementation requirements:
- Extract text from each page with pymupdf; concatenate page by page and keep a
  page-number mapping (for debugging).
- Cleaning rules (enumerable and configurable; needed for the report):
  1) Truncate the reference section: detect the page where "References" starts and
     discard all body text after it;
  2) Remove headers/footers (arXiv template lines, journal names, page-number lines);
  3) Remove pure-formula noise lines (lines with a high ratio of non-letter characters
     and no period; threshold configurable);
  4) Remove URL and DOI lines;
  5) Merge lines/paragraphs split across page breaks (optional hyphenation repair).
- Section-title detection: match known subsection patterns at line starts
  ("\\d+\\.\\s+[A-Z]" or a list of common heading words) and emit
  `(section_title, text)` segments for the chunk module.
- QA: `quality_check(pdf_path, text)` returns a heuristic score (word count,
  bad-character ratio, number of sections) for sampled manual review.

Output format (internal to the parse stage): list[dict] = {"section": str, "text": str}
"""

from __future__ import annotations

from pathlib import Path


def parse_pdf(pdf_path: str | Path) -> list[dict[str, str]]:
    """Parse a single PDF into section segments.

    Args:
        pdf_path: path to a PDF under raw/pdf.

    Returns:
        [{"section": "Introduction", "text": "..."}, ...];
        on parse failure, raises PDFParseError (the caller skips and logs it).

    Raises:
        PDFParseError: when the file is corrupted or no valid text can be extracted.
    """
    raise NotImplementedError  # TODO: implement; see kb/README.md


def quality_check(pdf_path: str | Path, sections: list[dict[str, str]]) -> dict:
    """Heuristic score for parse quality.

    Returns:
        {"word_count": int, "bad_char_ratio": float, "num_sections": int,
         "ok": bool} (ok=False triggers manual review/removal).
    """
    raise NotImplementedError  # TODO: implement; see kb/README.md
