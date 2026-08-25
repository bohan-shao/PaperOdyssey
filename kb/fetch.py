"""arXiv metadata fetching, OpenAlex citation filtering, and PDF download. TODO.

Implementation requirements:
- Pull metadata from the arXiv API (http://export.arxiv.org/api/query) with paging by
  category and year, 100 records per page, request interval ≥3s (respect arXiv's terms
  of use and rate limits), with tenacity retries.
- Match DOI/arxiv_id against OpenAlex (https://api.openalex.org) to fill in
  citation_count; use the batch endpoint to reduce the number of calls; when the citation
  count is missing, fall back to the config default and log it.
- Filtering rules (config.yaml `kb.filter.*`): year range, category set, citation-count
  lower bound, and target number of papers (15k–30k).
- PDF download: concurrency ≤ the config-allowed value, resumable download (skip if the
  file already exists with a reasonable size), and record failures to
  `data/raw/download_failures.jsonl` (rerunnable to resume).
- Output `data/raw/metadata.csv`:
  paper_id, arxiv_id, title, abstract, year, categories, citation_count, pdf_path

Constraints: access public data only; respect arXiv/OpenAlex terms; do not hardcode paths.
"""

from __future__ import annotations

import pandas as pd


def fetch_arxiv_metadata(
    categories: list[str], year_min: int, year_max: int, max_papers: int
) -> pd.DataFrame:
    """Fetch arXiv metadata.

    Args:
        categories: e.g. ["cs.CL", "cs.AI", "cs.LG"].
        year_min/year_max: year range (inclusive).
        max_papers: metadata cap (before filtering).

    Returns:
        A DataFrame; columns as in the module docstring. Failed rows are dropped and logged.

    Raises:
        RuntimeError: when retries keep failing and no data is available at all.
    """
    raise NotImplementedError  # TODO: implement; see kb/README.md
