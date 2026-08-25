"""Download papers: metadata -> filter -> PDFs. TODO.

Usage:
    python scripts/download_papers.py --config config.yaml [--limit N] [--resume]

Pipeline: kb.fetch.fetch_arxiv_metadata -> kb.fetch.select_papers -> kb.fetch.download_pdfs
    -> kb.store.upsert_papers
Constraints: idempotent (resumable); `--limit` is for small-scale trial runs (default: full run).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the arXiv paper corpus")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=0, help="0=full run (use a small value for trial runs)")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    args = parser.parse_args()
    raise NotImplementedError  # TODO: implement, see kb/README.md


if __name__ == "__main__":
    main()
