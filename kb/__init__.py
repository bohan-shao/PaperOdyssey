"""kb package: knowledge base construction (Assignment Q1, 20 points).

Submodules:
- fetch:    arXiv API metadata + OpenAlex citation filtering + PDF download
- parse:    pymupdf full-text extraction and cleaning
- chunk:    section-aware fixed-size chunking (baseline) / semantic chunking / parent-document mapping (ablation)
- store:    passages.jsonl + papers.sqlite read/write
- stats:    corpus statistics (document/chunk/word/type counts), corpus funnel, QA review sheet
- merge:    merging the KB members' shard builds into one KB (D35)
- rechunk:  passages from the parsed sections in sections.jsonl, without the PDFs (D36)

Cross-module contract: see docs/SCHEMA.md §2–§3.
"""
