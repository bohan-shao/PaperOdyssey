# data/ Data directory (not in git; distributed via Dropbox)

> Everything in this directory is excluded by .gitignore (see the root .gitignore). Use Dropbox for
> collaboration syncing; package it with the final coursework submission.

| Subdirectory | Contents | Producer |
|---|---|---|
| `raw/` | downloaded metadata (metadata.csv), PDFs (pdf/), API caches (arxiv_slices/, openalex_citations.jsonl), download/parse failure lists, logs | kb |
| `processed/` | `passages.jsonl`, `sections.jsonl` (parent sections), `papers.sqlite` (SCHEMA.md §2-§3) | kb |
| `indices/` | `bm25/` (Pyserini), `faiss/` (FAISS + mapping files), `lookup.sqlite` (passage/section records by id) | retrieval |
| `eval/` | `eval_set.jsonl`, `annotation/` (annotation intermediate artifacts) | eval |
| `results/` | `retrieval_results/`, `final_results/`, `metrics/*.json` | everyone |

Conventions:
- Large files (PDFs/indexes) go only to Dropbox, not git; `*.jsonl`/`*.json` can be synced
- Result files named with a date suffix do not overwrite older versions (`metrics/retrieval_2026-09-30.json`)
- Every number cited in the report must be traceable to a file under `results/`
