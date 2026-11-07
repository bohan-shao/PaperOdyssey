# data/ Data directory

> Data, indices and results are excluded by .gitignore. The knowledge base (`raw/`, `processed/` and its
> `results/` statistics) is in the public Google Drive folder
> [PaperOdyssey-KB-merged](https://drive.google.com/drive/folders/1p2TmBcvMgRn88Bar-sGEIUH4VuNiFxY7)
> (download steps in the root [README](../README.md#3-download-the-knowledge-base)); indices are built locally
> with `scripts/build_indexes.py`; evaluation results are shared as the results archive linked from the report.
> The reviewed `eval/eval_set.jsonl` is the sole dataset exception, alongside directory documentation and
> placeholders, so every evaluation uses the same version.

| Subdirectory | Contents | Producer |
|---|---|---|
| `raw/` | downloaded metadata (metadata.csv), PDFs (pdf/), API caches (arxiv_slices/, openalex_citations.jsonl), download/parse failure lists, logs | kb |
| `processed/` | `passages.jsonl`, `sections.jsonl` (parent sections), `papers.sqlite` (SCHEMA.md §2-§3) | kb |
| `indices/` | `bm25/` (Pyserini), `faiss/` (FAISS + mapping files), `lookup.sqlite` (passage/section records by id) | retrieval |
| `eval/` | `eval_set.jsonl`, `annotation/` (annotation intermediate artifacts) | eval |
| `results/` | `retrieval_results/`, `final_results/`, `metrics/*.json` | everyone |

## Fixed evaluation set

`eval/eval_set.jsonl` contains the round-1 set built with `scripts/eval_set.py build` (D39/D42):
1,027 accepted questions, split into 912 eval and 115 dev. Each question has one gold source paper.
The pilot contributes 49 questions marked by 3 people; the main part contributes 978 marked by 2.
The 66 ties and 7 rejected questions were excluded; no tie-break or rewrite round was run.

SHA-256: `83418cc4cfdf25483b77b8bb843a033958c71d7d265f8b14da537819e45471b6`.

Changes to this dataset require review and an updated hash/count description. Record the dataset hash
with evaluation results. Raw annotation sheets and IAA reports remain shared artifacts, not git data.

Conventions:
- Large files (PDFs/indexes) never go into git; share them on Google Drive or Dropbox
- Result files named with a date suffix do not overwrite older versions (`metrics/retrieval_2026-09-30.json`)
- Every number cited in the report must be traceable to a file under `results/`
