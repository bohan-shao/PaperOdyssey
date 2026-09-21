# Data Schema (Cross-Module Contract)

> **This document is the single source of truth for cross-module data formats.** Any field change requires a PR that updates this document,
> together with the type definitions in `retrieval/types.py` and `orchestration/types.py`.
> All files are **UTF-8, JSON/JSONL** (except SQLite and index binaries).

## 1. File Layout (`data/`)

```
data/
├── raw/          # downloaded metadata, PDFs (not committed)
├── processed/    # passages.jsonl, sections.jsonl, papers.sqlite (KB outputs)
├── indices/      # bm25/ (Pyserini), faiss/ (FAISS index + vectors)
├── eval/         # eval_set.jsonl (evaluation set), annotation/ (annotation intermediates)
└── results/      # per-stage results: retrieval_results/, final_results/, metrics/*.json
```

## 2. Passages (`data/processed/passages.jsonl`, one JSON per line)

| Field | Type | Description |
|---|---|---|
| `passage_id` | str | Globally unique; format `P{paper_id}_{chunk_index:04d}` |
| `paper_id` | str | Unique paper ID (internal: arxiv_id without version) |
| `arxiv_id` | str | Original arXiv ID (with version; used for display links) |
| `title` | str | Paper title |
| `section` | str | Section title the chunk belongs to (`""` if there is no section) |
| `year` | int | Year |
| `categories` | list[str] | Categories, e.g. `["cs.CL"]` |
| `chunk_index` | int | Chunk index within the paper |
| `text` | str | Chunk text. The prefix `[{title}] {section}` is **embedded as the first line** (`"[Title] Section\n<body>"`), see kb/README.md |
| `parent_section_id` | str | Parent unit = the whole parsed section, format `{paper_id}_S{section_index:03d}`; always present (lets the parent-document ablation run without rebuilding) |

`section` is the parsed heading without its number; subsections are `"Parent / Child"` (top two levels).

**Parent-document retrieval** (Q3 ablation, D22): parent texts live in
`data/processed/sections.jsonl`, one record per parent:
`{parent_section_id, paper_id, arxiv_id, title, section, year, categories, section_index, text}`
(`text` carries the same prefix line). Read via `kb.store.iter_parents()` / `kb.store.load_parents(ids)`.

## 3. Paper Metadata (`data/processed/papers.sqlite`)

Table `papers`:

| Column | Type | Description |
|---|---|---|
| `paper_id` | TEXT PK | |
| `arxiv_id` | TEXT | Latest arxiv ID |
| `title` | TEXT | |
| `abstract` | TEXT | |
| `year` | INTEGER | |
| `categories` | TEXT | JSON array string |
| `citation_count` | INTEGER | OpenAlex citation count (filter criterion) |
| `pdf_path` | TEXT | Path relative to raw/ |
| `num_chunks` | INTEGER | Number of chunks |
| `source` | TEXT | `arxiv` / `s2orc` |

## 4. Query Plan (produced by the orchestration layer; in-memory object + JSON log)

`QueryPlan`:

```json
{
  "input_text": "the user's input paragraph",
  "sub_queries": [
    {"query": "natural-language sub-query 1", "keywords": ["kw1", "kw2"], "filters": {"year_min": 2023, "categories": ["cs.CL"]}}
  ],
  "criteria": [
    {"criterion": "must be a LoRA variant method", "source": "extracted"},
    {"criterion": "published after 2023", "source": "fixed_recency"}
  ]
}
```

## 5. Review Record

`ReviewRecord`:

```json
{
  "paper_id": "P1234",
  "verdict": "pass",              // "pass" | "reject"
  "score": 4,                     // 1–5
  "reasons": "covers LoRA variants but lacks comparison experiments",
  "covered": ["efficient fine-tuning methods"],
  "missing": ["comparison experiment conclusions"],
  "round": 1,
  "evidence": {"title": "...", "abstract_snippet": "...", "top_passages": ["P1234_0007"]}
}
```

## 6. Final Result — Paper Search output (written to `data/results/final_results/`)

`FinalResult` is the **Paper Search** output (`orchestration.run_query()`, D24). It contains no
generated answer; the downstream task (`downstream/`) consumes it and produces its own
`DownstreamResult` (schema TBD with the task type — see `downstream/types.py`):

```json
{
  "query_id": "Q0001",
  "input_text": "...",
  "query_plan": { ... see §4 ... },
  "papers": [ {"paper_id": "...", "score": 4, "verdict": "pass", "title": "...", "arxiv_id": "...", "year": 2024, "url": "https://arxiv.org/abs/..."} ],
  "coverage_report": {"total_aspects": 3, "covered_aspects": 3, "rounds": 2, "papers_reviewed": 12},
  "latency_by_stage": {"understanding_ms": 1500, "retrieval_ms": 420, "review_ms": 3800, "rewrite_ms": 600, "total_ms": 6500}
}
```

## 7. Evaluation Set (`data/eval/eval_set.jsonl`; assignment Q4 requires ≥1,000 items, 3 annotators, IAA ≥ 80%)

```json
{
  "query_id": "Q0001",
  "input_text": "the user's input (a paragraph)",
  "gold_paper_ids": ["P1234", "P5678"],
  "gold_passage_ids": ["P1234_0007"],     // annotated only for a subset (100–200 items); [] otherwise
  "gold_answer": "reference answer (for generation evaluation)",
  "annotations": [
    {"annotator": "A", "papers": ["P1234"], "answer": "..."},
    {"annotator": "B", "papers": ["P1234", "P5678"], "answer": "..."}
  ],
  "majority_paper_ids": ["P1234"],
  "split": "eval"                          // "dev" (for tuning, ~10%) | "eval"
}
```

Constraints:
- **Deduplication** (keep only one of similar inputs), **balance** (query types: definitional / method-comparison / survey / paper-locating / technical-detail, as balanced as possible).
- Annotation guidelines: `eval/annotation/guidelines.md`; IAA computation: Fleiss' κ over 3 annotators + percentage agreement (usable only at ≥80%).

## 8. Retrieval/Evaluation Intermediates (`data/results/retrieval_results/`)

```json
{
  "query_id": "Q0001",
  "sub_query": "sub-query 1",
  "stage": "bm25_only",                 // "bm25" | "dense" | "hybrid_rrf" | "hybrid_rerank"
  "top_k": 10,
  "hits": [{"passage_id": "...", "score": 0.83, "rank": 1, "paper_id": "..."}],
  "latency_ms": 35
}
```

## 9. Type Definition Locations

- Passages/papers: read/write defined in `kb/store.py`; type definitions in `retrieval/types.py` (`ScoredPassage`)
- Orchestration layer: `orchestration/types.py` (`QueryPlan`, `ReviewRecord`, `FinalResult`)
- Evaluation set: defined within `eval/`

> ⚠️ When changing a contract: update SCHEMA.md → update the corresponding types.py → run `scripts/sanity_check.py` (schema round-trips) → PR with review.
