# Data Schema (Cross-Module Contract)

> **This document is the single source of truth for cross-module data formats.** Any field change requires a PR that updates this document,
> together with the type definitions in `retrieval/types.py` and `orchestration/types.py`.
> All files are **UTF-8, JSON/JSONL** (except SQLite and index binaries).

## 1. File Layout (`data/`)

```
data/
├── raw/          # downloaded metadata, PDFs (not committed)
├── processed/    # passages.jsonl, sections.jsonl, papers.sqlite (KB outputs)
├── indices/      # bm25/ (Pyserini), faiss/ (FAISS index + vectors), lookup.sqlite (passage/section records by id)
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
    {"query": "natural-language sub-query 1", "keywords": "kw1 kw2", "filters": {"year_min": 2023, "categories": ["cs.CL"]}}
  ],
  "criteria": [
    {"criterion": "must be a LoRA variant method", "source": "extracted"},
    {"criterion": "published after 2023", "source": "extracted"}
  ]
}
```

Fixed review rubric entries are configuration-driven scoring dimensions and are merged into the
review prompt; they are not counted as input coverage aspects. With HyDE enabled, the generated
hypothetical document is appended as one additional `SubQuery` with empty `keywords`.

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
`DownstreamResult` (section 10; see `downstream/types.py`):

```json
{
  "query_id": "Q0001",
  "input_text": "...",
  "query_plan": { ... see §4 ... },
  "papers": [ {"paper_id": "...", "verdict": "pass", "score": 4, "reasons": "...", "covered": ["..."], "missing": [], "round": 1, "evidence": {"title": "...", "top_passages": ["P1234_0007"]}} ],
  "coverage_report": {"total_aspects": 3, "covered_aspects": 3, "covered_criteria": ["..."], "uncovered_aspects": [], "rounds": 2, "papers_reviewed": 12, "papers_passed": 5, "termination_reason": "coverage_met", "cache_hit": false, "template_version": "0.2.0", "review_backend": "llm"},
  "latency_by_stage": {"understanding_ms": 1500, "hyde_ms": 0, "retrieval_ms": 420, "review_ms": 3800, "rewrite_ms": 600, "total_ms": 6500}
}
```

## 7. Evaluation Set (`data/eval/eval_set.jsonl`)

The original documented Q4 target was ≥1,000 items, 3 independent annotators and IAA ≥80%.
The current D42 protocol uses 3 annotators for the pilot and 2 for the main set, with an optional
third for ties; it does not meet the original three-annotator target. See EVALUATION.md §1 for
the actual counts and the remaining check against the coursework instructions.

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
- Annotation guidelines: `eval/annotation/guidelines.md`; IAA computation: Fleiss' κ + percentage agreement, per number of annotators (D42) (usable only at ≥80%).
- Built by `eval/annotation/questions.py` (D39): one gold paper per item, the paper the question was drafted
  from; `annotations` lists the annotators the question went to (D42: 3 in the pilot, 2 after it, plus
  `"tiebreak": true` on a third who broke a tie), and `papers` is `[that paper]` when that annotator accepted the question, else `[]`.
  `gold_passage_ids` is `[]` and `gold_answer` is `""` (not collected). Extra fields: `question_type` (one of the
  five types, or `"other"`; for a rewritten question the type its writer gave), `question_source` (`"llm:<model>"` for a drafted question, `"human"` for a rewritten
  one) `round` (1 = the LLM draft, 2+ = rewritten by a team member and marked again) and `question_form`
  (`"question"` or `"request"`, the form the LLM was asked to write; null for a rewritten one).

## 8. Retrieval/Evaluation Intermediates (`data/results/retrieval_results/`)

```json
{
  "query_id": "Q0001",
  "sub_query": "sub-query 1",
  "stage": "bm25",                      // "bm25" | "dense" | "hybrid_rrf" | "hybrid_rerank"
  "top_k": 10,
  "hits": [{"passage_id": "...", "score": 0.83, "rank": 1, "paper_id": "..."}],
  "latency_ms": 35
}
```

The retrieval stage of `scripts/run_eval.py` (`eval/retrieval_eval.py`) writes one JSONL file per
configuration, `<stage>_<date>.jsonl`, with one record per eval-set question in eval-set order:
`sub_query` is the question's `input_text` (retrieved as it is, no filters), `hits` the returned
passages, `latency_ms` the `retrieve()` total.

## 9. Type Definition Locations

Evaluation rankings sort all supplied hits by `rank`, deduplicate the selected identifier
(paper or passage), and only then apply a positive integer K. At paper level, a paper takes
its best passage's position in the deduplicated ranking. The D41 probe uses these same
Recall/MRR/NDCG functions; its separate `distinct_papers_in_top_10_passages` statistic applies
a passage cutoff. Probe report `format_version: 2` / `metric_version: paper_ranking_v2`
identifies this convention; historical reports retain their original definitions. Retrieval
still returns a passage list, so ten returned passages need not contain ten distinct papers.

Retrieval's existing public signature remains
`retrieve(query: str, filters: RetrievalFilters | None = None, top_k: int | None = None) -> RetrieveResult`.
No fields change in D38/D40/D41. `total_candidates` counts the selected pool before reranking/final
truncation, capped by `fusion.top_k` for both hybrid and single-channel runs, including when
reranking is disabled; it is not the raw recall union size.
`latency_ms` may additionally contain `query_ms` and `selection_ms`; timings remain integer
milliseconds. Selection preserves passage IDs and scores, and assigns contiguous ranks.
With parent-document mode enabled, `ScoredPassage.text` is a child-anchored parent excerpt
bounded by `retrieval.parent_doc.max_chars` (0 = legacy unbounded section). An unlocatable
or oversized child is returned unchanged; ambiguous anchors also preserve the child. Whitespace
differences alone can be matched. Fallbacks emit a warning with `parent_excerpt_outcome` and
`passage_id`; no counters or log metadata are added to `RetrieveResult.latency_ms`.
Downstream QA still reads the original child
record by `passage_id`, as specified in section 10.

- Passages/papers: read/write defined in `kb/store.py`; type definitions in `retrieval/types.py` (`ScoredPassage`)
- Orchestration layer: `orchestration/types.py` (`QueryPlan`, `ReviewRecord`, `FinalResult`)
- Evaluation set: defined within `eval/`
- Downstream QA: `downstream/types.py` (`EvidenceSource`, `DownstreamResult`)

> ⚠️ When changing a contract: update SCHEMA.md → update the corresponding types.py → run `scripts/sanity_check.py` (schema round-trips) → PR with review.

## 10. Cited Paper QA

`downstream.run(input_text: str, query_id: str | None = None) -> DownstreamResult`:

| Field | Type | Meaning |
|---|---|---|
| `query_id` | str | ID returned by Paper Search |
| `input_text` | str | Question, stripped of surrounding whitespace |
| `answer` | str | Natural-language answer with `[n]` citations, or an insufficient-evidence explanation |
| `status` | str | `answered` or `insufficient_evidence`; operational failures raise `DownstreamError` |
| `sources` | list[EvidenceSource] | All excerpts actually supplied to the model, numbered from 1 |
| `cited_source_numbers` | list[int] | Distinct cited numbers in first-appearance order; all must exist in `sources` |
| `search_result` | FinalResult | Original Paper Search result (section 6), unchanged |
| `latency_by_stage` | dict[str, int] | `search_ms`, `evidence_ms`, `generation_ms`, `total_ms`; wall-clock milliseconds |
| `prompt_version` | str | Downstream prompt version, currently `1.0.0` |

`EvidenceSource` contains `number: int`, `paper_id: str`, `passage_id: str`, `title: str`,
`year: int | None`, `section: str`, `url: str | None`, and `text: str`. The URL is constructed
only from a valid arXiv ID; unknown IDs have no link. Text is the exact bounded excerpt fed
to generation, loaded from the stored child passage. `[n]` identifies a passage, not a paper.

Answered results require at least one valid citation. An insufficient-evidence result may
have no citations; any citations present still undergo validation. `answer` stores citations
as individual `[n]`: grouped forms (`[1, 2]`, `[1-3]`, `[01]`) are rewritten, and brackets
whose numbers are not all supplied sources (e.g. the interval `[0, 1]`) are plain text.
Missing individual passages are skipped. A missing/corrupt lookup store is an error, not an
empty search.
No new evidence fields are required on `FinalResult` or `ReviewRecord`.

Passage records are read with `retrieval.lookup.get_lookup_store().get_passages(passage_ids)`
(D28): primary-key lookups in `lookup.sqlite`, the store that also hydrates retrieval hits.
Missing IDs are omitted; an empty selection does not open the store. Results serialize with
`asdict()` and JSON; the UI supports downloading this full payload for later evaluation.
