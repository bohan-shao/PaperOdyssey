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
  "reviews": [ { ... see §5: one pass or reject record per reviewed paper, its latest review ... } ],
  "retrieval_ranks": {"P1234": 1, "P5678": 3},
  "coverage_report": {"total_aspects": 3, "covered_aspects": 3, "covered_criteria": ["..."], "uncovered_aspects": [], "rounds": 2, "papers_reviewed": 12, "papers_passed": 5, "re_reviews": 0, "termination_reason": "coverage_met", "cache_hit": false, "template_version": "0.2.0", "review_backend": "llm"},
  "latency_by_stage": {"understanding_ms": 1500, "hyde_ms": 0, "retrieval_ms": 420, "review_ms": 3800, "rewrite_ms": 600, "total_ms": 6500}
}
```

`papers` contains only passing records, ordered by `orchestration.types.passing_order_key` (D47):
descending score, then ascending `retrieval_ranks`, then paper ID. `retrieval_ranks` maps every
reviewed paper to its best position (1 = first) among the passages of any sub-query's `retrieve()`
result in the run, including sub-queries that returned it after its review; review scores are
1–5 integers, so this keeps the retriever's order among equal scores. Results written before D47
have no ranks (`{}`) and fall back to score then paper ID. `reviews`
contains one `ReviewRecord` per reviewed paper, including rejects, in round order;
the order within a round is not a relevance ranking. Each record retains its round, verdict,
reasons, covered/missing aspects and evidence for review evaluation. A paper reviewed again in a
later round, because that round retrieved passages of it the reviewer had not seen (D53), keeps
only its latest record; `coverage_report.re_reviews` counts these (absent before D53). On a fresh
run, `len(reviews)` equals `coverage_report.papers_reviewed`.

`reviews` defaults to `[]` for existing callers and cached results written before this field
was introduced. An empty trace from an old cache means the review records are unavailable;
it must not be reconstructed from `papers`, which omits rejects. New cache entries preserve
the complete trace. Consumers needing review records must run with `orchestration.loop.cache=false`
or use a result known to contain the complete trace.

## 7. Evaluation Set (`data/eval/eval_set.jsonl`)

The original documented Q4 target was ≥1,000 items, 3 independent annotators and IAA ≥80%.
The current D42 protocol uses 3 annotators for the pilot and 2 for the main set, with an optional
third for ties. The coursework brief allows this ("it is recommended to have 3 annotators, but 2 is
also OK"); EVALUATION.md §1 has the actual counts.

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
| `prompt_version` | str | Downstream prompt version, currently `1.1.0` |

`EvidenceSource` contains `number: int`, `paper_id: str`, `passage_id: str`, `title: str`,
`year: int | None`, `section: str`, `url: str | None`, and `text: str`. The URL is constructed
only from a valid arXiv ID; unknown IDs have no link. Text is the exact bounded excerpt fed
to generation, loaded from the stored child passage. `[n]` identifies a passage, not a paper.

Answered results require at least one valid citation. An insufficient-evidence result may
have no citations; any citations present still undergo validation. `answer` stores citations
as individual `[n]`: grouped forms (`[1, 2]`, `[1-3]`, `[01]`) are rewritten, and brackets
whose numbers are not all supplied sources (e.g. the interval `[0, 1]`) are plain text. So are
a lone `[0]` and brackets attached to a name, i.e. index notation such as `X[0]`, `x^{[0]}` or
`A[i][1]` (D50); `downstream.sub_citations` finds the citations of a stored answer.
Missing individual passages are skipped. A missing/corrupt lookup store is an error, not an
empty search.
No new evidence fields are required on `FinalResult` or `ReviewRecord`.

Passage records are read with `retrieval.lookup.get_lookup_store().get_passages(passage_ids)`
(D28): primary-key lookups in `lookup.sqlite`, the store that also hydrates retrieval hits.
Missing IDs are omitted; an empty selection does not open the store. Results serialize with
`asdict()` and JSON; the UI supports downloading this full payload for later evaluation.

## 11. Review evaluation artifacts (D45)

`paths.results/review_runs/<tag>/` contains evaluation-only artifacts, not online API types:

- `manifest.json`: format identity, dataset SHA-256, selected query IDs/split/limit, effective
  search config with credentials omitted, index metadata, source-code hash, environment and
  creation time. `fingerprint` must match before resuming.
- `attempts/NNNNNN.json`: one attempt, numbered in the order attempts start, with the original
  eval question, `status` (`running`, `complete`, `failed`), `started_at`, `cold_start`, `workers`
  (questions allowed at once in its process, D48; absent in older attempts, meaning 1), and, when
  available, `wall_ms` and `usage` (`calls`, `prompt_tokens`, `completion_tokens`, this attempt's
  calls alone). Completed attempts contain a
  complete §6 `result`, including `reviews`. Failed attempts record query ID/error type.
  A crash can leave `running`; resume retries it while retaining the original attempt.
- `report.json`: `stage=review`, `format_version=1`, run `status`, selected/completed counts,
  failed/interrupted attempt counts, incomplete IDs, `cold_start_query_ids`, `workers_used` (sorted
  worker counts of all attempts), protocol, manifest provenance, observed
  usage and per-split metrics. Metrics include completed queries only. Known-source ranking
  and source passed/rejected/not-reviewed counts do not imply candidate negative labels.
  Coverage is model-reported; undefined ratios are JSON null.
- `metrics/llm_usage.jsonl`: the shared client's best-effort raw usage log, useful for examining
  interrupted attempts. It is not guaranteed to include all billed work.

Completed checkpoints are reused without running or timing the query again. Queries started before
the process finished its first query are flagged as cold starts (with one worker, only the first);
they may still reuse OS/model caches.
A prefix selected with `--limit` is a development smoke test, not an unbiased quality sample.

## 12. Automatic ablation artifacts (D46)

`eval.ablation.run_ablation(eval_set_path, results_dir, tag, *, factors, stage, split="all",
limit=None, resume=False, dry_run=False) -> dict` plans or executes one or two factors. The
CLI passes `paths.results/ablation_runs` as `results_dir`. These are evaluation dictionaries;
online `RetrieveResult`, `FinalResult` and downstream dataclasses are unchanged.

- `manifest.json`: `fingerprint`, `plan`, `code_sha256`, `indexes`, `input_files`, `runtime`,
  `environment` and `created_at`. The fingerprint binds the dataset hash, selected query IDs,
  split/limit, factor order, row overrides/effective redacted configuration, seed, code and
  index/corpus metadata, Python and dependency versions. Large index contents are identified
  by their manifests and file size/mtime metadata, not a new full-content hash.
- `selected_eval.jsonl`: exact selected question records in execution order (split before
  limit), reused by every cell. Resume verifies it against the source selection.
- `rows/<baseline|a|b|ab>/<retrieval|review>/`: cell `manifest.json`, atomic sequential
  `attempts/NNNNNN.json` and `report.json`. Review cells use §11. Retrieval attempts contain
  the exact `question`, matrix-derived `fingerprint`, `status`, `started_at`, `cold_start`,
  and elapsed `wall_ms` when available. Complete attempts add a §8 retrieval `record`,
  `total_candidates` and `latency_ms`; failures add `error` with query ID and exception type.
  Failed/running attempts can be retried; attempts after a completed query are invalid.
  Sequence, question identity, result structure and finite nonnegative timings/usage are
  checked for every saved cell before starting new queries.
- Matrix `report.json`: `stage="ablation"`, `format_version=1`, `status` (`complete`,
  `partial`, `failed`), plan fields, manifest `provenance`, selected/completed cell counts,
  `error`, observed `usage`, `comparisons`, `interaction`, and `updated_at`. Dry-run returns
  only a `planned` report and writes nothing. Rows record `enabled_factors`, `overrides`,
  `effective_config`, and stage-keyed `results` with status, completed count, report path,
  usage and normalized `per_query` observations. The CLI also atomically publishes the matrix
  report at `metrics/ablation_<factor_names_joined_underscore>_<tag>.json`.

Normalized observations contain `first_gold_rank` (positive integer or null), `source_success`
(Hit@10 for retrieval; any accepted source for review), `wall_ms`, `cold_start`, and optional
`source_recall`, `coverage_met`, `rounds`. Review usage sums `calls`, `prompt_tokens` and
`completion_tokens` across all attempts, with derived `total_tokens`; retrieval uses zero.
This is returned API usage, not an invoice, and may omit usage lost during transport/crashes.

`comparisons[stage]` contains single-switch edges with `before`, `after` and `splits.dev/eval`.
Each split has `n_selected`, `n_paired`, paired/unpaired IDs, and per-metric `{n, before,
after, delta}`. Undefined means/deltas are null. It also stores fixed/broken/moved-up/down
counts and IDs, up to five fixed/broken examples, and paired cold-start counts. Review adds
separate `top10_fixed`/`top10_broken` counts, IDs and examples. Example fields are query ID,
input text, gold paper IDs and before/after ranks.

For two factors, `interaction[stage]` uses the same four-way completed subset per split.
Each selected metric records `{n, baseline, A, B, AB, interaction}`, where interaction is
AB−A−B+baseline. This is a descriptive contrast, not a statistical significance test.
Incomplete queries never contribute a zero or an inferred negative label.

## 13. Answer evaluation artifacts (D49)

`eval.answer_eval.run_answer_eval(source_dir, results_dir, tag, *, split="all", limit=None,
resume=False, workers=1) -> dict` answers the questions a finished §11 review run (or each review
row of a §12 ablation) completed, from the stored `result`, with `downstream.answer_from_search`.
The CLI passes `paths.results/answer_runs` as `results_dir`. Online types are unchanged.

- `manifest.json`: `fingerprint`, `source` (kind `review_run` or `ablation`, path, each row's
  search-run fingerprint and dataset SHA-256), `selection` (split, limit, query IDs in source
  order), `searched` (selected IDs each row's search completed), `settings` (`downstream`,
  `eval.answer`, redacted `llm`, dense model), `prompt_versions` (answer, judge), `code_sha256`,
  `environment`, `created_at`. Index manifests and corpus file metadata must equal every source
  row's, because answers read passage text from the lookup store.
- `rows/<row>/answers/<query_id>.json` (row `run` for a review run, `baseline|a|b|ab` for an
  ablation): the eval `question`, `status` (`complete` or `failed`), `workers`, `wall_ms`,
  `usage.generation` / `usage.judge` (`calls`, `prompt_tokens`, `completion_tokens`), and
  `earlier_usage` (same shape) when it retries failed attempts. Complete checkpoints add `answer`
  (§10 `DownstreamResult` without `search_result`), `task` (`gold_in_sources`, `gold_cited`),
  `scores` (`claims`, `claim_verdicts` [{claim, supported, reason}], `faithfulness`,
  `generated_questions`, `noncommittal`, `answer_relevance`, `source_verdicts`
  [{source, relevant, reason}], `context_relevance`, `context_precision`) and `judge_ms`.
  Failed ones add `error` with query ID and exception type.
- `rows/<row>/answers.jsonl`: one line per complete question in source order: `query_id`,
  `split`, `input_text`, `gold_paper_ids`, `status`, `answer`, `cited_source_numbers`, `sources`
  (`number`, `paper_id`, `passage_id`, `title`, `year`, `section`, `url`; excerpt text stays in
  the checkpoint) and the six metric values.
- `report.json`: `stage="answer"`, `format_version=1`, `status` (`complete`, `partial`: some
  question failed, `failed`: stopped after `MAX_FAILURES` failures), `source_kind`,
  `selected_queries`, `usage` (generation/judge), protocol notes, manifest `provenance`,
  `workers_used`, and `rows.<row>` with `source_dir`, searched/completed counts,
  `failed_query_ids`, `not_searched_query_ids`, `usage` and `splits.dev/eval`. A split has
  `n_queries`, `answered`, `insufficient_evidence`, `no_sources`, `{n, mean}` for
  `gold_in_sources`, `gold_cited`, `context_relevance`, `context_precision`,
  `sources_per_answer`, faithfulness (plus `claims`, `supported_claims`, `pooled`), answer
  relevance (plus `noncommittal`), `generation_ms` (`n`, `mean`, `p50`, `p95`, `max` over answers
  with excerpts), `tokens_per_query` (`{n, mean}` per part),
  and `per_query`. Undefined values are JSON null and are not counted in `n`.
- Ablation reports add `rows.<row>.enabled_factors` and `comparisons`: [{`before`, `after`,
  `splits.dev/eval`: {`n_paired`, `metrics.<metric>`: {`n`, `before`, `after`, `delta`}}}] for
  baseline→A, baseline→B, baseline→AB and A→AB, B→AB, over questions both rows completed.

The CLI also publishes the report at `metrics/answer_<tag>.json`.
