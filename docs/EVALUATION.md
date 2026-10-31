# Evaluation Plan

> Covers assignment Q3 (retrieval), Q4 (generation/RAG/performance), and Q5 (ablation).
> All results go to `data/results/`; how to run: see `scripts/run_eval.py`.

**Composition (D23/D24)**: the system has two presentation tabs. Retrieval/review metrics evaluate
**Paper Search** (`orchestration.run_query()`); the **downstream task** (`downstream.run()`) calls
Paper Search internally and returns cited answers and exact evidence for later task-level evaluation.
Both run on the same evaluation set.

## 1. Evaluation Set (Q4: original target and current protocol)

- Original documented target (D15): **≥1,000 items**; 3 independent annotators; **IAA ≥ 80%** (report percentage agreement + Fleiss' κ; κ target ≥0.6 preferred)
- Format: `data/eval/eval_set.jsonl` (see SCHEMA.md §7); QA tasks use JSONL (the assignment only requires eval.xls for classification tasks)
- Constraints: deduplication, type balance (definitional / method-comparison / survey / paper-locating / technical-detail)
- Original annotation plan: per item = paper-level relevance annotation (1,000 items) + passage-level relevance annotation (subset of 100–200 items, to validate evidence granularity) + gold answer (for generation evaluation; fine-grained annotation can be limited to the subset)
- dev/eval split: ~10% dev for tuning, 90% eval for the final reported metrics
- **Procedure in use (D39/D42)**: the LLM drafts one search request per randomly sampled paper;
  assigned annotators independently mark whether that paper is a good answer (y/n). The pilot's 50 questions
  have 3 annotators and the other 1,050 have 2 each. A majority accepts the question with its source paper as
  the only gold paper. The tooling supports an independent third annotator for ties and a rewrite round for
  rejected questions (`scripts/eval_set.py`, `eval/annotation/questions.py`). Agreement uses the original y/n
  marks, excludes tie-breakers, and is reported separately for the 2- and 3-annotator groups. Round 1 (2026-10-29):
  agreement 88% in the pilot (all 3 agree; Fleiss' κ 0.10) and 93.7% on the other 1,050 (κ 0.12; κ is low because
  about 96% of all marks are y); the tie-break and rewrite rounds were not run, so the 66 split and the
  7 rejected questions are left out and eval_set.jsonl has 1,027 questions (912 eval, 115 dev). Passage-level gold
  and gold answers are not collected.

The current main set does not meet the original three-annotator target. D42 records the team's choice;
the coursework brief allows it ("it is recommended to have 3 annotators, but 2 is also OK"). The 1,027 count includes dev
questions and must not be reported as 1,027 held-out eval questions.

## 2. Retrieval Metrics (Q3)

- **Hit@K** (K=1,3,5,10; the main number, as each question has one gold paper), **P@K, R@K**, **MRR@10**, **NDCG@10**
  (`eval/retrieval_metrics.py`). With one gold paper per question the scores are a lower bound: another relevant
  paper in the top K earns nothing.
- Comparison matrix (same evaluation set, same batch of queries):

| Configuration | Switch (config.yaml) |
|---|---|
| BM25 only | `retrieval.sparse.enabled=true, dense.enabled=false, rerank.enabled=false` |
| Dense only | `retrieval.sparse.enabled=false, dense.enabled=true, rerank.enabled=false` |
| Hybrid (RRF) | both on, `rerank.enabled=false` |
| Hybrid + rerank | everything on (system default) |

- Granularity: paper-level metrics are primary (gold_paper_ids); passage-level metrics are reported separately on the annotated subset
- **Run (D44)**: `python scripts/run_eval.py --config config.yaml --stage retrieval` (`eval/retrieval_eval.py`, a GPU job).
  Every eval-set question's `input_text` goes to `retrieve()` as it is (no filters, the configured top 10 passages) under
  the four rows, named `bm25`, `dense`, `hybrid_rrf` and `hybrid_rerank`. Each row sets exactly the three switches of the
  table, in memory; every other setting stays as configured (query views and diversity on), so `hybrid_rerank` is what
  Paper Search runs and `bm25` / `dense` are the system's own channels, not plain BM25/BGE rankings. Hits go to
  `data/results/retrieval_results/<row>_<date>.jsonl` (SCHEMA.md §8). `data/results/metrics/retrieval_<date>.json` has the
  metrics per split (eval = the reported number, dev = tuning) and per question type and form, each question's first gold
  position per row, and the questions whose gold paper enters (failed → succeeded) or leaves the top 10 papers for
  bm25 → hybrid_rrf, dense → hybrid_rrf and hybrid_rrf → hybrid_rerank (the rerank ablation of §3).
- **First run** (2026-10-30, TC2 A40, 658,871 passages, 10 passages per question), paper level; report
  `data/results/metrics/retrieval_2026-10-30.json`, hits `data/results/retrieval_results/<row>_2026-10-30.jsonl`:

| Configuration | Split (questions) | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| bm25 | eval (912) | 0.818 | 0.904 | 0.931 | 0.958 | 0.867 | 0.889 |
| dense | eval (912) | 0.828 | 0.940 | 0.956 | 0.974 | 0.885 | 0.907 |
| hybrid_rrf | eval (912) | 0.873 | 0.959 | 0.978 | 0.984 | 0.918 | 0.934 |
| hybrid_rerank | eval (912) | 0.919 | 0.975 | 0.986 | 0.991 | 0.947 | 0.958 |
| bm25 | dev (115) | 0.843 | 0.965 | 0.974 | 0.974 | 0.903 | 0.921 |
| dense | dev (115) | 0.800 | 0.904 | 0.965 | 0.991 | 0.865 | 0.896 |
| hybrid_rrf | dev (115) | 0.887 | 0.983 | 0.983 | 0.991 | 0.935 | 0.949 |
| hybrid_rerank | dev (115) | 0.957 | 1.000 | 1.000 | 1.000 | 0.975 | 0.982 |

- On the eval split, BM25 and dense leave the gold paper out of their top 10 on 38 and 24 questions, both on only 6.
  Fusion brings it into the top 10 on 27 questions BM25 misses (losing 4) and on 15 that dense misses (losing 6);
  reranking adds 9 more (losing 2) and moves the gold paper up on 86 questions, down on 43. The observed p50
  `retrieve()` time per question was 78 / 88 / 99 / 362 ms (page cache warmed; not the §7 benchmark). Every question
  was drafted from its gold paper's abstract (D39), which favors lexical matching, and another relevant paper earns
  nothing, so the scores are a lower bound.

## 3. Q3 Innovation Ablations

| Innovation | Compared against | Switch | Reporting Points |
|---|---|---|---|
| Semantic chunking | Fixed-chunking baseline | `kb.chunk.mode=section_fixed\|semantic` | Rebuild the index per configuration, then rerun the same batch of queries |
| Parent-document retrieval | Small chunks returned directly | `retrieval.parent_doc.enabled` | Review outcomes; text expansion occurs after ranking, so direct ranking metrics do not measure its effect |
| HyDE | No expansion | `orchestration.hyde.enabled` | Report the latency/cost increase as well |
| Metadata filtering | No filtering | `retrieval.filters.enabled` | Time-window/category scenarios |
| Cross-encoder rerank | No reranking | `retrieval.rerank.enabled` | See §2 matrix |
| Query views | Original query in both channels | `retrieval.query_processing.enabled` | Preserve dense semantics; deduplicate lexical tokens; retain negation/numbers |
| Paper/section diversity | Plain top-k passage truncation | `retrieval.diversity.enabled` | Source recall and relevant-paper precision alongside distinct-paper count |
| Sparse field weighting | Equal FTS5 field weights | `retrieval.sparse.field_weighting.enabled` | SQLite only; hold analyzer/backend/corpus fixed |

Each ablation must include: query instances that **failed before the improvement → succeeded after** (explicitly required by the assignment).

### Automatic matrix runner (D46)

`scripts/run_eval.py --stage retrieval --ablation query_views diversity --split eval --tag views-diversity`
runs four rows: `baseline` (both selected factors off), `a` (first on), `b` (second on), and
`ab` (both on). One factor produces only `baseline` and `a`. All other configured settings,
the selected questions, indexes and seed stay fixed. Split filtering precedes `--limit`.
Comparisons are baseline→A, baseline→B, A→AB and B→AB, so each edge changes one switch.
The descriptive interaction is AB−A−B+baseline, using the same four-way completed subset.

Supported factors are `rerank`, `query_views`, `diversity`, and `field_weighting` for retrieval,
review or both; `parent_doc`, `hyde` and `loop` require `--stage review`. Field weighting needs
the enabled SQLite backend with three positive, nonuniform weights. Loop experiments need
at least two allowed rounds and one rewrite query. Semantic chunking requires paired rebuilt
corpus/lookup/sparse/dense indexes, and filtering requires constrained query scenarios; neither
is automated by this runner. Jev evaluation is also excluded. Unsupported combinations fail
before model loading or API calls.

Add `--dry-run` to inspect the plan without loading models, inspecting index files, writing
artifacts or calling APIs. Use a distinct `--tag` per experiment; repeat the same command with
`--resume` after interruption. Every cell has isolated checkpoints. Resume checks the complete
matrix's saved manifests and attempts before any new query; changed data, selection, effective
settings, code, dependency versions or index/corpus metadata refuse reuse. Completed queries
are never called again. Failed attempts' observed usage remains in the cost total.

Reports separate dev/eval and compare only queries completed in both cells. Missing or failed
queries are unpaired, never scored as misses. Hit@1/3/5/10, MRR@10 and descriptive wall times
use identical paired denominators; review additionally reports source acceptance/recall,
model-reported coverage and rounds. Every metric records its denominator. Fixed/broken cases
mean Hit@10 changes for retrieval and source acceptance changes for review; review reports
top-10 changes separately. Full changed-query IDs and up to five examples are retained.

The local seed is reset per cell; remote LLM sampling is not seeded. Enabled features are not
necessarily exercised: HyDE can fall back, parent text can remain unchanged, and the loop can
stop after one round. Actual round counts are recorded, but this runner does not certify HyDE
or parent expansion per query. Timings include flagged component cold starts and may reuse OS
caches. These are observed source-finding differences, not proof of deterministic gains,
human relevance, or improved QA generation.

## 4. Review Module Metrics (framework level)

`scripts/run_eval.py --stage review --split dev --limit 5 --tag review_dev5` runs the
complete Paper Search workflow, including understanding and feedback, with the online cache
disabled. It preserves every pass/reject record in `FinalResult.reviews`; `papers` remains the
system's ranked list of passing papers. Existing caches without a review trace remain readable
but are not reused by this evaluator. `--resume` skips completed queries only when dataset,
selection, effective settings, code and index metadata match the original manifest.

The fixed eval set contains **positive source-paper labels only**, not relevance judgments
for every returned candidate. Its report therefore separates source papers passed, rejected,
and never reviewed; source acceptance uses all selected gold papers as the denominator,
while the conditional pass rate uses only reviewed gold papers. Other candidates remain
unjudged. Known-source Hit/R/MRR/NDCG use the actual final passing-paper order (score descending,
then best retrieval rank, then paper ID; D47), and dev/eval are always separate. A result whose
`retrieval_ranks` do not cover every reviewed paper, or whose `papers` order differs, is rejected. These are source-finding metrics, not overall
reviewer accuracy, precision or F1. `review_accuracy(records, human_labels)` supports those
classification metrics only against explicit per-query human labels; missing labels are never
inferred as negatives. Candidate-level positive and negative annotations are still needed.

Coverage is the LLM's self-reported covered/input aspect ratio, with macro and pooled values;
it does not measure human relevance, answer correctness or faithfulness. Report coverage-met
round counts separately from all termination reasons: stopping at `min_papers` does not prove
full coverage. Zero-aspect coverage and all zero-denominator metrics are undefined (`null`).

Runs write `paths.results/review_runs/<tag>/manifest.json`, `report.json`, and atomic
`attempts/NNNNNN.json` checkpoints with the question, full result, usage delta and elapsed time.
A single-process lock prevents concurrent writers. `--workers N` (D48, default 1) runs up to N
questions at once inside that process; attempts are numbered in the order questions start, and
each counts its own usage (`llm.client.usage_scope`). A query error stops the run and is reported
as incomplete, never as a retrieval miss; with several workers no new question starts after it,
while questions already running finish and are recorded. Completed-query metrics in an incomplete
run are not full-set results. Resume retains failed attempts' observed usage and may use another
worker count. Interrupted attempts can have additional usage in the best-effort
`metrics/llm_usage.jsonl`, and transport failures may be billable without returned usage; totals
are not an invoice. Only the shared `llm` review backend currently has supported usage accounting.
No warm-up LLM calls are hidden: queries started before the process finished its first query include
component loading (`cold_start`). With several workers, wall times and stage latencies include
waiting for the other questions' model and LLM calls, so per-query latency comes from a
`--workers 1` run. Prefix limits are smoke checks rather than representative quality samples. Generation evaluation remains
separate (§5).

## 5. Answer Metrics (Q4, D49)

The task is cited paper QA. `scripts/run_eval.py --stage answer --source <review run>`
(`eval/answer_eval.py`) answers every question a finished review run (§4) completed, from its stored
search result, with `downstream.answer_from_search`: the UI's evidence selection and its one
generation call, with no new search. Answers are paired with the search results the review
report scored. Outputs: SCHEMA §13 (`answers.jsonl` holds the generated answers for the data zip).

Task metrics against the eval set's labels (one source paper per question):

- **gold_in_sources**: the source paper is among the excerpts the answer model received
- **gold_cited**: the answer cites an excerpt of the source paper

The eval set has no reference answers (`gold_answer` is empty, §1), so ROUGE, BERTScore and
EM/F1 against a reference cannot be computed and are not reported. Runtime citation-number
validation only checks source identity; semantic support is measured by faithfulness below.

## 6. RAG Metrics (Q4, RAGAS-style)

Computed by `eval/answer_metrics.py` with an LLM judge, following the RAGAS definitions, for every
selected question whose search completed, over exactly the excerpts the answer model saw:

- **Faithfulness**: the judge splits the answer into standalone claims and checks each against
  the excerpts; supported claims / claims (undefined for an answer without claims)
- **Answer Relevance**: the judge writes `eval.answer.relevance_questions` (3) questions back from
  the answer; mean cosine similarity of their dense-encoder embeddings with the question, 0 when
  the judge finds the answer noncommittal
- **Context Relevance / Precision**: the judge marks each excerpt relevant or not to the question;
  relevant / excerpts, and average precision in source order

An answer without excerpts (no passing paper) is not judged: faithfulness and context scores are
undefined and answer relevance is 0. The judge defaults to the generator's model at temperature 0
(`eval.answer.judge_model`, `judge_temperature`); a model judging its own answers can be lenient,
so report the judge model. Judge and generation token usage are reported separately (§7).

## 7. Performance Metrics (Q2/Q4)

- Per-stage latency: `latency_by_stage` (understanding/retrieval/rerank/review/generation/total)
- Throughput: queries/sec (measured for the retrieval core alone, without mixing in LLM calls)
- Cost: tokens per query (in/out) + estimated price; total index-building time
- Scale: index size (BM25 directory, FAISS files), vector dimension, total number of chunks
- Scalability discussion: effect of ANN parameters (efSearch), sharding, and batch encoding on throughput (measured or estimated)

## 8. Q5 Downstream Innovation Ablations

> The QA baseline uses one generation call with numbered evidence. No extra self-check or
> CoT switches are implemented. The existing agentic loop can be evaluated on QA outcomes.

| Innovation | Compared against | Existing switch |
|---|---|---|
| Agentic loop (orchestration layer) | Single-round parallel | `orchestration.loop.enabled=false` |

Ablation matrix: baseline / +A / +B / +A+B (the assignment requires: add only A, add only B, add both, and report each gain).

Run the D46 matrix for Paper Search (for example `--stage review --ablation loop hyde`), then
`--stage answer --source <paths.results>/ablation_runs/<tag>` answers every row's questions and reports
paired answer-metric means and deltas (§5–§6) for baseline→A, baseline→B, baseline→AB and the
one-switch edges, next to the review report's source and coverage comparisons.

## 9. Running and Reproduction

- Unified entry point `scripts/run_eval.py --config config.yaml --stage <retrieval|review|all>`:
  retrieval (§2), review (§4), or both in sequence. `all` includes paid LLM calls; use an explicit
  stage for a bounded run. `--ablation FACTOR [FACTOR]` requires an explicit stage and follows
  the controlled protocol in §3. `--stage answer --source DIR` answers and scores a finished review
  run or review ablation (§5, §6, §8); it is not part of `all`.
- Retrieval writes `paths.results/metrics/retrieval_<tag>.json`; review writes
  `paths.results/review_runs/<tag>/report.json` with its manifest and per-attempt traces.
- Answers write `paths.results/answer_runs/<tag>/` (report, `rows/<row>/answers.jsonl`, checkpoints;
  SCHEMA §13) and publish `paths.results/metrics/answer_<tag>.json`.
- Ablations write `paths.results/ablation_runs/<tag>/report.json`, a selected-query snapshot,
  manifest and isolated row/stage checkpoints (SCHEMA §12). The CLI atomically publishes a
  copy at `paths.results/metrics/ablation_<factor_names_joined_underscore>_<tag>.json`.
- Cite the exact report artifact and its dataset/config provenance in the course report.
  Results stay gitignored and are included in the shared deliverables.

### D38 small-corpus diagnostic

`scripts/probe_retrieval.py --papers 300 --queries 60 [--hybrid] [--rerank]` samples complete
papers from the configured processed KB with `seed`, builds isolated sample indexes under
`paths.results/retrieval_probe/`, and writes
`metrics/ablation_retrieval_probe_<mode>_<UTC-date-time>_<run-id>.json`. Modes distinguish
sparse/hybrid and rerank on/off; every run has isolated sample/progress artifacts.
No temporary input directory is embedded
in code/config. The report includes query records, paper IDs, a passage SHA-256, versions,
per-row retrieval config, stage timing, hits, build costs, and per-query/per-query-type metrics.
`row_setup_ms` measures index loading plus one warm-up request, with potentially reused models
and OS caches. It is not a comparable process cold-start measurement. Retriever objects and
their lookup references are reset between rows; all experiment-owned components are cleared
on exit, including failures.

Queries alternate partial title terms and twelve distinct abstract terms excluding title
terms, using a fixed generation vocabulary independent of tuned retrieval stopwords. Every
source query has paired raw-term and `Find papers on ...` forms. Main metrics and stage
latencies use raw terms; `by_query_style` reports both forms separately. Historical 2026-10-13
runs used only the template, so their query-processing gain is confounded by removing its
stopwords. The revised diagnostic found raw-term MRR unchanged at 0.975 (see retrieval/REVIEW.md).
Neither query form represents independently collected natural user queries.
The source paper is a **weak, non-exhaustive label**. These are known-source probes,
not held-out human relevance judgments: source text remains indexed and source recall is
optimistic. In D41 report format 2 (`metric_version: paper_ranking_v2`), `source_recall@10`,
source MRR/NDCG call the canonical retrieval metric functions: sort all returned hits by rank,
deduplicate papers, then apply K. `distinct_papers_in_top_10_passages` separately counts unique
papers in the first ten passages. The probe requests at most ten passages, so this need not
fill ten paper positions. Diversity is not precision. Older reports retain their historical
passage-cutoff definitions; do not mix report versions without recomputing from saved hits.
The source formulas have hand-computed parity tests. The D39 annotation workflow remains
separate from this diagnostic, which does not satisfy the
original 1,000-item/three-annotator target in section 1.

Rows compare legacy 50/50/50 budgets, query processing alone, diversity alone, depth with
diversity, the combination, SQLite instead of rank-bm25, and SQLite field weighting. Hybrid
mode additionally tests 150/150 depth alone; sparse-only marks `depth_only` inapplicable
because the unchanged top-50 prefix cannot affect output. Compare `depth_with_diversity`
against `diversity_only` to test depth without increasing the 50-candidate scoring budget.
In hybrid mode the two channel budgets change together; this is not an isolated per-channel
ablation. `comparisons` records controls. Compare `sqlite_fields` only against `sqlite` to isolate field
weights. Baseline/combined rows keep the same model and index. The CPU MiniLM model pair
is a resource-limited proxy, not a BGE-vs-Qwen3 comparison. No LLM/API calls or token costs.

The latency runner uses protocol `legacy_50_v2` in a fresh process, fixing ranking switches,
budgets and scoring parameters before loading models. Model/backend/index identity remains
configured, not silently substituted. Every row saves effective settings; reports save full
queries, requested settings and baseline overrides. Rerank-depth sweeps expand the collection
budget to the largest requested depth and report actual pairs, so undersized pools are visible.
`cold_start_s` covers component loading and warm-up, not process creation or cold filesystem caches.

Before selecting a new production model/default, repeat on manually annotated dev queries,
then evaluate untouched held-out queries at paper and passage granularity. Include filtered,
acronym/model-name, negative constraint, comparison and technical-detail cases; report
P@10/R@10, MRR/NDCG, relevant evidence diversity, p50/p95, peak VRAM and build time. For
Qwen3 use separate dense index directories and record the instruction and token limit.

Parent-document latency rows also report `parent_text_counts` (`expanded`, `unchanged`,
`no_parent`) by comparing returned text with stored children outside the timed retrieval call.
An unchanged parent hit can mean fallback or a parent identical to its child; warnings carry
specific fallback reasons. Report actual expansion/fallback rates before attributing an effect
to parent retrieval. The default rerank batch size stays 64 until a reference-GPU sweep supports
a change; CPU probe batch size 16 is a separate experiment setting.
