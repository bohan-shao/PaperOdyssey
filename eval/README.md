# eval/ Evaluation (Q3)

## Goal

The D38 diagnostic is implemented in `retrieval_probe.py`; retrieval metrics, the eval-set
retrieval run (`retrieval_eval.py`, `scripts/run_eval.py --stage retrieval`), Paper Search review
metrics and its resumable runner (`review_eval.py`, `--stage review`), IAA and annotation tooling
are also implemented. `ablation.py` runs controlled one- or two-factor comparisons on the same
questions. `answer_eval.py` (`--stage answer`, D49) answers the questions of a finished review run
or review ablation and scores the answers (`answer_metrics.py`). These are evaluation tools; new
empirical results require a real run.
Run `python scripts/probe_retrieval.py --papers 300 --queries 60`
for a sparse comparison, or add `--hybrid --rerank` for real CPU MiniLM models selected by
`eval.retrieval_probe`. All retrieval goes through `retrieval.pipeline.retrieve()`. It writes
isolated sample indexes, queries, config snapshots, timings and per-query known-source metrics
under configured `paths.results`. Run in a fresh process, without an online server sharing
the components. Queries come from indexed source text; these proxy labels are not human
relevance judgments. See `docs/EVALUATION.md` and `retrieval/README.md` for measured results
and limitations. Set `HF_HUB_OFFLINE=1` when using locally cached models.
The revised probe pairs raw terms with the request template, reports their metrics separately,
and uses unique run directories/filenames for all four sparse/hybrid/rerank combinations.
`row_setup_ms` is a warm-up/setup measurement with possible model/cache reuse, not a process
cold start. See [retrieval/REVIEW.md](../retrieval/REVIEW.md) for the D40 audit and diagnostic results.
The D41 report format is version 2: source metrics delegate to `retrieval_metrics.py` (sort all
returned hits by rank, deduplicate papers, then apply positive integer K). Diversity is separately
named `distinct_papers_in_top_10_passages`. The pipeline still returns at most ten passages.
Sparse-only `depth_only` is omitted with an explicit reason; compare `depth_with_diversity`
against `diversity_only`. `comparisons` records the appropriate control for each row.
The shared internal `runtime.py` rejects previously loaded components and releases experiment
readers/models on exit. The probe also restores temporary Torch/FAISS thread settings.

Produce every number the report needs for the decided parts: retrieval metrics, review metrics,
performance/cost, ablation tables, and answer-level metrics (`answer_eval.py`).

## Implementation requirements

### retrieval_metrics.py — Retrieval metrics (Q3)
- `compute_retrieval_metrics(hits_per_query, gold_per_query, k_list)`: **implemented**
  - Hit@K, P@K, R@K (K=1,3,5,10 by default), MRR@10, NDCG@10, macro-averaged; `per_query` gives the
    position of each query's first gold paper (failed -> succeeded examples of an ablation)
  - gold supports two granularities: paper-level (gold_paper_ids) and passage-level (gold_passage_ids);
    every id counts once, at its best-ranked position (at paper level, a paper's best passage). The conventions are in the
    module docstring; checked against hand-computed values in `tests/test_eval.py`
  - hits come from `data/results/retrieval_results/` (run once for each of the four config variants)
- Output JSON: `data/results/metrics/retrieval_<date>.json`, written by `retrieval_eval.py` (below)
  (including a config snapshot + dependency versions, EVALUATION.md §9)

### retrieval_eval.py — The eval set through the §2 matrix (Q3, D44)
- `run_retrieval_eval(eval_set_path, results_dir, tag, configs, limit)`: **implemented**; run it with
  `python scripts/run_eval.py --config config.yaml --stage retrieval [--configs ...] [--limit N] [--tag T]`
  in a fresh process with the built indexes and models (a GPU job on TC2)
- Every question's `input_text` goes to `retrieve()` as it is (no filters, the configured top 10 passages) under
  `bm25`, `dense`, `hybrid_rrf` and `hybrid_rerank`. Each row sets exactly `sparse/dense/rerank.enabled` in memory
  (`latency.config_overrides`, restored afterwards); everything else stays as configured (query views, diversity),
  so `hybrid_rerank` is what Paper Search runs. One unrecorded warm-up request per row loads its components.
- Hits: `results/retrieval_results/<row>_<tag>.jsonl` (SCHEMA §8), each written as soon as its row finishes.
  Report: `results/metrics/retrieval_<tag>.json` with, per row and split (dev / eval), the `compute_retrieval_metrics`
  output, the same per `question_type` and `question_form`, every question's first gold position, the questions whose
  gold paper enters (failed → succeeded) or leaves the top 10 papers for bm25 → hybrid_rrf, dense → hybrid_rrf and
  hybrid_rrf → hybrid_rerank (with examples), each row's effective config and observed latency (incidental, not the
  §7 benchmark), the eval set's SHA-256, the index manifests, a config snapshot and dependency versions
- A retrieval error stops the run instead of being scored as misses. `--limit N` runs only the first N questions and
  tags its files `<date>_first<N>` by default, so a quick check has a different tag from a full run.
  `--stage all` runs this stage before Paper Search review. Named factors use the separate
  `--ablation` protocol below, rather than the four-channel retrieval matrix.
- Tests: `tests/test_retrieval_eval.py` (fake `retrieve()`, hand-computed metrics, the CLI)
- First run (2026-10-30, TC2 A40, 13 min for 1,027 questions × 4 rows): eval-split Hit@1 0.818 (bm25), 0.828
  (dense), 0.873 (hybrid_rrf), 0.919 (hybrid_rerank); Hit@10 0.958 / 0.974 / 0.984 / 0.991. Table, dev split and
  caveats: EVALUATION.md §2. On TC2, read `lookup.sqlite` once before the run (`cat ... > /dev/null`, 16 s): hit
  hydration then comes from the page cache (the bm25 row took ~0.1 s per question instead of ~1.2 s over NFS)

### review_metrics.py — Review metrics (framework level)
- `review_accuracy(records, human_labels)`: **implemented**, per query, only for reviewed papers
  with an explicit human label for that query. Reports confusion counts, support, label classes,
  Accuracy / Precision / Recall / F1; undefined denominators produce `null`.
- **Current labels are positive source papers only.** Other retrieved or reviewed papers are
  unjudged, not negative examples. This dataset cannot establish reviewer false-positive
  performance or general classification quality. Labeled papers never reviewed are retrieval
  misses, reported separately from classification false negatives. Preserve these limitations
  and the labeled support when using the metrics in the report.
- `coverage_stats(coverage_reports)`: **implemented**, macro and pooled aspect coverage, final-round
  distributions and termination reasons. Coverage is the system's own aspect judgment, not human
  relevance or answer faithfulness; reaching `min_papers` alone does not establish full coverage.

### review_eval.py — Resumable Paper Search and review evaluation
- `run_review_eval(eval_set_path, results_dir, tag, *, limit=None, split="all", resume=False, workers=1)`:
  runs `orchestration.run_query()` for each selected question and saves results under
  `data/results/review_runs/<tag>/`: `manifest.json`, numbered `attempts/*.json`, `report.json`,
  and best-effort raw `metrics/llm_usage.jsonl`. Completed attempts include the question, full
  `FinalResult`, token usage and wall time; failed/interrupted attempts remain identifiable.
  Unlike the retrieval stage, this
  includes query understanding, review and any configured rewrite rounds, and makes real LLM calls.
- Run in a fresh dedicated process with `orchestration.review.backend=llm`. Jev is not supported
  by this runner because its usage is not tracked. There is no hidden warm-up: queries started
  before the process finished its first query include component loading and are marked `cold_start`.
- `workers` (`--workers N`, D48): up to N questions at once in this one process, sharing its models
  and indexes; a question spends most of its time waiting for LLM replies. Usage is counted per
  question (`llm.client.usage_scope`), attempts are numbered in start order, and after a query error
  no new question starts. Wall times then include waiting for the other questions, so measure
  per-query latency with one worker. A run may resume with a different worker count.
  On TC2 (A40, 2026-10-30) 8 workers ran 24 dev questions in 172 s including model loading, about
  4x the one-at-a-time rate (~30 s per question), with ~25k tokens per question either way.
- Commands (built indexes, cached retrieval models and configured API credentials are required):
  ```bash
  python scripts/run_eval.py --config config.yaml --stage review --split dev --limit 3 --tag review-dev-smoke
  python scripts/run_eval.py --config config.yaml --stage review --split eval --tag review-eval
  python scripts/run_eval.py --config config.yaml --stage review --split eval --tag review-eval --resume
  python scripts/run_eval.py --config config.yaml --stage review --split eval --tag review-eval-8 --workers 8
  ```
  `--split` is applied before `--limit`. Keep the same tag, selection and configuration when
  resuming. `--resume` requires matching configuration, dataset, code and index metadata fingerprints, and reuses
  only completed query results. A different experiment needs a new tag.
- The runner disables the online search cache to collect fresh `FinalResult.reviews`, including
  all pass/reject records and round numbers. Existing online caches without a trace remain usable
  by the application but are not evidence for this evaluation.
- Reports distinguish `complete`, `partial` and `failed` and record completed/selected query
  counts. The CLI exits 1 and prints the summary to stderr for an incomplete run; its completed
  query artifacts remain available for resume. Partial metrics describe completed queries only
  and must not be presented as a full selected-set result.
- Token usage covers observed API responses with returned usage, including JSON retries and
  responses within query attempts that later fail. It does not establish total billed usage:
  interrupted attempts, transport failures and responses with missing provider usage can leave
  additional tokens unreported. No monetary cost estimate is produced.
- `--stage all` (also the CLI default) runs the four retrieval configurations and then review.
  `--configs` affects retrieval only; `--split`/`--resume` affect review only. To resume review
  without repeating retrieval, use `--stage review`. Answers are scored separately from a finished
  run (`--stage answer`, below).

### ablation.py — Controlled one- or two-factor experiments
- `run_ablation(eval_set_path, results_dir, tag, *, factors, stage, split="all", limit=None,
  resume=False, dry_run=False)` evaluates one or two selected factors with all other settings fixed.
  One factor produces `baseline` and `A`; two produce `baseline`, `A`, `B`, `AB`, where A and B
  follow the argument order. **The baseline disables all selected factors**, even if the supplied
  config enables them. A/B enable just that factor; AB enables both. This baseline is therefore
  defined by the chosen experiment, not necessarily the normal production configuration.
- Paired comparisons change exactly one factor: baseline→A, baseline→B, A→AB and B→AB (the
  Hamming-distance-one edges). They use the same selected questions and fixed settings; a
  baseline→AB change is not attributed to a single factor. Per-row metrics retain dev/eval
  separation and comparisons include questions fixed or broken by the change.

| Factor | Config setting | Supported stages |
|---|---|---|
| `rerank` | `retrieval.rerank.enabled` | retrieval, review, all |
| `query_views` | `retrieval.query_processing.enabled` | retrieval, review, all |
| `diversity` | `retrieval.diversity.enabled` | retrieval, review, all |
| `field_weighting` | `retrieval.sparse.field_weighting.enabled` (SQLite backend required) | retrieval, review, all |
| `parent_doc` | `retrieval.parent_doc.enabled` | review |
| `hyde` | `orchestration.hyde.enabled` | review |
| `loop` | `orchestration.loop.enabled` | review |

- Supply an explicit `--stage` with `--ablation`; `all` is accepted only when every selected
  factor supports both stages. Unsupported combinations fail before execution instead of silently
  ignoring a switch. `--configs` cannot be combined with ablations. Canonical factor names are
  used (for example `rerank`, not `rerank_off`).
  ```bash
  python scripts/run_eval.py --config config.yaml --stage retrieval --ablation rerank --split eval --tag rerank-eval
  python scripts/run_eval.py --config config.yaml --stage retrieval --ablation query_views diversity --split eval --tag views-diversity
  python scripts/run_eval.py --config config.yaml --stage review --ablation loop hyde --split dev --limit 3 --tag loop-hyde-smoke
  ```
  Add `--dry-run` to validate and print a plan without running queries or writing artifacts.
  `--dry-run` is only available with `--ablation`. For a real run, retrieval needs built indexes
  and models, and review also makes configured LLM calls. `--split` is applied before `--limit`
  for either stage; a prefix limit is a smoke check, not a representative quality estimate.
- Artifacts live under `data/results/ablation_runs/<tag>/`, including `report.json`. The CLI
  also atomically writes `data/results/metrics/ablation_<factor_names_joined_underscore>_<tag>.json`
  for complete, partial and failed runs. `planned` and `complete` exit successfully; incomplete
  runs exit 1 and print their summary to stderr. Dry runs print only. Resume with the same
  arguments and `--resume`; provenance checks reject changed data, selection, config or code.
  `--workers N` runs N questions at once inside each review cell (D48); it may differ on resume.
- Known-source retrieval metrics and review coverage have the same limitations as above:
  positive source-paper labels do not make other candidates negatives, and self-reported
  coverage is not answer quality. For answer-level comparisons of the same matrix, run
  `--stage answer --source` on the ablation directory (`answer_eval.py`, below).
- Enabled factors need not activate on every question: HyDE may fall back, parent text may
  remain unchanged, and the loop can stop in one round. Actual rounds are reported; per-query
  HyDE/parent expansion is not certified by this runner.
- Semantic chunking is excluded because it needs a protocol selecting separately prebuilt
  indexes for each chunking variant. Metadata filters are excluded because this dataset has
  no structured filter scenarios. Jev evaluation is excluded; this runner supports the shared
  `llm` review backend. Answer evaluation is the separate `answer_eval.py` stage.

### annotation/ — Evaluation set (Q4)
- Original documented target (D15): 1000 items, 3 independent annotators, majority vote,
  IAA>=80% (percentage agreement + Fleiss' kappa). The current D42 main set has 2 annotators per
  question, which the coursework brief allows ("it is recommended to have 3 annotators, but 2 is also OK").
- **Implemented (D39/D42)**: `questions.py` + `scripts/eval_set.py`: the LLM drafts one search request per
  sampled paper; 3 annotators mark each pilot question and 2 mark each main-set question y/n in CSV
  sheets (`data/eval/annotation/`), majority vote -> `data/eval/eval_set.jsonl` (SCHEMA.md §7).
  The tooling supports third-person tie-breaks and rewrites, but neither was run for round 1:
  1,027 accepted questions remain (912 eval, 115 dev); 66 ties and 7 rejected questions are excluded.
  Steps and commands: `annotation/README.md`; marking rules: `guidelines.md` §6
- IAA computation: `eval/annotation/iaa.py` (Fleiss' kappa + percentage agreement), **implemented**;
  unit = one (question, paper) pair, grouped by original annotator count with tie-breakers excluded;
  report in `data/results/metrics/iaa_<date>.json`

### latency.py — Retrieval-core latency benchmark (Q2, EVALUATION.md §7)
- Run: `./run_docker.sh python scripts/benchmark_retrieval.py [--queries 100]` (needs the built
  indexes and models) → `data/results/metrics/latency_retrieval_<date>.json` + a summary table
- Times `retrieve()` only (no LLM stages), warm, over paper titles sampled from papers.sqlite
  (seeded): per-stage p50/p95 for the four ablation rows + filtered + parent-doc, the DESIGN §4
  budget check (hybrid+rerank p95 < 500 ms), throughput at 1/4/8 concurrent callers, an efSearch
  sweep (HNSW time + recall@50 vs exact search) and a rerank-depth sweep (time + top-10
  agreement with depth 50); cold-start load time is reported separately
- Parent rows include actual expanded/unchanged/no-parent text counts. `mean_candidates` is
  the selected pool before reranking/final truncation, not the raw sparse/dense union.
- Configurations are switched in memory, never in config.yaml. Protocol `legacy_50_v2` fixes
  query processing/diversity/field weighting off, 50/50/50 candidates, ten outputs, equal RRF
  weights (k=60), BM25 0.9/0.4, efSearch 128, filtered exact threshold 50,000, parallel retrieval,
  and rerank batch size 64/max length 512 before loading components. Model/backend/index
  identity remains configured. Every row records its effective retrieval config, and the report
  includes requested/baseline configs and the full query batch. The CLI accepts `--config`.
  Actual index paths/manifests are recorded; the latest build-log path is unverified context
  and is not assumed to describe the measured indexes (`index_build` is null in v2).
- Run in a fresh dedicated process; loaded components from another configuration are rejected.
  `cold_start_s` is retained for compatibility but measures component load plus warm-up, with
  possibly warm OS/model file caches. Rerank sweeps collect up to the largest requested depth
  and record actual pair counts, including when the corpus cannot fill that depth.
- Percentiles: linear interpolation between ranks (numpy's default)
- First run on the 50-paper sample (RTX 5070): hybrid+rerank p95 410 ms (within budget, rerank
  ≈ 95% of it); throughput flat at ~2.6 qps from 1 to 8 callers (the reranker runs one call at a
  time, so concurrent sub-queries only queue); efSearch 128 → recall@50 0.998; rerank depth
  costs ~7.8 ms per pair (depth 30: 232 ms, top-10 agreement 0.77). Rerun after the full KB build
  before citing numbers: BM25 and HNSW times grow with the corpus

## answer_eval.py / answer_metrics.py — Answer-level evaluation (Q4/Q5, D49)

- `run_answer_eval(source_dir, results_dir, tag, *, split="all", limit=None, resume=False, workers=1)`:
  **implemented**. `source_dir` is a finished review run (`review_runs/<tag>`) or a review ablation
  (`ablation_runs/<tag>`, every row). For each question the search completed, the stored
  `FinalResult` goes through `downstream.answer_from_search` (the UI's evidence and generation code,
  one LLM call, no new search), so answers pair with the scored search results.
- Metrics: `gold_in_sources` / `gold_cited` against the eval set's source paper; RAGAS-style
  faithfulness, answer relevance, context relevance and context precision by an LLM judge
  (`answer_metrics.judge_answer`, definitions in its docstring and EVALUATION.md §6). The eval set
  has no reference answers, so ROUGE/BERTScore/EM are not computed. Judge scores are model
  judgments; by default the judge is the generator's model at temperature 0 (`eval.answer`).
- Resumable like the review runner: per-question checkpoints, a run lock, a fingerprint over
  sources, selection, settings, prompt versions and code. Index and corpus metadata must equal
  the source runs' manifests. Failed questions are listed and retried on `--resume` (their usage
  is kept); after `MAX_FAILURES` (5) failures no new question starts. `--workers N` as in D48;
  usage is counted per question and split into generation and judge.
- Outputs (SCHEMA §13): `answer_runs/<tag>/report.json`, `rows/<row>/answers.jsonl` (the
  generated answers for the data zip), checkpoints, and `metrics/answer_<tag>.json`. An ablation
  source adds paired comparisons: baseline→A, baseline→B, baseline→AB and the one-switch edges.
- Checked offline in `tests/test_answer_eval.py` (hand-computed scores, resume, failures,
  workers, ablation pairing). Needs the lookup store, the dense encoder and an LLM key to run.

## Ablation run conventions (Q3/Q5)
- Change only one switch (README: Ablation discipline); same evaluation set, same seed, same query batch
- Name results `ablation_<innovation>_<date>.json`, including failure -> success query instances

## Definition of Done
- Metric implementations verified against hand-computed P@K/NDCG etc. values on small constructed examples before use
- All metrics runnable offline (except RAGAS LLM evaluation)
