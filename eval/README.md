# eval/ Evaluation (Q3)

## Goal

The D38 diagnostic is implemented in `retrieval_probe.py`; retrieval metrics, the eval-set
retrieval run (`retrieval_eval.py`, `scripts/run_eval.py --stage retrieval`), IAA and
annotation tooling are also implemented, while review metrics remain pending.
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
performance/cost, and ablation tables. (Generation metrics and RAGAS are pending — see below.)

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
  tags its files `<date>_first<N>`, so a quick check never overwrites a full run. `review`, `all` and `--ablation`
  stop with an error until review metrics exist.
- Tests: `tests/test_retrieval_eval.py` (fake `retrieve()`, hand-computed metrics, the CLI)
- First run (2026-10-30, TC2 A40, 13 min for 1,027 questions × 4 rows): eval-split Hit@1 0.818 (bm25), 0.828
  (dense), 0.873 (hybrid_rrf), 0.919 (hybrid_rerank); Hit@10 0.958 / 0.974 / 0.984 / 0.991. Table, dev split and
  caveats: EVALUATION.md §2. On TC2, read `lookup.sqlite` once before the run (`cat ... > /dev/null`, 16 s): hit
  hydration then comes from the page cache (the bm25 row took ~0.1 s per question instead of ~1.2 s over NFS)

### review_metrics.py — Review metrics (framework level)
- LLM verdicts vs. human annotation: Accuracy / Precision / Recall (positive class = pass)
- Coverage: aggregated in coverage_report (covered_aspects/total_aspects)
- Round distribution: share meeting the threshold in rounds 1/2/3, unmet rate

### annotation/ — Evaluation set (Q4)
- Original documented target (D15): 1000 items, 3 independent annotators, majority vote,
  IAA>=80% (percentage agreement + Fleiss' kappa). The current D42 main set has 2 annotators per
  question; whether this is acceptable under the coursework instructions remains to be checked.
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

## Pending QA evaluation (Q4)

- **Generation metrics** (ROUGE/BERTScore) and **RAGAS** (faithfulness / answer relevance /
  context relevance) are not implemented in the minimal QA feature. The output contract is
  now `DownstreamResult` in SCHEMA §10: answer text, all source excerpts, citations used,
  nested search result, and timings. Implement metric runners and their dependencies as
  separate evaluation work, using the manually labeled held-out dataset.
- The evaluation plan for these is already written in docs/EVALUATION.md §5–§6.

## Ablation run conventions (Q3/Q5)
- Change only one switch (README: Ablation discipline); same evaluation set, same seed, same query batch
- Name results `ablation_<innovation>_<date>.json`, including failure -> success query instances

## Definition of Done
- Metric implementations verified against hand-computed P@K/NDCG etc. values on small constructed examples before use
- All metrics runnable offline (except RAGAS LLM evaluation)
