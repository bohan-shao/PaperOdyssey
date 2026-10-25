# eval/ Evaluation (Q3)

## Goal

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
- Output JSON: `data/results/metrics/retrieval_<stage>_<date>.json`
  (including a config snapshot + dependency versions, EVALUATION.md §9)

### review_metrics.py — Review metrics (framework level)
- LLM verdicts vs. human annotation: Accuracy / Precision / Recall (positive class = pass)
- Coverage: aggregated in coverage_report (covered_aspects/total_aspects)
- Round distribution: share meeting the threshold in rounds 1/2/3, unmet rate

### annotation/ — Evaluation set (Q4 hard requirement)
- Goal: 1000 items, 3 annotators, majority vote, IAA>=80% (percentage agreement + Fleiss' kappa)
- **Implemented (D39)**: `questions.py` + `scripts/eval_set.py`: the LLM drafts one search request per
  sampled paper, the 3 annotators mark it y/n in CSV sheets (`data/eval/annotation/`), majority vote ->
  `data/eval/eval_set.jsonl` (SCHEMA.md §7), rejected questions are rewritten and marked again.
  Steps and commands: `annotation/README.md`; marking rules: `guidelines.md` §6
- IAA computation: `eval/annotation/iaa.py` (Fleiss' kappa + percentage agreement), **implemented**;
  unit = one (question, paper) pair, report in `data/results/metrics/iaa_<date>.json`

### latency.py — Retrieval-core latency benchmark (Q2, EVALUATION.md §7)
- Run: `./run_docker.sh python scripts/benchmark_retrieval.py [--queries 100]` (needs the built
  indexes and models) → `data/results/metrics/latency_retrieval_<date>.json` + a summary table
- Times `retrieve()` only (no LLM stages), warm, over paper titles sampled from papers.sqlite
  (seeded): per-stage p50/p95 for the four ablation rows + filtered + parent-doc, the DESIGN §4
  budget check (hybrid+rerank p95 < 500 ms), throughput at 1/4/8 concurrent callers, an efSearch
  sweep (HNSW time + recall@50 vs exact search) and a rerank-depth sweep (time + top-10
  agreement with depth 50); cold-start load time is reported separately
- Configurations are switched in memory, never in config.yaml; every row sets all switches
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
