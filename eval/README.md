# eval/ Evaluation (Q3)

## Goal

Produce every number the report needs for the decided parts: retrieval metrics, review metrics,
performance/cost, and ablation tables. (Generation metrics and RAGAS are parked — see below.)

## Implementation requirements

### retrieval_metrics.py — Retrieval metrics (Q3)
- `compute_retrieval_metrics(hits_per_query, gold_per_query, k_list)`:
  - P@K, R@K (K=5,10), MRR@10, NDCG@10
  - gold supports two granularities: paper-level (gold_paper_ids) and passage-level (gold_passage_ids)
  - hits come from `data/results/retrieval_results/` (run once for each of the four config variants)
- Output JSON: `data/results/metrics/retrieval_<stage>_<date>.json`
  (including a config snapshot + dependency versions, EVALUATION.md §9)

### review_metrics.py — Review metrics (framework level)
- LLM verdicts vs. human annotation: Accuracy / Precision / Recall (positive class = pass)
- Coverage: aggregated in coverage_report (covered_aspects/total_aspects)
- Round distribution: share meeting the threshold in rounds 1/2/3, unmet rate

### annotation/ — Evaluation set (Q4 hard requirement)
- Goal: 1000 items, 3 annotators, majority vote, IAA>=80% (percentage agreement + Fleiss' kappa)
- `guidelines.md`: annotation guidelines (relevance definition, examples, workflow)
- Tooling: start with manual annotation in Excel/spreadsheet templates (`data/eval/annotation/`),
  optionally upgrade to a simple annotation web page later (stretch, does not block the main line)
- Output: `data/eval/eval_set.jsonl` (SCHEMA.md §7)
- IAA computation: `eval/annotation/iaa.py` (Fleiss' kappa + percentage agreement)

## Parked (come back with the downstream task, Q4)

- **Generation metrics** (ROUGE/BERTScore) and **RAGAS** (faithfulness / answer relevance /
  context relevance) depend on the downstream task output schema — `downstream/` is an empty
  reserved folder (task type TBD). Re-add `generation_metrics.py` and `rag_metrics.py`
  (and the rouge-score / bert-score / ragas dependencies) when the task is decided.
- The evaluation plan for these is already written in docs/EVALUATION.md §5–§6.

## Ablation run conventions (Q3/Q5)
- Change only one switch (README: Ablation discipline); same evaluation set, same seed, same query batch
- Name results `ablation_<innovation>_<date>.json`, including failure -> success query instances

## Definition of Done
- Metric implementations verified against hand-computed P@K/NDCG etc. values on small constructed examples before use
- All metrics runnable offline (except RAGAS LLM evaluation)
