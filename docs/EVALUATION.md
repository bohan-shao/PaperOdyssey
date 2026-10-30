# Evaluation Plan

> Covers assignment Q3 (retrieval), Q4 (generation/RAG/performance), and Q5 (ablation).
> All results go to `data/results/`; how to run: see `scripts/run_eval.py`.

**Composition (D23/D24)**: the system has two presentation tabs. Retrieval/review metrics evaluate
**Paper Search** (`orchestration.run_query()`); the **downstream task** (`downstream.run()`) calls
Paper Search internally and returns cited answers and exact evidence for later task-level evaluation.
Both run on the same evaluation set.

## 1. Evaluation Set (Q4 hard requirement)

- Size: **≥1,000 items**; 3 independent annotators; **IAA ≥ 80%** (report percentage agreement + Fleiss' κ; κ target ≥0.6 preferred)
- Format: `data/eval/eval_set.jsonl` (see SCHEMA.md §7); QA tasks use JSONL (the assignment only requires eval.xls for classification tasks)
- Constraints: deduplication, type balance (definitional / method-comparison / survey / paper-locating / technical-detail)
- Annotation content: per item = paper-level relevance annotation (mandatory, 1,000 items) + passage-level relevance annotation (subset of 100–200 items, to validate evidence granularity) + gold answer (for generation evaluation; fine-grained annotation can be limited to the subset)
- dev/eval split: ~10% dev for tuning, 90% eval for the final reported metrics
- **Procedure in use (D39)**: the LLM drafts one search request per randomly sampled paper; all 3 annotators
  mark independently whether that paper is a good answer (y/n); questions most of them accept enter the set with
  their paper as the only gold paper, rejected ones are rewritten by a person and marked again
  (`scripts/eval_set.py`, `eval/annotation/questions.py`). Agreement is over these y/n marks. From D42 on, the pilot's 50 questions have 3 annotators and the other
  1,050 have 2 each (a third breaks a tie); agreement is reported for each group. Round 1 (2026-10-29):
  agreement 88% in the pilot (all 3 agree; Fleiss' κ 0.10) and 93.7% on the other 1,050 (κ 0.12; κ is low because
  about 96% of all marks are y); the tie-break and rewrite rounds were not run, so the 66 split and the
  7 rejected questions are left out and eval_set.jsonl has 1,027 questions (912 eval, 115 dev). Passage-level gold
  and gold answers are not collected; the bullets above describe the original plan.

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

## 3. Q3 Innovation Ablations

| Innovation | Compared against | Switch | Reporting Points |
|---|---|---|---|
| Semantic chunking | Fixed-chunking baseline | `kb.chunking.mode=section_fixed\|semantic` | Rebuild the index per configuration, then rerun the same batch of queries |
| Parent-document retrieval | Small chunks returned directly | `retrieval.parent_doc.enabled` | Retrieval metrics + downstream generation effect |
| HyDE | No expansion | `orchestration.hyde.enabled` | Report the latency/cost increase as well |
| Metadata filtering | No filtering | `retrieval.filters.enabled` | Time-window/category scenarios |
| Cross-encoder rerank | No reranking | `retrieval.rerank.enabled` | See §2 matrix |
| Query views | Original query in both channels | `retrieval.query_processing.enabled` | Preserve dense semantics; deduplicate lexical tokens; retain negation/numbers |
| Paper/section diversity | Plain top-k passage truncation | `retrieval.diversity.enabled` | Source recall and relevant-paper precision alongside distinct-paper count |
| Sparse field weighting | Equal FTS5 field weights | `retrieval.sparse.field_weighting.enabled` | SQLite only; hold analyzer/backend/corpus fixed |

Each ablation must include: query instances that **failed before the improvement → succeeded after** (explicitly required by the assignment).

## 4. Review Module Metrics (framework level)

- LLM decisions vs. human annotations: Accuracy / Precision / Recall (positive class = pass)
- Coverage: proportion of input aspects covered (`coverage_report`)
- Round distribution: how many queries met the bar in 1 / 2 / 3 rounds, and the failure rate

## 5. Generation Metrics (Q4)

> The task is now cited paper QA. Generation/RAGAS metric runners and the manually labeled
> dataset remain separate follow-up work; offline functional tests are not quality scores.
> Use `DownstreamResult.answer`, `sources[].text`, and `cited_source_numbers` as inputs.

- **ROUGE-1/2/L** (`rouge-score`), **BERTScore** (vs. gold_answer)
- Abstractive QA relies mainly on ROUGE/BERTScore; if the answer is extractive, EM/F1 can be added
- Citation quality: manual checks that the cited passage supports the associated claim.
  Runtime citation-number validation only verifies identity, not semantic faithfulness.

## 6. RAG Metrics (Q4, RAGAS)

Run on every evaluation query (or a representative subset):

- **Faithfulness**: whether each claim in the answer is grounded in the retrieved passages
- **Answer Relevance**: whether the answer addresses the user's input
- **Context Relevance / Precision**: the proportion of retrieved evidence relevant to the question

> RAGAS uses LLM-based evaluation (consumes API tokens); running it on all 1,000 items is affordable (DeepSeek),
> so we recommend running the full set and recording token costs.

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
Metrics: the ROUGE/BERTScore/RAGAS trio + coverage.

## 9. Running and Reproduction

- Planned unified entry point `scripts/run_eval.py --config config.yaml --stage <retrieval|review|all>`; the script remains a stub, and generation/ragas stages are not implemented.
- Each evaluation run outputs: `data/results/metrics/<stage>_<date>.json` (including a config snapshot + dependency versions)
- When the report cites metrics, it must include the `metrics/*.json` file paths (included in the Dropbox deliverables)

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
1,000-item/three-annotator requirement.

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
