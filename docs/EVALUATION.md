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

## 2. Retrieval Metrics (Q3)

- **P@K, R@K** (K=5,10), **MRR@10**, **NDCG@10**
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

### D36 small-corpus diagnostic

`scripts/probe_retrieval.py --papers 300 --queries 60 [--hybrid] [--rerank]` samples complete
papers from the configured processed KB with `seed`, builds isolated sample indexes under
`paths.results/retrieval_probe/`, and writes
`metrics/ablation_retrieval_probe_<mode>_<date>.json`. No temporary input directory is embedded
in code/config. The report includes query records, paper IDs, a passage SHA-256, versions,
per-row retrieval config, stage timing, hits, build costs, and per-query/per-query-type metrics.

Queries alternate partial title terms and twelve distinct abstract terms excluding title
terms. The source paper is a **weak, non-exhaustive label**. These are known-source probes,
not held-out human relevance judgments: source text remains indexed and source recall is
optimistic. `source_recall@10`, source MRR/NDCG and distinct-paper count are computed over
distinct papers represented in the first ten returned passages. Diversity is not precision.
The source formulas have hand-computed regression tests. Production retrieval metric stubs
remain unchanged; this diagnostic does not satisfy the 1,000-item/three-annotator requirement.

Rows compare legacy 50/50/50 budgets, query processing alone, 150/150 depth alone, diversity
alone, the combination, SQLite instead of rank-bm25, and SQLite field weighting. The depth
row changes the two channel budgets as one coupled recall-budget experiment; it is not an
isolated per-channel ablation. Compare `sqlite_fields` only against `sqlite` to isolate field
weights. Baseline/combined rows keep the same model and index. The CPU MiniLM model pair
is a resource-limited proxy, not a BGE-vs-Qwen3 comparison. No LLM/API calls or token costs.

Before selecting a new production model/default, repeat on manually annotated dev queries,
then evaluate untouched held-out queries at paper and passage granularity. Include filtered,
acronym/model-name, negative constraint, comparison and technical-detail cases; report
P@10/R@10, MRR/NDCG, relevant evidence diversity, p50/p95, peak VRAM and build time. For
Qwen3 use separate dense index directories and record the instruction and token limit.
