# Evaluation Plan

> Covers assignment Q3 (retrieval), Q4 (generation/RAG/performance), and Q5 (ablation).
> All results go to `data/results/`; how to run: see `scripts/run_eval.py`.

**Composition (D23/D24)**: the system has two presentation tabs. Retrieval/review metrics evaluate
**Paper Search** (`orchestration.run_query()`); the **downstream task** (`downstream.run()`) calls
Paper Search internally and adds task-level metrics (finalized when the task type is decided).
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
| BM25 only | `retrieval.sparse.enabled=true, dense.enabled=false` |
| Dense only | reverse of the above |
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

Each ablation must include: query instances that **failed before the improvement → succeeded after** (explicitly required by the assignment).

## 4. Review Module Metrics (framework level)

- LLM decisions vs. human annotations: Accuracy / Precision / Recall (positive class = pass)
- Coverage: proportion of input aspects covered (`coverage_report`)
- Round distribution: how many queries met the bar in 1 / 2 / 3 rounds, and the failure rate

## 5. Generation Metrics (Q4)

> Parked: implemented together with the downstream task (downstream/ is empty, task type TBD;
> see eval/README.md). The plan below is fixed for when the task is decided.

- **ROUGE-1/2/L** (`rouge-score`), **BERTScore** (vs. gold_answer)
- Abstractive QA relies mainly on ROUGE/BERTScore; if the answer is extractive, EM/F1 can be added
- Citation quality: whether the `[n]` citations in the answer match the evidence actually used (manual spot-checks + self-check module output)

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

> The downstream task type is TBD; the innovation switches below are defined together with the task
> (likely under a future `downstream.*` config section). The agentic loop ablation is already live.

| Innovation | Compared against | Switch (when the task is defined) |
|---|---|---|
| Citation grounding + faithfulness self-check | No self-check | `downstream.faithfulness_check.enabled` |
| CoT prompting | Direct generation | `downstream.cot.enabled` |
| Agentic loop (orchestration layer) | Single-round parallel | `orchestration.loop.enabled=false` |

Ablation matrix: baseline / +A / +B / +A+B (the assignment requires: add only A, add only B, add both, and report each gain).
Metrics: the ROUGE/BERTScore/RAGAS trio + coverage.

## 9. Running and Reproduction

- Unified entry point `scripts/run_eval.py --config config.yaml --stage <retrieval|review|all>` (generation/ragas stages return with the downstream task)
- Each evaluation run outputs: `data/results/metrics/<stage>_<date>.json` (including a config snapshot + dependency versions)
- When the report cites metrics, it must include the `metrics/*.json` file paths (included in the Dropbox deliverables)
