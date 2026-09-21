# Design (System Architecture + Decision Log)

> This is the system's **authoritative design document**. Any design change goes through a PR and must append a decision-log entry below.
> Corresponding assignments: knowledge base construction (Q1, 20 points), retrieval (Q2/Q3, 40 points), generation (Q4/Q5, 40 points).

## 1. Design Principles

1. **Retriever–Generator contract** (core of the assignment): build the corpus and indices offline, retrieve evidence online, and generation must be conditioned on the evidence and traceable.
2. **Not just assembling off-the-shelf services**: each stage of retrieval and generation is implemented by us and explainable in the report; libraries (Pyserini/SBERT/FAISS/OpenAI client) are allowed, but hosted RAG APIs are not.
3. **Cheap filtering first, expensive judgment last**: cross-encoder prefiltering → only a small candidate set goes to the LLM for review.
4. **Parallelism first**: each sub-query runs independently through the retrieval core; the orchestration layer uses an asyncio producer–consumer pattern so retrieval and review run in parallel.
5. **Contracts first**: all cross-module data follows the formats defined in `docs/SCHEMA.md`; modules do not depend on each other's internals.

## 2. Components and Responsibilities

### 2.1 Knowledge Base Pipeline (`kb/`, Q1: 20 points)

| Stage | Requirements |
|---|---|
| Acquisition | arXiv API (cs.CL/cs.AI/cs.LG metadata) → OpenAlex to supplement citation counts → filter 15k–30k high-impact papers from 2015–2025 → rate-limited PDF download (resumable) |
| Parsing | pymupdf full-text extraction; remove reference sections/headers/footers/formula noise; keep section titles; sample-based quality checks |
| Deduplication | Keep the latest version by `arxiv_id` |
| Chunking | Baseline: section-aware fixed chunking (300–400 tokens, 100-token overlap, chunk prefix "paper title + section name"); ablation: semantic chunking, parent-document retrieval mapping (child→section) |
| Storage | `passages.jsonl` (passage-level) + SQLite (paper-level metadata) |
| Statistics | Document count, chunk count, total word count, type counts (required by Q1 item 3) |

Fallback: S2ORC filtered subset (if mass PDF download/parsing costs get out of control).

### 2.2 Retrieval Core (`retrieval/`, Q2/Q3: 40 points)

```
query + filters
  ├─ sparse: Pyserini (Lucene BM25)               ~10ms
  ├─ dense: BGE-large-en-v1.5 (retrieval instruction prefix on the query side) → FAISS HNSW   ~20–50ms
  └─ RRF fusion (k=60, weights tunable) → top-50 → cross-encoder rerank → top-10
```

- Forbidden: SQL `LIKE`-style text search.
- Interface: `retrieval.pipeline.retrieve(query, filters, top_k) -> list[ScoredPassage]` (unified entry point; runs sparse and dense retrieval in parallel internally).
- Per-stage timing is a hard requirement (used for Q2 latency measurement and the performance section of the report).
- Metadata filtering (year range, categories) must be supported (Q3 innovation) and applied on both the sparse and dense sides.

### 2.3 Orchestration Layer (`orchestration/`, Q3/Q5 innovation) — Paper Search

This module is **Paper Search**: a standalone, task-agnostic capability (Q2/Q3 presentation).
The downstream task (`downstream/`, §2.6) **calls** it via `run_query()` to obtain evidence (D24);
Paper Search itself never generates answers.

```
user input paragraph
  └─► query_understanding: LLM ──► QueryPlan{sub_queries[], keywords[], criteria[]}
        │
        ▼  asyncio multi-path parallel (producer–consumer)
  sub-query i ──► retrieval.pipeline.retrieve ──► dedup & merge ──► review queue
        ▲                                             │
        └── insufficient coverage: rewrite generates supplementary queries ◄── review: LLM judges against criteria ──┘
        │
        ▼  FinalResult (paper candidates + review verdicts + coverage report)
```

- **Review criteria**: fixed rubric items (relevance to the requirement / coverage of input aspects / recency / source credibility / technical depth) + personalized requirements extracted from the input.
- **Review output**: `ReviewRecord{verdict, score 1–5, reasons, covered[], missing[]}`; batched review (5 papers per batch: title + abstract + top 2–3 passages).
- **Termination conditions**: coverage-first (all aspects mentioned in the input are covered) → round cap of 2–3 → at least 5 candidates; the three serve as fallbacks for one another.
- **Cost control**: cross-path dedup, per-identical-input caching, round cap, batched review.
- This is the Q5 "Iterative/Agentic RAG" innovation; it must support ablation (loop disabled = single-round parallel).

### 2.4 LLM Layer (`llm/`)

- **`llm/client.py`**: the single LLM entry point for the whole system (DeepSeek API,
  OpenAI-compatible; `LLM_*` env vars can switch provider) — used by orchestration's own LLM
  calls (query understanding / review / rewriting) and later by the downstream task.
- **`llm/prompt.py`**: prompt templates (query understanding / review / rewriting).
- Task-specific helpers (answer generation, faithfulness self-check) are **parked** — they will be
  defined with the downstream task (`downstream/`) once its type is decided.

### 2.5 Downstream Task (`downstream/`, Q4/Q5: 40 points) — reserved, task type TBD

The package is currently an **empty reserved folder** (task type not yet decided — candidates:
QA with cited answers, summarization, classification over papers). When decided:

```
downstream.run(input_text)
   ├─ 1) orchestration.run_query(input_text)   ← Paper Search: papers + review verdicts + coverage
   ├─ 2) task-specific processing on the evidence (TBD)
   └─ 3) task output
```

- Composition decision already fixed (D24): the downstream task **CALLS Paper Search**.
- When the task is decided, only this package + its evaluation change; the framework stays untouched.
- No circular dependency: `orchestration → llm.client`; `downstream → orchestration + llm`.

### 2.6 Interface (`ui/`, Streamlit, Q2)

- Two tabs (D23): **Paper Search** (`orchestration.run_query()`) and **Downstream**
  (`downstream.run()`, which internally calls Paper Search), sharing the same input box.
- Paper Search tab: paragraph input → understanding results (sub-queries/criteria, editable) → filter conditions (year/category) → paper list (title/year/link/review score and reasons) → expandable hit passages.
- Downstream tab: everything above + the task output (schema TBD with the task).
- Per-stage latency display in both tabs.
- 5 preset example inputs (Q2 requirement: write 5 queries, produce results, measure latency).
- Keep the interface simple (the assignment explicitly does not require a complex UI).

## 3. Data Flow and Public Interfaces

```
kb.store output ──► retrieval builds dual indices ──► orchestration (Paper Search)
                                                          │
                    ┌─────────────────────────────────────┘
                    ▼
              downstream (task output) ──► ui
              orchestration ──► llm.client (LLM infra)
              downstream ──► orchestration + llm (task logic; reserved until the task is defined)
```

Modules depend only on: the data formats in `docs/SCHEMA.md` + the interface contracts in each package's `README.md` + the `retrieve()` signature in `retrieval/pipeline.py`.

## 4. Latency Budget (Q2 Latency-Measurement Targets)

| Stage | Budget |
|---|---|
| Query understanding LLM | 1–2 s |
| Parallel retrieval (incl. rerank, per path) | < 0.5 s |
| Review (5 papers per batch) | 2–4 s / round |
| **Paper Search total** | **3–8 s** |
| Downstream task processing | TBD (depends on the task; generative tasks +2–5 s) |
| **End-to-end (Downstream tab)** | **5–15 s** (single round within budget ~5 s) |

Latency is recorded per stage (`latency_by_stage`: understanding/retrieval/review/rewrite[/task]/total); the report must include a breakdown analysis.

## 5. Evaluation Integration Points

- Component level: retrieval metrics (P@K/R@K/MRR/NDCG), review-decision accuracy — `eval/retrieval_metrics.py`, `eval/review_metrics.py`
- Framework level: coverage, round distribution, end-to-end candidate quality — `eval/`
- Generation level: ROUGE/BERTScore + RAGAS — parked, defined together with the downstream task (Q4; see eval/README.md)
- Evaluation data and results go to `data/eval/` and `data/results/`; formats in SCHEMA.md

## 6. Decision Log (ADR)

> Records key design decisions (date, conclusion, alternatives, rationale). Cite this document in the report.
> Adding a decision: append an entry with consecutive numbering; keep SCHEMA.md / EVALUATION.md in sync.

| # | Date | Decision | Conclusion | Alternatives | Rationale |
|---|---|---|---|---|---|
| D1 | 2026-08-24 | Downstream task | Paper QA (cited answer) | Pure semantic retrieval / survey generation | The retriever–generator contract is the most complete; the 40 generation points are easy to secure; dual-level retrieval granularity also serves the search experience |
| D2 | 2026-08-24 | Corpus source | arXiv cs.CL/cs.AI/cs.LG full text, 15k–30k papers, 2015–2025 | S2ORC / other domains | Open data in abundance, the assignment-recommended source, and full-text passage retrieval works well |
| D3 | 2026-08-24 | Paper filtering | OpenAlex citation-count filter | Pure random sampling | Ensures corpus quality (avoids low-quality papers diluting retrieval/review quality) |
| D4 | 2026-08-24 | Chunking baseline | Section-aware fixed chunking, 300–400 tokens, 100-token overlap | Start with semantic chunking | Get the baseline working first; semantic chunking / parent-document retrieval are Q3 ablations (D21/D22) |
| D5 | 2026-08-24 | Sparse retrieval | Pyserini (Lucene BM25) | rank-bm25 / Elasticsearch | Tool named in the assignment, reproducible; rank-bm25 as a fallback option |
| D6 | 2026-08-24 | Dense retrieval | BGE-large-en-v1.5 (1024-dim) → FAISS HNSW | BGE-M3 / E5 / GTE | Strong on English, mature instruction prefix; BGE-M3 has no advantage for English monolingual retrieval |
| D7 | 2026-08-24 | Fusion | RRF (k=60, weights tunable) | Weighted score fusion | Simple, robust, no training required |
| D8 | 2026-08-24 | Reranking | BGE-reranker-v2-m3, top-50 → top-10 | No reranking | Q3 innovation; cheap prefiltering keeps cost down |
| D9 | 2026-08-25 | Orchestration layer | Agentic: query understanding → multi-path parallel retrieval → LLM review → feedback-based rewriting loop | Single-round parallel | Driven by user requirements; corresponds to the Q5 iterative-RAG innovation; asyncio producer–consumer for parallelism |
| D10 | 2026-08-25 | Review criteria | Fixed rubric items + input-extracted requirements | Fixed-only / extracted-only | Balances consistency and personalization; explainable in the report |
| D11 | 2026-08-25 | Loop termination | Coverage-first, round cap 2–3, ≥5 candidates as fallback | Fixed rounds / quality-gain monitoring | Controllable cost, explainable behavior |
| D12 | 2026-08-24 | Generation model | Cloud API (DeepSeek by default, OpenAI-compatible) | Local Qwen/Llama | Cheap, stable, fast evaluation; A40 can run local models as a fallback |
| D13 | 2026-08-24 | Retrieval granularity | Dual-level: passage-level retrieval + paper-level display (paper score = best passage score) | Passage-only / paper-only | Evidence granularity and search experience in one; evaluate at both levels |
| D14 | 2026-08-24 | UI | Streamlit, all-English | Gradio / FastAPI | The assignment only asks for something simple; lightweight and demo-friendly |
| D15 | 2026-08-24 | Evaluation set | 1,000 items, 3 annotators, IAA≥80% | 2 annotators | The assignment recommends 3; Fleiss' κ is more convincing |
| D16 | 2026-08-24 | Annotation granularity | Paper-level fully annotated + passage-level subset (100–200 items) + gold answer | All passage-level | Paper-level is easier for annotators to judge and cost-effective; passage-level validates evidence precision |
| D17 | 2026-08-25 | Generation evaluation | ROUGE + BERTScore + RAGAS three metrics | EM/F1 only | Abstractive generation relies mainly on ROUGE/BERTScore; RAGAS covers the assignment requirement |
| D18 | 2026-08-25 | Q5 innovations | Citation grounding + faithfulness self-check, CoT, agentic loop | Fine-tuning the generator (stretch) | Explainable, clean ablations; fine-tuning listed as a stretch goal |
| D19 | 2026-08-25 | Cost control | Rerank prefiltering first, review batches of 5, cross-path dedup, per-input caching, round cap | Full LLM review of everything | Optimal latency/cost (cheap filtering → expensive judgment) |
| D20 | 2026-08-25 | Evaluation split | dev/eval = 10%/90% | Single set | Separates tuning from reported metrics to prevent overfitting |
| D21 | 2026-08-25 | Q3 ablation ① | Semantic chunking vs. fixed chunking | — | Chunk self-containedness → retrieval precision |
| D22 | 2026-08-25 | Q3 ablation ② | Parent-document retrieval (child→section) vs. small-chunk return | — | Decouples retrieval precision from context completeness |
| D23 | 2026-08-25 | Presentation | Two tabs: Paper Search (`orchestration.run_query()`, standalone) and Downstream (`downstream.run()`) | Single combined flow | Retrieval (Q2/Q3) and downstream (Q4/Q5) become independently demoable and evaluable; Paper Search is faster and shows the agentic loop clearly |
| D24 | 2026-08-25 | Composition | The downstream task CALLS Paper Search: `downstream.run()` invokes `orchestration.run_query()`, then does task-specific processing | Orchestration generates internally | Keeps Paper Search a standalone, task-agnostic capability; downstream type (QA/summarization/classification) can change without touching the framework; avoids circular dependencies |
| D25 | 2026-09-21 | KB acquisition | arXiv API sliced by (category, year, month), at most 150 records per slice, then OpenAlex citations matched on `locations.landing_page_url` (batched OR filter), then year-stratified top-cited selection | Single arXiv query; OpenAlex DOI lookup; global top-cited | arXiv paging breaks on deep offsets; arXiv DOIs are mostly unresolvable in OpenAlex; a global citation ranking crowds out recent (low-citation) years |
| D26 | 2026-09-21 | Chunk text format | Sentence-packed chunks (350 body tokens, up to 100 tokens of sentence-aligned overlap, tiktoken cl100k_base), `[title] section` prefix embedded in `text`; every passage carries `parent_section_id`, parents stored in `sections.jsonl` | Raw token windows; prefix as separate field | Chunks stay readable for the LLM; BM25 and dense both see title/section; the parent-document ablation needs no KB rebuild |
| D27 | 2026-09-21 | Semantic chunking threshold (Q3 ablation A) | BAAI/bge-small-en-v1.5 sentence embeddings, cut where adjacent cosine < 0.55, chunks clamped to 100-500 tokens | 0.8 (initial guess) | Measured on 2.5k adjacent-sentence pairs: median 0.65, 10th percentile 0.53. At 0.8, ~90% of sentence boundaries became cuts (91-word chunks), which would mix up the ablation with a chunk-size change; 0.55 gives 239 vs 286 mean body tokens for the fixed baseline |
