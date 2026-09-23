# orchestration/ Orchestration Layer (Q3/Q5 innovation: agentic RAG)

## Goal

The user provides **a paragraph of text** → query understanding (sub-queries + keywords + criteria) → **parallel multi-path retrieval** →
**review loop** (unified criteria gate + feedback-based rewriting) → final paper candidate set (with review comments).

Coursework mapping: Q5 "Iterative/Agentic RAG" innovation; also the carrier of the Q3 query-rewriting assignment.

## Processing Flow (fixed)

```
input_text
  └─ query_understanding(input) → QueryPlan{sub_queries[], keywords[], criteria[]}
       └─ optional HyDE → one hypothetical-document retrieval path
       └─ asyncio parallel: each sub-query → retrieval.retrieve() → results enter the review queue
            └─ review.batch_review(candidate batch) → ReviewRecord[] (pass/reject/1-5 score/covered/missing)
                 └─ coverage insufficient? → rewrite(missing aspects) → supplementary sub-queries → re-retrieve (next round)
                      └─ terminate (coverage met / round limit 2–3 / candidates ≥ 5) → FinalResult
```

## Implementation Requirements

### types.py
- `QueryPlan` (SCHEMA.md §4): input_text / sub_queries[{query, keywords, filters}] / criteria[{criterion, source}]
- `ReviewRecord` (§5): paper_id / verdict / score 1–5 / reasons / covered[] / missing[] / round / evidence
- `FinalResult` (§6): query_id / input_text / query_plan / papers[] / coverage_report / latency_by_stage

### query_understanding.py
- Calls llm.client; prompt requirements (concrete templates provided by llm/prompt.py):
  - Output must be JSON (strict schema; retry once on parse failure)
  - Number of sub-queries from config `orchestration.num_sub_queries` (default 3, allowed 1–5)
  - Keywords for BM25; filters extracted from the input (e.g. "after 2023" → year_min=2023)
  - Query planning emits input-specific criteria with source `extracted`; fixed rubric entries are merged only during review
- Failure strategy: LLM output is not JSON → retry → if it still fails, fall back to a single sub-query (the original text)

### HyDE
- `orchestration.hyde.enabled=true` adds one hypothetical academic abstract as an extra retrieval path
- The HyDE call returns plain text and is recorded as `hyde_ms`; a failure skips only this optional path
- The generated path is included in `QueryPlan.sub_queries` so a run remains reproducible

### review.py — Review module (core, D10)
- Fixed rubric items (config `orchestration.review.rubric`, default 5 items):
  relevance to the requirement / coverage of input aspects / recency / source credibility / technical depth
- Input-extracted criteria are merged with the fixed items into the review prompt; each paper outputs verdict/score/reasons/covered/missing
- **Batched** review: 5 papers per batch (config `review.batch_size`); each paper is given
  title + abstract (truncated) + top 2–3 passages (truncated), token budget controlled by config
- Review concurrency is bounded by `review.max_concurrency`; paper metadata is enriched through
  the read-only `kb.store.load_paper()` interface and safely falls back to retrieval evidence
- Deduplication: a paper hit by multiple paths is reviewed only once (by paper_id)
- Outputs a list of ReviewRecord; papers with `verdict="reject"` do not enter the candidate set
- Invalid or unavailable review output is retried once, then fails the query with `OrchestrationError`

### rewrite.py — Feedback-based rewriting
- Input: list of missing aspects (missing[], possibly accumulated across multiple papers)
- Output: 1–2 supplementary sub-queries (LLM), added to the next round of retrieval; round+1
- Constraint: supplementary queries must not duplicate the semantics of existing sub-queries (the LLM instruction carries the existing query list for deduplication)

### loop.py — Loop control (D11)
- asyncio producer-consumer: retrieval tasks run in parallel and enqueue as soon as they finish; review workers consume in parallel
- Termination conditions (stop when any one is met):
  1) Coverage met: all input criteria/aspects are covered (`coverage_report`)
  2) Round limit: config `loop.max_rounds` (default 2–3)
  3) Candidate threshold met: passed papers ≥ config `loop.min_papers` (default 5)
- Ablation switch: `orchestration.loop.enabled=false` → single-round parallel (Q5 ablation baseline)
- Same-input cache (config `loop.cache`): identical input_text returns the cached FinalResult directly
- Latency: record understanding/retrieval/review/rewrite/total (`latency_by_stage`)
- Coverage applies only to input-extracted criteria. Fixed rubric entries are scoring dimensions,
  not content aspects. A started round finishes before a termination condition is evaluated.
- Cache identity is SHA-256 over the whitespace-normalized input plus the settings that can change
  the result (`orchestration.*`, `retrieval.*`, `llm.*`, `kb.chunk` / `kb.store` and
  `TEMPLATE_VERSION`), stored under `data/results/final_results/`. Flipping an ablation switch
  therefore runs the search again instead of returning a result made under other settings.
  Auto-generated query IDs depend on the input only, so they stay stable across settings; an
  explicitly supplied query ID overrides the returned ID while reusing cached content. Writes are
  atomic and failed queries are never cached.

## Interface
```python
run_query(input_text: str, query_id: str | None = None) -> FinalResult
```
- This module is **Paper Search** — a standalone capability (Q2/Q3 presentation).
- The **downstream task** (see `downstream/`) calls this function to obtain evidence,
  then produces its own task output (D24). This module never generates answers itself.
- `OrchestrationError` is exported for unrecoverable retrieval, review, or workflow failures.

## Definition of Done
- Sanity: QueryPlan/ReviewRecord/FinalResult round-trip covered by `scripts/sanity_check.py`
- Smoke: `python -m unittest discover -v` runs the loop end-to-end with deterministic offline dependencies
- Integration: real LLM calls verified via a small end-to-end run
