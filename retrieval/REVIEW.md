# Retrieval review verification (2026-10-24)

Scope: the 14 leads in the supplied `pr_review_checklist.md`, verified against `903c9ac`.
The initial tracked worktree was clean; the supplied checklist and reference corpus were
untracked and remain untouched. Remote refs were fetched: `origin/main` is `55cef08`.
Its independent KB changes were inspected, not merged or committed. D36/D37 are reproduced
from that commit for numbering compatibility; retrieval uses D38 and review corrections D40.
Dense + sparse hybrid retrieval and all public retrieval fields/signatures remain intact.

Merge integration on 2026-10-26 brings in main `64e3ef9`, including the KB and evaluation
implementations. The review decision was renumbered from D39 to D40 to retain the upstream
evaluation workflow's D39. The verification results below describe the original 2026-10-24 audit.

## Item-by-item disposition

| Item | Evidence and action |
|---|---|
| 1. Index destination deletion | Confirmed: `indices` collapsed to the shared root in both sparse and dense. A shared path helper rejects roots, relative escapes, input-containing directories, links, unknown artifacts and missing ownership manifests. Both builders validate before construction and replacement. Direct absolute destinations and valid legacy relative paths still work. The related lookup builder now rejects input/output or input/temporary-file collisions. |
| 2. Dotted lexical terms | Confirmed in preprocessing: dots were stripped before the backend analyzer saw the query. Preserve dotted names/versions and underscores; leave backend analysis in charge. Lucene's StandardTokenizer rules preserve decimal and dotted-letter terms. Added a real Pyserini analyzer parity test, skipped locally because Java/Pyserini are unavailable; see source verification below. |
| 3. Single-channel budget | Confirmed: diversity-off sparse/dense-only runs could send 150 pairs to a 50-pair budget. Enforce `fusion.top_k` before scoring for every channel combination; final `top_k` remains a separate cap. Regression covers sparse/dense/hybrid, diversity on/off, rerank on/off, and empty/nonempty pools. |
| 4. SQLite category aliases | Confirmed: casefold alone differed from the other backends. Build and query use `category_token`; duplicate normalized aliases are deduplicated. SQLite format v2 refuses v1 indexes with an explicit rebuild message instead of silently returning incorrect filtered results. |
| 5. Query-only dense changes | Confirmed: documents use an empty prompt. Query instruction is retained as build provenance but no longer invalidates document vectors. Model, dimension, document prompt and effective token limit remain checked. Old unprompted manifests remain supported. |
| 6. Probe mode/overwrites | Confirmed: sparse + rerank used the sparse-only filename. Four explicit mode labels plus unique UTC/run IDs now distinguish reports, sample indexes, queries and progress files, including repeat runs on the same day. |
| 7. ADR collision | Confirmed against fetched main: D36/D37 already exist. Renumbered retrieval D36 to D38 throughout repository documentation and added D39 for this review. No unrelated KB merge or history rewrite. |
| 8. Template bias | Confirmed by paired diagnostics below. Generate queries independently of tuned stopwords; run raw terms and the request template separately. Headline metrics and latency use raw terms. Historical template-only results remain labeled as confounded diagnostics. |
| 9. Setup timing/lifecycle | Confirmed: old dense retained a closed lookup reference and models were reused. Drop both retrievers and close every referenced lookup between rows; retain models explicitly for warm experiments, release them on exit, and reject preloaded components at entry. Rename `cold_start_ms` to `row_setup_ms` and describe cache/model reuse. |
| 10. Encoder configuration | Confirmed: direct construction of another model inherited dense max length, instruction and device. Apply those settings only to the configured model; other models use their own defaults. Cap configured lengths at the tokenizer/model ceiling, warn on capping, and reject nonpositive limits. |
| 11. Parent fallback | Confirmed: 20 of 9,994 sampled passages cannot be located in long parents. Preserve the child, emit structured fallback reasons, accept whitespace-only anchors, and reject ambiguous anchors instead of choosing the first occurrence. Parent latency rows count expanded/unchanged/no-parent output. No approximate matching invents surrounding evidence. |
| 12. SQLite serialization | Confirmed: independent read-only connections were serialized by a shared lock. Remove the lock; concurrent-query regression verifies overlapping execution and identical results. Local backend measurements below support this change. Rebuild remains an offline operation with readers stopped. |
| 13. Rerank batch default | The reduction from 64 to 16 had no reference-A40 benchmark. Restore 64 to preserve the existing baseline. CPU probes explicitly retain their separate batch size 16; no GPU speed claim is made. |
| 14. Type documentation | Confirmed stale docstrings. `total_candidates` means the selected pool before reranking/final truncation, even when reranking is disabled; parent text may be an excerpt or unchanged child. Updated types, Schema, package docs and the latency consumer's documented meaning. No public fields changed. |

The downstream test fixture also kept SQLite connections open between subtests before
replacing their database on Windows. Closing those fixture-owned readers fixes the two
baseline errors without altering production QA behavior. Empty lookup requests no longer
open a connection unnecessarily.

For item 2, the repository uses Anserini's default analyzer. Its
[official implementation](https://github.com/castorini/anserini/blob/master/src/main/java/io/anserini/analysis/DefaultEnglishAnalyzer.java)
uses StandardTokenizer followed by possessive/lowercase/stopword/stemming filters. Lucene's
[tokenizer rules](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/analysis/standard/StandardTokenizerImpl.jflex)
keep internal decimal/dotted-letter punctuation. The optional runtime test checks the installed
backend directly rather than assuming every Lucene version behaves identically.

## Diagnostics and limits

Paired sparse probe: seed 42, 300 complete papers, 9,994 passages, 60 source queries, each
with raw and template forms. Seven ablation rows share the same sample and queries. The fixed
generation vocabulary differs from the historical generator; compare rows within this run.

| Query form | Legacy source MRR@10 | Query processing only | Source recall@10, both rows |
|---|---:|---:|---:|
| Raw terms | 0.9750 | 0.9750 | 0.9833 |
| Request template | 0.9313 | 0.9750 | 0.9833 |

This supports removal of template noise, not a general gain on natural user questions.
The source labels remain weak and incomplete. Artifact:
`data/results/metrics/ablation_retrieval_probe_sparse_2026-10-24T152206_4eb310eb.json`.

Real cached CPU MiniLM hybrid + reranker smoke run: 12 papers, four paired source queries,
all seven rows completed. All raw-source recall/MRR values are 1.0; this tiny sample checks
integration and lifecycle, not comparative quality or latency (timings were noisy).
Artifact: `data/results/metrics/ablation_retrieval_probe_hybrid_rerank_2026-10-24T152348_3c35ae59.json`.

The previously mislabeled sparse + rerank mode also completed all seven rows with real cached
MiniLM reranking (four papers, one paired query). It wrote a separate `sparse_rerank` artifact:
`data/results/metrics/ablation_retrieval_probe_sparse_rerank_2026-10-24T153120_866f6f58.json`.

SQLite backend throughput: same existing 9,994-passage index and 60 queries, no metadata
filter, top 50, three repeats, median wall time. These exclude hydration, model inference and
orchestration, and are local warm-cache measurements, not production throughput promises.

| Concurrent readers | Shared-lock QPS | Independent-connection QPS |
|---:|---:|---:|
| 1 | 117.2 | 122.0 |
| 4 | 116.7 | 441.7 |
| 8 | 117.1 | 589.7 |

Artifacts: `data/results/review_sqlite_before.json` and
`data/results/review_sqlite_parent_after.json`. Parent scan at 6,000 characters found 7,777
full parents, 2,197 exact windows and 20 unmatched fallbacks (0.20% of all sampled passages).
Whitespace matching did not repair those 20 content mismatches; they deliberately retain the
child and are now observable. Whitespace and ambiguous-anchor cases are covered synthetically.

## Validation and follow-up

- `scripts/sanity_check.py`: passes, including all public schema round trips.
- Targeted retrieval/latency/downstream tests: 118 run, 110 pass, eight skip (seven Pyserini
  cases including the new analyzer test, plus uncached BGE reranker).
- Full discovery: 137 run, five errors, eight skip. All five remaining errors are missing
  local dependencies (`httpx2`, `tenacity`, `pandas`, `streamlit`), including the
  downstream integration test's import of the HTTP test transport. The two Windows SQLite
  fixture errors from the baseline are resolved.
- PyCharm build reports success with limited diagnostic collection. Inspected new helper,
  query, SQLite and probe files have no IDE errors; SQL data-source and private experiment
  lifecycle access warnings remain. Runtime tests are the meaningful check.
- Logs: `data/results/review_baseline_tests.log`, `review_targeted_tests.log`,
  `review_full_tests.log`, `review_hybrid_smoke.log`. Experiment outputs remain gitignored.

Before a production run, rebuild SQLite v1 indexes with the configured paths. Existing
compatible FAISS indexes do not need rebuilding for query instruction changes. Restart
processes after configuration/index changes. Repeat Pyserini tests in the Java reference
environment, install the full project dependencies for the full suite, and measure BGE/A40
batch sizes and human-labeled relevance separately. No models or dependencies were installed
for this review, and no runtime code depends on the temporary reference corpus directory.
