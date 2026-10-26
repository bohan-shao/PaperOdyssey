# retrieval/ Hybrid retrieval (Q2/Q3)

`retrieval.pipeline.retrieve(query, filters=None, top_k=None) -> RetrieveResult` remains the
only retrieval entry point for orchestration, downstream and quality experiments. The default
still combines sparse and dense retrieval. D38 changes candidate handling, not public fields
or signatures. D40 corrects verified review findings; see [REVIEW.md](REVIEW.md) for evidence,
regression coverage and remaining environment limits. Contracts: `docs/SCHEMA.md`;
decisions: `docs/DESIGN.md`.

## Shard 0 findings (2026-10-13)

The complete supplied KB contains 6,650 metadata records, of which **6,643 papers** have
**217,294 passages** in **100,174 parent sections**, covering 2015-2025. Counts came from
streaming the complete JSONL files and querying the metadata database. The supplied files
were copied to configured `data/processed/` for local experiments. The temporary reference
folder is not used by runtime code, committed config, or tests.

| Distribution | Median | p90 | p99 | Maximum |
|---|---:|---:|---:|---:|
| Passages per paper | 28 | 51 | 108 | 482 |
| Passages per section | 1 | 4 | 10 | 330 |
| Words per passage, including prefix | 239 | 287 | 314 | 368 |
| Words per parent section | 280 | 799 | 1,781 | 45,870 |

All passages carry the `[title] section` prefix. Categories overlap: cs.LG appears on 129,994
passages, cs.AI on 97,121, cs.CL on 73,852, and cs.CV on 32,179. This is a mixed AI/ML/NLP
collection. Introductions account for 23,327 passages; abstracts for 6,814. Scientific acronyms,
model names, hyphenated terms and numbers should survive query processing. Word statistics
are not model-token statistics.

The previous flow was BM25 top-50 + BGE top-50 -> RRF top-50 -> BGE reranker top-10:

- Repeated title matches and overlapping chunks from long papers occupy many candidate slots.
  Passage-ID deduplication alone cannot improve paper coverage. Truncation before considering
  paper/section membership loses alternative evidence before reranking.
- Both channels received the same query. The rank-bm25 fallback repeats query tokens and
  includes stopwords; long polite/natural-language queries can change lexical ranks.
- Every dense model received a BGE prefix, even after selecting another model. Compatibility
  checks covered model/dimension, but not the instruction or document token limit.
- Whole parent sections were returned without a bound, including sections tens of thousands
  of words long. Head truncation could hide the child that caused retrieval.
- rank-bm25 holds the corpus in Python and scans every document per query. Serialized k1/b
  values also ignored later config changes. Pyserini remains appropriate in Docker.
- A custom CLI config did not reliably govern later no-argument `get_config()` calls. Index
  paths were based on the parent of `paths.indices`, potentially defeating relocation.

## Current architecture

```text
query + optional metadata filters
  |-- normalized semantic query -> BGE encoder -> FAISS HNSW top-150
  `-- deduplicated lexical terms -> BM25 top-150
                parallel; filters applied before top-k
                         |
              full weighted RRF union (k=60)
                         |
          paper/section selection -> at most 50 candidates
                         |
           BGE cross-encoder -> diversity selection -> top-10
                         |
             optional child-anchored parent excerpt
```

`query_processing.enabled` normalizes Unicode/dashes/whitespace, deduplicates lexical terms
and removes configured stopwords. Dense/reranker retain the complete normalized question,
including negation. No automatic acronym expansion, LLM call or implicit filter is introduced.

`diversity.enabled` applies soft quotas: before rerank, four passages per paper and two per
parent; finally, two per paper and one per parent. Deferred passages fill spare slots for
single-paper queries. Exact repeated body text within a paper is removed. Selection preserves
scores and relative relevance order, then assigns contiguous ranks. Different papers are not
necessarily relevant papers. The full RRF union reaches selection before the top-50 cut.
Channel recall depth and rerank budget are independent. `top_k` is an upper bound and cannot
return more than the available selected pool. Single-channel ablations also apply selection.

For the legacy baseline disable query processing/diversity, set both channel depths and
fusion depth to 50, and rerank batch size to 64. RRF weights remain `[1, 1]`: no uncalibrated
sum of BM25/cosine scores. Zero-weight channels contribute no candidates; ties are stable.

## Components and choices

| Module | Responsibilities |
|---|---|
| `query.py` | Pure semantic/lexical query views and known-prefix removal |
| `sparse.py` | Pyserini (default), SQLite or rank-bm25; no silent backend fallback |
| `sqlite_bm25.py` | Contentless disk-backed FTS5 index over title/section/body; exact metadata filters before LIMIT |
| `dense.py` | Unit float32 vectors, FAISS HNSW, model/document-prompt/token-limit validation |
| `fusion.py` | Full weighted RRF union with stable ties and finite nonnegative weights |
| `selection.py` | Paper/section quotas and exact body deduplication before/after reranking |
| `rerank.py` | CrossEncoder sigmoid scoring, serialized inference, reference batch size 64 |
| `lookup.py` | Primary-key hydration, missing-ID errors, bounded parent excerpts |
| `pipeline.py` | Parallel sparse/dense, sequential cold loading, selected-pool count, stage timings |

Pyserini uses Porter/stopwords and configurable k1=0.9, b=0.4. rank-bm25 retains its lowercase
alphanumeric analyzer and applies configured k1/b when loaded, including old pickles. SQLite
uses Porter + unicode61 and **fixed k1=1.2, b=0.75**; its scores differ from Lucene/rank-bm25.
FTS5 negative scores are negated for higher-is-better output. Retrieval uses the inverted
index `MATCH`, not SQL LIKE. See the [SQLite BM25 specification](https://www.sqlite.org/fts5.html#the_bm25_function).

Optional SQLite `sparse.field_weighting.enabled` weights title/section/body `[2, 0.5, 1]`.
The known prefix is removed before indexing the body, avoiding duplicate title weight.
The switch defaults **off** pending labeled evaluation; enabling it on another backend raises.
SQLite is a scalable no-Java alternative; rank-bm25 remains useful for small baseline samples.
Pyserini remains the production default until a labeled backend comparison supports a switch.
All backends normalize category aliases such as `cs.CL`, `cs_CL`, and `cs CL` identically.
SQLite format v2 applies that normalization at build and search time; rebuild v1 SQLite indexes.
Each SQLite query owns a read-only connection, allowing concurrent readers without a shared
query lock. Stop online readers before an offline index replacement.

Lexical query processing preserves dots and underscores within technical names/versions
(`GPT-3.5`, `Llama 3.1`, `cs.CL`), leaving final tokenization to the selected backend. It must
not split a Lucene index term into separate query terms.

**Production models remain BGE-large-en-v1.5 + bge-reranker-v2-m3.** Changing models without
measuring this corpus would not establish a gain. The machine has an 8 GiB RTX 4070 Laptop
GPU, but the installed probe runtime is CPU-only. Qwen3-Embedding-0.6B is supported as an
optional experiment through SentenceTransformer with model-specific query instructions:

```yaml
# Change these fields in a full copy of config.yaml; use a separate dense index.
retrieval:
  dense:
    model_name: Qwen/Qwen3-Embedding-0.6B
    query_format: qwen3
    query_instruction: null
    task_instruction: "Given a scientific question, retrieve relevant passages from research papers"
    max_length: 1024
    batch_size: 8
    faiss_dir: faiss_qwen3_06b
```

Qwen queries use `Instruct: ...\nQuery: ...`; documents have no instruction. Auto mode recognizes
BGE/Qwen3 model names; plain mode supports MiniLM. Local paths should specify `query_format`
or `query_instruction`. An explicit empty SentenceTransformer prompt prevents accidental
additional model prompts. The manifest records the query instruction for provenance; query-only
instruction/format/task changes do not require rebuilding unprompted document vectors. Model,
document prompt and effective document token limit must match. Direct `Encoder(other_model)`
calls retain that model's defaults instead of inheriting unrelated dense settings; configured
token limits are capped at the model/tokenizer limit. Mapping row order is validated. Qwen3 requires transformers
>=4.51.0. See the [official embedding card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
Instruction/manifest behavior is covered by offline tests; Qwen weights were not downloaded
or evaluated in this change. At 1,024 float32 dimensions, Shard 0 alone needs about 0.83 GiB
for dense vectors before HNSW graph overhead. A model swap at the same dimension does not
reduce this index footprint.

Qwen3-Reranker is **not** a drop-in sequence-classification CrossEncoder: its official adapter
uses causal yes/no token logits. Passing its model name to this wrapper raises an actionable
error rather than using an untrained classification head. Implement and measure that adapter
separately before selecting it. See the [official reranker card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B).

Dense filtering retains exact search for <=50,000 eligible vectors, and fallback when filtered
ANN returns too few results. HNSW remains M=32, efConstruction=200, per-query efSearch >= k.
Models serialize tokenizer/inference access; concurrent orchestration paths can still queue
on the reranker. Timings include query/selection, both channels, fusion, rerank, parent and total.

## Storage and compatibility

Inputs use `paths.processed` plus `kb.store.*` filenames. Sparse/dense subdirectories resolve
under `paths.indices`; defaults are `bm25` and `faiss`. Absolute paths and legacy
`indices/bm25` / `indices/faiss` spellings work. Lookup remains
`paths.indices/retrieval.lookup.db_file`. The selected CLI config is reused by package calls.
Index destinations must be dedicated directories: the shared root (`indices`, `.`, or an
absolute root), relative escapes, links, input-containing directories, and directories with
foreign artifacts or no index manifest are rejected before a build/replacement. Lookup output
and its temporary file cannot overwrite the KB inputs. Separate absolute index locations remain supported.
Restart after model/index/backend changes; online singletons are not hot-reloaded.

Layouts: `lookup.sqlite`; sparse `meta.json` + `lucene/`, `rankbm25.pkl`, or `bm25.sqlite`;
dense `index.faiss`, `mapping.jsonl`, `meta.json`. Legacy BGE indexes with 512-token settings
remain readable. Changing model/document token limit/backend requires rebuilding that index.
Candidate/query-view changes do not. Compare in separate directories and close readers before
replacing indexes on Windows. Build with `python scripts/build_indexes.py --config <full-config.yaml>`.

Parent mode defaults off. When on, return up to 6,000 characters around the matched child,
keeping the title prefix. Whitespace-only differences can be anchored without changing the
parent's original text. Unlocatable, ambiguous or oversized children remain unchanged and
emit structured warnings (`parent_excerpt_outcome`, `passage_id`); `max_chars: 0`
restores whole sections. ID/score/rank still identify the child; reranking always sees the child.
Downstream QA reloads original children by ID, so parent mode currently changes retrieval/review
context, not QA's evidence-loading contract.

## Historical small-scale validation (2026-10-13)

The seeded probe uses 300 complete papers (9,994 passages, 4,635 sections), 60 queries and
seed 42. Queries alternate partial title and abstract terms. Source labels are weak and
non-exhaustive: this diagnoses source finding/candidate crowding, not general relevance or
answer quality. Source metrics deduplicate papers within the first ten returned passages.
Every historical query also prepended `Find papers on`, which the configured stopword list
removes. The query-only MRR gain below is confounded by that template and must not be claimed
as a gain on natural user queries. The revised paired diagnostic is summarized in [REVIEW.md](REVIEW.md).

Sparse-only results:

| Variant | Source recall@10 | Source MRR@10 | Distinct papers@10 | Warm p50 ms |
|---|---:|---:|---:|---:|
| Legacy rank-bm25 | 0.9833 | 0.9456 | 3.45 | 36.0 |
| Query processing only | 0.9833 | 0.9750 | 3.37 | 31.0 |
| Diversity only | 0.9833 | 0.9456 | 7.70 | 37.0 |
| Combined, rank-bm25 | 0.9833 | 0.9750 | 7.53 | 34.5 |
| Combined, SQLite | 0.9833 | 0.9722 | 7.48 | 11.0 |
| SQLite + field weighting | 0.9833 | 0.9833 | 7.37 | 12.0 |

Per-query records and the isolated depth row:
`data/results/metrics/ablation_retrieval_probe_sparse_2026-10-13.json`.
Full corpus profile: `data/results/metrics/shard0_profile.json`. Artifacts are gitignored.

Real MiniLM dense + sparse + MiniLM cross-encoder on the same sample and queries:

| Variant | Source recall@10 | Source MRR@10 | Distinct papers@10 | Warm p50 / p95 ms |
|---|---:|---:|---:|---:|
| Legacy budgets/selection | 0.9833 | 0.9833 | 3.07 | 1,710.5 / 2,199.6 |
| Diversity only | 0.9833 | 0.9833 | 7.28 | 1,696.5 / 1,938.1 |
| Combined, rank-bm25 | 1.0000 | 0.9861 | 7.38 | 1,689.0 / 2,434.5 |
| Combined, SQLite | 1.0000 | 0.9861 | 7.32 | 1,645.5 / 3,458.7 |

Report: `data/results/metrics/ablation_retrieval_probe_hybrid_rerank_2026-10-13.json`.
The field-weighted row also has recall 1.0 and MRR 0.9861: no additional source gain after
reranking. It stays disabled by default. Query-only/depth-only rows do not recover the
one missed source; the combined candidate changes do. `PROBE0057` (abstract terms about
spam classifiers) previously misses source `1502.03581`; the combined run includes its
abstract as passage 9, distinct-paper rank 6. This is one diagnostic recovery, not a
statistically established relevance improvement.

Median latency is similar, but combined p95 is worse and CPU reranking remains the bottleneck;
there is no general speedup claim. Keep the rerank budget at 50 and measure GPU tails before
raising it. Dense sample build: 193.9 s encoding, 0.4 s HNSW insertion, 19.1 MB total index.
These MiniLM figures must not be extrapolated to Qwen/BGE as measured costs. Final validation:
sanity config/import/schema checks pass; 88 retrieval/latency tests ran, 81 passed and seven
optional Pyserini/BGE-reranker tests skipped.

```bash
python scripts/probe_retrieval.py --papers 300 --queries 60
python scripts/probe_retrieval.py --papers 300 --queries 60 --hybrid --rerank
python scripts/sanity_check.py
python -m unittest tests.test_retrieval tests.test_retrieval_redesign tests.test_latency
```

The probe makes no LLM/API calls. Neural mode uses real cached MiniLM embedding/cross-encoder
models on CPU; it cannot establish BGE/Qwen superiority. Reports preserve stage latency,
row setup, versions, corpus hash, queries and config snapshots. `row_setup_ms` includes loading
the row's indexes and a warm-up request; models/OS caches can be reused, so it is not comparable
process cold-start latency. New runs use unique directories and four distinct mode names
(`sparse`, `sparse_rerank`, `hybrid`, `hybrid_rerank`). Each source query runs as raw terms and
with a request template; main metrics/latencies use raw terms and `by_query_style` reports
the template separately. The query generator's vocabulary is independent of retrieval tuning.
Set `HF_HUB_OFFLINE=1` for
cached-model runs. Tests never download models. See `docs/EVALUATION.md` for the ablation
protocol and separate human-label requirement.

The historical full repository test attempt also encountered missing non-retrieval dependencies
(`httpx2`, `tenacity`, `pandas`, `streamlit`) and a downstream fixture that replaces an open
SQLite database on Windows. It is not reported as a clean full-suite pass. Detailed output:
`data/results/tests_retrieval_redesign.log`. Pyserini/BGE-reranker live tests require the
optional Java/model environment; the local retrieval tests skip those explicitly.

## Next measurements

Keep field weighting off and BGE unchanged until annotated dev data supports switching.
Compare Qwen3-0.6B embedding vs BGE with separate indexes, fixed queries and equal budgets;
measure passage/paper relevance, truncation, p95 latency and peak VRAM. Sweep channel depths
50/100/150 and rerank budgets 30/50. A paper-level title/abstract route is a next experiment
if title queries still retrieve incidental mentions. Do not tune on the held-out evaluation set.
