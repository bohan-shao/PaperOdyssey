# retrieval/ Retrieval core (Assignment Q2/Q3, 40 points)

## Goal

Each sub-query goes through: `BM25 ∥ BGE dense → RRF → cross-encoder reranking → top-10`.
Supports metadata filtering (year/category). The four ablation configurations are
toggleable (EVALUATION.md §2).

## Implementation Requirements

### types.py — Type contracts (read SCHEMA.md first)
- `ScoredPassage`: passage_id / paper_id / text / score / rank / source stage / metadata (title/year/categories)
- `RetrievalFilters`: year_min / year_max / categories
- `RetrieveResult`: hits + latency_ms + per-stage breakdown (needed for Q2 timing and report performance)

### lookup.py — Lookup store (D28)
- `data/indices/lookup.sqlite` (config `retrieval.lookup.db_file`), built by `scripts/build_indexes.py`
  from the same `passages.jsonl` + `sections.jsonl` as the indexes; rebuild = temp file + atomic replace
- Tables `passages` (SCHEMA.md §2 columns, keyed by `passage_id`) and `sections` (keyed by `parent_section_id`)
- `get_lookup_store().hydrate([(passage_id, score), ...], stage)` → `ScoredPassage` list, ranks 1..n;
  sparse and dense both return hits through this
- `with_parent_text(hits)`: parent-document mode, swaps in the section text, `passage_id` keeps the child
- Primary-key lookups only (not retrieval); an id missing from the store raises (indexes and store out of sync)
- Verified on the local KB sample (1,997 passages): build 0.03 s, 7 MB; hydrating 50 hits + 10 parent swaps ≈ 0.3 ms

### sparse.py — Sparse (D5)
- Build a Lucene BM25 index with Pyserini (`data/indices/bm25/`); indexed fields: text + metadata
  (year/categories as filter fields)
- Query: BM25 scoring + filters (add year/category constraints to the Lucene query)
- Return top-k (default 50, before reranking) with raw BM25 scores
- Fallback: if Java is unavailable → rank-bm25 (already listed in `requirements.txt`), note it in the report
- Forbidden: SQL-LIKE retrieval

Implementation notes:
- Index layout: `data/indices/bm25/meta.json` (backend, doc count) + `lucene/` (Pyserini) or
  `rankbm25.pkl`; opening an index built by the other backend raises
- Pyserini: Anserini's default English analyzer (Porter, stopwords), BM25 `k1=0.9, b=0.4`
  (config `retrieval.sparse.bm25`); queries go through Anserini's bag-of-words generator, so
  punctuation in LLM-written sub-queries cannot break query parsing
- Filters are Lucene FILTER clauses on the extra `year` / `categories` fields: they restrict the
  candidates and leave scores unchanged. Categories are indexed as `cs_cl`-style tokens, because
  the analyzer splits `q-bio.NC` at the hyphen and stems `eess.AS` to `eess.a`
- rank-bm25: lowercase alphanumeric tokens, no stemming/stopwords (scores differ from Lucene);
  the pickle is held in memory and every query scores all passages in Python, so it is only
  practical on small corpora
- Verified on the local KB sample (1,997 passages): Pyserini build 6.5 s, 444 KB;
  warm top-50 query ≈ 2.6 ms including lookup hydration (first query ≈ 50 ms, JVM warm-up)

### dense.py — Dense (D6)
- Encoding model: BGE-large-en-v1.5 (sentence-transformers); the model name is config-replaceable
- **The query side must prepend the instruction prefix**: "Represent this sentence for searching relevant passages: "
  (BGE retrieval convention; omitting it significantly hurts scores)
- Index: FAISS HNSW (M=32, efConstruction=200, efSearch config-adjustable),
  vectors saved under `data/indices/faiss/`; the passage_id ↔ vector-row mapping is stored separately (JSON/parquet)
- Encoding: batched (config `retrieval.dense.batch_size`); fp16 available on A40; record total encoding time
- Query: encode the query (same model, same prefix) → FAISS search → cosine similarity scores

Implementation notes:
- Index layout: `data/indices/faiss/index.faiss` (IndexHNSWFlat, inner product on unit vectors, so
  score = cosine; the file also holds the float32 vectors), `mapping.jsonl` (row → passage_id,
  year, categories), `meta.json` (model, dim, HNSW params, query instruction, faiss version);
  opening an index built with another model or dim raises
- Build streams passages in chunks of 16k (encode → add), so texts are never all in memory;
  encoding and HNSW insertion times are reported separately in the build report
- Filters use a FAISS ID selector during the search (same semantics as sparse, never a post-cut):
  selections of ≤ `retrieval.dense.filter_exact_max` passages (default 50k) are searched exactly
  over the stored vectors, because HNSW with a selective filter returns too few hits (in a check
  keeping 3 of 100 vectors it found only 1); larger selections use HNSW + selector with an exact
  fallback if it comes back short
- The query encoder is behind a lock (HF fast tokenizers are not thread-safe); the retriever
  encodes a warm-up query on open so the first CUDA call is not in the measured latency
- Verified on the local KB sample (1,997 passages, RTX 5070, fp16): encoding 11.7 s (~170/s,
  so roughly 1-1.5 h for the full ~800k), index 8.9 MB; warm top-50 query ≈ 15 ms including
  query encoding and hydration

### fusion.py — Hybrid fusion (D7)
- RRF: `score = Σ 1/(k + rank_i)`, k=60 (config-adjustable); sparse and dense each feed in their top-50
- Support a weighted variant (config `fusion.weights`) as an ablation comparison
- Output the sorted, deduplicated top-50 (cross-path merging happens here; see pipeline)

### rerank.py — Reranking (D8)
- BGE-reranker-v2-m3 cross-encoder; top-50 → top-10
- Scoring pairs: (query, passage.text); output the reranked list + rerank scores
- Record latency (~100–300ms for 50 pairs on A40)

### pipeline.py — Unified entry point (interface discipline)
```python
def retrieve(query: str, filters: RetrievalFilters | None = None,
             top_k: int = 10) -> RetrieveResult
```
- Internally: run sparse/dense in parallel (threads/asyncio) → fusion → (optional) rerank
- All toggles come from config: `retrieval.{sparse,dense,fusion,rerank}.enabled` (ablation matrix)
- Must record per-stage latency (sparse_ms/dense_ms/fusion_ms/rerank_ms/total_ms)
- The orchestration layer may only call `retrieve()`; evaluation scripts may only call it too (to keep ablations fair)

## Ablation Checklist (EVALUATION.md §2–§3)
1. Four-level matrix: bm25 / dense / hybrid / hybrid+rerank
2. Parent-document retrieval toggle: `retrieval.parent_doc.enabled` (on a child hit, return the parent text; SCHEMA.md §2)
3. Metadata filtering toggle: `retrieval.filters.enabled`
4. HyDE is implemented in the orchestration layer (this package only gets called)

## Definition of Done (see README: Definition of Done)
- Sanity: ScoredPassage/RetrieveResult round-trip covered by `scripts/sanity_check.py`
- Smoke: sparse/dense/fusion/rerank each runnable end-to-end on a small corpus (offline)
- Integration: real index build and query verified with small `--limit` runs
