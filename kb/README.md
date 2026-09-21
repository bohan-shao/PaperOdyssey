# kb/ Knowledge base construction (Assignment Q1, 20 points)

## Goal

Full text of 15k–30k arXiv papers (cs.CL/cs.AI/cs.LG, 2015–2025) → cleaning → chunking → storage.
Meet the assignment's hard requirements: ≥10,000 documents, ≥100,000 words, and a statistics report (Q1 item 3).

## Implementation Requirements

### fetch.py — Fetching
- Pull metadata from the arXiv API (title/abstract/categories/year/arxiv_id), with paged rate limiting (`tenacity` retries)
- Use the OpenAlex API (free, no key required) to fill in `citation_count` by arxiv_id
- Filtering: years 2015–2025 + categories ∈ {cs.CL, cs.AI, cs.LG} + citation-count threshold (config-adjustable) → target 15k–30k papers
- PDF download: rate limiting (respect arXiv terms and rate limits), resumable download, retry on failure, save to `raw/pdf/`
- Output: `data/raw/metadata.csv` (paper metadata, including pdf_path)
- Fallback: S2ORC (HuggingFace datasets) — if bulk PDF download/parsing gets out of hand, explain in the report

### parse.py — Parsing and Cleaning
- Extract full text with pymupdf; preserve the section-title hierarchy
- Cleaning rules (needed for the report; rules must be enumerable):
  - Remove the reference list (truncate at the end of the body after "References")
  - Remove headers/footers, arXiv template remnants, pure-formula noise lines, URLs
  - Keep: body text, figure/table captions (`Figure N:`/`Table N:` lines can be kept or dropped per config)
- QA: manually inspect parse quality on a sample of 50 papers; report the parse failure rate (disclose in the report)

### chunk.py — Chunking
- Baseline (D4): **section-aware fixed-size chunking** — split by section boundaries first, 300–400 tokens per chunk, 100 tokens of overlap between adjacent chunks; chunk prefix format `[Title] Section name` (put it inside `text` or as a separate field — pick one and document it)
- Ablation A (D21): semantic chunking — cut where sentence-level embedding similarity drops sharply (config: `similarity_threshold`)
- Ablation B (D22): parent-document mapping — `parent_section_id` (child→section), see SCHEMA.md §2
- Constraints: chunk text must be directly readable by an LLM (clean, self-contained); `chunk_index` must be monotonic; never cross paper boundaries

### store.py — Storage
- `passages.jsonl` append-only writes (resumable); `papers.sqlite` (table structure in SCHEMA.md §3)
- Idempotent: rerunning with the same paper_id must not produce duplicate passages (upsert, or delete-then-insert)
- Provide read interfaces (needed by retrieval to build indices): `iter_passages()`, `load_paper(paper_id)`

### stats.py — Statistics (Q1 item 3, must deliver)
- Output: document count, chunk count, total word count (token or word; state the convention), type count (unique words)
- Output `data/results/corpus_stats.json` + print a report table

## Interface Constraints

- All paths/parameters come from `config.yaml` (the `kb.*` section); hardcoding is forbidden
- Write only under `data/raw|processed/`; do not touch other modules' directories
- Download/parse/chunk script entry points live in `scripts/` (see scripts/README.md)
- Validate the pipeline on a small sample first (`--limit 100`); sanity checks live in `scripts/sanity_check.py`

## Implementation Notes (status: implemented)

### How to run

```bash
pip install -r requirements.txt        # kb needs pymupdf, tiktoken, httpx, tenacity, pandas
cp .env.example .env                   # optional: OPENALEX_API_KEY, KB_CONTACT_EMAIL

# Trial (~3 min): 25 papers end to end
python scripts/download_papers.py --limit 25
python scripts/build_kb.py

# Full run — split across the 3 KB members (arXiv rate limits are per IP)
python scripts/download_papers.py --metadata-only          # one person: ~40 min arXiv + OpenAlex
#   -> share data/raw/metadata.csv (Dropbox); every member downloads from the SAME file:
python scripts/download_papers.py --shard 0/3              # member 1 (1/3, 2/3 for the others), ~6 h each
#   -> collect all PDFs into one data/raw/pdf/, then on that machine:
python scripts/download_papers.py --resume                 # records every pdf_path, retries failures
python scripts/build_kb.py                                 # ~30 min with 4 workers; resumable
python scripts/build_kb.py --stats-only                    # recompute results/corpus_stats.json

# Q3 ablation A (semantic chunking, D21; needs sentence-transformers) — baseline files untouched
python scripts/build_kb.py --chunk-mode semantic --passages-file passages_semantic.jsonl
```

Budget: ~20k PDFs × 3 s ≈ 17 h from one IP, ~30 GB of PDFs. The candidate pool (~45–55k papers)
needs ~1000 OpenAlex calls — the anonymous daily cap — so set `OPENALEX_API_KEY` for full runs.

**Failure handling.** Everything is cached and resumable (arXiv slices, OpenAlex lookups, PDFs,
built papers). arXiv/OpenAlex 429s back off for up to ~10 min. If any arXiv slice or OpenAlex
batch still fails, `download_papers.py` exits **before selection** (never selects from partial
data) — just rerun it. A failed PDF download or parse is logged (`raw/download_failures.jsonl`,
`raw/parse_failures.jsonl`) and skipped; `build_kb.py --retry-failed` retries parse failures.

### Decisions (report material; see DESIGN.md D25/D26)

| Stage | What we do |
|---|---|
| Fetch | arXiv API per (category, year, month) slice, ≤ `kb.fetch.per_slice` (150) records each, cached in `raw/arxiv_slices/`; dedup by paper_id (cross-lists), latest version kept |
| Citations | OpenAlex matched on `locations.landing_page_url = http://arxiv.org/abs/<id>` (50 ids/call; arXiv DOIs don't resolve in OpenAlex; https URLs don't match); max over matching works |
| Selection | `citation_min` filter → per-year quota (target / #years) by citation count → leftover slots by global citation rank. Note: "top-cited" is within the sampled pool (150/category/month), not all of arXiv |
| Parse | Span-level pymupdf; cleaning rules R1–R8 in the `kb/parse.py` docstring (references cut, headers/footers, formula/figure/table noise, URLs, front matter, acknowledgments, captions, repair); thresholds in `kb.parse.*` |
| Headings | Bold/large + numbered or known heading word; also IEEE small-caps (`II. RELATED WORK`), Springer ALL-CAPS, SIAM run-in (`2.1. Title.` + text); subsections named `Parent / Child` |
| Quality | `quality_check`: ≥ `min_words` words, broken-glyph ratio ≤ `max_bad_char_ratio`, mean word length 3–9; failures excluded and logged; 50 random papers → `results/parse_qa_sample.csv` for manual review (fill `ok`/`notes`) |
| Abstract | The arXiv metadata abstract is inserted as an "Abstract" section when the parser found none |
| Chunk | Sentence packing to 350 **body** tokens (tiktoken cl100k_base), ≤100-token sentence-aligned overlap, tail < 50 new tokens merged, paragraph breaks kept; prefix `[title] section` embedded as the first line of `text` |
| Semantic (D21) | bge-small adjacent-sentence cosine < 0.55 → boundary (calibrated: median 0.65, 10th pct 0.53), chunks clamped to 100–500 tokens; mean 239 vs 286 body tokens for the baseline on the same papers |
| Parents (D22) | Every passage has `parent_section_id`; parent texts in `processed/sections.jsonl` |
| Store | JSONL append + per-paper rewrite (idempotent), torn-line recovery; SQLite upsert touches only the given columns |
| Stats | Words/types over `sections.jsonl` (no overlap, no prefix); lowercase, punctuation-stripped types; the corpus funnel (selected → PDFs → built/failed + reasons) is included |

### Verification (2026-09-21)

- **Diverse sample**: 90 papers from 18 (category, year, month) slices covering cs.CL/AI/LG and
  every year 2015–2025 → 89 PDFs (1 has no PDF on arXiv) → **89/89 built, 0 parse failures**,
  2,672 chunks, 459k words. Chunk bodies: median 322 tokens (mean 281, max 397; short chunks are
  short sections).
- **Parser regression set**: 119 PDFs (incl. ACL, NeurIPS, ICLR, IEEE, LREC and SIAM-style
  templates, single- and two-column) re-parsed after every rule change; section lists diffed and
  checked by hand.
- **Content audit**: no reference lists left in chunks (after the "Bibliographical References"
  fix), no low-letter-ratio chunks; the table rule removes ~0.6% of words (table columns,
  pseudo-code), checked by reading the dropped blocks.
- **Resume**: interrupted build (30 papers) + resume → no duplicate passages/sections, contiguous
  chunk indices, `num_chunks` in SQLite matches passages exactly.
- Unit tests: `python -m unittest tests.test_kb` (offline; synthetic PDF, mocked HTTP).

### Interfaces for other modules

- `kb.store.iter_passages(path)` — build indexes (retrieval)
- `kb.store.load_paper(paper_id, sqlite_path)` — metadata + abstract, categories as a list (orchestration review)
- `kb.store.load_parents(ids, sections_path)` / `iter_parents(path)` — parent-document mode
- `num_chunks = 0` in papers.sqlite means the paper has no passages (no PDF or parse failure)

## Definition of Done (see README: Definition of Done)
