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

## Definition of Done (see README: Definition of Done)
