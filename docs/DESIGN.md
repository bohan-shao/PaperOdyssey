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
  ├─ lexical query view → sparse: Pyserini BM25 top-150 (SQLite FTS5 alternative)
  ├─ semantic query → BGE-large-en-v1.5 → FAISS HNSW top-150
  └─ full RRF union → paper/section selection to 50 → cross-encoder → diverse top-10
```

- Forbidden: SQL `LIKE`-style text search.
- Interface: `retrieval.pipeline.retrieve(query, filters, top_k) -> RetrieveResult` (unified entry point; runs sparse and dense retrieval in parallel internally).
- Per-stage timing is a hard requirement (used for Q2 latency measurement and the performance section of the report).
- Metadata filtering (year range, categories) must be supported (Q3 innovation) and applied on both the sparse and dense sides.
- D38 keeps BGE as the production model; Qwen3 embeddings have a model-specific instruction
  adapter for explicit experiments. Parent expansion is bounded around the matched child.

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
  calls (query understanding / review / rewriting) and downstream answer generation.
- **`llm/prompt.py`**: prompt templates (query understanding / review / rewriting).
- Answer prompts are defined in `downstream/prompts.py`; QA makes one generation call
  and validates citation numbers in code, without an additional model self-check.

### 2.5 Downstream Task (`downstream/`, Q4/Q5) — cited paper QA

The task is simple question answering from retrieved paper evidence (D29):

```
downstream.run(input_text)
   ├─ 1) orchestration.run_query(input_text)   ← Paper Search: papers + review verdicts + coverage
   ├─ 2) lookup store .get_passages(...)       ← original passages named in review evidence (D28)
   └─ 3) llm.client.chat_json(...)             → answer + numbered passage citations
```

- Composition decision already fixed (D24): the downstream task **CALLS Paper Search**.
- Defaults: 5 passing papers by review score, 2 passages per paper, 2,400 characters per
  passage, 1,024 answer tokens; all configured under `downstream.*`.
- Return `DownstreamResult` (SCHEMA §10), including all model context and actual citations.
  No usable evidence produces an explicit insufficient-evidence response. Grouped citations
  (`[1, 2]`, `[1-3]`) are rewritten as `[1][2]`; operational failures and invalid citation
  output (unknown sources, no citation) raise `DownstreamError`.
- No forced comparison table, evidence matrix, or extra model verification pass.
- No circular dependency: `orchestration → llm.client`; `downstream → orchestration + retrieval.lookup + llm`.

### 2.6 Interface (`ui/`, Streamlit, Q2)

- Two tabs: **Paper QA** and **Paper Search**, sharing one question input.
- **Ask** calls `downstream.run()` and populates both tabs using the same search result;
  **Search papers only** calls `orchestration.run_query()` without generating an answer.
- QA displays the answer, expandable citation excerpts, arXiv links, timing, and a JSON
  download. Search displays reviewed papers, matched passages, coverage, and stage timings.
- Editable query plans, separate filter controls, and live round streaming remain deferred;
  explicit year/category requirements can still be expressed in the question.
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
              downstream ──► orchestration + retrieval.lookup + llm (cited answer generation)
```

Modules depend only on: the data formats in `docs/SCHEMA.md` + the interface contracts in each package's `README.md` + the `retrieve()` signature in `retrieval/pipeline.py`.

## 4. Latency Budget (Q2 Latency-Measurement Targets)

| Stage | Budget |
|---|---|
| Query understanding LLM | 1–2 s |
| Parallel retrieval (incl. rerank, per path) | < 0.5 s |
| Review (5 papers per batch) | 2–4 s / round |
| **Paper Search total** | **3–8 s** |
| Downstream task processing | Evidence loading + one generation call; +2–5 s target, not yet measured live |
| **End-to-end (Downstream tab)** | **5–15 s** (single round within budget ~5 s) |

Latency is recorded per stage (`latency_by_stage`: understanding/retrieval/review/rewrite[/task]/total); the report must include a breakdown analysis.

## 5. Evaluation Integration Points

- Component level: retrieval metrics (P@K/R@K/MRR/NDCG), review-decision accuracy — `eval/retrieval_metrics.py`, `eval/review_metrics.py`
- Framework level: coverage, round distribution, end-to-end candidate quality — `eval/`
- Generation level: ROUGE/BERTScore + RAGAS — planned evaluation of the QA result, not yet implemented (see eval/README.md)
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
| D28 | 2026-09-28 | Passage lookup store | SQLite `data/indices/lookup.sqlite` (passages + sections tables keyed by id), built by `build_indexes.py` from the same passages file as BM25/FAISS; sparse and dense hydrate hits through it | In-memory dict from `passages.jsonl`; Pyserini stored raw docs; `kb.store.load_parents` scans | ~600k passages ≈ 1 GB of text is too heavy to hold in the UI process; stored docs would tie dense to the BM25 index and break the rank-bm25 fallback; `load_parents` scans `sections.jsonl` on every call |
| D29 | 2026-09-26 | Downstream task and minimal UI | Simple cited paper QA: Paper Search, stored passage evidence (read from the D28 lookup store), one answer-generation call; shared QA/search UI | Cross-paper comparison matrices / multi-pass verification / long surveys | Keep the question-to-answer flow simple; citations remain inspectable and evidence is retained for later evaluation. Generation metrics and a live end-to-end run on the built indexes remain pending. |
| D30 | 2026-09-27 | KB reference cut and headings | Cut at "References" only when bibliography entries follow (entry labels, years, page ranges, venues, author lists); >= 3 separate lists (chapters, report compilations) are skipped one by one, a running "Bibliography" header is not a new list; one-level numbered body-style lines are not headings when they are list items (after "...:", before a lowercase run-on, next to the adjacent number), code or captions; TeX bold fonts named only by convention ("CMBX10") count as bold | Cut at the first "Reference" line; no validation; bold = PDF bold flag or "bold" in the font name | A table column named "Reference" cut papers in half, and running headers made one thesis bibliography look like many chapter lists, so it stayed in the text; list items and algorithm lines ("1 Initialize K = 0") mislabelled the sections after them; CMBX headings at body size (about 5% of papers) were missed |
| D31 | 2026-09-27 | KB text repair | Accents drawn as separate glyphs are attached to the letter under them by glyph position; a large drop-cap initial goes back in front of its paragraph; a caption continued in the next block stays a caption; a hyphenated line break keeps the hyphen per TeX's hyphenation limits, then the paper's own spelling, then "both halves are words of the paper" (not a function word / prefix + word, not a suffix), else joins; the space lost after an "ff" ligature is restored; when the metadata abstract is inserted, the paper's own copy is removed from leading front matter | Always drop the hyphen; an English word list (web2) | The accent split ("J ́egou") affected 2,543 papers; always joining broke 6.5% of line-break hyphens (27,239 checked against the corpus spelling: "stateof-the-art", "selfattention"); the rule gets 99.2%. web2 would add little and is not on every machine, and shards must build identically; 69 papers had their abstract twice |
| D32 | 2026-09-28 | KB fixes from the 50-paper parse review | Captions and ALL-CAPS table titles set between the two halves of a paragraph broken by a column/page are moved after the joined paragraph (only complete captions are passed over; a caption never continues body text); a paragraph ending on a word no sentence ends on ("as shown in") runs on into a block starting with a capital unless that block starts a new sentence, list item, caption or label; appendix titles ("APPENDIX A", "Appendix B. Title", bold letter sequences "A ...", "B ..."); IEEE italic "A." subsections (also sharing a block or wrapped), small-caps Roman titles, "I." after "H." is a subsection; numbered lines ending in a relation sign are math; body-style lettered options are headings only in IEEE papers; bold "Our ..." titles; bold subsection names may start lowercase; long bold titles (<= 14 words) with a subtitle colon or in title case; list-item neighbours seen past indented item lines and rows of numbers; an ethics/impact title as a numbered subsection of > 400 words is content; title-cased names keep the paper's own spelling of words with inner capitals ("DNN", "SpectralFormer"); header/footer zone 12% of the page | Keep build H's rules | The review found 13 of 50 papers with section errors or lost text and 18 with small defects; the caption split alone affected 4,517 of 6,644 papers (11,985 broken sentences). Each rule was checked on two independent 800-paper samples: every paper whose section structure changed was read (about 100 per sample), 25 random caption joins were read (22 right, 3 were already wrong before) |
| D33 | 2026-09-28 | KB fixes for what the first 2026-09-28 build left | A complete "Fig. N." caption set in body-size text is passed over like a "Figure N:" caption when a paragraph of text ending mid-sentence comes before the run of captions and a paragraph starting lowercase after it; an unfinished one is never passed over (body text can start with "Table 3. Overall, ..."). In a short paper a margin line of >= 2 words on every odd or every even page after the first is a running header. A footnote mark set as a superscript after a section title is not part of the name (an exponent after a variable, "R2", is). A numbered bold heading wrapped onto a second bold line keeps up to 20 words (was 14) | Keep the first 2026-09-28 build's rules | Sentences split by a caption: 399 in 204 papers -> 85 in 71; words broken at a line end and glued to a caption label ("exam-Fig." + "ple"): 115 in 76 papers -> 88 in 56. Checked on two 800-paper samples, every changed paper read (43); this found two errors in a first version (body text starting "Table 3." passed over; the 2 of R² dropped), both fixed and in the unit tests. Header rule: 87 candidate lines in the 1,822 PDFs of 3-9 pages read; one-word lines (table cells such as "Hindi") excluded. Full build: 200 of 6,643 papers changed, every class of change read: 146 captions moved after a rejoined sentence, 29 split words rejoined, 16 titles completed, 15 running headers removed, 11 footnote marks removed from names; the 50 reviewed papers keep their verdicts |
| D34 | 2026-09-28 | KB fixes from the review of PR #12 | A one-line numbered reference entry ("1. Smith, J.: ...") is tested as a list item before it can end a reference list; when the paper's own copy of the abstract is removed, a 6-word sequence the copy already had never extends it (the reviewer's fix for the 150-word bridge over a figure, applied to the 25-word gap too); `--rebuild` first sets every `num_chunks` to 0; an error in the abstract cleanup keeps the parsed text instead of stopping the build; drop-cap candidates are looked up per page; pymupdf pinned to 1.28.2. The > 400-word exception of R6 stays limited to ethics/impact/reproducibility titles | Keep the rules of 0771162; extend the R6 exception to every statement title | Both bugs reproduced with the reviewer's scripts, which are now unit tests (they fail on 0771162). Full rebuild against the build the PR was opened with: the reviewer's two fixes changed 1 of 6,643 papers (1804.10390 keeps the first 141 words of its introduction); reading it showed its introduction's first sentence, which repeats the abstract's right after the copy, was still removed through the 25-word gap, so the rule was applied there too (unit test added). With that, 2 papers changed: 1804.10390 keeps 163 words, 2403.09718 its introduction's first sentence and a 9-word keyword line; the reference-list fix changed no paper of shard 0; the 50 QA papers are unchanged. R6: extending the exception changes 1 of 6,650 PDFs and would keep an appendix "Author Contributions" list; a text scan finds 3 numbered statement subsections in shard 0 (conflict of interest, acknowledgements, ethical considerations), none for data or code availability. Drop caps: 2.2 s -> 0.02 s on a synthetic 200-page PDF with 7,200 single-capital lines, same output. pymupdf: the parser relies on dict/rawdict alignment, span flags and glyph boxes, and shards must build identically (D31) |
| D35 | 2026-10-12 | KB build split across members | Each KB member downloads and builds their own third of the shared selection (`download_papers.py --shard I/3`, `build_kb.py --rebuild`, same commit and pymupdf), and `kb/merge.py` merges the three builds: lines copied in shard order (CRLF from a build on Windows becomes LF), every shard's `pdf_path`, `num_chunks` recounted, shard 0's reviewed QA sample, the chunking settings the shards recorded. It refuses before writing anything when the selection differs, a paper has a PDF in two shards, a shard holds papers without a PDF in its own metadata (a stale folder), the chunking differs, or a passages/sections file is missing | Collect every PDF (~20k, about 48 GB at shard 0's mean of 2.4 MB per PDF) on one machine and build there | Each member already holds their third of the PDFs, so only the build outputs travel. The shards build identically: every member sends 20 random PDFs, re-parsed on the machine that built shard 0 (shards 1 and 2: 20/20 byte-identical passages and sections). Ablation A (D21) does not need the PDFs again: section_fixed passages rebuilt from `sections.jsonl` alone match the build for 1,500 random papers (500 per shard), so semantic chunks can be made from the merged `sections.jsonl` on one machine |
| D36 | 2026-10-12 | Ablation A passages from the parsed sections | `build_kb.py --from-sections` (`kb/rechunk.py`) chunks the parsed sections stored in `sections.jsonl` instead of the PDFs, so `passages_semantic.jsonl` is built on one machine from the merged KB (D35). Semantic chunking embeds the sentences of 200 papers per encoder call (`chunk.chunk_semantic_batch`; each section is still cut on its own), on Apple MPS | Every member builds semantic passages for their own shard from their PDFs, and the shards are merged again | No PDFs and no work by the other members. Checked on the merged KB: section_fixed passages rebuilt this way equal the build's `passages.jsonl` byte for byte (all 658,869). On 200 random papers the semantic chunks are identical with MPS and CPU and with 200 papers or 1 paper per encoder call (200/200); MPS took 80 s and CPU 283 s for them |
| D37 | 2026-10-12 | Sentence splitter keeps a closing bracket or quote | `split_sentences` matches a closing quote or bracket after the stop in a lookbehind, so it stays with its sentence ("(see Fig. 3.) Next" kept its ")"); abbreviations are judged on the word before it, so the split decisions do not change. The merged KB's `passages.jsonl` and `num_chunks` were rebuilt from `sections.jsonl` (D36) | Keep the shared passages and list the loss as a known issue | The old pattern took the closer as part of the gap between sentences: 3,950 characters lost in 2,895 of 302,135 sections. On every section the fixed splitter gives the same sentences except for the restored characters (3,931 sentences changed), and every section's sentences now hold exactly its words (before: 1% of sections lost one). Rebuilt passages: 4,179 only gain the characters, 26 move a chunk or overlap edge by one sentence, 2 papers get one more passage (658,871 in all); the hard checks pass |
| D38 | 2026-10-13 | Shard-aware hybrid retrieval | Keep BGE + BM25 + RRF + cross-encoder; separate normalized semantic/lexical queries, recall 150 per channel, select 50 across papers/sections before rerank, return diverse top-10; bounded child-anchored parent excerpts; optional disk-backed SQLite FTS5 with independently switchable field weights; model-specific BGE/Qwen3/plain instructions and dense manifests; active CLI config and relocatable index roots | Replace both models with Qwen3 immediately; keep fixed top-50 passage truncation; load all fallback BM25 postings in Python | Shard 0 has 217,294 passages from 6,643 papers, up to 482 passages per paper and 45,870 words per section. Candidate crowding and unbounded parents are measurable structural issues. CPU MiniLM and lexical probes are reproducible diagnostics, not evidence of BGE/Qwen superiority or human relevance. Keep production BGE and field weighting disabled pending labeled evaluation. Public retrieval fields/signature unchanged; see retrieval/README.md and EVALUATION.md |
| D39 | 2026-10-25 | Evaluation set from LLM-drafted questions | For each of N papers sampled with the config seed (built, abstract >= 200 characters) the LLM writes one search request from the title and abstract (one sentence of 12-25 words asking one thing, no title copying, no names the paper introduces, no results; a question for 60% of the papers and a search request such as "Papers on ..." for 40%, by a hash of the paper id); all 3 annotators mark alone whether the paper is a good answer (y/n); a question most of them accept enters eval_set.jsonl with its paper as the only gold paper; a rejected one is rewritten by a person and marked again by all 3. Agreement (percentage + Fleiss' kappa) is over these marks. Retrieval is scored by Hit@K (K = 1, 3, 5, 10), with P/R/MRR/NDCG. (D38 is left to PR #16.) | Pooled judging: the 3 annotators mark the top results of several retrieval runs as relevant or not (TREC pooling); people write all 1,000 questions | Agreed by the team on 2026-10-24: pooled judging of ~5 papers x 1,000 questions x 3 people is too much work before 2026-11-07, and writing 1,000 questions by hand is slower and less uniform. Another relevant paper in the top K earns nothing, so the scores are a lower bound; the drafting rules keep a question specific to its paper. Prompt q3 replaced q1 after the 50-question pilots of 2026-10-26: about 9 of 50 q1 questions asked two or three things or stated the paper's results; on 10 of those papers q2 fixed them with deepseek-v4.1-flash, and deepseek-v4-pro with q2 was about as good (it reused title words such as "ternary") at ~12x the cost per question; with q2, 45 of 50 items were how-questions, so q3 asks for a search request for 40% of the papers |
| D40 | 2026-10-24 | Retrieval review correctness and reproducibility | Validate dedicated index destinations before build/replacement; cap single-channel and hybrid selected pools equally; preserve dotted lexical tokens; normalize SQLite categories with a versioned index; query instructions do not invalidate unprompted document embeddings; isolate direct encoder configuration; paired raw/template probes with unique artifacts and honest setup timing; observable parent fallback; independent SQLite read connections | Treat every review suggestion as a redesign; force dense rebuilds for query-only changes; call reused model timings cold starts | Confirmed by targeted offline regressions and small local diagnostics. Restore rerank batch size 64 pending reference-GPU measurements. Retrieval fields and signatures remain unchanged; SQLite v1 requires a sparse rebuild. See retrieval/REVIEW.md for item-by-item evidence and limits |
| D41 | 2026-10-28 | Final PR #16 correctness and experiment controls | Separate readable index aliases from safe replacement checks; resolve ancestor aliases and publish with rollback; validate string stopwords; pin legacy_50_v2 before loading components and save effective inputs; skip ineffective sparse-only depth and add a diversity-controlled depth comparison; use canonical paper metrics with separate passage diversity | Reject every ancestor link; inherit live innovation settings in baseline rows; report a no-op as a depth ablation; duplicate metric formulas | Fixes verified path/YAML failures and evaluation confounds without changing retrieval contracts or model defaults. Also retain nonpositive-IDF lexical matches, drain failed parallel requests, and expose actual rerank sweep sizes. See retrieval/REVIEW.md for regressions, local diagnostics and remaining platform/dependency limits |
