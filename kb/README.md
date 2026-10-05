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

# Full run, split across the 3 KB members (arXiv rate limits are per IP)
python scripts/download_papers.py --metadata-only          # one person: ~40 min arXiv + OpenAlex
#   -> share data/raw/metadata.csv (Dropbox); every member downloads from the SAME file:
python scripts/download_papers.py --shard 0/3              # member 1 (1/3, 2/3 for the others), ~6 h each
#   -> either collect all PDFs into one data/raw/pdf/, then on that machine:
python scripts/download_papers.py --resume                 # records every pdf_path, retries failures
python scripts/build_kb.py                                 # ~30 min with 4 workers; resumable
python scripts/build_kb.py --stats-only                    # recompute results/corpus_stats.json
#   -> or each member builds their own shard (build_kb.py --rebuild) and the shards are merged:
python scripts/merge_kb_shards.py --out merged/data shard0/data shard1/data shard2/data

# Q3 ablation A (semantic chunking, D21; needs sentence-transformers). Baseline files are left alone
python scripts/build_kb.py --chunk-mode semantic --passages-file passages_semantic.jsonl
```

All shards must be built with the same code and the same pymupdf (pinned in `requirements.txt`, 1.28.2, which
built shard 0): the parser relies on extraction details (dict/rawdict alignment, superscript and italic span
flags, glyph boxes for accent placement) that can change between pymupdf releases.

Budget: ~20k PDFs at 3 s each is about 17 h from one IP, ~30 GB of PDFs. The candidate pool
(~45-55k papers) needs ~1000 OpenAlex calls, which is the whole anonymous daily cap, so set `OPENALEX_API_KEY` for full runs.
Downloads use one connection (`kb.download.max_concurrency: 1`), as arXiv's terms ask. For a
bulk run of this size, arXiv's official bulk PDF access on S3 (requester pays) is the sanctioned
alternative to splitting the download across IPs.

**Failure handling.** Everything is cached and resumable (arXiv slices, OpenAlex lookups, PDFs,
built papers). arXiv/OpenAlex 429s back off for up to ~10 min. If any arXiv slice or OpenAlex
batch still fails, `download_papers.py` exits **before selection** (never selects from partial
data); just rerun it. A failed PDF download or parse is logged (`raw/download_failures.jsonl`,
`raw/parse_failures.jsonl`) and skipped; `build_kb.py --retry-failed` retries parse failures.

### Decisions (report material; see DESIGN.md D25/D26, D30-D34)

| Stage | What we do |
|---|---|
| Fetch | arXiv API per (category, year, month) slice, at most `kb.fetch.per_slice` (150) records each, cached in `raw/arxiv_slices/`; dedup by paper_id (cross-lists), latest version kept |
| Citations | OpenAlex matched on `locations.landing_page_url = http://arxiv.org/abs/<id>` (50 ids/call; arXiv DOIs don't resolve in OpenAlex; https URLs don't match); max over matching works |
| Selection | `citation_min` filter, then a per-year quota (target / #years) by citation count, then leftover slots by global citation rank. Note: "top-cited" is within the sampled pool (150/category/month), not all of arXiv |
| Parse | Span-level pymupdf; cleaning rules R0-R8 in the `kb/parse.py` docstring (contents pages, references cut, headers/footers incl. a thesis chapter's running header (numbered or ALL CAPS, on >= 3 pages, same size as its repeats) and, in a short paper, an author/title line on every odd or every even page, formula/figure/table/mojibake noise, URLs, front matter, statement sections, captions, repair); header/footer zone = top/bottom 12% of the page; thresholds in `kb.parse.*` |
| References | Cut at the reference list only when bibliography entries follow the title (entry labels, "2008." years, page ranges, DOIs, venues, author lists incl. AAAI "Abadi, M.; Agarwal, A.;"), so a table column named "Reference(s)" no longer ends the paper; titles in any style (small bold ACM, inside a block, "References:"); untitled lists ("[1] ...", "[2] ...") too. Books, theses and report compilations with >= 3 lists skip each list instead; a running "Bibliography" header on each page of one list is not a new list. The body font size and numbered-section mode are measured on the text before the list |
| Headings | Bold (incl. TeX bold fonts named only by convention, `CMBX10`)/large + numbered or known heading word; also IEEE small-caps (`II. RELATED WORK`, `A.` subsections nested under Roman sections), Springer ALL-CAPS, ACM numbered bold headings set below body size, SIAM run-in (`2.1. Title.` + text), numbered run-in (`7.2.2 Method. Common ...`), appendix titles (`APPENDIX A`, `Appendix B. Title`; bare-letter `B Proofs` when `B.1` follows or when consecutive letters are set bold in one size); IEEE italic `A.` subsections also when they share a block with the section title or wrap onto a second line, small-caps Roman titles set below body size, `I.` after `H.` is a subsection; bold `1.2 Our Contributions`; bold subsection names starting lowercase (`2.2 word2vec and GloVe`); bold numbered titles up to 14 words with a subtitle colon or in title case, and up to 20 when wrapped onto a second bold line; a footnote mark set as a superscript after a title is not part of the name (`Comparison to Mayr et al.`), an exponent is (`R2`); wrapped titles ending in `?`; lettered run-ins (`A. Zachary's karate club. The ...`) nested below the current section; large unnumbered chapter titles (`INTRODUCTION`) in numbered theses; rejected: one-level numbered list items (after a line ending in ":", before a lowercase run-on sentence, or right after item n-1 / before item n+1, looking past the indented lines of an item, formulas and rows of numbers; they also do not count towards numbered-section mode), body-style lettered options (`A. Absolutely must be false`) outside IEEE papers, lines ending in a relation sign (`X. Assume σ|s| ≥`), long bold list items (`2. Test the model against ..., ...`), sentence fragments, code and algorithm lines (`1 Initialize K = 0`, `4 HIDDEN_SIZE = 20`), captions (`Supplementary Figure 2.`), URLs, ALL-CAPS formulas, name lists, contents entries; subsections named `Parent / Child` (the Abstract and dropped statement sections never act as parents); ALL-CAPS names title-cased with small words in lowercase and words the paper writes with inner capitals spelled as in its text (`Analysis of DNN Inference Time`, `SpectralFormer`) |
| Statements | Acknowledgments, funding, ethics/impact statements, declarations, conflicts of interest, data/code availability (`kb.parse.drop_sections`) are dropped: the first paragraph or two; >= 300 more words before the next heading are kept unnamed (a missed heading). An ethics/impact/reproducibility title set as a numbered subsection with > 400 words (a survey's `6.1.2 Ethical Considerations`) is content and kept. Extending this to every statement title was measured on shard 0 and not done: it would keep one more section in 6,650 PDFs, an appendix `Author Contributions` list, and a text scan of the PDFs finds no numbered data/code availability subsection (D34) |
| Repair | Accents that TeX draws as separate glyphs are attached to the letter they sit on, by glyph position (`Man´e` -> `Mané`, `l̸ =` -> `l ≠`); a large decorative initial (IEEE drop cap) goes back in front of its paragraph (`A` + `ccording` -> `According`); zero-width spaces removed; the space lost after an "ff" ligature restored (`trade-offbetween`); a hyphenated line or column break keeps its hyphen when TeX could not have hyphenated there (`state-of-the-` + `art`, `q-` + `values`), else the paper's own spelling decides, else it stays before a capital/digit (`CIFAR-10`) and between two words the paper uses (`manually-written`) unless the first is a function word/prefix or the second a suffix (`with-out`, `allow-ing` join), else the syllable break is joined (`representation`); an accent left without its letter (unmapped base glyph) is kept in spacing form; suspended hyphens stay (`pre- and post-training`); a paragraph broken by a column or page is joined across the complete captions and ALL-CAPS table titles set between its halves, which then follow it (`the most significant` + `Table 6: ...` + `change in the network`), also across complete `Fig. 2. ...` captions set in body-size text when the other half starts lowercase (so a split word such as `exam-` + `Fig. 2. ...` + `ple` is rejoined); a caption never continues body text; a paragraph ending on a word no sentence ends on (`... as shown in`) runs on into a block starting with a capital (`Figure 2, there is ...`) unless that block starts a new sentence, a list item, a caption or a label; table header cells in a caption's block are dropped |
| Quality | `quality_check`: at least `min_words` words, broken-glyph ratio at most `max_bad_char_ratio`, mean word length 3-9; failures excluded and logged; 50 random papers go to `results/parse_qa_sample.csv` for review (`ok`/`notes`; see the review below) |
| Abstract | The arXiv metadata abstract is inserted as an "Abstract" section when the parser found none; the paper's own copy (with the title and author lines before it) is then removed from a leading unnamed front-matter section, so it is not there twice (`drop_leading_abstract`); a 6-word sequence the copy already had never extends it, so an introduction that repeats a sentence of the abstract keeps it |
| Chunk | Sentence packing to 350 **body** tokens (tiktoken cl100k_base), up to 100 tokens of sentence-aligned overlap, tail < 50 new tokens merged, paragraph breaks kept; prefix `[title] section` embedded as the first line of `text` |
| Semantic (D21) | Cut where the bge-small cosine between adjacent sentences is < 0.55 (calibrated: median 0.65, 10th pct 0.53), chunks clamped to 100-500 tokens; mean 239 vs 286 body tokens for the baseline on the same papers |
| Parents (D22) | Every passage has `parent_section_id`; parent texts in `processed/sections.jsonl` |
| Store | JSONL append + per-paper rewrite (idempotent), torn-line recovery; SQLite upsert touches only the given columns |
| Stats | Words/types over `sections.jsonl` (no overlap, no prefix); lowercase, punctuation-stripped types; the corpus funnel (selected, PDFs, built, failed + reasons) is included |

### Verification (2026-09-21)

- **Diverse sample**: 90 papers from 18 (category, year, month) slices covering cs.CL/AI/LG and
  every year 2015-2025: 89 PDFs (1 has no PDF on arXiv), **89/89 built, 0 parse failures**,
  2,672 chunks, 459k words. Chunk bodies: median 322 tokens (mean 281, max 397; short chunks are
  short sections).
- **Parser regression set**: 119 PDFs (incl. ACL, NeurIPS, ICLR, IEEE, LREC and SIAM-style
  templates, single- and two-column) re-parsed after every rule change; section lists diffed and
  checked by hand.
- **Content audit**: no reference lists left in chunks (after the "Bibliographical References"
  fix), no low-letter-ratio chunks; the table rule removes ~0.6% of words (table columns,
  pseudo-code), checked by reading the dropped blocks.
- **Resume**: a build interrupted after 30 papers and then resumed has no duplicate passages or
  sections, contiguous chunk indices, and `num_chunks` in SQLite matches the passages exactly.
- Unit tests: `python -m unittest tests.test_kb` (offline; synthetic PDF, mocked HTTP).

### Verification of the full shard 0 build (2026-09-28)

Shard 0/3 (6,667 papers): 6,650 PDFs, 6,643 built, 7 rejected by the quality check, 17 without a PDF on
arXiv; 217,295 passages, 100,174 sections, 37.5M words. Hard checks, all passed: ids unique, schema
keys, every `parent_section_id` present, funnel arithmetic, `num_chunks` = passages per paper (0 for a
rejected paper), category/year filter, metadata title on page 1 of 200/200 random PDFs, and 0 passages
with "(cid:", U+FFFD, ligatures, control or zero-width characters or accents split from their letter.

Measured against the first build of the same shard (same measures on both):

| Measure | first build | this build |
|---|---|---|
| Passages with an accent split from its letter | 11,843 in 2,543 papers | 0 |
| Passages with control characters | 1,621 | 0 |
| Tokens spelled against the paper's own majority spelling (line-break hyphens) | 29,480 | 7,555 |
| Missing space after an "ff" ligature ("trade-offbetween") | 591 | 0 |
| Papers with the abstract twice (inserted + front matter) | 81 | 4 |
| Reference-list-like paragraphs (the 13 left are tables citing papers, one caption) | 66 in 28 papers | 13 in 12 papers |
| Section names ending in a number (the 127 left are real headings) | 564 in 123 papers | 127 in 54 papers |
| Words in unnamed sections | 942,697 | 448,065 (1.2%) |

Sentences split by a caption set between their two halves ("the most significant" / "Table 6: ..." /
"change in the network"): 11,985 in 4,517 papers in the 2026-09-27 build, 399 in 204 papers in the first
2026-09-28 build, 85 in 71 papers now (not measured on the first build). Words broken at a line end and glued to a
caption label ("exam-Fig. 2. ..." + "ple"): 115 in 76 papers in the first 2026-09-28 build, 88 in 56 now.

Parse QA sample (50 random papers, `results/parse_qa_sample.csv`, `ok`/`notes` filled in): reviewed by
Claude (an AI assistant), not by a person. Each paper's sections were compared with its PDF (the PDF
outline where there is one, else the heading lines and page images), every PDF text line was searched
in the parsed text, and parsed text not in the PDF was listed: 31 yes / 18 minor / 1 no (the
2026-09-27 build: 19 / 18 / 13). The one "no" (1502.03581) is a PDF that stores each line in two or
three pieces. The errors found were fixed as general rules (DESIGN.md D32, D33), not per paper, and checked
on two more random samples of 800 papers each.

One paper that passed before is now rejected (1701.03571): the parser keeps three more of its
paragraphs, whose inline math comes out as runs of single letters, and its broken-character share
rose from 0.0144 to 0.0205, over `max_bad_char_ratio` (0.02). A `--rebuild` now first sets every paper's
`num_chunks` to 0 and counts them again as papers are built (before, a paper that failed, lost its PDF or was
left out by `--limit` kept the previous build's count).

Fixes from the review of PR #12 (DESIGN.md D34), measured on a full rebuild against the build the PR was opened
with: 2 of 6,643 papers changed, both keeping text they lost with the abstract copy before (1804.10390: the
first 163 words of its introduction; 2403.09718: its introduction's first sentence and a 9-word keyword line);
the reference-list fix changed no paper of this shard; the 50 QA papers are unchanged. Indexes built from an earlier output of this parser (`scripts/build_indexes.py`) must be rebuilt.

Shard 1/3 (built on 2026-10-02 by a second member with the same code, pymupdf 1.28.2 and tiktoken 0.14.0, on
Python 3.14): 6,652 PDFs (15 not on arXiv), 6,649 built, 3 rejected by the quality check, 218,900 passages,
100,491 sections; its metadata.csv is the shared selection with PDFs only in its own rows. The same hard checks
pass, 20 of its PDFs re-parsed on the machine that built shard 0 give byte-identical passages and sections, and
no character in its text is treated differently by Python 3.11 and 3.14 (letter class, category, NFKC).
Merged with shard 0 (`merge_kb_shards.py`): 13,292 papers, 436,195 passages, 200,665 sections, all checks pass.

How the rules were checked: every paper whose section names changed between successive full builds
was listed and reviewed (311, then 116); random samples of 500-900 papers were re-parsed after each
rule change and diffed; hyphenation was scored on 27,239 real line breaks from 300 papers against the
corpus-wide spelling (93.5% -> 99.2%); accent repair on 2,890 broken words from 300 papers.

### Interfaces for other modules

- `kb.store.iter_passages(path)`: build indexes (retrieval)
- `kb.store.load_paper(paper_id, sqlite_path)`: metadata + abstract, categories as a list (orchestration review)
- `kb.store.load_parents(ids, sections_path)` / `iter_parents(path)`: parent-document mode
- `num_chunks = 0` in papers.sqlite means the paper has no passages (no PDF or parse failure)

## Definition of Done (see README: Definition of Done)
