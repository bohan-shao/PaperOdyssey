# PaperOdyssey — AI6127 Paper-QA RAG System

> Course assignment: AI6127 Deep Neural Networks for NLP (2026). Deadline **2026-11-07 23:59 SGT**, submit via Blackboard.
> Status (2026-11-07): the knowledge base, retrieval core, Paper Search, cited Paper QA, Streamlit UI and evaluation
> tooling are implemented and were evaluated on the full knowledge base (19,941 arXiv papers, 658,871 passages).
> The knowledge base, the evaluation datasets and every run behind the report's numbers (retrieved results, review
> verdicts, generated answers, metrics, latency, examples and UI screenshots) are in the data archive linked from the
> report (`paperodyssey_data_2026-11-07.zip`; its `README.md` maps each folder to the report).
> To install and run the system, see [How to Run](#how-to-run).

## What the System Does

Two presentation tabs over the same agentic retrieval core (D23/D24):

**Tab 1 — Paper Search** (`orchestration.run_query()`): a standalone, task-agnostic capability.
The user enters **a research requirement** (a paragraph) and gets relevant papers with review verdicts:

1. **Query understanding**: an LLM splits the input into multiple natural-language sub-queries plus BM25 keywords, and extracts the user's criteria
2. **Parallel retrieval**: the sub-queries run in parallel through the retrieval core (BM25 ∥ dense → RRF → rerank)
3. **Review loop**: a review LLM judges each paper against a **fixed rubric + input-extracted criteria** (pass/reject, 1–5 score, reasons, covered/missing aspects); if coverage is insufficient, feedback-based rewriting triggers supplementary retrieval (at most 2–3 rounds)

**Paper QA** (`downstream.run()`): **calls Paper Search**, reads the retrieved paper passages,
and generates a concise answer with numbered evidence citations. Expand a citation to read
the exact excerpt and open the paper on arXiv. Insufficient evidence is stated explicitly.
The UI shares one question box and offers **Ask** and **Search papers only** actions; an Ask
request populates both tabs from the same search. See [downstream/README.md](downstream/README.md).

## Architecture Overview

```
Offline build:
  arXiv metadata+PDF ─► pymupdf parse/clean ─► section-aware chunking ─► passages.jsonl + SQLite
        │                                        │
        ▼                                        ▼
  Pyserini BM25 index                     BGE-large-en-v1.5 → FAISS HNSW

Online query:
  user input paragraph
    └─► [query understanding LLM] ──► sub-queries + keywords + criteria
          │
          ▼  asyncio multi-path parallel
    sub-query1 ─► retrieval core (BM25∥dense→RRF→rerank top-10) ─┐
    sub-query2 ─► retrieval core ─────────────────────────────┼─► dedup ─► [review queue]
    sub-query3 ─► retrieval core ─────────────────────────────┘      │
          ◄─── insufficient coverage? [feedback rewriting LLM] ── [review LLM] judges per criteria
          (coverage met / 2–3 rounds / ≥5 papers)                   │
                                                                   ▼
     final candidate set (with review notes) ─► [downstream task] task output ─► Streamlit UI
```

Detailed design and decision log: see [docs/DESIGN.md](docs/DESIGN.md).

The D38 retriever keeps dense+sparse search, broadens each channel to 150, selects 50 candidates
across papers/sections, and reranks to ten diverse passages. Corpus analysis, switches,
SQLite BM25 (no Java), Qwen3 embedding configuration, and the reproducible small-sample probe
are documented in [retrieval/README.md](retrieval/README.md). `get_config(path)` selects the
active configuration for later package calls; relative index subdirectories now resolve under
`paths.indices` (legacy `indices/bm25` / `indices/faiss` spellings still work). Restart the process
after changing model/index configuration so loaded components are replaced consistently.

## Repository Structure

| Directory | Responsibility |
|---|---|
| `kb/` | Knowledge base construction: acquisition/parsing/cleaning/chunking/storage/statistics (assignment Q1, 20 points) |
| `retrieval/` | Retrieval core: BM25/dense/RRF/rerank/unified entry point (assignment Q2/Q3, 40 points) |
| `orchestration/` | Orchestration layer (Paper Search): query understanding/parallel retrieval/review/rewriting loop (agentic RAG, Q3/Q5 innovation) |
| `downstream/` | Cited paper QA: calls Paper Search, reads evidence, generates an answer (Q4/Q5) |
| `llm/` | Shared LLM client and orchestration prompts; answer prompts live in `downstream/` |
| `eval/` | Retrieval/review/answer evaluation (RAGAS-style answer metrics) and annotation tooling |
| `ui/` | Streamlit interface (Q2) |
| `scripts/` | One-click scripts: download/build index/run evaluation/sanity checks |
| `tests/` | Offline unit tests (no network, no API calls) |
| `docs/` | Architecture, schema, conventions, evaluation plan, decision records |
| `data/` | Data and indices, not in git (the knowledge base is on Google Drive, see [How to Run](#3-download-the-knowledge-base)); the fixed `eval/eval_set.jsonl` dataset is versioned |
| `config.yaml` | Every setting and ablation switch; read only through `config.py` |

## How to Run

PaperOdyssey is pure Python, so there is nothing to compile. Installing the dependencies may build
Pyserini's JVM bridge (pyjnius) from source, which needs a C compiler and JDK 21; the Docker image
has both. Running the system takes five steps: install the environment, set an LLM key, download the
knowledge base, build the indices, then start the UI or the evaluation scripts.

### Requirements

| | |
|---|---|
| Machine | Linux x86_64, or any host with Docker. The reported runs used Rocky Linux 9.5 on NTU's TC2 cluster with an NVIDIA A40 |
| Python | 3.12 (Pyserini ≥ 2.0 requires it; the code itself supports ≥ 3.11) |
| Java | JDK 21 for the Pyserini BM25 index (included in the Docker image) |
| GPU | Recommended. Encoding the passages for the dense index took 52 minutes on an A40 and is much slower on CPU; searching also runs on CPU, more slowly |
| Memory | 16 GB RAM (an evaluation job running four questions at once peaked at about 7 GB) |
| Disk | 2.1 GB knowledge base, 5.5 GB indices, about 4 GB model weights (downloaded on first use), and 10 GB more for the Docker image |
| LLM | A key for an OpenAI-compatible chat API (step 2) |

### 1. Install

**Option A — Docker** (the reference environment: Python 3.12, Temurin JDK 21, CUDA torch and `requirements.txt`)

```bash
./run_docker.sh build                              # once; builds a ~10 GB image
./run_docker.sh python scripts/sanity_check.py
```

Run any command below inside the container by prefixing it with `./run_docker.sh`, or open a shell
there with `./run_docker.sh`. The repository is bind-mounted at `/workspace`, so code edits need no
rebuild; rebuild only when `requirements.txt` or the `Dockerfile` changes. Model weights land in
`.cache/` (gitignored) and survive restarts. GPUs are passed through when the NVIDIA container runtime
is present (`NO_GPU=1` forces CPU); on a machine without an NVIDIA GPU, build with
`TORCH_VARIANT=cpu ./run_docker.sh build`.

**Option B — conda** (how the environment of the reported runs was made)

```bash
conda create -n paperodyssey -c conda-forge python=3.12 openjdk=21
conda activate paperodyssey
pip install -r requirements-lock.txt               # the exact versions of the reported runs (Linux x86_64, CUDA 12.8)
```

`requirements-lock.txt` pins every package of that environment. For newer versions within the
supported ranges, install torch for your hardware and then the lower bounds in `requirements.txt`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or .../whl/cpu without an NVIDIA GPU
pip install -r requirements.txt
```

Without Java, install everything in `requirements.txt` except `pyserini` and set
`retrieval.sparse.backend: sqlite` in `config.yaml` before building the indices; this uses a disk-backed
BM25 index in SQLite (see [retrieval/README.md](retrieval/README.md)). The reported results used Pyserini.

Check the installation (both need no data, network or API key):

```bash
python scripts/sanity_check.py                     # config loads, packages import, data contracts round-trip
python -m unittest discover -s tests -t .          # ~400 offline tests in under a minute
```

### 2. Set the LLM key

```bash
cp .env.example .env
```

Then edit `.env`. The reported results used `deepseek/deepseek-v4.1-flash` through ZenMux, an
OpenAI-compatible gateway:

```bash
LLM_API_KEY=<your ZenMux key>
LLM_MODEL=deepseek/deepseek-v4.1-flash
LLM_BASE_URL=https://zenmux.ai/api/v1
```

With a DeepSeek key, keep the defaults in `.env.example` (`DEEPSEEK_API_KEY`, `deepseek-chat` at
`https://api.deepseek.com`). Any other OpenAI-compatible endpoint works through the same three `LLM_*`
variables; another model gives somewhat different results from the report. Every other setting is in
`config.yaml`, and all its paths are relative to the repository root, so nothing else needs editing.

### 3. Download the knowledge base

The processed knowledge base (19,941 papers: 658,871 passages and 302,135 sections) is in the public
Google Drive folder [PaperOdyssey-KB-merged](https://drive.google.com/drive/folders/1p2TmBcvMgRn88Bar-sGEIUH4VuNiFxY7).
Put its three `processed/` files in `data/processed/` and check them against the folder's `SHA256SUMS`:

```bash
dl() { curl -L -C - -o "$2" "https://drive.usercontent.google.com/download?id=$1&export=download&confirm=t"; }
dl 1y23DhIPjAWK52jFXlcnYwkTIC7VnFGqL data/processed/passages.jsonl    # 1.2 GB
dl 1ueUYeTCtniZt3uL2u5B7LTnmxu546tvo data/processed/sections.jsonl    # 0.9 GB
dl 1iC50Mn1tyeBCLjkOXauL3sZKSsTg8ix5 data/processed/papers.sqlite     # 33 MB
dl 1RQk2kMY8liF2TLPercGXYR6NCmn3Av0_ data/SHA256SUMS
(cd data && sha256sum -c SHA256SUMS --ignore-missing)                # 3 x OK; on macOS: shasum -a 256 -c
```

The data archive linked from the report holds the same files under `knowledge_base/`, so copying them
from there works too. The Drive folder's `raw/` (paper metadata) and `results/` (corpus statistics,
parse-quality checks) are not needed to run the system. The fixed evaluation set,
`data/eval/eval_set.jsonl` (1,027 questions), is in this repository. The 34-question challenge set is in
the data archive as `eval/challenge_set.jsonl`; copy it to `data/eval/` to rerun the challenge-set
experiments.

Rebuilding the knowledge base from arXiv instead means downloading 20,000 PDFs (about 48 GB, roughly
18 hours from one IP address); the steps are in [kb/README.md](kb/README.md).

### 4. Build the indices

```bash
python scripts/build_indexes.py --config config.yaml
```

This writes `data/indices/lookup.sqlite` (2.4 GB, passage and section text by id), `data/indices/bm25/`
(120 MB, Pyserini) and `data/indices/faiss/` (2.9 GB, BGE-large-en-v1.5 embeddings in an HNSW index).
The lookup store and the BM25 index took under 4 minutes; encoding the 658,871 passages took 52 minutes
on an A40. `--only lookup|sparse|dense` rebuilds one part. The first run downloads the encoder from
Hugging Face; the reranker follows on the first search.

### 5. Run

**Web UI**

```bash
bash scripts/run_ui.sh                             # or: ./run_docker.sh ui
```

Open http://localhost:8501 and type a research question. **Ask** runs Paper Search and writes an
answer with numbered citations; **Search papers only** stops after Paper Search. On the first visit
the page loads the retrieval models in the background (about half a minute). A question takes about
40 seconds end to end, most of it in the LLM review of candidate papers.

**Command line** (the same pipeline without the UI)

```bash
python scripts/run_examples.py --config config.yaml                       # the five UI example inputs
python scripts/run_examples.py --config config.yaml --input "How does DPO differ from PPO?"   # adds your own question
```

Each run writes the papers, answers, sources, per-stage latency and token use to
`data/results/metrics/examples_<date>.json`.

**Reproduce the evaluation**

`scripts/run_eval.py` runs the experiments in the report on the fixed evaluation set and writes
metrics to `data/results/metrics/`. For example:

```bash
# Retrieval (Q2/Q3): BM25, dense, hybrid RRF and hybrid + rerank on all 1,027 questions; no LLM calls
python scripts/run_eval.py --config config.yaml --stage retrieval
# Paper Search (Q5): plain RAG vs agentic search on the 912 eval questions
python scripts/run_eval.py --config config.yaml --stage review --ablation agentic --split eval --tag agentic --workers 4
# Paper QA answers with RAGAS-style scores for those searches
python scripts/run_eval.py --config config.yaml --stage answer --source data/results/ablation_runs/agentic --split eval --tag agentic --workers 16
# Query understanding x LLM review (D61)
python scripts/run_eval.py --config config.yaml --stage review --ablation understanding review --split eval --tag understanding-review --workers 4
```

Agentic search uses about 30–35k LLM tokens per question, so a full 912-question run is about 30M
tokens. Add `--limit 3` with its own `--tag` for a quick check, or `--dry-run` to print an ablation's
plan; an interrupted run continues with `--resume` and the same tag. `--eval-set
data/eval/challenge_set.jsonl` runs the challenge set, and `scripts/challenge_set.py score` scores it.
[scripts/README.md](scripts/README.md) and [eval/README.md](eval/README.md) describe every stage,
ablation factor and output file; the data archive's `RESULTS_INDEX.md` lists the run behind each
reported result.

## Conventions

Before changing code, read the contracts: [docs/SCHEMA.md](docs/SCHEMA.md) (data contracts, the single
source of truth across modules), [docs/DESIGN.md](docs/DESIGN.md) (system design and decision log), and
the `README.md` of the package you change, which states its implementation requirements, constraints
and status. Coding agents (Codex, Claude Code) read [AGENTS.md](AGENTS.md); `CLAUDE.md` imports it, so
edit `AGENTS.md` only.

**Interface discipline** (the most important constraint):
- Cross-module data may **only** use the formats defined in `docs/SCHEMA.md`
- Retrieval goes **only** through `retrieval.pipeline.retrieve()`; the orchestration layer must not bypass it to use BM25/FAISS directly
- LLM calls go **only** through `llm.client`
- Changing SCHEMA.md or a public signature: update SCHEMA.md → update `types.py` → update all callers → PR review
- Record `latency_ms` on critical paths (Q2 latency measurement and the report's performance section need it)

**Git workflow**: `main` is protected; feature branches `module/description`; Conventional Commits (`feat:` `fix:` `docs:` `refactor:` `chore:`); every PR needs at least 1 reviewer who is not the author; `scripts/sanity_check.py` must pass before merging.

**Sanity checks**: before any PR run `python scripts/sanity_check.py`, which verifies ① `config.yaml` loads with `${VAR}` placeholders expanded, ② every package imports cleanly, ③ schema round-trips (ScoredPassage/RetrieveResult/QueryPlan/ReviewRecord/FinalResult/EvidenceSource/DownstreamResult serialize and read back with identical fields). It needs only pyyaml (via `config.py`) — no heavy dependencies.

**Tests**: offline `unittest` tests live in `tests/` (no network, no real API calls; tests whose optional dependency — Java, FAISS, cached models — is missing are skipped). Run `python -m unittest discover -s tests -t .`, or `./run_docker.sh python -m unittest discover -s tests -t .` for the full run. GitHub Actions (`.github/workflows/tests.yml`) runs the sanity check and these tests on every pull request and push to `main`, without Pyserini, sentence-transformers and transformers, so their tests skip there. End-to-end pipeline behavior is still checked with small real runs (e.g. `--limit 100`), not mocks alone.

**Ablation discipline** (Q3/Q5): every innovation is an enable/disable switch in `config.yaml`; ablation runs change only one switch at a time on the same evaluation set and seed; results go to `data/results/metrics/ablation_<innovation>_<date>.json` with a config snapshot and failed→succeeded query instances.

**Definition of Done** — a module/feature is done only when:
- [ ] Contract (SCHEMA.md/types.py) updated or confirmed unchanged
- [ ] `scripts/sanity_check.py` passes (and relevant `tests/` pass)
- [ ] Latency/cost recorded (when the feature touches the online path)
- [ ] Package README implementation notes updated
- [ ] PR passed review
- [ ] Q3/Q5 innovation: ablation switch exists with an explicit default

**Weekly meetings**: sync on progress and blockers. **Data distribution**: the knowledge base is shared
on Google Drive; the data archive linked from the report packs it together with both evaluation sets and the
formal results; indices are built locally. The reviewed
`data/eval/eval_set.jsonl` is versioned so evaluations use the same questions; its provenance and hash
are in `data/README.md`.

## Key Design Decisions Summary

| Decision | Conclusion |
|---|---|
| Downstream task | Simple cited paper QA: Paper Search → original passages → one answer-generation call (D29) |
| Corpus | arXiv cs.CL/cs.AI/cs.LG, 2015–2025, OpenAlex citation filter: 20,000 papers selected, 19,941 built |
| Sparse/dense | Pyserini BM25 / BGE-large-en-v1.5 + FAISS HNSW |
| Fusion/rerank | RRF(k=60) → BGE-reranker-v2-m3 (top-50 → top-10) |
| Orchestration | Paper Search: agentic query understanding → parallel retrieval → LLM review → feedback-based rewriting (coverage-first, 2–3 rounds) |
| LLM | DeepSeek V4.1 Flash (`deepseek/deepseek-v4.1-flash` via ZenMux) through an OpenAI-compatible client; swappable |

Full decision log (with alternatives and rationale): [docs/DESIGN.md §Decision Log](docs/DESIGN.md).
