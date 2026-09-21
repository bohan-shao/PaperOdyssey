# PaperOdyssey — AI6127 Paper-QA RAG System

> Course assignment: AI6127 Deep Neural Networks for NLP (2026). Deadline **2026-11-07 23:59 SGT**, submit via Blackboard.
> Current status: **skeleton repository** (interface contracts and implementation requirements are fixed; the concrete logic is a placeholder stub, to be implemented).

## What the System Does

Two presentation tabs over the same agentic retrieval core (D23/D24):

**Tab 1 — Paper Search** (`orchestration.run_query()`): a standalone, task-agnostic capability.
The user enters **a research requirement** (a paragraph) and gets relevant papers with review verdicts:

1. **Query understanding**: an LLM splits the input into multiple natural-language sub-queries plus BM25 keywords, and extracts the user's criteria
2. **Parallel retrieval**: the sub-queries run in parallel through the retrieval core (BM25 ∥ dense → RRF → rerank)
3. **Review loop**: a review LLM judges each paper against a **fixed rubric + input-extracted criteria** (pass/reject, 1–5 score, reasons, covered/missing aspects); if coverage is insufficient, feedback-based rewriting triggers supplementary retrieval (at most 2–3 rounds)

**Tab 2 — Downstream task** (`downstream.run()`): **calls Paper Search** (Tab 1) and then produces
the task output from the evidence. Task type TBD (QA with cited answers / summarization / classification);
only this part changes when the task is decided (Q4/Q5).

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

## Repository Structure

| Directory | Responsibility |
|---|---|
| `kb/` | Knowledge base construction: acquisition/parsing/cleaning/chunking/storage/statistics (assignment Q1, 20 points) |
| `retrieval/` | Retrieval core: BM25/dense/RRF/rerank/unified entry point (assignment Q2/Q3, 40 points) |
| `orchestration/` | Orchestration layer (Paper Search): query understanding/parallel retrieval/review/rewriting loop (agentic RAG, Q3/Q5 innovation) |
| `downstream/` | Downstream task (reserved folder; calls Paper Search — task type TBD, Q4/Q5, 40 points) |
| `llm/` | LLM layer: single LLM client + prompt templates (used by orchestration and, later, the downstream task) |
| `eval/` | Evaluation: retrieval/review metrics + annotation tooling (Q3); generation/RAGAS metrics come with the downstream task (Q4) |
| `ui/` | Streamlit interface (Q2) |
| `scripts/` | One-click scripts: download/build index/run evaluation/sanity checks |
| `docs/` | Architecture, schema, conventions, evaluation plan, decision records |
| `data/` | Data and indices (gitignored, distributed via Dropbox) |

## Quick Start (Skeleton Phase)

### Option A — Docker (recommended; bundles the JDK that Pyserini needs)

```bash
./run_docker.sh build            # one-time (~10 GB image: Temurin 21 + CUDA torch + requirements)
./run_docker.sh                  # interactive shell, repo bind-mounted at /workspace
./run_docker.sh ui               # Streamlit on http://localhost:8501
./run_docker.sh python scripts/sanity_check.py
```

The repo is bind-mounted, so **code edits need no rebuild** — rebuild only when
`requirements.txt` or the `Dockerfile` changes. Model weights and HF caches land in
`.cache/` (gitignored) and survive rebuilds. GPU is passed through automatically when the
NVIDIA container runtime is present; `NO_GPU=1` forces CPU. Build with `TORCH_VARIANT=cpu`
on a machine without an NVIDIA card, or `INSTALL_PYSERINI=0` to skip the JVM path and use
the rank-bm25 fallback.

### Option B — local venv

```bash
# 1. Environment (Python ≥ 3.11)
#    Note: current Pyserini declares requires-python >= 3.12, so on 3.11 pip either fails
#    or silently backtracks to a very old release. Pyserini also needs a JDK 21 on PATH.
#    Use Option A, or set retrieval.sparse.backend=rankbm25, if that bites.
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. API key
cp .env.example .env    # fill in DEEPSEEK_API_KEY

# 3. Read the contracts (required before writing code)
docs/SCHEMA.md          # data contracts (single source of truth across modules)
docs/DESIGN.md          # system design and decision log
README.md               # conventions / git workflow / Definition of Done (below)
```

Each module's `README.md` states that module's **implementation requirements and constraints**; the `.py` files are contract-bearing placeholder stubs (`NotImplementedError` + TODO markers) — implement them according to the contract.

## Conventions

**Interface discipline** (the most important constraint):
- Cross-module data may **only** use the formats defined in `docs/SCHEMA.md`
- Retrieval goes **only** through `retrieval.pipeline.retrieve()`; the orchestration layer must not bypass it to use BM25/FAISS directly
- LLM calls go **only** through `llm.client`
- Changing SCHEMA.md or a public signature: update SCHEMA.md → update `types.py` → update all callers → PR review
- Record `latency_ms` on critical paths (Q2 latency measurement and the report's performance section need it)

**Git workflow**: `main` is protected; feature branches `module/description`; Conventional Commits (`feat:` `fix:` `docs:` `refactor:` `chore:`); every PR needs at least 1 reviewer who is not the author; `scripts/sanity_check.py` must pass before merging.

**Sanity checks** (no test framework): before any PR run `python scripts/sanity_check.py`, which verifies ① `config.yaml` loads with `${VAR}` placeholders expanded, ② every package imports cleanly, ③ schema round-trips (ScoredPassage/RetrieveResult/QueryPlan/ReviewRecord/FinalResult serialize and read back with identical fields). It needs only pyyaml (via `config.py`) — no heavy dependencies. Pipeline correctness is validated by small real runs (e.g. `--limit 100`), not mocks.

**Ablation discipline** (Q3/Q5): every innovation is an enable/disable switch in `config.yaml`; ablation runs change only one switch at a time on the same evaluation set and seed; results go to `data/results/metrics/ablation_<innovation>_<date>.json` with a config snapshot and failed→succeeded query instances.

**Definition of Done** — a module/feature is done only when:
- [ ] Contract (SCHEMA.md/types.py) updated or confirmed unchanged
- [ ] `scripts/sanity_check.py` passes
- [ ] Latency/cost recorded (when the feature touches the online path)
- [ ] Package README implementation notes updated
- [ ] PR passed review
- [ ] Q3/Q5 innovation: ablation switch exists with an explicit default

**Weekly meetings**: sync on progress and blockers. **Data distribution**: `data/` is not committed; shared via Dropbox.

## Key Design Decisions Summary

| Decision | Conclusion |
|---|---|
| Downstream task | **TBD** — calls Paper Search (D24); candidates: QA with cited answers / summarization / classification |
| Corpus | arXiv cs.CL/cs.AI/cs.LG, 15k–30k papers, 2015–2025, OpenAlex citation filter |
| Sparse/dense | Pyserini BM25 / BGE-large-en-v1.5 + FAISS HNSW |
| Fusion/rerank | RRF(k=60) → BGE-reranker-v2-m3 (top-50 → top-10) |
| Orchestration | Paper Search: agentic query understanding → parallel retrieval → LLM review → feedback-based rewriting (coverage-first, 2–3 rounds) |
| LLM | DeepSeek API (OpenAI-compatible, swappable) |

Full decision log (with alternatives and rationale): [docs/DESIGN.md §Decision Log](docs/DESIGN.md).
