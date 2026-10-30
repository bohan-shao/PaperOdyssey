# AGENTS.md

Instructions for coding agents (Codex, Claude Code, ...) working in this repository.
Humans should start with [README.md](README.md); this file is the agent-oriented summary.

## Project

PaperOdyssey is the AI6127 (Deep Neural Networks for NLP) course assignment: a RAG system over
arXiv cs.CL/cs.AI/cs.LG papers. Deadline **2026-11-07 23:59 SGT**.

- **Paper Search** — `orchestration.run_query()`: query understanding → parallel retrieval → LLM review → feedback rewriting loop.
- **Paper QA** — `downstream.run()`: calls Paper Search, reads passage evidence, generates one answer with numbered citations.
- **UI** — Streamlit (`ui/app.py`), one question box feeding both tabs.

Read before changing code:

| Document | What it governs |
|---|---|
| [docs/SCHEMA.md](docs/SCHEMA.md) | Cross-module data contracts — the single source of truth |
| [docs/DESIGN.md](docs/DESIGN.md) | Architecture and the decision log (D1, D2, ...) |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Evaluation plan, metrics, ablation protocol |
| `<package>/README.md` | Implementation requirements, constraints, and status of that package |

## Layout and data flow

| Path | Responsibility |
|---|---|
| `kb/` | Fetch, parse, clean, chunk, store papers (Q1) |
| `retrieval/` | BM25 ∥ dense → RRF → rerank; single entry point `retrieval.pipeline.retrieve()` (Q2/Q3) |
| `orchestration/` | Paper Search agentic loop (Q3/Q5) |
| `downstream/` | Cited paper QA (Q4/Q5); answer prompts in `downstream/prompts.py` |
| `llm/` | Shared OpenAI-compatible client (`llm/client.py`), Jev client, orchestration prompts (`llm/prompt.py`) |
| `eval/` | Retrieval/review/answer metrics, latency, annotation and IAA |
| `ui/` | Streamlit app |
| `scripts/` | Thin CLI wrappers around package functions |
| `config.py` / `config.yaml` | The only config entry point / all tunables and ablation switches |
| `tests/` | Offline `unittest` tests |
| `data/` | Data and indices — shared via Dropbox; only the fixed `eval/eval_set.jsonl` dataset is versioned |

Offline: `scripts/download_papers.py` → `scripts/build_kb.py` (`passages.jsonl`, `sections.jsonl`, `papers.sqlite`)
→ `scripts/build_indexes.py` (`lookup.sqlite`, BM25, FAISS).
Online: `retrieval.pipeline.retrieve()` → `orchestration.run_query()` → `downstream.run()` → `ui/app.py`.

## Environment

- Python ≥ 3.11. Docker is the reference environment (JDK 21 for Pyserini, CUDA torch); the repo is
  bind-mounted, so code edits need no rebuild:
  `./run_docker.sh build` once, then `./run_docker.sh python <script>`.
- Local venv: `pip install -r requirements.txt`. Pyserini needs JDK 21 and Python ≥ 3.12; without it set
  `retrieval.sparse.backend: rankbm25` in `config.yaml`.
- Secrets live in `.env` (copy from `.env.example`). Never commit `.env`, `.cache/`, or model weights.
  Data and indices stay out of git, except the reviewed `data/eval/eval_set.jsonl` dataset and directory
  documentation/placeholders. Annotation intermediates, raw data, indices and experiment results remain
  gitignored and are shared via Dropbox. See `data/README.md` for the fixed dataset's provenance and hash.

## Commands

```bash
python scripts/sanity_check.py                     # merge gate: config loads, packages import, schema round-trips
python -m unittest discover -s tests -t .          # all offline tests (heavy-dependency tests skip when deps are missing)
python -m unittest tests.test_retrieval            # one test module
./run_docker.sh python -m unittest discover -s tests -t .   # full run with Java/FAISS/cached models
bash scripts/run_ui.sh                             # Streamlit on :8501 (needs built indices and an LLM key)
```

Scripts take `--config config.yaml` and support `--help`. Use small real runs (e.g.
`python scripts/build_kb.py --limit 100`) to check pipeline changes end to end.

## Hard rules

These are interface contracts between teammates' modules. Do not work around them.

1. **Schemas.** Cross-module data uses only the formats in `docs/SCHEMA.md`. To change a field or a public
   signature: update `docs/SCHEMA.md` → the matching `types.py` (`retrieval/`, `orchestration/`,
   `downstream/`) → every caller → the round-trip checks in `scripts/sanity_check.py`.
2. **Retrieval** goes only through `retrieval.pipeline.retrieve()`. Orchestration and downstream never call
   BM25/FAISS/rerank directly (reading passages by id from `retrieval.lookup` is fine).
3. **LLM calls** go only through `llm.client` (`get_client()` and its `chat` / `chat_json` methods). No direct
   `openai`/`httpx` calls in business code.
4. **Config** comes only from `config.get_config()` / `config.get_path()`. Never open `config.yaml` or read
   environment variables directly in business code; output paths come from config, not hard-coded strings.
5. **Latency.** Record `latency_ms` on online-path stages (needed for Q2 and the report).
6. **Ablations.** Every Q3/Q5 innovation is a switch in `config.yaml` with an explicit default. Ablation runs
   change one switch at a time on the same eval set and seed; results go to
   `data/results/metrics/ablation_<innovation>_<date>.json` with a config snapshot.
7. **Design changes** append the next numbered entry to the decision log in `docs/DESIGN.md` and keep
   `SCHEMA.md` / `EVALUATION.md` in sync.
8. **Scripts are thin**: parse args → call a package function → write output. Logic lives in packages.
9. **Stubs**, when present (`raise NotImplementedError  # TODO`), are
   implemented to the contract in their docstring and package README; don't change their signatures silently.

## Code style

- `.editorconfig`: UTF-8, LF, 4-space indent for Python, 2 spaces for YAML/JSON/Markdown.
- `from __future__ import annotations`, type hints, dataclasses for contract types, a module docstring
  stating the module's purpose and contract.
- Import heavy or optional dependencies (pyserini, faiss, torch, sentence-transformers, httpx, ...) lazily
  inside functions, so every package imports with only `pyyaml` installed and `sanity_check.py` stays light.
- Tests stay offline: no network, no real API calls, no model downloads. Mock the LLM transport and
  retrieval, and `skipUnless` tests whose optional dependency is missing.
- Code, comments, docs, and UI text are in English.
- When behavior or status changes, update the package `README.md` in the same change.

## Git and PRs

- `main` is protected. Work on a feature branch named `module/description`.
- Conventional Commits with a scope: `feat(retrieval): ...`, `fix(llm): ...`, `docs: ...`, `refactor: ...`, `chore: ...`.
- Every PR needs at least one reviewer who is not the author, and `scripts/sanity_check.py` must pass.
- Only commit or push when asked.

**Definition of Done**

- [ ] Contract (`SCHEMA.md` / `types.py`) updated or confirmed unchanged
- [ ] `scripts/sanity_check.py` passes (and relevant `tests/` pass)
- [ ] Latency/cost recorded if the change touches the online path
- [ ] Package README implementation notes updated
- [ ] Q3/Q5 innovation: ablation switch exists with an explicit default
