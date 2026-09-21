# scripts/ One-click scripts

> All scripts are **thin wrappers**: parse CLI args -> call the corresponding package function -> write to disk.
> Keep logic in the packages (testable); keep logic out of the scripts.

| Script | Purpose |
|---|---|
| `download_papers.py` | metadata fetch + OpenAlex citations + filtering + PDF download (resumable, `--shard I/N`) |
| `build_kb.py` | parse + clean + chunk PDFs → passages.jsonl / sections.jsonl / papers.sqlite + corpus stats (resumable) |
| `build_indexes.py` | build BM25 + FAISS indexes from passages.jsonl |
| `run_eval.py` | unified evaluation entry point (--stage retrieval/review/all) |
| `sanity_check.py` | dependency-free sanity checks: config loads, packages import, schema round-trips (see README: Sanity checks) |
| `run_ui.sh` | start Streamlit |

Conventions:
- Usage `python scripts/xxx.py --config config.yaml [args]`; `--help` must work
- Output paths all come from config.yaml; print stage progress (tqdm) + write log files
- Re-runnable (idempotent): resumable downloads, index rebuild overwrites, evaluation results carry a date suffix
