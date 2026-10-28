# scripts/ One-click scripts

> All scripts are **thin wrappers**: parse CLI args -> call the corresponding package function -> write to disk.
> Keep logic in the packages (testable); keep logic out of the scripts.

| Script | Purpose |
|---|---|
| `download_papers.py` | metadata fetch + OpenAlex citations + filtering + PDF download (resumable, `--shard I/N`) |
| `build_kb.py` | parse + clean + chunk PDFs into passages.jsonl / sections.jsonl / papers.sqlite + corpus stats (resumable); `--from-sections` chunks the parsed sections instead (ablation passages without PDFs, D36) |
| `merge_kb_shards.py` | merge the KB members' shard builds into one KB (`kb.merge.merge_shards`, D35); refuses before writing anything when shards don't fit together |
| `build_indexes.py` | build the lookup store + BM25 + FAISS indexes from passages.jsonl (`--only lookup\|sparse\|dense\|all`) |
| `benchmark_retrieval.py` | retrieval-core latency benchmark (per-stage latency per config, throughput, efSearch and rerank-depth sweeps) → `metrics/latency_retrieval_<date>.json` |
| `probe_retrieval.py` | paired raw/template source diagnostic; four sparse/hybrid/rerank modes, unique run artifacts and one-switch comparisons (not human relevance evaluation) |
| `eval_set.py` | evaluation set (D39): `draft` (LLM questions) -> `sheets` (one CSV per annotator) -> `build` (majority vote, eval_set.jsonl, agreement report, to_rewrite.csv) -> `rewrites` (next round) |
| `run_eval.py` | unified evaluation entry point (--stage retrieval/review/all) |
| `sanity_check.py` | dependency-free sanity checks: config loads, packages import, schema round-trips (see README: Sanity checks) |
| `run_ui.sh` | start Streamlit |

Conventions:
- Usage `python scripts/xxx.py --config config.yaml [args]`; `--help` must work
- Output paths all come from config.yaml; print stage progress (tqdm) + write log files
- Re-runnable (idempotent): resumable downloads, index rebuild overwrites, evaluation results carry a date suffix

`benchmark_retrieval.py --config <full-config.yaml>` uses the fixed `legacy_50_v2` ranking
protocol with the configured models/backends/indexes, and saves complete queries and per-row
effective settings. Use `--out` to retain multiple runs on the same day. Start it in a fresh
process; see `eval/README.md` for baseline and startup timing definitions.
Probe format v2 shares the canonical paper metrics. Sparse-only depth is tested by comparing
`depth_with_diversity` with `diversity_only`; `depth_only` is explicitly skipped for that mode.
