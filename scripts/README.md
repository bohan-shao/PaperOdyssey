# scripts/ One-click scripts

> All scripts are **thin wrappers**: parse CLI args -> call the corresponding package function -> write to disk.
> Keep logic in the packages (testable); keep logic out of the scripts.

| Script | Purpose |
|---|---|
| `download_papers.py` | metadata fetch + OpenAlex citations + filtering + PDF download (resumable, `--shard I/N`) |
| `build_kb.py` | parse + clean + chunk PDFs into passages.jsonl / sections.jsonl / papers.sqlite + corpus stats (resumable); `--from-sections` chunks the parsed sections instead (ablation passages without PDFs, D36) |
| `merge_kb_shards.py` | merge the KB members' shard builds into one KB (`kb.merge.merge_shards`, D35); refuses before writing anything when shards don't fit together |
| `build_indexes.py` | build the lookup store + BM25 + FAISS indexes from passages.jsonl (`--only lookup\|sparse\|dense\|all`) |
| `benchmark_retrieval.py` | retrieval-core latency benchmark (per-stage latency per config, throughput, efSearch and rerank-depth sweeps, five example queries) → `metrics/latency_retrieval_<date>.json`; `--protocol production` times the configured settings (D51) → `metrics/latency_retrieval_production_<date>.json` |
| `run_examples.py` | the five UI example inputs through Paper Search + cited answer with the configured settings (D51): papers, answers, sources, per-stage latency and tokens → `metrics/examples_<date>.json` |
| `probe_retrieval.py` | paired raw/template source diagnostic; four sparse/hybrid/rerank modes, unique run artifacts and one-switch comparisons (not human relevance evaluation) |
| `eval_set.py` | evaluation set (D39): `draft` (LLM questions) -> `sheets` (one CSV per annotator) -> `build` (majority vote, eval_set.jsonl, agreement report, to_rewrite.csv) -> `rewrites` (next round); `sheets --per-question 2` and `tiebreak` for two annotators per question (D42) |
| `run_eval.py` | `--stage retrieval`: four retrieval configurations → `retrieval_results/<configuration>_<tag>.jsonl` + `metrics/retrieval_<tag>.json`; `--stage review`: resumable Paper Search/review → `review_runs/<tag>/report.json`; `--stage all` (default): retrieval then review. `--ablation FACTOR [FACTOR]` with an explicit stage runs controlled baseline/A/B/AB experiments → `ablation_runs/<tag>/report.json` and `metrics/ablation_<factors>_<tag>.json`; `--workers N` runs N review questions at once (D48); `--stage answer --source DIR`: answers and scores a finished review run or review ablation → `answer_runs/<tag>/` + `metrics/answer_<tag>.json` (D49) |
| `sanity_check.py` | dependency-free sanity checks: config loads, packages import, schema round-trips (see README: Sanity checks) |
| `run_ui.sh` | start Streamlit |

Conventions:
- Usage `python scripts/xxx.py --config config.yaml [args]`; `--help` must work
- Output paths all come from config.yaml; print stage progress (tqdm) + write log files
- Re-runnable (idempotent): resumable downloads, index rebuild overwrites, evaluation results carry a date suffix

`run_eval.py --stage review --split eval --tag review-eval` runs the held-out Paper Search set;
add `--limit 3` with a separate tag for a smoke check, or `--resume` with the original tag and
selection to continue an interrupted run. Resume requires matching data/config/code fingerprints.
`--workers N` (review stage and review ablation cells, D48) runs N questions at once in the process;
resume may use another value, and per-query latency should come from a `--workers 1` run.
Without `--ablation`, `--split` and `--resume` affect review only, including within `--stage all`; `--configs` affects
retrieval only. Use `--stage review` when resuming to avoid repeating retrieval. Incomplete review
runs exit 1 and print their summary on stderr; artifacts remain under `review_runs/<tag>/`.
Review labels currently identify positive source papers only; unjudged papers are never negatives.
Usage includes successfully returned, reported LLM tokens only and is not a billing total or a
monetary cost estimate. See `eval/README.md` for interpretation and limitations.

Named ablations require an explicit stage and one or two canonical factor names:

```bash
python scripts/run_eval.py --stage retrieval --ablation rerank --split eval --tag rerank-eval
python scripts/run_eval.py --stage retrieval --ablation query_views diversity --split eval --tag views-diversity
python scripts/run_eval.py --stage review --ablation loop hyde --split dev --limit 3 --tag loop-hyde-smoke
```

After a review run or a review ablation finishes, answer and score its questions (D49; LLM calls, no
new search; needs the same lookup store and the dense encoder):

```bash
python scripts/run_eval.py --stage answer --source data/results/review_runs/review-eval --tag answers-eval --workers 8
python scripts/run_eval.py --stage answer --source data/results/ablation_runs/loop-hyde --tag answers-loop-hyde --workers 8
```

Append `--dry-run` to print the plan without querying or writing files. The baseline disables
all selected factors; A/B/AB enable the named factors while retaining other config settings.
Comparisons change exactly one factor at a time. In ablation mode `--split` and `--resume` also
apply to retrieval, and `--configs` is forbidden. `parent_doc`, `hyde` and `loop` require review;
`all` requires every selected factor to support both stages. `field_weighting` requires SQLite.
The catalog excludes chunking (separate prebuilt indexes needed), structured filter scenarios,
and Jev evaluation. It does not add generation/RAG metrics. See `eval/README.md` for the complete
factor catalog, result interpretation and resume rules.

`benchmark_retrieval.py --config <full-config.yaml>` uses the fixed `legacy_50_v2` ranking
protocol with the configured models/backends/indexes, and saves complete queries and per-row
effective settings. Use `--out` to retain multiple runs on the same day. Start it in a fresh
process; see `eval/README.md` for baseline and startup timing definitions.
`--protocol production` (D51) times the configured settings instead, for the deployed system's
Q2 latency; with `dense.top_k` 150 pass `--ef` values of at least 150.
Probe format v2 shares the canonical paper metrics. Sparse-only depth is tested by comparing
`depth_with_diversity` with `diversity_only`; `depth_only` is explicitly skipped for that mode.
