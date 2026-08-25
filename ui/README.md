# ui/ Streamlit UI (Q2)

## Goal (coursework requirement: keep the UI simple; it is not a focus for extra credit)

Two presentation tabs (D23) sharing the same input box:

1. **Paper Search tab** — natural-language paper search: calls `orchestration.run_query()` directly.
   Shows the full agentic retrieval process and the paper candidate set (with review verdicts).
2. **Downstream tab** — RESERVED: the downstream task type is TBD (`downstream/` is an empty
   folder). Keep the tab **hidden/disabled** until the task is defined; when defined, it calls
   `downstream.run()`, which **internally calls Paper Search** (D24) and produces the task output.

## Page layout requirements

1. **Input area**: multi-line text box (a paragraph) + a "Search" button + 5 example inputs (dropdown/buttons, Q2 requirement)
2. **Paper Search tab only** (until the downstream task is defined): show the agentic retrieval process and results
3. **Understanding results area** (editable): shows the sub-queries/keywords/criteria produced by query understanding; allow the user to add/remove/edit and re-retrieve
4. **Filter options**: year range, category (multi-select), mapped to RetrievalFilters
5. **Retrieval process area** (optionally expandable): round progress, per-round sub-queries, papers under review (live streaming refresh)
6. **Results area**:
   - Paper list: title/year/category/arXiv link/review score and reason (verdict pass first)
   - Expand matched passages (including section names)
7. **Performance area**: per-stage latency (understanding/retrieval/review/rewrite/total) + token cost

## Implementation requirements

- Only call `orchestration.run_query()` (including the `orchestration.loop.enabled` switch demo)
- Config comes from config.yaml (`ui.*`); do not hardcode models/paths in app.py
- Layout with st.expander, keep the style simple; English UI (D14)
- Results land in `data/results/final_results/` (run_query is responsible; the UI does not write again)
- Latency/timing display reads `latency_by_stage` directly (source of Q2 timing data)
- When the downstream task is defined: add the Downstream tab calling `downstream.run()`
