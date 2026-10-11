# ui/ Streamlit UI (Q2)

## Goal (coursework requirement: keep the UI simple; it is not a focus for extra credit)

Two presentation tabs (D23/D29) sharing the same input box:

1. **Paper QA** — **Ask** calls `downstream.run()`, which calls Paper Search once and
   generates an answer with evidence citations. Displays expandable original excerpts,
   arXiv links, latency, and a JSON download for evaluation.
2. **Paper Search** — displays the search result from the same Ask request. The separate
   **Search papers only** action calls `orchestration.run_query()` without answer generation.

## Implemented minimal page

1. Multi-line question, 5 configured examples, Ask and Search papers only buttons.
2. Answer with numbered citations; each citation expands to the exact excerpt sent to the model.
3. Paper list with review scores/reasons and stored passages; expandable coverage/timings.
4. Result state persists across rerenders; a failed or empty new submission clears stale results.
5. Clear insufficient-evidence messages and actionable operational errors.

Editable query plans, filter widgets, and live round streaming are deferred. Users can
include date/category constraints in their question for the existing query planner.

## Implementation requirements

- Call `downstream.run()` for QA and `orchestration.run_query()` for standalone search.
- Config comes from config.yaml (`ui.*`); do not hardcode models/paths in app.py
- Layout with st.expander, keep the style simple; English UI (D14)
- Search cache remains owned by `run_query`. QA results stay in session state and can be
  downloaded as JSON; the UI does not write them to disk automatically.
- Latency/timing display reads `latency_by_stage` directly (source of Q2 timing data)
- Run `bash scripts/run_ui.sh`. Live requests require processed passages, built indices
  (`scripts/build_indexes.py`), and an LLM key; a missing index or lookup store is surfaced as
  an actionable error, not a generated answer.
- Test interactions offline with `python -m unittest tests.test_ui -v`.
