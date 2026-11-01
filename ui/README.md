# ui/ Streamlit UI (Q2)

## Goal (coursework requirement: keep the UI simple; it is not a focus for extra credit)

Two presentation tabs (D23/D29) sharing the same input box:

1. **Paper QA**: **Ask** calls `downstream.run()`, which calls Paper Search once and
   generates an answer with evidence citations. Displays expandable original excerpts,
   arXiv links, latency, and a JSON download for evaluation.
2. **Paper Search**: displays the search result from the same Ask request. The separate
   **Search papers only** action calls `orchestration.run_query()` without answer generation.

## Implemented minimal page

1. A title block set like a paper's: title, the corpus (number of papers with passages, from
   `kb.store.count_papers`, and the configured arXiv categories) and an abstract of the method,
   written from the config switches so it stays true when one is turned off.
2. Multi-line question with Ask and Search papers only; the 5 configured examples sit in the
   margin and copy themselves into the question box.
3. Answer with numbered citations ([1][2], never 12); on wide screens a References margin beside
   it lists each cited source (title, year, section, the exact excerpt sent to the model, arXiv).
4. Paper Search: each accepted paper with review score, covered aspects, reasons and stored
   passages; an "About this search" margin says in words why the search ended (rounds, papers
   reviewed and accepted, re-reviews, aspects covered, cache), times each stage that ran, and
   offers the search record as JSON.
5. While a request runs, a status box under the question lists each step as it happens (planning,
   each round with papers reviewed and accepted, follow-up searches, writing the answer; the
   `progress` events of SCHEMA.md §6.1) and the seconds so far. It ends as "Answer ready in X s",
   "Search finished in X s: N papers accepted" or says there is no answer, and stays (collapsed)
   on later reruns. The tab of the latest result opens by itself, and the examples' expander,
   open on the first visit, folds after the first question (D58). The request runs only after
   the rest of the page is drawn, with its status and results starting empty, so nothing of the
   previous result stays greyed out while it runs. The first page view loads the retrieval
   models in the background, so the first question does not wait for them.
6. Result state persists across rerenders; a failed or empty new submission clears stale results.
7. No answer is explained: how many rounds ran and papers were read out of the knowledge base's
   size, what to try, and the closest papers the reviewer turned down with its reasons.
   Operational errors name the recovery.

Look (`.streamlit/config.toml`, read from the repository root where `scripts/run_ui.sh` starts the
app): a page set like a LaTeX paper. Latin Modern (LaTeX's own face, served from `ui/static/fonts`,
licence in the README there) for the title, answer, references and excerpts; sans for controls;
a paper ground with one ink-red accent for the primary action, links and selection. One grid of a
main column and a margin (after Tufte's handouts, https://github.com/edwardtufte/tufte-css) that
stacks on phones. Every text colour passes WCAG AA. A style block in `app.py` covers what the
theme cannot (layout widths, reading measure, excerpt rules, selection colour, caption contrast).
The work followed impeccable's product-UI guidance (https://github.com/pbakaus/impeccable):
standard controls, one accent, no decorative chrome; the fonts come from LaTeX.css
(https://github.com/vincentdoerig/latex-css).

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
