# ui/ Streamlit UI (Q2)

## Goal (coursework requirement: keep the UI simple; it is not a focus for extra credit)

Two presentation tabs (D23/D29) sharing the same input box:

1. **Paper QA**: **Ask** calls `downstream.run()`, which calls Paper Search once and
   generates an answer with evidence citations. Displays expandable original excerpts,
   arXiv links, latency, and a JSON download for evaluation.
2. **Paper Search**: displays the search result from the same Ask request. The separate
   **Search papers only** action calls `orchestration.run_query()` without answer generation.

## Implemented page (D58, layout D60)

1. **Landing, before the first question.** One centred column holds:
   - the title;
   - the corpus: the number of papers with passages (from `kb.store.count_papers`) and the
     configured arXiv categories;
   - the question box, with Ask and Search papers only;
   - the 5 configured examples; a click copies one into the box;
   - "How it works": three numbered steps written from the config switches (keyword/semantic
     search, agentic review, follow-up loop), so they stay true when one is turned off.

   There are no tabs yet.
2. **Results layout, from the first question on.**
   - A slim bar holds the title, the corpus and an Examples popover with the same examples.
   - The question box sits over the main column, with the status under it, then the tabs.
   - `_mark_asked`, the callback of both submit buttons, switches the layout before the run, so the run
     that starts the request already draws it.
3. **Paper QA.**
   - The question becomes the heading, and the answer is set in Latin Modern.
   - Citations are numbered superscripts: [1][2], never 12. A word joiner attaches each one to the
     word before it, so notation such as W[0] is never parsed as a reference link. Each citation
     carries its source's title and year as a tooltip and jumps to its reference.
   - On wide screens a References margin lists each cited source like a bibliography: number, title,
     year, section, arXiv link, and the exact excerpt sent to the model behind a compact toggle. The
     entry a citation jumped to is highlighted.
   - Under the answer: the time it took and the JSON download.
4. **Paper Search.**
   - Each accepted paper, best first, shows its rank, a title linked to arXiv, its arXiv id, year and
     reviewer score, the reviewer's reasons and the aspects it covers. The passages the reviewer read
     sit behind a toggle.
   - The "About this search" margin:
     - says in words why the search ended;
     - gives papers reviewed, accepted and reviewed again, and aspects covered, in a table;
     - lists the aspects not covered;
     - times each stage that ran;
     - offers the search record as JSON.
5. **Status box.**
   - While a request runs, a status box under the question lists each step as a checklist: planning,
     each round with papers reviewed and accepted, follow-up searches, writing the answer. These are
     the `progress` events of SCHEMA.md §6.1. The step under way shimmers, and the box shows the
     seconds so far.
   - It ends as "Answer ready in X s", as "Search finished in X s: N papers accepted", or by saying
     there is no answer, and it stays (collapsed) on later reruns.
   - The request lives in session state while it runs, so a rerun (a tab or an example click) picks it
     up again with its steps so far. A new question cancels it: its progress callback raises, which
     stops the search at its next step.
   - The tab of the latest result opens by itself (D58).
   - The request runs only after the rest of the page is drawn. Its status, and everything below the
     question box, start as empty slots, so nothing of the previous result (or of the landing) stays
     greyed out while it runs.
   - The first page view loads the retrieval models in the background, so the first question does not
     wait for them.
   - The file watcher is off (`.streamlit/config.toml`) because it froze the app while the models
     loaded, so restart the server after changing the code.
6. Result state persists across rerenders; a failed or empty new submission clears stale results.
7. **No answer.** The page says why:
   - how many rounds ran and how many papers were read, out of the knowledge base's size;
   - what to try;
   - the closest papers the reviewer turned down, with its reasons.

   Operational errors name the recovery.

Needs Streamlit 1.58 or later (compact expanders; keyed `st.tabs` with `on_change` came in 1.55); tested
with 1.66.1. The style block keys off Streamlit's `data-testid` attributes and `st-key-*` classes, so check
the page after an upgrade.

## Look

`.streamlit/config.toml` is read from the repository root, where `scripts/run_ui.sh` starts the app. The
page is set like a LaTeX paper:
- **Type.** Latin Modern, LaTeX's own face, served from `ui/static/fonts` (licence in the README there),
  for the title, headings, answer, references and excerpts; sans for controls. At 18px Latin Modern
  read well in the 2026-11-04 screenshots, so the planned switch of the answer to Streamlit's bundled
  Source Serif was not needed.
- **Colour.** A paper ground with one ink-red accent for the primary action, citations, links and
  selection. Every text colour passes WCAG AA.
- **Grid.** A main column and a margin (after Tufte's handouts, https://github.com/edwardtufte/tufte-css)
  that stack on phones.
- **Lists and tables.** Results and references are entries divided by hairlines, not bordered boxes.
  Tables follow booktabs: a rule above, a rule below, a rule under the head, no vertical lines.
- **The question box** is a white sheet on the paper ground.

A style block in `app.py` covers what the theme cannot: layout widths, the reading measure, paragraph
type, citations, entry rules, tables, the selection colour and caption contrast. It writes its paragraph
rules as `.stMarkdown p.po-*`, because Streamlit styles paragraphs with a class and an element.

The work followed impeccable's product-UI guidance (https://github.com/pbakaus/impeccable): standard
controls, one accent, no decorative chrome. The fonts come from LaTeX.css
(https://github.com/vincentdoerig/latex-css).

Editable query plans and filter widgets are deferred. Users can include date and category constraints
in their question for the existing query planner.

## Implementation requirements

- Call `downstream.run()` for QA and `orchestration.run_query()` for standalone search.
- Config comes from config.yaml (`ui.*`); do not hardcode models/paths in app.py
- Lay the page out with Streamlit elements (compact expanders for excerpts and passages) and keep the
  style simple; the UI is in English (D14).
- Search cache remains owned by `run_query`. QA results stay in session state and can be
  downloaded as JSON; the UI does not write them to disk automatically.
- Latency/timing display reads `latency_by_stage` directly (source of Q2 timing data)
- Run `bash scripts/run_ui.sh`. Live requests require processed passages, built indices
  (`scripts/build_indexes.py`), and an LLM key; a missing index or lookup store is surfaced as
  an actionable error, not a generated answer.
- Test interactions offline with `python -m unittest tests.test_ui -v`.
