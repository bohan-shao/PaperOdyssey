"""Streamlit paper search and cited QA. Run: bash scripts/run_ui.sh (theme: .streamlit/config.toml).

The page has two layouts (D60). Before the first question it is a landing page: the title, the
question box, the example inputs and how the system works, in one centred column. From the first
question on, a slim bar holds the title, the corpus and the examples, and the question box sits over
the Paper QA and Paper Search tabs.
"""

from __future__ import annotations

import html
import json
import queue
import re
import sqlite3
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import streamlit as st

# Streamlit may place only ui/ on sys.path when launched by the CLI.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import downstream
import orchestration
from config import get_config, get_path
from kb.store import count_papers
from orchestration.types import FinalResult, ReviewRecord, passing_order_key
from retrieval.lookup import get_lookup_store
from retrieval.pipeline import retrieve

# Layout, reading type and browser surfaces. Colours match .streamlit/config.toml; Latin Modern
# is served from ui/static/fonts (see the README there).
_STYLE = """
<style>
::selection { background: #F1D9D5; color: #1F1C18; }
textarea { caret-color: #9B2C2C; }
[data-testid="stMainBlockContainer"] { max-width: 1120px; padding: 2.25rem 2rem 5rem; }
@media (max-width: 640px) { [data-testid="stMainBlockContainer"] { padding: 1.25rem 1rem 4rem; } }
[data-testid="stCaptionContainer"] { color: #5E584F; opacity: 1; }
/* Streamlit styles paragraphs with a class and an element (.x p), so the rules below say .stMarkdown p.po-*. */

/* Landing: the title block over the question box, then the examples and how it works. */
.st-key-hero { margin-top: 4vh; }
.st-key-hero h1 { font-weight: 400; font-size: 3rem; line-height: 1.1; padding: 0; }
.stMarkdown p.po-sub { font-family: "Latin Modern Roman", serif; font-size: 1.125rem; line-height: 1.5;
  color: #3D3832; max-width: 34rem; margin: 0 auto 0.75rem; text-wrap: balance; }
.stMarkdown p.po-label { font-size: 0.875rem; font-weight: 600; color: #5E584F; margin: 0; }
.st-key-method { border-top: 1px solid #E7E1D6; padding-top: 1rem; margin-top: 0.25rem; }
.stMarkdown p.po-step-t { font-weight: 600; margin: 0 0 0.125rem; }
.po-step-n { font-family: "Latin Modern Roman", serif; font-weight: 700; color: #9B2C2C; margin-right: 0.5rem; }
.stMarkdown p.po-step-d { font-size: 0.875rem; line-height: 1.5; color: #5E584F; margin: 0; }
@media (max-width: 640px) { .st-key-hero { margin-top: 0.5rem; } .st-key-hero h1 { font-size: 2.25rem; } }

/* Results: a slim bar with the title, the corpus and the examples. */
.st-key-topbar { border-bottom: 1px solid #DDD6CA; padding-bottom: 0.5rem; margin-bottom: 0.5rem; }
.stMarkdown p.po-wordmark { font-family: "Latin Modern Roman", serif; font-size: 1.625rem; line-height: 1.2;
  margin: 0; }

/* The question box: a white sheet on the paper ground. */
.st-key-ask [data-testid="stTextAreaRootElement"] { background: #FFFFFF; border-radius: 0.75rem; }
.st-key-ask textarea { font-size: 1.0625rem; line-height: 1.5; }
/* On phones the empty margin beside the question box would stack under it as a gap. */
@media (max-width: 640px) { .st-key-top [data-testid="stColumn"] + [data-testid="stColumn"] { display: none; } }

/* Examples: rows that copy themselves into the question box. */
.st-key-examples button { justify-content: flex-start; align-items: baseline; padding: 0.25rem 0.625rem;
  min-height: 0; border-radius: 0.5rem; }
.st-key-examples button * { justify-content: flex-start; text-align: left; }
.st-key-examples button:hover { background: #F0ECE4; }
.st-key-examples button p { font-size: 0.9375rem; line-height: 1.45; }

/* Progress: a tight checklist. */
.st-key-progress .stMarkdown p { margin-bottom: 0.375rem; }
.st-key-progress [data-testid="stIconMaterial"] { color: #8A8276; }

/* Results: the question as the heading, the answer set like a paper. */
.stMarkdown p.po-question { font-family: "Latin Modern Roman", serif; font-size: 1.5rem; line-height: 1.3;
  color: #1F1C18; margin: 0.25rem 0 0.75rem; max-width: 34em; text-wrap: balance; }
.st-key-answer p, .st-key-answer li { font-family: "Latin Modern Roman", serif; font-size: 1.125rem;
  line-height: 1.7; max-width: 68ch; text-wrap: pretty; }
.st-key-answer a[href^="#source-"] { font-size: 0.72em; vertical-align: super; line-height: 0;
  text-decoration: none; font-variant-numeric: tabular-nums; white-space: nowrap; padding: 0 0.04em; }
.st-key-answer a[href^="#source-"]:hover, .st-key-answer a[href^="#source-"]:focus-visible {
  text-decoration: underline; }
.st-key-timing { font-variant-numeric: tabular-nums; }

/* References and papers: entries divided by hairlines, not boxes. */
.st-key-sources h3, .st-key-about h3 { font-size: 1.375rem; font-weight: 700; padding: 0.25rem 0; }
[class*="st-key-ref_"], [class*="st-key-paper_"] { border-top: 1px solid #E7E1D6; padding-top: 0.75rem; }
[class*="st-key-ref_"]:has(:target) { background: #F6EBE3; box-shadow: 0 0 0 0.5rem #F6EBE3; }
[id^="source-"] { scroll-margin-top: 5rem; }
.stMarkdown p.po-entry-title { font-family: "Latin Modern Roman", serif; font-size: 1.0625rem; line-height: 1.4;
  margin: 0 0 0.125rem; }
.stMarkdown .po-paper p.po-entry-title { font-size: 1.1875rem; }
.po-n { color: #9B2C2C; font-variant-numeric: tabular-nums; margin-right: 0.4em; }
.stMarkdown p.po-entry-title a { color: inherit; text-decoration: none; }
.stMarkdown p.po-entry-title a:hover { color: #9B2C2C; text-decoration: underline; text-underline-offset: 0.15em; }
.stMarkdown p.po-meta { font-size: 0.875rem; color: #5E584F; margin: 0; font-variant-numeric: tabular-nums; }
.stMarkdown p.po-reasons { font-size: 0.9375rem; line-height: 1.55; margin: 0.375rem 0 0; max-width: 72ch; }
.stMarkdown p.po-covers { font-size: 0.875rem; color: #5E584F; margin: 0.25rem 0 0; max-width: 72ch; }
.st-key-sources [data-testid="stText"], .st-key-papers [data-testid="stText"] {
  padding-left: 0.875rem; border-left: 1px solid #DDD6CA; margin: 0.25rem 0 0.75rem; }
.st-key-sources [data-testid="stText"] *, .st-key-papers [data-testid="stText"] * {
  font-family: "Latin Modern Roman", serif; font-size: 1.0625rem; line-height: 1.55; white-space: pre-wrap; }

/* Tables with booktabs rules: no vertical lines; rules above, below and under the head. */
.stMarkdown table.po-table { border-collapse: collapse; width: 100%; margin: 0.25rem 0 0.5rem;
  border: none; border-top: 1.5px solid #1F1C18 !important; border-bottom: 1.5px solid #1F1C18 !important;
  font-variant-numeric: tabular-nums; }
.stMarkdown .po-table tr, .stMarkdown .po-table th, .stMarkdown .po-table td { border: none !important;
  background: none; }
.stMarkdown .po-table th, .stMarkdown .po-table td { padding: 0.3rem 0.25rem; text-align: left; font-size: 0.9375rem; }
.stMarkdown .po-table thead th { border-bottom: 1px solid #1F1C18 !important; font-weight: 600; }
.stMarkdown .po-table tr.po-total td { border-top: 1px solid #DDD6CA !important; font-weight: 600; }
.stMarkdown .po-table td:last-child, .stMarkdown .po-table th:last-child { text-align: right; }
</style>
"""

_CATEGORIES = {"cs.CL": "computation and language", "cs.AI": "artificial intelligence", "cs.LG": "machine learning"}
_STAGES = {"understanding_ms": "Query understanding", "hyde_ms": "Hypothetical abstract (HyDE)",
           "retrieval_ms": "Retrieval", "review_ms": "Review", "rewrite_ms": "Follow-up queries",
           "total_ms": "Total"}
_ENDINGS = {"coverage_met": "every aspect of the question was covered",
            "min_papers": "enough papers passed review",
            "max_rounds": "the round limit was reached",
            "loop_disabled": "the follow-up loop is switched off",
            "rewrite_exhausted": "no new follow-up queries were left",
            "plain_rag": "one retrieval, without review"}
# The pattern downstream/qa.py links sources with: new-style and old-style arXiv identifiers.
_ARXIV_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z.\-]*/\d{7})(?:v\d+)?")
# Progress steps: finished, the one under way, and the one a failed request stopped at.
_STEP_DONE, _STEP_NOW, _STEP_FAILED = ":material/check_circle:", ":material/pending:", ":material/cancel:"


def _seconds(ms: float) -> str:
    return f"{ms / 1000:.1f} s"


def _series(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _count(n: int, singular: str, plural: str = "") -> str:
    return f"{n:,} {singular if n == 1 else plural or singular + 's'}"


def _one_line(text: str) -> str:
    """Text with its line breaks and runs of spaces collapsed (titles and reasons from parsed PDFs)."""
    return " ".join(str(text).split())


def _arxiv_url(arxiv_id: str) -> str | None:
    return f"https://arxiv.org/abs/{arxiv_id}" if _ARXIV_ID.fullmatch(arxiv_id) else None


def _link(url: str, text: str) -> str:
    """An external link in raw HTML (escaped), opened in a new tab so the session stays."""
    return f'<a href="{html.escape(url)}" target="_blank" rel="noopener noreferrer">{text}</a>'


def _table(rows: Sequence[tuple[str, str]], header: tuple[str, str] | None = None, total: bool = False) -> str:
    """A two-column HTML table with booktabs rules; `total` sets the last row off as a total."""
    head = ("<thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in header) + "</tr></thead>"
            if header else "")
    body = "".join(
        ('<tr class="po-total">' if total and index == len(rows) - 1 else "<tr>")
        + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for index, row in enumerate(rows))
    return f'<table class="po-table">{head}<tbody>{body}</tbody></table>'


@st.cache_data(show_spinner=False)
def _paper_count(path: str, stamp: int) -> int:
    """Papers with passages in the configured store; `stamp` (mtime) refreshes it after a rebuild."""
    try:
        return count_papers(path)
    except sqlite3.Error:
        return 0


def _corpus_size(cfg: dict) -> int:
    database = get_path("processed") / cfg["kb"]["store"]["sqlite_file"]
    return _paper_count(str(database), database.stat().st_mtime_ns if database.exists() else 0)


def _categories(cfg: dict) -> list[str]:
    return list(cfg["kb"].get("filter", {}).get("categories", []))


def _hero(cfg: dict, papers: int) -> None:
    """The landing's title block: the title and what the corpus is."""
    fields = _series([_CATEGORIES.get(c, c) for c in _categories(cfg)] or ["computer science"])
    corpus = f"{papers:,} arXiv papers" if papers else "arXiv papers"
    with st.container(key="hero", horizontal_alignment="center"):
        st.title(cfg["ui"].get("title", "PaperOdyssey"), anchor=False, text_alignment="center")
        st.markdown(f'<p class="po-sub">Cited answers from {corpus} in {html.escape(fields)}</p>',
                    unsafe_allow_html=True, width="stretch", text_alignment="center")


def _topbar(cfg: dict, papers: int) -> None:
    """The slim bar of the results layout: the title, the corpus and the example inputs."""
    corpus = " · ".join([f"{papers:,} arXiv papers" if papers else "arXiv papers", *_categories(cfg)])
    with st.container(key="topbar", horizontal=True, vertical_alignment="center", gap="medium"):
        st.markdown(f'<p class="po-wordmark">{html.escape(cfg["ui"].get("title", "PaperOdyssey"))}</p>',
                    unsafe_allow_html=True, width="content")
        st.space("stretch")
        st.caption(corpus, width="content")
        with st.popover("Examples", type="tertiary", icon=":material/lightbulb:"):
            _examples(cfg)


def _method_steps(cfg: dict) -> list[tuple[str, str]]:
    """How it works, written from the config switches so it stays true when one is turned off."""
    retrieval = cfg.get("retrieval", {})
    kinds = [kind for kind, on in (("keyword", retrieval.get("sparse", {}).get("enabled", True)),
                                   ("semantic", retrieval.get("dense", {}).get("enabled", True))) if on]
    steps = [("Search", f"{_series(kinds or ['passage']).capitalize()} search finds candidate passages.")]
    orchestration_cfg = cfg.get("orchestration", {})
    if orchestration_cfg.get("agentic", {}).get("enabled", True):
        again = (" and searches again for any part no paper covers yet"
                 if orchestration_cfg.get("loop", {}).get("enabled", True) else "")
        steps.append(("Review", f"A language model reviews each paper against your question{again}."))
    steps.append(("Answer", "Ask writes an answer in which every number links to the excerpt it rests on."))
    return steps


def _landing(cfg: dict) -> None:
    """Below the question box until the first question: the examples and how it works."""
    with st.container(horizontal_alignment="center"), st.container(width=720, key="landing"):
        st.markdown('<p class="po-label">Examples</p>', unsafe_allow_html=True)
        _examples(cfg)
        steps = _method_steps(cfg)
        with st.container(key="method"):
            st.markdown('<p class="po-label">How it works</p>', unsafe_allow_html=True)
            for column, (number, (name, text)) in zip(st.columns(len(steps), gap="medium"),
                                                     enumerate(steps, start=1)):
                column.markdown(f'<p class="po-step-t"><span class="po-step-n">{number}</span>{name}</p>'
                                f'<p class="po-step-d">{html.escape(text)}</p>', unsafe_allow_html=True)


def _step_line(event: str, details: Mapping[str, object]) -> str | None:
    """One progress event of the search or answer (SCHEMA.md §6.1) in words; None for unknown events."""
    n = details.get
    if event == "understanding":
        return "Reading the question and planning the searches."
    if event == "planned":
        return (f"Planned {_count(int(n('queries', 0)), 'search', 'searches')} for "
                f"{_count(int(n('aspects', 0)), 'aspect')} of the question.")
    if event == "round":
        return f"Round {n('round')}: running {_count(int(n('queries', 0)), 'search', 'searches')}."
    if event == "round_done":
        return (f"Round {n('round')} done: {_count(int(n('reviewed', 0)), 'paper')} reviewed so far, "
                f"{n('passed')} accepted, {n('covered')} of {_count(int(n('aspects', 0)), 'aspect')} covered.")
    if event == "follow_up":
        missing = [str(item).rstrip(". ") for item in details.get("missing", []) or []]  # type: ignore[union-attr]
        return "Not covered yet: " + "; ".join(missing) + ". Writing new searches for " + (
            "it." if len(missing) == 1 else "them.")
    if event == "retrieved":
        return f"Found {_count(int(n('papers', 0)), 'paper')} (plain retrieval, no review)."
    if event == "cache_hit":
        return "Asked before with the same settings, so the search came from the cache."
    if event == "answering":
        return f"Writing the answer from {_count(int(n('sources', 0)), 'excerpt')}."
    return None


def _steps_markdown(lines: Sequence[str], *, running: bool = False, failed: bool = False) -> str:
    """The steps as a checklist: each done, and the last one under way (shimmering) or where it failed."""
    shown = []
    for number, line in enumerate(lines, start=1):
        if number < len(lines) or not (running or failed):
            shown.append(f"{_STEP_DONE} {line}")
        elif running:
            # A directive's label ends at the first unescaped bracket, and aspects are free text.
            label = re.sub(r"([\\\[\]])", r"\\\1", line)
            shown.append(f"{_STEP_NOW} :shimmer[{label}]")
        else:
            shown.append(f"{_STEP_FAILED} {line}")
    return "\n\n".join(shown)


class _Cancelled(Exception):
    """Raised by a request's progress callback once a newer request replaced it (SCHEMA.md §6.1)."""


_FAILED = ("The request could not be completed. Check that the knowledge base and indexes are built "
           "and that the LLM key is set in .env, then try again.")


@dataclass
class _Request:
    """An Ask or Search request, kept in session state while it runs.

    A rerun during the request (a tab or an example click) stops the script run, not the request:
    the next run follows it again from the steps shown so far. Only a new submit replaces it.
    """

    ask: bool
    future: Future
    events: queue.Queue
    cancel: threading.Event
    started: float = field(default_factory=time.perf_counter)
    lines: list[str] = field(default_factory=list)


def _start(question: str, ask: bool) -> _Request:
    """Start the request in a worker thread. Its progress callback queues each step and, once the
    request is cancelled, raises, which ends the search at its next step."""
    events: queue.Queue[tuple[str, dict[str, object]]] = queue.Queue()
    cancel = threading.Event()

    def progress(event: str, details: Mapping[str, object]) -> None:
        if cancel.is_set():
            raise _Cancelled
        events.put((event, dict(details)))

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ask")
    work = downstream.run if ask else orchestration.run_query
    future = pool.submit(work, question, progress=progress)
    pool.shutdown(wait=False)  # the worker ends on its own; nothing here waits for it
    return _Request(ask, future, events, cancel)


def _keep_outcome(request: _Request) -> None:
    """Move a finished request's result (or error) and steps into session state and drop it, before
    the page shows them, so a rerun from here on finds the outcome instead of losing it."""
    try:
        result = request.future.result()
    except downstream.DownstreamError as exc:
        st.session_state["error"] = str(exc)
        st.session_state["steps"] = ("Stopped: the request failed", request.lines)
    except Exception:
        # Do not expose provider responses, credentials, or tracebacks in the UI.
        st.session_state["error"] = _FAILED
        st.session_state["steps"] = ("Stopped: the request failed", request.lines)
    else:
        if request.ask:
            st.session_state["qa_result"] = result
            st.session_state["search_result"] = result.search_result
        else:
            st.session_state["search_result"] = result
        st.session_state["steps"] = (_finished_label(result, time.perf_counter() - request.started), request.lines)
    del st.session_state["request"]


def _follow(request: _Request) -> None:
    """Show the request's steps and the seconds so far until it ends, then its outcome."""
    status = st.status(f"Working on your question ({int(time.perf_counter() - request.started)} s)",
                       expanded=True)
    # One text element for all steps: it replaces the previous request's steps at once, where
    # one element per line would leave its later lines greyed out until this run caught up.
    body = status.empty()
    if request.lines:
        body.markdown(_steps_markdown(request.lines, running=True))
    shown = -1
    while True:
        done = request.future.done()  # checked first: every event of a finished run is queued by now
        while not request.events.empty():
            line = _step_line(*request.events.get())
            if line:
                request.lines.append(line)
                body.markdown(_steps_markdown(request.lines, running=True))
        if done:
            break
        elapsed = int(time.perf_counter() - request.started)
        if elapsed != shown:  # an update without `expanded` would fold the box away
            status.update(label=f"Working on your question ({elapsed} s)", expanded=True)
            shown = elapsed
        time.sleep(0.2)
    _keep_outcome(request)
    label, lines = st.session_state["steps"]
    failed = "error" in st.session_state
    if lines:  # the last step is no longer under way
        body.markdown(_steps_markdown(lines, failed=failed))
    status.update(label=label, state="error" if failed else "complete", expanded=False)
    if failed:
        st.error(st.session_state["error"])


def _finished_label(result: downstream.DownstreamResult | FinalResult, seconds: float) -> str:
    took = _seconds(seconds * 1000)
    if isinstance(result, downstream.DownstreamResult):
        if result.status == "answered":
            return f"Answer ready in {took}"
        return f"Finished in {took}: no paper covers this question well enough to answer it"
    if result.papers:
        return f"Search finished in {took}: {_count(len(result.papers), 'paper')} accepted"
    return f"Search finished in {took}: no paper accepted"


def _replay_steps() -> None:
    """The finished request's status box (and error), kept on later reruns (a tab click, a download)."""
    label, lines = st.session_state["steps"]
    failed = "error" in st.session_state
    with st.status(label, state="error" if failed else "complete", expanded=False):
        st.markdown(_steps_markdown(lines, failed=failed))
    if failed:
        st.error(st.session_state["error"])


def _explain_empty(search: FinalResult, papers_in_kb: int) -> None:
    """Why a search accepted no paper, and the closest papers the reviewer turned down."""
    report = search.coverage_report
    kb = f"the {_count(papers_in_kb, 'paper')} in the knowledge base" if papers_in_kb else "the knowledge base"
    if report.get("termination_reason") == "plain_rag":
        how = "Retrieval found no passage for it."
    else:
        rounds = int(report.get("rounds", 1) or 1)  # type: ignore[call-overload]
        how = (f"The search ran {_count(rounds, 'round')} and the reviewer read "
               f"{_count(int(report.get('papers_reviewed', 0) or 0), 'paper')}, "  # type: ignore[call-overload]
               "but none covered the question well enough to cite.")
    st.info(f"No paper from {kb} answers this question. {how} Try naming the method, task or dataset you "
            "mean; questions outside computation and language, AI and machine learning have no papers here.")
    turned_down = sorted((record for record in search.reviews if record.verdict != "pass"),
                         key=lambda record: passing_order_key(record.paper_id, record.score, search.retrieval_ranks))
    if turned_down:
        with st.expander(f"Closest papers the reviewer turned down ({min(len(turned_down), 3)})", type="compact"):
            for record in turned_down[:3]:
                title = html.escape(_one_line(str(record.evidence.get("title") or record.paper_id)))
                st.markdown(f'<p class="po-entry-title">{title}</p>'
                            f'<p class="po-meta">Reviewer score {record.score}/5</p>', unsafe_allow_html=True)
                if record.reasons:
                    st.caption(record.reasons)


@st.cache_resource(show_spinner=False)
def _warm_up() -> Future[object]:
    """Load the retrieval models and indexes in the background once per server, on the first page view.

    The first question then does not wait about 20 s for them (D58). A failure here is left for
    the first real request to report.
    """
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="warm-up")
    future = pool.submit(retrieve, "warm up")
    pool.shutdown(wait=False)
    return future


def _examples(cfg: dict) -> None:
    """The configured example inputs; a click copies one into the question box."""
    with st.container(key="examples", gap="xxsmall"):
        for number, example in enumerate(cfg["ui"].get("example_inputs", [])):
            st.button(example, key=f"example_{number}", type="tertiary", icon=":material/north_west:",
                      width="stretch", on_click=_use_example, args=(example,))


def _use_example(example: str) -> None:
    st.session_state["question"] = example


def _mark_asked() -> None:
    """Before the run a submit starts: a question switches the page to the results layout, so the
    run that starts the request already draws it."""
    if str(st.session_state.get("question") or "").strip():
        st.session_state["asked"] = True


def _question_line(text: str) -> None:
    st.markdown(f'<p class="po-question">{html.escape(text)}</p>', unsafe_allow_html=True)


def _citation(match: re.Match[str], sources: Mapping[int, downstream.EvidenceSource]) -> str:
    """One [n] of the answer as a link to its reference, with the source's title as its tooltip.

    The brackets stay visible, so neighbouring citations read [1][2], not 12. Set as a superscript,
    a citation joins the word before it: a word joiner replaces the space, which also keeps the two
    on one line and keeps notation such as W[0] from reading as a reference link with the citation.
    """
    space, number = match[1], match[2]
    source = sources.get(int(number))
    tip = ""
    if source is not None:
        label = _one_line(source.title) + (f" ({source.year})" if source.year else "")
        tip = ' "' + label.replace("\\", "\\\\").replace('"', '\\"') + '"'
    joiner = "⁠" if space and "\n" not in space else space
    return f"{joiner}[\\[{number}\\]](#source-{number}{tip})"


def _short_section(section: str, limit: int = 40) -> str:
    """A section heading short enough for a reference's one-line details."""
    section = _one_line(section)
    return section if len(section) <= limit else section[:limit - 1].rstrip(" ,;:/-") + "…"


def _references(cited: Sequence[downstream.EvidenceSource]) -> None:
    """The cited sources in the margin, like a paper's reference list; each excerpt opens on demand."""
    with st.container(key="sources"):
        st.subheader("References", anchor=False)
        for source in cited:
            meta = [html.escape(part) for part in (str(source.year) if source.year else "",
                                                   _short_section(source.section) if source.section else "") if part]
            if source.url:
                meta.append(_link(source.url, "arXiv"))
            details = f'<p class="po-meta">{" · ".join(meta)}</p>' if meta else ""
            with st.container(key=f"ref_{source.number}", gap="xsmall"):
                st.markdown(f'<div id="source-{source.number}"><p class="po-entry-title">'
                            f'<span class="po-n">[{source.number}]</span>{html.escape(_one_line(source.title))}</p>'
                            f"{details}</div>", unsafe_allow_html=True)
                with st.expander("Excerpt", type="compact"):
                    st.text(source.text)


def _show_answer(result: downstream.DownstreamResult, papers_in_kb: int) -> None:
    cited = [source for source in result.sources if source.number in result.cited_source_numbers]
    # On wide screens the references sit in the margin beside the answer; on phones they follow it.
    main, margin = st.columns([1.55, 1], gap="large")
    with main:
        _question_line(result.input_text)
        if result.status == "insufficient_evidence":
            _explain_empty(result.search_result, papers_in_kb)
        else:
            by_number = {source.number: source for source in result.sources}
            answer = downstream.sub_citations(result.answer, lambda match: _citation(match, by_number))
            with st.container(key="answer"):
                st.markdown(answer)
        latency = result.latency_by_stage
        parts = [f"{_seconds(latency['search_ms'])} searching papers"] if "search_ms" in latency else []
        if latency.get("generation_ms"):  # no excerpts, no generation call
            parts.append(f"{_seconds(latency['generation_ms'])} writing the answer")
        timing = f"Took {_seconds(latency['total_ms'])}" + (f": {', '.join(parts)}." if len(parts) > 1 else ".")
        if result.search_result.coverage_report.get("cache_hit"):
            timing += " The paper search came from the cache."
        with st.container(key="timing", gap="xxsmall"):
            st.caption(timing)
            st.download_button("Download answer and sources (JSON)",
                               json.dumps(asdict(result), ensure_ascii=False, indent=2),
                               file_name="paper-qa.json", mime="application/json", type="tertiary",
                               icon=":material/download:")
    if cited:
        with margin:
            _references(cited)


def _search_summary(result: FinalResult) -> str:
    papers = _count(len(result.papers), "paper")
    order = ", best first." if len(result.papers) > 1 else "."
    report = result.coverage_report
    if report.get("termination_reason") == "plain_rag":
        return f"{papers} from one retrieval, without review{order}"
    reviewed = report.get("papers_reviewed")
    if isinstance(reviewed, int) and reviewed:
        return f"{papers} passed review out of {reviewed:,} reviewed{order}"
    return f"{papers} passed review{order}"


def _paper_card(rank: int, paper: ReviewRecord, passages: Mapping[str, Mapping[str, object]]) -> None:
    """One accepted paper: rank, title (linked to arXiv), year, score, the reviewer's reasons and the
    aspects it covers; the passages the reviewer read open on demand."""
    found = [passages[pid] for pid in paper.evidence.get("top_passages", []) if pid in passages]
    title = html.escape(_one_line(str(paper.evidence.get("title") or paper.paper_id)))
    year = next((record.get("year") for record in found if record.get("year")), None)
    arxiv_id = str(next((record.get("arxiv_id") for record in found if record.get("arxiv_id")), None)
                   or paper.paper_id)
    url = _arxiv_url(arxiv_id)
    # Score 0 marks the plain retrieval baseline, which reviews nothing (D55).
    meta = [f"arXiv {arxiv_id}" if url else "", str(year) if year else "",
            f"Reviewer score {paper.score}/5" if paper.score else "Not reviewed (plain retrieval)"]
    parts = [f'<p class="po-entry-title"><span class="po-n">{rank}</span>{_link(url, title) if url else title}</p>',
             f'<p class="po-meta">{html.escape(" · ".join(part for part in meta if part))}</p>']
    if paper.reasons:
        parts.append(f'<p class="po-reasons">{html.escape(_one_line(paper.reasons))}</p>')
    if paper.covered:  # the reviewer writes aspects as sentences; drop their full stops
        covers = "; ".join(_one_line(aspect).rstrip(". ") for aspect in paper.covered)
        parts.append(f'<p class="po-covers">Covers: {html.escape(covers)}.</p>')
    with st.container(key=f"paper_{rank}", gap="xsmall"):
        st.markdown(f'<div class="po-paper">{"".join(parts)}</div>', unsafe_allow_html=True)
        if found:
            with st.expander(f"{_count(len(found), 'passage')} the reviewer read", type="compact"):
                for record in found:
                    if record.get("section"):
                        st.caption(str(record["section"]))
                    st.text(str(record.get("text", "")))


def _about(result: FinalResult) -> None:
    """The margin of Paper Search: why the search ended, what it read, and how long each stage took."""
    report = result.coverage_report
    st.subheader("About this search", anchor=False)
    rounds = report.get("rounds", 1)
    ending = _ENDINGS.get(str(report.get("termination_reason")), str(report.get("termination_reason", "")))
    st.markdown(f"Ended after {rounds} round{'s' if rounds != 1 else ''}: {ending}.")
    rows = []
    if "papers_reviewed" in report:
        rows += [("Papers reviewed", str(report["papers_reviewed"])),
                 ("Papers accepted", str(report.get("papers_passed", len(result.papers))))]
        if report.get("re_reviews"):
            rows.append(("Reviewed again with new passages", str(report["re_reviews"])))
    if report.get("total_aspects"):
        rows.append(("Aspects of the question covered", f"{report.get('covered_aspects', 0)} of {report['total_aspects']}"))
    if rows:
        st.markdown(_table(rows), unsafe_allow_html=True)
    missing = report.get("uncovered_aspects") or []
    if report.get("total_aspects") and missing:
        st.markdown("Not covered: " + "; ".join(missing) + ".")  # type: ignore[arg-type]
    if report.get("cache_hit"):
        st.markdown("Loaded from the search cache, so the times below are those of the original search.")
    # Stages that did not run (HyDE off, no follow-up round) are left out.
    stages = [(_STAGES.get(stage, stage.removesuffix("_ms")), f"{ms / 1000:.2f}")
              for stage, ms in result.latency_by_stage.items()
              if isinstance(ms, int | float) and (ms > 0 or stage == "total_ms")]
    if stages:
        st.markdown(_table(stages, header=("Stage", "Seconds"), total="total_ms" in result.latency_by_stage),
                    unsafe_allow_html=True)
        st.caption("Retrieval and review overlap, so the stages can add up to more than the total.")
    st.download_button("Download search record (JSON)", json.dumps(asdict(result), ensure_ascii=False, indent=2),
                       file_name="paper-search.json", mime="application/json", type="tertiary",
                       icon=":material/download:")


def _show_search(result: FinalResult, papers_in_kb: int) -> None:
    main, margin = st.columns([1.55, 1], gap="large")
    with main:
        _question_line(result.input_text)
        if result.papers:
            st.caption(_search_summary(result))
        else:
            _explain_empty(result, papers_in_kb)
        ids = [pid for paper in result.papers for pid in paper.evidence.get("top_passages", [])]
        try:
            passages = get_lookup_store().get_passages(ids)  # runs on every rerun: keep it a key lookup
        except (OSError, RuntimeError, sqlite3.Error, ValueError, KeyError):
            passages = {}
            st.warning("Paper excerpts are unavailable: the retrieval lookup store could not be read.")
        with st.container(key="papers"):
            for rank, paper in enumerate(result.papers, start=1):
                _paper_card(rank, paper, passages)
    with margin, st.container(key="about"):
        _about(result)


def _question_column(results: bool):
    """Where the question box goes: centred on the landing, over the main column once there are results."""
    if results:
        return st.container(key="top").columns([1.55, 1], gap="large")[0]
    return st.container(horizontal_alignment="center").container(width=720)


def render() -> None:
    cfg = get_config()
    st.set_page_config(page_title="PaperOdyssey", page_icon=":material/menu_book:", layout="wide")
    st.html(_STYLE)
    _warm_up()
    papers_in_kb = _corpus_size(cfg)
    # The landing turns into the results layout with the first question (`_mark_asked`).
    results = bool(st.session_state.get("asked")) or "request" in st.session_state
    if results:
        _topbar(cfg, papers_in_kb)
    else:
        _hero(cfg, papers_in_kb)
    top = _question_column(results)
    with top, st.container(key="ask"), st.form("question_form", border=False):
        question = st.text_area("Your question", key="question", height=104, label_visibility="collapsed",
                                placeholder="Ask a research question, for example: how do retrieval-augmented "
                                            "models handle conflicting evidence?")
        with st.container(horizontal=True, gap="small"):
            ask = st.form_submit_button("Ask", key="ask_button", type="primary", on_click=_mark_asked,
                                        help="Search the papers, then write an answer that cites them.")
            search_only = st.form_submit_button("Search papers only", key="search_button", on_click=_mark_asked,
                                                help="List the papers that pass review, without an answer.")
    submitted = ask or search_only
    if submitted:
        replaced = st.session_state.get("request")
        if replaced is not None:
            replaced.cancel.set()  # its search stops at its next step; its result is never shown
        for key in ("request", "qa_result", "search_result", "steps", "error"):
            st.session_state.pop(key, None)
        if question.strip():
            # Set before the tabs are drawn: the tab of the new result opens by itself, so a
            # search-only result is never behind Paper QA (tracking the tab needs on_change).
            st.session_state["tab"] = "Paper QA" if ask else "Paper Search"
            st.session_state["asked"] = True
            st.session_state["request"] = _start(question, ask)
            if not results:  # only if `_mark_asked` did not run: draw the results layout instead
                st.rerun()
    running = "request" in st.session_state
    # A request is followed only after the whole page is drawn. Its status and everything below the
    # question box start as empty slots, which drops the previous result (or the landing) at once
    # instead of leaving it greyed out (stale) while it runs; other reruns keep plain containers, so
    # nothing is redrawn.
    fresh = submitted or running
    with top:
        status_slot = st.empty() if fresh else st.container()
    body = st.empty() if fresh else st.container()
    with body.container() if fresh else body:
        if results:
            qa_tab, search_tab = st.tabs(["Paper QA", "Paper Search"], key="tab", on_change="rerun")
            qa_slot = qa_tab.empty() if fresh else qa_tab.container()
            search_slot = search_tab.empty() if fresh else search_tab.container()
        else:
            _landing(cfg)
    with status_slot.container() if fresh else status_slot, st.container(key="progress"):
        if submitted and not question.strip():
            st.warning("Enter a question first.")
        elif running:
            _follow(st.session_state["request"])
        elif "steps" in st.session_state:
            _replay_steps()
    if not results:
        return
    with qa_slot.container() if fresh else qa_slot:
        if "qa_result" in st.session_state:
            _show_answer(st.session_state["qa_result"], papers_in_kb)
        else:
            st.caption("Select Ask to get an answer that cites its sources by number.")
    with search_slot.container() if fresh else search_slot:
        if "search_result" in st.session_state:
            _show_search(st.session_state["search_result"], papers_in_kb)
        else:
            st.caption("Papers that pass review appear here, best first, with the passages the reviewer read.")


if __name__ == "__main__":
    render()
