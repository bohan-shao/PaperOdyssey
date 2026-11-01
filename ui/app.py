"""Streamlit paper search and cited QA. Run: bash scripts/run_ui.sh (theme: .streamlit/config.toml)."""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

import streamlit as st

# Streamlit may place only ui/ on sys.path when launched by the CLI.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import downstream
import orchestration
from config import get_config
from orchestration.types import FinalResult
from retrieval.lookup import get_lookup_store

# Browser surfaces and the reading column. Colours match .streamlit/config.toml; the serif is
# the face Streamlit bundles, so nothing is downloaded.
_STYLE = """
<style>
::selection { background: #F1D9D5; color: #1F1C18; }
textarea { caret-color: #9B2C2C; }
[data-testid="stMainBlockContainer"] { padding-top: 3rem; }
[data-testid="stCaptionContainer"] { color: #5E584F; opacity: 1; }
.st-key-answer p, .st-key-answer li { font-family: "Source Serif", serif; font-size: 1.0625rem;
  line-height: 1.65; max-width: 68ch; text-wrap: pretty; }
.st-key-answer a { text-underline-offset: 0.15em; font-variant-numeric: tabular-nums; white-space: nowrap; }
.st-key-sources [data-testid="stText"], .st-key-papers [data-testid="stText"] {
  padding-left: 0.875rem; border-left: 1px solid #DDD6CA; margin: 0.25rem 0 0.75rem; }
.st-key-sources [data-testid="stText"] *, .st-key-papers [data-testid="stText"] * {
  font-family: "Source Serif", serif; font-size: 0.9375rem; line-height: 1.6; white-space: pre-wrap; }
.st-key-details td, .st-key-timing { font-variant-numeric: tabular-nums; }
</style>
"""

_STAGES = {"understanding_ms": "Query understanding", "hyde_ms": "Hypothetical abstract (HyDE)",
           "retrieval_ms": "Retrieval", "review_ms": "Review", "rewrite_ms": "Follow-up queries",
           "total_ms": "Total"}
_ENDINGS = {"coverage_met": "every aspect of the question was covered",
            "min_papers": "enough papers passed review",
            "max_rounds": "the round limit was reached",
            "loop_disabled": "the follow-up loop is switched off",
            "rewrite_exhausted": "no new follow-up queries were left",
            "plain_rag": "one retrieval, without review"}


def _seconds(ms: float) -> str:
    return f"{ms / 1000:.1f} s"


def _use_example() -> None:
    example = st.session_state.get("example")
    if example:
        st.session_state["question"] = example


def _show_answer(result: downstream.DownstreamResult) -> None:
    st.caption(f"Question: {result.input_text}")
    if result.status == "insufficient_evidence":
        st.info(result.answer)
    else:
        # Brackets stay visible, so neighbouring citations read [1][2], not 12.
        answer = downstream.sub_citations(result.answer, lambda m: f"{m[1]}[\\[{m[2]}\\]](#source-{m[2]})")
        with st.container(key="answer"):
            st.markdown(answer)
    cited = [source for source in result.sources if source.number in result.cited_source_numbers]
    if cited:
        st.markdown("#### Sources")
    with st.container(key="sources"):
        for source in cited:
            st.markdown(f'<span id="source-{source.number}"></span>', unsafe_allow_html=True)
            with st.expander(f"[{source.number}] {source.title}"):
                details = ", ".join(part for part in (f"Published {source.year}" if source.year else "",
                                                      f"{source.section} section" if source.section else "") if part)
                if details:
                    st.caption(details)
                st.text(source.text)
                if source.url:
                    st.link_button("Open on arXiv", source.url)
    latency = result.latency_by_stage
    parts = [f"{_seconds(latency['search_ms'])} searching papers"] if "search_ms" in latency else []
    if latency.get("generation_ms"):  # no excerpts, no generation call
        parts.append(f"{_seconds(latency['generation_ms'])} writing the answer")
    timing = f"Took {_seconds(latency['total_ms'])}" + (f": {', '.join(parts)}." if len(parts) > 1 else ".")
    if result.search_result.coverage_report.get("cache_hit"):
        timing += " The paper search came from the cache."
    with st.container(key="timing"):
        st.caption(timing)
    st.download_button("Download answer and sources (JSON)", json.dumps(asdict(result), ensure_ascii=False, indent=2),
                       file_name="paper-qa.json", mime="application/json")


def _show_search(result: FinalResult) -> None:
    st.caption(f"Search: {result.input_text}")
    if not result.papers:
        st.info("No paper passed review for this question. Try naming the method, task or dataset you mean.")
    ids = [pid for paper in result.papers for pid in paper.evidence.get("top_passages", [])]
    try:
        passages = get_lookup_store().get_passages(ids)  # runs on every rerun: keep it a key lookup
    except (OSError, RuntimeError, sqlite3.Error, ValueError, KeyError):
        passages = {}
        st.warning("Paper excerpts are unavailable: the retrieval lookup store could not be read.")
    with st.container(key="papers"):
        for paper in result.papers:
            pids = [pid for pid in paper.evidence.get("top_passages", []) if pid in passages]
            year = next((passages[pid].get("year") for pid in pids if passages[pid].get("year")), None)
            with st.expander(str(paper.evidence.get("title") or paper.paper_id)):
                # Score 0 marks the plain retrieval baseline, which reviews nothing (D55).
                verdict = f"Reviewer score {paper.score} of 5" if paper.score else "Not reviewed (plain retrieval)"
                st.caption(verdict + (f", published {year}." if year else "."))
                if paper.covered:  # the reviewer writes aspects as sentences; drop their full stops
                    st.caption("Covers: " + "; ".join(aspect.rstrip(". ") for aspect in paper.covered) + ".")
                if paper.reasons:
                    st.write(paper.reasons)
                for pid in pids:
                    st.text(passages[pid]["text"])
    report = result.coverage_report
    with st.expander("Search details"), st.container(key="details"):
        rounds = report.get("rounds", 1)
        ending = _ENDINGS.get(str(report.get("termination_reason")), str(report.get("termination_reason", "")))
        lines = [f"Search ended after {rounds} round{'s' if rounds != 1 else ''}: {ending}."]
        if "papers_reviewed" in report:
            line = f"Papers reviewed: {report['papers_reviewed']}, accepted: {report.get('papers_passed', len(result.papers))}"
            lines.append(line + (f", reviewed again with new passages: {report['re_reviews']}." if report.get("re_reviews")
                                 else "."))
        if report.get("total_aspects"):
            lines.append(f"Aspects of the question covered: {report.get('covered_aspects', 0)} of {report['total_aspects']}.")
            missing = report.get("uncovered_aspects") or []
            if missing:
                lines.append("Not covered: " + "; ".join(missing) + ".")
        if report.get("cache_hit"):
            lines.append("Loaded from the search cache, so the times below are those of the original search.")
        st.markdown("\n\n".join(lines))
        # Stages that did not run (HyDE off, no follow-up round) are left out.
        rows = [f"| {_STAGES.get(stage, stage.removesuffix('_ms'))} | {ms / 1000:.2f} |"
                for stage, ms in result.latency_by_stage.items()
                if isinstance(ms, int | float) and (ms > 0 or stage == "total_ms")]
        if rows:
            st.markdown("| Stage | Seconds |\n|---|---:|\n" + "\n".join(rows))
            st.caption("Retrieval and review overlap, so the stages can add up to more than the total.")
        st.download_button("Download search record (JSON)", json.dumps(asdict(result), ensure_ascii=False, indent=2),
                            file_name="paper-search.json", mime="application/json")


def render() -> None:
    cfg = get_config()
    st.set_page_config(page_title="PaperOdyssey", page_icon=":material/menu_book:", layout="centered")
    st.html(_STYLE)
    st.title(cfg["ui"].get("title", "PaperOdyssey"))
    st.caption("Ask a research question about NLP, AI or machine learning. "
               "The answer cites the arXiv paper excerpts it is based on.")
    st.selectbox("Example questions", cfg["ui"].get("example_inputs", []), index=None,
                 placeholder="Choose one to copy it into the question box", key="example", on_change=_use_example)
    with st.form("question_form"):
        question = st.text_area("Your question", key="question", height=104,
                                placeholder="For example: How do retrieval-augmented models handle conflicting evidence?")
        with st.container(horizontal=True, gap="small"):
            ask = st.form_submit_button("Ask", type="primary")
            search_only = st.form_submit_button("Search papers only")
    if ask or search_only:
        st.session_state.pop("qa_result", None)
        st.session_state.pop("search_result", None)
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            waiting = ("Searching papers and writing a cited answer. This can take a minute." if ask
                       else "Searching papers. This can take a minute.")
            try:
                with st.spinner(waiting):
                    if ask:
                        answer = downstream.run(question)
                        st.session_state["qa_result"] = answer
                        st.session_state["search_result"] = answer.search_result
                    else:
                        st.session_state["search_result"] = orchestration.run_query(question)
            except downstream.DownstreamError as exc:
                st.error(str(exc))
            except Exception:
                # Do not expose provider responses, credentials, or tracebacks in the UI.
                st.error("The request could not be completed. Check that the knowledge base and indexes are built "
                         "and that the LLM key is set in .env, then try again.")
    qa_tab, search_tab = st.tabs(["Paper QA", "Paper Search"])
    with qa_tab:
        if "qa_result" in st.session_state:
            _show_answer(st.session_state["qa_result"])
        else:
            st.caption("Select Ask to get an answer that cites its sources by number.")
    with search_tab:
        if "search_result" in st.session_state:
            _show_search(st.session_state["search_result"])
        else:
            st.caption("Papers that pass review appear here, best first, with the passages the reviewer read.")


if __name__ == "__main__":
    render()
