"""Minimal Streamlit paper search and cited QA. Run: streamlit run ui/app.py."""

from __future__ import annotations

import json
import re
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


def _use_example() -> None:
    example = st.session_state.get("example", "")
    if example:
        st.session_state["question"] = example


def _show_answer(result: downstream.DownstreamResult) -> None:
    st.caption(f"Question: {result.input_text}")
    if result.status == "insufficient_evidence":
        st.info(result.answer)
    else:
        answer = re.sub(r"\[(\d+)\]", r"[\1](#source-\1)", result.answer)
        st.markdown(answer)
    for source in result.sources:
        if source.number not in result.cited_source_numbers:
            continue
        st.markdown(f'<span id="source-{source.number}"></span>', unsafe_allow_html=True)
        with st.expander(f"[{source.number}] {source.title}"):
            details = " · ".join(str(v) for v in (source.year, source.section) if v)
            if details:
                st.caption(details)
            if source.url:
                st.link_button("Open paper on arXiv", source.url)
            st.text(source.text)
    st.caption(f"Total: {result.latency_by_stage['total_ms'] / 1000:.2f}s · "
               f"Answer generation: {result.latency_by_stage['generation_ms'] / 1000:.2f}s")
    st.download_button("Download answer and sources", json.dumps(asdict(result), ensure_ascii=False, indent=2),
                       file_name="paper-qa.json", mime="application/json")


def _show_search(result: FinalResult) -> None:
    st.caption(f"Search: {result.input_text}")
    if not result.papers:
        st.info("No papers matched this question.")
        return
    ids = [pid for paper in result.papers for pid in paper.evidence.get("top_passages", [])]
    try:
        passages = get_lookup_store().get_passages(ids)  # runs on every rerun: keep it a key lookup
    except (OSError, RuntimeError, sqlite3.Error, ValueError, KeyError):
        passages = {}
        st.warning("Paper excerpts are unavailable: the retrieval lookup store could not be read.")
    for paper in result.papers:
        with st.expander(str(paper.evidence.get("title") or paper.paper_id)):
            st.caption(f"Review: {paper.verdict} · Score: {paper.score}/5")
            st.write(paper.reasons)
            for pid in paper.evidence.get("top_passages", []):
                if pid in passages:
                    st.text(passages[pid]["text"])
    with st.expander("Search details"):
        st.json(result.coverage_report)
        st.json(result.latency_by_stage)


def render() -> None:
    cfg = get_config()
    st.set_page_config(page_title="PaperOdyssey", layout="centered")
    st.title(cfg["ui"].get("title", "PaperOdyssey"))
    st.caption("Ask a question and read an answer with evidence from research papers.")
    st.selectbox("Example question", [""] + cfg["ui"].get("example_inputs", []),
                 key="example", on_change=_use_example)
    with st.form("question_form"):
        question = st.text_area("Your question", key="question")
        ask = st.form_submit_button("Ask", type="primary")
        search_only = st.form_submit_button("Search papers only")
    if ask or search_only:
        st.session_state.pop("qa_result", None)
        st.session_state.pop("search_result", None)
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Searching papers and preparing your answer…" if ask else "Searching papers…"):
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
                st.error("The request could not be completed. Check the knowledge base, retrieval index, and LLM configuration.")
    qa_tab, search_tab = st.tabs(["Paper QA", "Paper Search"])
    with qa_tab:
        if "qa_result" in st.session_state:
            _show_answer(st.session_state["qa_result"])
        else:
            st.info("Enter a question and select Ask to generate a cited answer.")
    with search_tab:
        if "search_result" in st.session_state:
            _show_search(st.session_state["search_result"])
        else:
            st.info("Retrieved papers will appear here.")


if __name__ == "__main__":
    render()
