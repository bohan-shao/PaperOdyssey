"""Paper Search -> stored evidence -> a single cited answer generation."""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Any

from config import get_config
from llm.client import get_client
from orchestration import run_query
from orchestration.types import FinalResult, passing_order_key
from retrieval.lookup import get_lookup_store

from .prompts import ANSWER_SCHEMA, PROMPT_VERSION, build_answer_prompt
from .types import DownstreamResult, EvidenceSource

_CITATION = re.compile(r"\[(\d+)\]")
_NUMERIC_BRACKET = re.compile(r"\[\s*\d[\d\s,;\-–]*\]")
_SINGLE = re.compile(r"\[\s*(\d+)\s*\]")
_RANGE = re.compile(r"\s*(\d+)\s*[-–]\s*(\d+)\s*")
_NUMBERS = re.compile(r"\s*\d+(?:\s+\d+)*\s*")
_ARXIV_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z.\-]*/\d{7})(?:v\d+)?")


class DownstreamError(RuntimeError):
    """Search, corpus, generation, or answer-format failure (not lack of evidence)."""


def _limit(settings: dict[str, Any], key: str, default: int) -> int:
    value = settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"downstream.{key} must be a positive integer")
    return value


def _sources(search: FinalResult, cfg: dict[str, Any]) -> list[EvidenceSource]:
    settings = cfg.get("downstream", {})
    max_papers = _limit(settings, "max_papers", 5)
    per_paper = _limit(settings, "passages_per_paper", 2)
    char_limit = _limit(settings, "max_chars_per_passage", 2400)
    selected: list[tuple[str, str]] = []
    seen_papers: set[str] = set()
    ranks = search.retrieval_ranks
    for paper in sorted(search.papers, key=lambda p: passing_order_key(p.paper_id, p.score, ranks)):
        if paper.verdict != "pass" or paper.paper_id in seen_papers:
            continue
        seen_papers.add(paper.paper_id)
        ids = paper.evidence.get("top_passages", [])
        if isinstance(ids, list):
            ids = list(dict.fromkeys(p for p in ids if isinstance(p, str) and p.strip()))
            selected.extend((paper.paper_id, pid) for pid in ids[:per_paper])
        if len(seen_papers) >= max_papers:
            break
    if not selected:
        return []

    # Primary-key lookups in the store the retrieval hits came from (D28), not a corpus scan.
    passages = get_lookup_store().get_passages(pid for _, pid in selected)
    sources: list[EvidenceSource] = []
    seen: set[str] = set()
    for paper_id, passage_id in selected:
        passage = passages.get(passage_id)
        if not passage or passage_id in seen:
            continue
        # Never attach another paper's text to this paper, including stale review caches.
        if passage.get("paper_id") != paper_id:
            continue
        raw_text = passage.get("text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue
        seen.add(passage_id)
        arxiv_id = str(passage.get("arxiv_id") or paper_id)
        url = f"https://arxiv.org/abs/{arxiv_id}" if _ARXIV_ID.fullmatch(arxiv_id) else None
        year = passage.get("year")
        sources.append(EvidenceSource(
            number=len(sources) + 1,
            paper_id=paper_id,
            passage_id=passage_id,
            title=str(passage.get("title") or paper_id),
            year=year if isinstance(year, int) and not isinstance(year, bool) else None,
            section=str(passage.get("section") or ""),
            url=url,
            text=raw_text[:char_limit],
        ))
    return sources


def _group_numbers(inner: str, source_count: int) -> list[int] | None:
    """Sources named by a bracket body such as "1, 2" or "1-3"; None if it is not a citation.

    Every number must be a supplied source, so an interval such as [0, 1] stays plain text.
    """
    numbers: list[int] = []
    for part in re.split(r"[,;]", inner):
        if bounds := _RANGE.fullmatch(part):
            low, high = int(bounds[1]), int(bounds[2])
            if not 1 <= low <= high <= source_count:
                return None
            numbers.extend(range(low, high + 1))
        elif _NUMBERS.fullmatch(part):
            values = [int(n) for n in part.split()]
            if not all(1 <= n <= source_count for n in values):
                return None
            numbers.extend(values)
        else:
            return None
    return numbers


def _normalize_citations(answer: str, source_count: int) -> str:
    """Rewrite [1, 2] / [1-3] / [01] as [1][2] / [1][2][3] / [1]; keep non-citation brackets."""
    def rewrite(match: re.Match[str]) -> str:
        if single := _SINGLE.fullmatch(match[0]):
            return f"[{int(single[1])}]"  # a lone number is always a citation; bounds checked later
        numbers = _group_numbers(match[0][1:-1], source_count)
        return match[0] if numbers is None else "".join(f"[{n}]" for n in dict.fromkeys(numbers))

    return _NUMERIC_BRACKET.sub(rewrite, answer)


def _parse_answer(result: object, source_count: int) -> tuple[str, str, list[int]]:
    if not isinstance(result, dict):
        raise ValueError("answer response must be an object")
    status, answer = result.get("status"), result.get("answer")
    if status not in {"answered", "insufficient_evidence"}:
        raise ValueError("answer response has an invalid status")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer response is empty")
    answer = _normalize_citations(answer.strip(), source_count)
    numbers = list(dict.fromkeys(int(n) for n in _CITATION.findall(answer)))
    if any(n < 1 or n > source_count for n in numbers):
        raise ValueError("answer cites a source that was not supplied")
    if status == "answered" and not numbers:
        raise ValueError("answer is missing evidence citations")
    return status, answer, numbers


def run(input_text: str, query_id: str | None = None) -> DownstreamResult:
    """Answer a question from Paper Search evidence; never bypass the retrieval contract.

    Empty input raises ValueError. Operational failures raise DownstreamError. Missing
    evidence returns status=insufficient_evidence without calling the answer model.
    """
    if not isinstance(input_text, str) or not input_text.strip():
        raise ValueError("Enter a question first.")
    question = input_text.strip()
    started = time.perf_counter()
    _limit(get_config().get("downstream", {}), "max_tokens", 1024)  # fail before a paid search
    try:
        search = run_query(question, query_id=query_id)
    except Exception as exc:
        raise DownstreamError("Paper search failed. Check the retrieval index and search configuration.") from exc
    return answer_from_search(question, search, started=started)


def answer_from_search(question: str, search: FinalResult, *, started: float | None = None) -> DownstreamResult:
    """The second half of `run`: evidence and one cited answer for a finished Paper Search result.

    The answer evaluation (eval/answer_eval.py, D49) calls it on stored search results, so the UI
    and the evaluation share one evidence and generation path. `started` is the caller's
    `time.perf_counter()` before the search (search_ms is 0 without it). Raises like `run`.
    """
    search_end = time.perf_counter()
    started = search_end if started is None else started
    question = question.strip()
    cfg = get_config()
    max_tokens = _limit(cfg.get("downstream", {}), "max_tokens", 1024)
    try:
        sources = _sources(search, cfg)
    except (OSError, RuntimeError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
        raise DownstreamError("Could not read paper evidence. Check the lookup store (scripts/build_indexes.py).") from exc
    evidence_end = time.perf_counter()
    if sources:
        try:
            reply = get_client().chat_json(build_answer_prompt(question, sources), ANSWER_SCHEMA,
                                           max_tokens=max_tokens)
            status, answer, cited = _parse_answer(reply, len(sources))
        except Exception as exc:
            raise DownstreamError("Could not produce a valid cited answer. Check the LLM configuration or try again.") from exc
    else:
        status, cited = "insufficient_evidence", []
        answer = (
            "未找到足够的论文证据回答该问题。请尝试补充关键词或调整问题。"
            if re.search(r"[\u4e00-\u9fff]", question)
            else "No sufficient paper evidence was found to answer this question. Try adding keywords or rephrasing it."
        )
    finished = time.perf_counter()
    return DownstreamResult(
        query_id=search.query_id, input_text=question, answer=answer, status=status,
        sources=sources, cited_source_numbers=cited, search_result=search,
        latency_by_stage={
            "search_ms": round((search_end - started) * 1000),
            "evidence_ms": round((evidence_end - search_end) * 1000),
            "generation_ms": round((finished - evidence_end) * 1000) if sources else 0,
            "total_ms": round((finished - started) * 1000),
        },
        prompt_version=PROMPT_VERSION,
    )
