"""Parallel retrieval, review, and feedback loop for Paper Search."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Coroutine

from config import get_config, get_path
from kb.store import load_paper
from llm.client import get_client
from llm.prompt import TEMPLATE_VERSION, build_hyde_prompt
from retrieval.pipeline import retrieve
from retrieval.types import RetrievalFilters, RetrieveResult, ScoredPassage

from .errors import OrchestrationError
from .query_understanding import understand
from .review import batch_review
from .rewrite import rewrite
from .types import Criterion, FinalResult, QueryPlan, ReviewRecord, SubQuery


def _normalise_input(input_text: str) -> str:
    if not isinstance(input_text, str) or not (normalised := " ".join(input_text.split())):
        raise ValueError("input_text must be a non-empty string")
    return normalised


def _input_digest(input_text: str) -> str:
    return hashlib.sha256(input_text.encode("utf-8")).hexdigest()


def _auto_query_id(digest: str) -> str:
    return f"Q{digest[:12].upper()}"


def _config_fingerprint(cfg: dict[str, Any]) -> str:
    """The settings that can change a search result, as a stable string.

    It is part of the cache key, so flipping an ablation switch (loop on/off, HyDE, reviewer,
    retrieval settings, LLM model, chunking) or a prompt template change never returns a
    result that was computed under different settings.
    """
    kb_cfg = cfg.get("kb", {})
    relevant = {
        "orchestration": cfg.get("orchestration", {}),
        "retrieval": cfg.get("retrieval", {}),
        "llm": cfg.get("llm", {}),
        "kb": {"chunk": kb_cfg.get("chunk", {}), "store": kb_cfg.get("store", {})},
        "template_version": TEMPLATE_VERSION,
    }
    return json.dumps(relevant, sort_keys=True, default=str)


def _cache_digest(input_text: str, cfg: dict[str, Any]) -> str:
    return _input_digest(f"{input_text}\x00{_config_fingerprint(cfg)}")


def _cache_path(digest: str) -> Path:
    return get_path("results") / "final_results" / f"{digest}.json"


def _result_from_dict(raw: object) -> FinalResult:
    if not isinstance(raw, dict):
        raise ValueError("cached result must be an object")
    raw_plan = raw["query_plan"]
    if not isinstance(raw_plan, dict):
        raise ValueError("cached query_plan must be an object")
    plan = QueryPlan(
        input_text=str(raw_plan["input_text"]),
        sub_queries=[SubQuery(**item) for item in raw_plan.get("sub_queries", [])],
        criteria=[Criterion(**item) for item in raw_plan.get("criteria", [])],
    )
    return FinalResult(
        query_id=str(raw["query_id"]),
        input_text=str(raw["input_text"]),
        query_plan=plan,
        papers=[ReviewRecord(**item) for item in raw.get("papers", [])],
        coverage_report=dict(raw.get("coverage_report", {})),
        latency_by_stage={
            str(key): int(value) for key, value in raw.get("latency_by_stage", {}).items()
        },
    )


def _read_cache(path: Path, input_text: str, query_id: str) -> FinalResult | None:
    try:
        result = _result_from_dict(json.loads(path.read_text(encoding="utf-8")))
        if _normalise_input(result.input_text) != input_text:
            return None
        report = dict(result.coverage_report)
        report["cache_hit"] = True
        return replace(result, query_id=query_id, input_text=input_text, coverage_report=report)
    except Exception:
        # Cache data is untrusted and disposable; any corruption becomes a clean miss.
        return None


def _write_cache(path: Path, result: FinalResult) -> None:
    """Best-effort atomic cache write; cache failures do not fail a valid search."""
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(result), ensure_ascii=False, indent=2)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError):
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _to_filters(raw: dict[str, object] | None) -> RetrievalFilters | None:
    if not raw:
        return None
    return RetrievalFilters(
        year_min=raw.get("year_min") if isinstance(raw.get("year_min"), int) else None,
        year_max=raw.get("year_max") if isinstance(raw.get("year_max"), int) else None,
        categories=list(raw["categories"]) if isinstance(raw.get("categories"), list) else None,
    )


def _paper_database_path(cfg: dict[str, Any]) -> Path:
    filename = cfg.get("kb", {}).get("store", {}).get("sqlite_file", "papers.sqlite")
    return get_path("processed") / str(filename)


def _candidate_from_hits(
    paper_id: str,
    hits: list[ScoredPassage],
    database_path: Path,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    try:
        loaded = load_paper(paper_id, database_path)
        if isinstance(loaded, dict):
            metadata = loaded
    except Exception:
        # Metadata enrichment is optional; retrieval evidence remains sufficient to review.
        metadata = {}

    ranked = sorted(hits, key=lambda hit: (hit.rank, -hit.score, hit.passage_id))
    unique_passages: list[dict[str, str]] = []
    seen_passages: set[str] = set()
    for hit in ranked:
        if hit.passage_id in seen_passages:
            continue
        seen_passages.add(hit.passage_id)
        unique_passages.append({"passage_id": hit.passage_id, "text": hit.text})
        if len(unique_passages) == 3:
            break
    first = ranked[0]
    categories = metadata.get("categories", first.categories)
    if isinstance(categories, str):
        try:
            decoded = json.loads(categories)
            categories = decoded if isinstance(decoded, list) else [categories]
        except json.JSONDecodeError:
            categories = [categories]
    return {
        "paper_id": paper_id,
        "title": metadata.get("title") or first.title,
        "abstract": metadata.get("abstract") or "",
        "top_passages": unique_passages,
        "year": metadata.get("year", first.year),
        "categories": categories or [],
    }


def _candidates_from_result(result: RetrieveResult, database_path: Path) -> list[dict[str, Any]]:
    grouped: dict[str, list[ScoredPassage]] = {}
    for hit in result.hits:
        grouped.setdefault(hit.paper_id, []).append(hit)
    return [
        _candidate_from_hits(paper_id, hits, database_path)
        for paper_id, hits in grouped.items()
    ]


async def _run_round(
    queries: list[SubQuery],
    criteria: list[Criterion],
    input_text: str,
    round_no: int,
    seen_papers: set[str],
    cfg: dict[str, Any],
) -> tuple[list[ReviewRecord], int, int]:
    review_cfg = cfg["orchestration"].get("review", {})
    batch_size = max(1, int(review_cfg.get("batch_size", 5)))
    worker_count = max(1, int(review_cfg.get("max_concurrency", 2)))
    database_path = _paper_database_path(cfg)
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    seen_lock = asyncio.Lock()
    retrieval_intervals: list[tuple[float, float]] = []
    review_intervals: list[tuple[float, float]] = []

    async def producer(sub_query: SubQuery) -> None:
        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                retrieve,
                sub_query.query,
                filters=_to_filters(sub_query.filters),
            )
        except Exception as exc:
            raise OrchestrationError(
                f"retrieval failed for query {sub_query.query!r}: {exc}"
            ) from exc
        retrieval_intervals.append((started, time.perf_counter()))
        candidates = await asyncio.to_thread(_candidates_from_result, result, database_path)
        for candidate in candidates:
            async with seen_lock:
                paper_id = candidate["paper_id"]
                if paper_id in seen_papers:
                    continue
                seen_papers.add(paper_id)
            await queue.put(candidate)

    async def review_worker() -> list[ReviewRecord]:
        records: list[ReviewRecord] = []
        batch: list[dict[str, Any]] = []

        async def flush() -> None:
            if not batch:
                return
            started = time.perf_counter()
            reviewed = await asyncio.to_thread(
                batch_review,
                list(batch),
                criteria,
                input_text,
                round_no,
            )
            finished = time.perf_counter()
            review_intervals.append((started, finished))
            records.extend(reviewed)
            batch.clear()

        while True:
            candidate = await queue.get()
            if candidate is None:
                await flush()
                break
            batch.append(candidate)
            if len(batch) >= batch_size:
                await flush()
        return records

    workers = [asyncio.create_task(review_worker()) for _ in range(worker_count)]
    producers = [asyncio.create_task(producer(query)) for query in queries]
    try:
        await asyncio.gather(*producers)
    except Exception:
        for task in producers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*producers, return_exceptions=True)
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers, return_exceptions=True)
        raise

    for _ in workers:
        await queue.put(None)
    outputs = await asyncio.gather(*workers)
    records = [record for worker_records in outputs for record in worker_records]
    retrieval_ms = (
        round((max(end for _, end in retrieval_intervals) - min(start for start, _ in retrieval_intervals)) * 1000)
        if retrieval_intervals
        else 0
    )
    review_ms = (
        round((max(end for _, end in review_intervals) - min(start for start, _ in review_intervals)) * 1000)
        if review_intervals
        else 0
    )
    return records, retrieval_ms, review_ms


async def _generate_hyde(input_text: str) -> tuple[str | None, int]:
    started = time.perf_counter()
    try:
        text = await asyncio.to_thread(get_client().chat, build_hyde_prompt(input_text))
        text = " ".join(text.split()) if isinstance(text, str) else ""
        return (text or None), round((time.perf_counter() - started) * 1000)
    except Exception:
        return None, round((time.perf_counter() - started) * 1000)


async def _execute_workflow(input_text: str, query_id: str, cfg: dict[str, Any]) -> FinalResult:
    understanding_started = time.perf_counter()
    plan = await asyncio.to_thread(understand, input_text)
    understanding_ms = round((time.perf_counter() - understanding_started) * 1000)

    orchestration_cfg = cfg["orchestration"]
    hyde_ms = 0
    if orchestration_cfg.get("hyde", {}).get("enabled", False):
        hypothetical, hyde_ms = await _generate_hyde(input_text)
        if hypothetical:
            shared_filters = plan.sub_queries[0].filters if plan.sub_queries else None
            plan = replace(
                plan,
                sub_queries=[*plan.sub_queries, SubQuery(query=hypothetical, filters=shared_filters)],
            )

    loop_cfg = orchestration_cfg.get("loop", {})
    loop_enabled = bool(loop_cfg.get("enabled", True))
    max_rounds = max(1, int(loop_cfg.get("max_rounds", 3)))
    min_papers = max(1, int(loop_cfg.get("min_papers", 5)))
    coverage_criteria = list(
        dict.fromkeys(
            criterion.criterion
            for criterion in plan.criteria
            if criterion.source == "extracted"
        )
    )
    if not coverage_criteria:
        coverage_criteria = [input_text]

    current_queries = list(plan.sub_queries)
    existing_queries = [query.query for query in current_queries]
    seen_papers: set[str] = set()
    all_reviews: list[ReviewRecord] = []
    covered: set[str] = set()
    retrieval_ms = 0
    review_ms = 0
    rewrite_ms = 0
    rounds = 0
    termination_reason = "max_rounds"

    for round_no in range(1, max_rounds + 1):
        rounds = round_no
        reviews, round_retrieval_ms, round_review_ms = await _run_round(
            current_queries,
            plan.criteria,
            input_text,
            round_no,
            seen_papers,
            cfg,
        )
        retrieval_ms += round_retrieval_ms
        review_ms += round_review_ms
        all_reviews.extend(reviews)
        for record in reviews:
            if record.verdict == "pass":
                covered.update(item for item in record.covered if item in coverage_criteria)

        passed = [record for record in all_reviews if record.verdict == "pass"]
        if covered.issuperset(coverage_criteria):
            termination_reason = "coverage_met"
            break
        if len(passed) >= min_papers:
            termination_reason = "min_papers"
            break
        if not loop_enabled:
            termination_reason = "loop_disabled"
            break
        if round_no >= max_rounds:
            termination_reason = "max_rounds"
            break

        missing = [criterion for criterion in coverage_criteria if criterion not in covered]
        rewrite_started = time.perf_counter()
        current_queries = await asyncio.to_thread(rewrite, missing, existing_queries)
        rewrite_ms += round((time.perf_counter() - rewrite_started) * 1000)
        if not current_queries:
            termination_reason = "rewrite_exhausted"
            break
        existing_queries.extend(query.query for query in current_queries)

    passed_papers = sorted(
        (record for record in all_reviews if record.verdict == "pass"),
        key=lambda record: (-record.score, record.paper_id),
    )
    uncovered = [criterion for criterion in coverage_criteria if criterion not in covered]
    coverage_report: dict[str, object] = {
        "total_aspects": len(coverage_criteria),
        "covered_aspects": len(coverage_criteria) - len(uncovered),
        "covered_criteria": [criterion for criterion in coverage_criteria if criterion in covered],
        "uncovered_aspects": uncovered,
        "rounds": rounds,
        "papers_reviewed": len(all_reviews),
        "papers_passed": len(passed_papers),
        "termination_reason": termination_reason,
        "cache_hit": False,
        "template_version": TEMPLATE_VERSION,
    }
    return FinalResult(
        query_id=query_id,
        input_text=input_text,
        query_plan=plan,
        papers=passed_papers,
        coverage_report=coverage_report,
        latency_by_stage={
            "understanding_ms": understanding_ms,
            "hyde_ms": hyde_ms,
            "retrieval_ms": retrieval_ms,
            "review_ms": review_ms,
            "rewrite_ms": rewrite_ms,
            "total_ms": 0,
        },
    )


def _run_async(factory: Callable[[], Coroutine[Any, Any, FinalResult]]) -> FinalResult:
    """Run the async workflow from both ordinary and already-async callers."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(factory())).result()


def run_query(input_text: str, query_id: str | None = None) -> FinalResult:
    """Run one complete Paper Search flow and return reviewed paper candidates."""
    input_text = _normalise_input(input_text)
    if query_id is not None and (not isinstance(query_id, str) or not query_id.strip()):
        raise ValueError("query_id must be a non-empty string when provided")
    digest = _input_digest(input_text)
    resolved_query_id = query_id.strip() if query_id is not None else _auto_query_id(digest)
    cfg = get_config()
    loop_cfg = cfg["orchestration"].get("loop", {})
    cache_enabled = bool(loop_cfg.get("cache", True))
    path = _cache_path(_cache_digest(input_text, cfg)) if cache_enabled else None
    if cache_enabled:
        assert path is not None
        cached = _read_cache(path, input_text, resolved_query_id)
        if cached is not None:
            return cached

    started = time.perf_counter()
    try:
        result = _run_async(lambda: _execute_workflow(input_text, resolved_query_id, cfg))
    except OrchestrationError:
        raise
    except Exception as exc:
        raise OrchestrationError(f"orchestration workflow failed: {exc}") from exc
    latency = dict(result.latency_by_stage)
    latency["total_ms"] = round((time.perf_counter() - started) * 1000)
    result = replace(result, latency_by_stage=latency)
    if cache_enabled:
        assert path is not None
        _write_cache(path, result)
    return result
