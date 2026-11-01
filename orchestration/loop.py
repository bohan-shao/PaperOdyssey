"""Parallel retrieval, review, and feedback loop for Paper Search."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Coroutine

from config import get_config, get_path
from kb.store import load_paper
from llm.client import get_client
from llm.prompt import TEMPLATE_VERSION, build_hyde_prompt
from retrieval.index_paths import index_directory
from retrieval.lookup import lookup_path
from retrieval.pipeline import retrieve
from retrieval.types import RetrievalFilters, RetrieveResult, ScoredPassage

from .errors import OrchestrationError
from .query_understanding import understand
from .review import batch_review
from .rewrite import rewrite
from .types import Criterion, FinalResult, Progress, QueryPlan, ReviewRecord, SubQuery, passing_order_key


def _normalise_input(input_text: str) -> str:
    if not isinstance(input_text, str) or not (normalised := " ".join(input_text.split())):
        raise ValueError("input_text must be a non-empty string")
    return normalised


def _input_digest(input_text: str) -> str:
    return hashlib.sha256(input_text.encode("utf-8")).hexdigest()


def _auto_query_id(digest: str) -> str:
    return f"Q{digest[:12].upper()}"


def _file_stamp(path: Path) -> list[object] | None:
    """Path, size and modification time of a built file; None while it does not exist."""
    try:
        info = path.stat()
    except OSError:
        return None
    return [str(path), info.st_size, info.st_mtime_ns]


def _corpus_identity(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """The built files a search reads, stamped so that a rebuild changes the cache key (D52).

    The lookup store, both index manifests and the paper store. A rebuild writes them anew, so
    their modification times change even at the same paths and corpus size. None when the
    configuration names no corpus.
    """
    try:
        retrieval_cfg = cfg["retrieval"]
        files = {
            "lookup": lookup_path(cfg),
            "sparse": index_directory(retrieval_cfg["sparse"]["index_dir"], cfg) / "meta.json",
            "dense": index_directory(retrieval_cfg["dense"]["faiss_dir"], cfg) / "meta.json",
            "papers": _paper_database_path(cfg),
        }
    except (KeyError, TypeError, ValueError):
        return None
    return {name: _file_stamp(path) for name, path in files.items()}


def _config_fingerprint(cfg: dict[str, Any]) -> str:
    """The settings and built corpus that can change a search result, as a stable string.

    It is part of the cache key, so flipping an ablation switch (loop on/off, HyDE, reviewer,
    retrieval settings, LLM model, chunking), a prompt template change or a rebuilt KB or index
    never returns a result that was computed under different settings or another corpus.
    """
    kb_cfg = cfg.get("kb", {})
    relevant = {
        "orchestration": cfg.get("orchestration", {}),
        "retrieval": cfg.get("retrieval", {}),
        "llm": cfg.get("llm", {}),
        "kb": {"chunk": kb_cfg.get("chunk", {}), "store": kb_cfg.get("store", {})},
        "template_version": TEMPLATE_VERSION,
        "corpus": _corpus_identity(cfg),
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
        reviews=[ReviewRecord(**item) for item in raw.get("reviews", [])],
        retrieval_ranks={str(key): int(value) for key, value in raw.get("retrieval_ranks", {}).items()},
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


_EVIDENCE_PASSAGES = 3  # passages per review, as in _candidate_from_hits


@dataclass
class _Reviewed:
    """One paper's review state across rounds (D53)."""

    record: ReviewRecord  # the kept record (`_merge_review`)
    basis: list[dict[str, str]]  # the passages shown in the review the kept record rests on
    seen: set[str]  # ids of every passage the reviewer was shown


def _claim(
    candidate: dict[str, Any],
    queued: dict[str, list[dict[str, str]]],
    reviewed: dict[str, _Reviewed],
    uncovered: set[str],
    re_review: bool = True,
) -> dict[str, Any] | None:
    """The candidate to review now, or None (D53). Call under the round's lock.

    A paper is reviewed at most once per round, so round 1 behaves as before D53; `queued` maps
    each paper claimed this round to the passages its review is shown. A paper reviewed in an
    earlier round is reviewed again when this round retrieved passages of it the reviewer has not
    seen, and its kept record is a reject or lists as missing an aspect still `uncovered`. The
    re-review gets the kept record's passages with the new ones interleaved after the best of
    them (at most 3; a full set drops its weakest). `re_review=False`
    (`orchestration.loop.re_review`) restores the pre-D53 behaviour: a paper is never reviewed twice.
    """
    paper_id = candidate["paper_id"]
    if paper_id in queued:
        return None
    earlier = reviewed.get(paper_id)
    if earlier is not None:
        record = earlier.record
        if not re_review or (record.verdict == "pass" and not uncovered.intersection(record.missing)):
            return None
        new = [passage for passage in candidate["top_passages"] if passage["passage_id"] not in earlier.seen]
        if not new:
            return None
        merged = [p for pair in zip_longest(earlier.basis, new) for p in pair if p is not None]
        candidate = {**candidate, "top_passages": merged[:_EVIDENCE_PASSAGES]}
    queued[paper_id] = list(candidate["top_passages"])
    return candidate


def _merge_review(earlier: ReviewRecord, new: ReviewRecord) -> tuple[ReviewRecord, bool]:
    """The record kept when a paper is reviewed again (D53), and whether it rests on the new review.

    A re-review can only add. A reject never replaces the kept record, so a pass stays a pass; a
    pass replaces a reject; a second pass keeps the union of covered aspects and the higher score,
    and takes the new review's reasons and evidence only when it covers more. `round` stays the
    round in which the kept verdict was first reached; `evidence["review_rounds"]` lists every
    round in which the paper was reviewed.
    """
    rounds = sorted({*earlier.evidence.get("review_rounds", [earlier.round]), new.round})
    if new.verdict != "pass":
        kept, newer = earlier, False
    elif earlier.verdict != "pass":
        kept, newer = new, True
    else:
        added = [item for item in new.covered if item not in earlier.covered]
        covered = [*earlier.covered, *added]
        missing = [item for item in dict.fromkeys([*earlier.missing, *new.missing]) if item not in covered]
        base, newer = (new, True) if added else (earlier, False)
        reasons = f"{earlier.reasons} Round {new.round} re-review: {new.reasons}".strip() if added else earlier.reasons
        kept = replace(base, score=max(earlier.score, new.score), reasons=reasons, covered=covered,
                       missing=missing, round=earlier.round)
    return replace(kept, evidence={**kept.evidence, "review_rounds": rounds}), newer


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
    reviewed: dict[str, _Reviewed],
    uncovered: set[str],
    best_ranks: dict[str, int],
    cfg: dict[str, Any],
) -> tuple[list[ReviewRecord], dict[str, list[dict[str, str]]], int, int]:
    review_cfg = cfg["orchestration"].get("review", {})
    batch_size = max(1, int(review_cfg.get("batch_size", 5)))
    worker_count = max(1, int(review_cfg.get("max_concurrency", 2)))
    re_review = bool(cfg["orchestration"].get("loop", {}).get("re_review", True))
    database_path = _paper_database_path(cfg)
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    seen_lock = asyncio.Lock()
    queued: dict[str, list[dict[str, str]]] = {}
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
        # Best position over every sub-query of the run, also for papers already seen (D47);
        # no await in between, so concurrent producers cannot interleave this update.
        for hit in result.hits:
            if hit.rank < best_ranks.get(hit.paper_id, hit.rank + 1):
                best_ranks[hit.paper_id] = hit.rank
        candidates = await asyncio.to_thread(_candidates_from_result, result, database_path)
        for candidate in candidates:
            async with seen_lock:
                claimed = _claim(candidate, queued, reviewed, uncovered, re_review)
            if claimed is not None:
                await queue.put(claimed)

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
    return records, queued, retrieval_ms, review_ms


async def _generate_hyde(input_text: str) -> tuple[str | None, int]:
    started = time.perf_counter()
    try:
        text = await asyncio.to_thread(get_client().chat, build_hyde_prompt(input_text))
        text = " ".join(text.split()) if isinstance(text, str) else ""
        return (text or None), round((time.perf_counter() - started) * 1000)
    except Exception:
        return None, round((time.perf_counter() - started) * 1000)


def _notify(progress: Progress | None, event: str, **details: object) -> None:
    """Report one step to the caller's progress callback (SCHEMA.md §6.1), if there is one."""
    if progress is not None:
        progress(event, details)


def _plain_rag(input_text: str, query_id: str, progress: Progress | None = None) -> FinalResult:
    """The non-agentic baseline (D55): the question as typed -> retrieve() -> papers in rank order.

    No LLM call: no query understanding, HyDE, review or rewriting. Every paper among the
    retrieved passages is kept ("pass", score 0 = not reviewed) in the order of its best passage,
    with its passages in rank order as evidence, so Paper QA answers it with the same prompt and
    evidence budget (`downstream.max_papers`, `passages_per_paper`) as the agentic search.
    """
    _notify(progress, "round", round=1, queries=1)
    started = time.perf_counter()
    try:
        result = retrieve(input_text)
    except Exception as exc:
        raise OrchestrationError(f"retrieval failed for query {input_text!r}: {exc}") from exc
    retrieval_ms = round((time.perf_counter() - started) * 1000)
    passages: dict[str, list[str]] = {}
    titles: dict[str, str] = {}
    ranks: dict[str, int] = {}
    for hit in sorted(result.hits, key=lambda hit: (hit.rank, -hit.score, hit.passage_id)):
        ids = passages.setdefault(hit.paper_id, [])
        if hit.passage_id not in ids:
            ids.append(hit.passage_id)
        titles.setdefault(hit.paper_id, hit.title)
        ranks.setdefault(hit.paper_id, hit.rank)
    records = [
        ReviewRecord(paper_id, "pass", 0, reasons="Plain RAG baseline: retrieved, not reviewed.", round=1,
                     evidence={"title": titles[paper_id], "top_passages": ids})
        for paper_id, ids in passages.items()
    ]
    _notify(progress, "retrieved", papers=len(records))
    return FinalResult(
        query_id=query_id,
        input_text=input_text,
        query_plan=QueryPlan(input_text, [SubQuery(input_text)], []),
        papers=sorted(records, key=lambda record: passing_order_key(record.paper_id, record.score, ranks)),
        coverage_report={
            "total_aspects": 0, "covered_aspects": 0, "covered_criteria": [], "uncovered_aspects": [],
            "rounds": 1, "papers_reviewed": len(records), "papers_passed": len(records), "re_reviews": 0,
            "termination_reason": "plain_rag", "cache_hit": False, "template_version": TEMPLATE_VERSION,
            "review_backend": "none",
        },
        reviews=records,
        retrieval_ranks=ranks,
        latency_by_stage={"understanding_ms": 0, "hyde_ms": 0, "retrieval_ms": retrieval_ms,
                          "review_ms": 0, "rewrite_ms": 0, "total_ms": 0},
    )


async def _execute_workflow(
    input_text: str, query_id: str, cfg: dict[str, Any], progress: Progress | None = None,
) -> FinalResult:
    if not cfg["orchestration"].get("agentic", {}).get("enabled", True):
        return await asyncio.to_thread(_plain_rag, input_text, query_id, progress)
    _notify(progress, "understanding")
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
    _notify(progress, "planned", queries=len(plan.sub_queries), aspects=len(coverage_criteria))

    current_queries = list(plan.sub_queries)
    existing_queries = [query.query for query in current_queries]
    best_ranks: dict[str, int] = {}
    reviewed: dict[str, _Reviewed] = {}  # one kept record per paper, in order of first review (D53)
    re_reviews = 0
    covered: set[str] = set()
    missing: list[str] = list(coverage_criteria)
    retrieval_ms = 0
    review_ms = 0
    rewrite_ms = 0
    rounds = 0
    termination_reason = "max_rounds"

    for round_no in range(1, max_rounds + 1):
        rounds = round_no
        _notify(progress, "round", round=round_no, queries=len(current_queries))
        reviews, shown, round_retrieval_ms, round_review_ms = await _run_round(
            current_queries,
            plan.criteria,
            input_text,
            round_no,
            reviewed,
            set(missing),
            best_ranks,
            cfg,
        )
        retrieval_ms += round_retrieval_ms
        review_ms += round_review_ms
        for record in reviews:
            passages = shown.get(record.paper_id, [])
            ids = {passage["passage_id"] for passage in passages}
            earlier = reviewed.get(record.paper_id)
            if earlier is None:
                reviewed[record.paper_id] = _Reviewed(record, passages, ids)
                continue
            re_reviews += 1
            kept, newer = _merge_review(earlier.record, record)
            reviewed[record.paper_id] = _Reviewed(kept, passages if newer else earlier.basis, earlier.seen | ids)
        passed = [state.record for state in reviewed.values() if state.record.verdict == "pass"]
        covered = {item for record in passed for item in record.covered if item in coverage_criteria}
        _notify(progress, "round_done", round=round_no, reviewed=len(reviewed), passed=len(passed),
                covered=len(covered), aspects=len(coverage_criteria))
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
        _notify(progress, "follow_up", missing=missing)
        rewrite_started = time.perf_counter()
        current_queries = await asyncio.to_thread(rewrite, missing, existing_queries)
        rewrite_ms += round((time.perf_counter() - rewrite_started) * 1000)
        if not current_queries:
            termination_reason = "rewrite_exhausted"
            break
        existing_queries.extend(query.query for query in current_queries)

    all_reviews = [state.record for state in reviewed.values()]
    retrieval_ranks = {
        paper_id: best_ranks[paper_id]
        for paper_id in sorted({record.paper_id for record in all_reviews})
        if paper_id in best_ranks
    }
    passed_papers = sorted(
        (record for record in all_reviews if record.verdict == "pass"),
        key=lambda record: passing_order_key(record.paper_id, record.score, retrieval_ranks),
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
        "re_reviews": re_reviews,
        "termination_reason": termination_reason,
        "cache_hit": False,
        "template_version": TEMPLATE_VERSION,
        "review_backend": str(orchestration_cfg.get("review", {}).get("backend", "llm")),
    }
    return FinalResult(
        query_id=query_id,
        input_text=input_text,
        query_plan=plan,
        papers=passed_papers,
        coverage_report=coverage_report,
        reviews=all_reviews,
        retrieval_ranks=retrieval_ranks,
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


def run_query(input_text: str, query_id: str | None = None, *, progress: Progress | None = None) -> FinalResult:
    """Run one complete Paper Search flow and return reviewed paper candidates.

    `progress`, if given, is called with each step as it happens (SCHEMA.md §6.1).
    """
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
            _notify(progress, "cache_hit")
            return cached

    started = time.perf_counter()
    try:
        result = _run_async(lambda: _execute_workflow(input_text, resolved_query_id, cfg, progress))
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
