"""One resumable retrieval ablation cell under the caller's effective configuration.

Only retrieval.pipeline.retrieve() executes retrieval. No configuration switches are changed
and no hidden warm-up is run: the first newly executed query includes component loading.
The caller supplies the experiment fingerprint; this cell additionally binds checkpoints to
the exact ordered question records. Failed/incomplete queries are excluded from metrics,
never turned into retrieval misses. Completed queries are immutable on resume.
"""

from __future__ import annotations

import copy
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from config import get_config
from retrieval.pipeline import retrieve
from retrieval.types import RetrieveResult
from .retrieval_eval import score_config, to_record
from .review_eval import _lock, _write
from .runtime import clear_runtime, require_fresh_runtime


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, *, nonnegative: bool = True) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (nonnegative and value < 0)):
        raise ValueError(f"{name} must be a finite {'nonnegative ' if nonnegative else ''}number")


def _questions(questions: list[dict]) -> list[dict]:
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a nonempty list")
    seen = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("each question must be a dictionary")
        qid = _text(question.get("query_id"), "query_id")
        _text(question.get("input_text"), "input_text")
        if qid in seen:
            raise ValueError(f"duplicate query_id {qid!r}")
        seen.add(qid)
        if question.get("split") not in ("dev", "eval"):
            raise ValueError(f"{qid}: split must be dev or eval")
        for key in ("gold_paper_ids", "gold_passage_ids"):
            ids = question.get(key, [] if key == "gold_passage_ids" else None)
            if not isinstance(ids, list) or (key == "gold_paper_ids" and not ids):
                raise ValueError(f"{qid}: {key} must be a list (paper gold cannot be empty)")
            for paper_id in ids:
                _text(paper_id, key)
            if len(ids) != len(set(ids)):
                raise ValueError(f"{qid}: duplicate {key}")
    # Check JSON serializability/NaN before creating an experiment directory or using models.
    json.dumps(questions, allow_nan=False)
    return copy.deepcopy(questions)


def _validate_complete(row: dict, question: dict, top_k: int) -> None:
    record = row.get("record")
    if not isinstance(record, dict) or any(record.get(k) != v for k, v in {
        "query_id": question["query_id"], "sub_query": question["input_text"],
        "stage": "retrieval", "top_k": top_k,
    }.items()):
        raise ValueError("checkpoint retrieval record identity/config mismatch")
    _integer(record.get("top_k"), "top_k", 1)
    hits = record.get("hits")
    if not isinstance(hits, list) or len(hits) > top_k:
        raise ValueError("checkpoint hits must be a list within top_k")
    passages = set()
    for rank, hit in enumerate(hits, 1):
        if not isinstance(hit, dict):
            raise ValueError("checkpoint hit must be a dictionary")
        pid = _text(hit.get("passage_id"), "passage_id")
        _text(hit.get("paper_id"), "paper_id")
        if pid in passages:
            raise ValueError("duplicate checkpoint passage_id")
        passages.add(pid)
        if _integer(hit.get("rank"), "rank", 1) != rank:
            raise ValueError("checkpoint hit ranks must be consecutive and ordered")
        _number(hit.get("score"), "score", nonnegative=False)
    candidates = _integer(row.get("total_candidates"), "total_candidates")
    if candidates < len(hits):
        raise ValueError("checkpoint has more hits than candidates")
    latency = row.get("latency_ms")
    if not isinstance(latency, dict) or "total_ms" not in latency:
        raise ValueError("checkpoint stage latency must include total_ms")
    for stage, value in latency.items():
        _text(stage, "latency stage")
        _number(value, "latency_ms")
    _integer(record.get("latency_ms"), "record latency_ms")
    if record["latency_ms"] != int(latency["total_ms"]):
        raise ValueError("checkpoint total latency differs from the retrieval record")
    _number(row.get("wall_ms"), "wall_ms")


def _load_attempts(run_dir: Path, questions: list[dict], fingerprint: str, top_k: int) -> list[dict]:
    by_id = {q["query_id"]: q for q in questions}
    attempts, completed = [], set()
    for number, path in enumerate(sorted((run_dir / "attempts").glob("*.json")), 1):
        if path.name != f"{number:06d}.json":
            raise ValueError("checkpoint attempt sequence is not contiguous")
        row = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(row, dict) or not isinstance(row.get("question"), dict):
            raise ValueError("invalid checkpoint object/question")
        qid = row["question"].get("query_id")
        if (not isinstance(qid, str) or row["question"] != by_id.get(qid)
                or row.get("fingerprint") != fingerprint
                or row.get("status") not in ("running", "failed", "complete")
                or type(row.get("cold_start")) is not bool):
            raise ValueError("checkpoint identity/status mismatch")
        if qid in completed:
            raise ValueError("checkpoint follows an already completed query")
        if row["status"] == "complete":
            _validate_complete(row, by_id[qid], top_k)
            completed.add(qid)
        elif row["status"] == "failed":
            error = row.get("error")
            if (not isinstance(error, dict) or error.get("query_id") != qid
                    or not isinstance(error.get("type"), str) or not error["type"]):
                raise ValueError("invalid failed checkpoint error")
            _number(row.get("wall_ms"), "wall_ms")
        elif "wall_ms" in row:
            _number(row["wall_ms"], "wall_ms")
        attempts.append(row)
    return attempts


def _report(questions: list[dict], attempts: list[dict], fingerprint: str, error: dict | None) -> dict:
    complete = {row["question"]["query_id"]: row for row in attempts if row["status"] == "complete"}
    rows = [complete[q["query_id"]] for q in questions if q["query_id"] in complete]
    scores, ranks = score_config([r["record"] for r in rows], [r["question"] for r in rows])
    return {
        "stage": "retrieval", "format_version": 1, "fingerprint": fingerprint,
        "status": "failed" if error else "complete" if len(rows) == len(questions) else "partial",
        "selected_queries": len(questions), "completed_queries": len(rows), "error": error,
        "usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "splits": scores["metrics"],
        "per_query": {
            r["question"]["query_id"]: {
                "first_gold_rank": ranks[r["question"]["query_id"]],
                "source_success": ranks[r["question"]["query_id"]] is not None
                and ranks[r["question"]["query_id"]] <= 10,
                "latency_ms": r["latency_ms"], "wall_ms": r["wall_ms"], "cold_start": r["cold_start"],
            } for r in rows
        },
        "incomplete_query_ids": [q["query_id"] for q in questions if q["query_id"] not in complete],
        "failed_attempts": sum(r["status"] == "failed" for r in attempts),
        "interrupted_attempts": sum(r["status"] == "running" for r in attempts),
        "cold_start_query_ids": [r["question"]["query_id"] for r in attempts if r["cold_start"]],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def run_retrieval_cell(questions: list[dict], run_dir: Path, fingerprint: str, *, resume: bool = False) -> dict:
    """Execute one cell without changing any effective retrieval configuration.

    The caller owns the data/config/code/index fingerprint. manifest.json also records exact
    ordered questions; attempts/*.json records question, retrieval record, stage/wall timing
    and cold_start. Resume validates all checkpoints before the first retrieve() call.
    Query failures return a failed report and preserve successful checkpoints for resume;
    malformed inputs/checkpoints raise. No API tokens are consumed by this retrieval stage.
    """
    questions = _questions(questions)
    _text(fingerprint, "fingerprint")
    run_dir = Path(run_dir).resolve()
    require_fresh_runtime()
    try:
        cfg = get_config()
        top_k = _integer(cfg["retrieval"]["rerank"].get("top_k", 10), "retrieval.rerank.top_k", 1)
        run_dir.mkdir(parents=True, exist_ok=True)
        with _lock(run_dir / ".lock"):
            metadata = run_dir / "manifest.json"
            if metadata.exists():
                if not resume:
                    raise FileExistsError("retrieval cell already exists; use resume or another directory")
                manifest = json.loads(metadata.read_text(encoding="utf-8"))
                if (not isinstance(manifest, dict) or manifest.get("stage") != "retrieval"
                        or manifest.get("format_version") != 1 or manifest.get("fingerprint") != fingerprint
                        or manifest.get("questions") != questions):
                    raise ValueError("retrieval cell fingerprint/questions mismatch")
            else:
                if resume:
                    raise FileNotFoundError("cannot resume retrieval cell without manifest.json")
                if any((run_dir / "attempts").glob("*.json")):
                    raise ValueError("retrieval checkpoints exist without a manifest")
                _write(metadata, {"stage": "retrieval", "format_version": 1, "fingerprint": fingerprint,
                                  "questions": questions, "created_at": datetime.now(timezone.utc).isoformat()})
            attempts = _load_attempts(run_dir, questions, fingerprint, top_k)
            completed = {r["question"]["query_id"] for r in attempts if r["status"] == "complete"}
            cold_start, error = True, None
            for question in questions:
                qid = question["query_id"]
                if qid in completed:
                    continue
                path = run_dir / "attempts" / f"{len(attempts) + 1:06d}.json"
                row = {"question": question, "fingerprint": fingerprint, "status": "running",
                       "cold_start": cold_start, "started_at": datetime.now(timezone.utc).isoformat()}
                _write(path, row)
                attempts.append(row)
                started = time.perf_counter()
                try:
                    result = retrieve(question["input_text"])
                    if not isinstance(result, RetrieveResult):
                        raise ValueError("retrieve must return RetrieveResult")
                    candidate = {**row, "record": to_record(question, "retrieval", top_k, result),
                                 "total_candidates": result.total_candidates,
                                 "latency_ms": dict(result.latency_ms),
                                 "wall_ms": round((time.perf_counter() - started) * 1000, 2)}
                    _validate_complete(candidate, question, top_k)
                    row.update(candidate, status="complete")
                    completed.add(qid)
                except Exception as exc:
                    error = {"query_id": qid, "type": type(exc).__name__}
                    row.update(status="failed", error=error)
                finally:
                    row["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
                    _write(path, row)
                cold_start = False
                report = _report(questions, attempts, fingerprint, error)
                _write(run_dir / "report.json", report)
                if error:
                    return report
            report = _report(questions, attempts, fingerprint, error)
            _write(run_dir / "report.json", report)
            return report
    finally:
        clear_runtime(release_models=True)
