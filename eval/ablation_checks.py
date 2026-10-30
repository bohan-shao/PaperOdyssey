"""Validate all saved review attempts before an ablation resumes paid search work.

This read-only check also covers failed and interrupted attempts, whose observed usage
must remain trustworthy when the matrix aggregates costs. A crash may leave a running
attempt without final timing/usage. Complete and failed attempts must have both.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .review_eval import _validate_result


def _usage(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("checkpoint usage must be a dictionary")
    for key in ("calls", "prompt_tokens", "completion_tokens"):
        count = value.get(key)
        if type(count) is not int or count < 0:
            raise ValueError(f"checkpoint usage.{key} must be a nonnegative integer")


def _wall(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("checkpoint wall_ms must be a finite nonnegative number")
    try:
        valid = math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("checkpoint wall_ms must be a finite nonnegative number")


def validate_review_attempts(directory: Path, questions: list[dict]) -> None:
    """Validate exact question identity, sequence, status, traces and observed usage.

    Unknown questions, non-contiguous filenames, malformed records, duplicate completions,
    and any attempt following a completed query raise ValueError. Missing directories or
    attempts are valid before a cell starts. Nothing is written and no client is created.
    """
    if not isinstance(questions, list):
        raise ValueError("questions must be a list")
    selected = {}
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("each selected question must be a dictionary")
        qid = question.get("query_id")
        if not isinstance(qid, str) or not qid.strip() or qid in selected:
            raise ValueError("selected query IDs must be nonempty and unique")
        selected[qid] = question
    completed: set[str] = set()
    for number, path in enumerate(sorted((Path(directory) / "attempts").glob("*.json")), 1):
        if path.name != f"{number:06d}.json":
            raise ValueError("review checkpoint attempt sequence is not contiguous")
        row = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(row, dict) or not isinstance(row.get("question"), dict):
            raise ValueError("invalid review checkpoint object/question")
        qid = row["question"].get("query_id")
        if not isinstance(qid, str) or row["question"] != selected.get(qid):
            raise ValueError("review checkpoint question differs from the selected question")
        status = row.get("status")
        if status not in ("running", "failed", "complete") or type(row.get("cold_start")) is not bool:
            raise ValueError("invalid review checkpoint status/cold_start")
        if qid in completed:
            raise ValueError("review checkpoint follows an already completed query")
        if status in ("complete", "failed") or "usage" in row:
            _usage(row.get("usage"))
        if status in ("complete", "failed") or "wall_ms" in row:
            _wall(row.get("wall_ms"))
        if status == "complete":
            result = row.get("result")
            if not isinstance(result, dict):
                raise ValueError("complete review checkpoint must include a result")
            try:
                _validate_result(selected[qid], result)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError("complete review checkpoint has an invalid result") from exc
            completed.add(qid)
        elif status == "failed":
            error = row.get("error")
            if (not isinstance(error, dict) or error.get("query_id") != qid
                    or not isinstance(error.get("type"), str) or not error["type"].strip()):
                raise ValueError("failed review checkpoint must identify its query and error type")
