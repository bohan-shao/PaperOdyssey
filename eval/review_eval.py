"""Resumable Paper Search evaluation with positive-source gold and full review traces.

Run in a dedicated process, one question at a time or up to `workers` at once (D48). Completed
queries are immutable checkpoints; resume requires the same dataset, selection, effective config,
code and index metadata. Only observed API usage is reported, not an invoice or a hard token budget.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from config import PROJECT_ROOT, get_config, get_path
from llm import client as llm_client
from llm.client import get_client
from orchestration import run_query
from orchestration.types import passing_order_key
from .latency import config_overrides, environment, summarize
from .retrieval_eval import index_manifests, load_eval_set
from .retrieval_metrics import compute_retrieval_metrics
from .review_metrics import coverage_stats
from .runtime import clear_runtime, require_fresh_runtime

log = logging.getLogger(__name__)
_USAGE_KEYS = ("calls", "prompt_tokens", "completion_tokens")
MAX_WORKERS = 32


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def _safe_config(cfg: dict) -> dict:
    """Only settings used by search; never export fetch credentials or URL credentials."""
    def clean(value, key=""):
        if any(word in key.lower() for word in ("api_key", "secret", "password", "credential")):
            return "<redacted>"
        if isinstance(value, dict):
            return {k: clean(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str) and key.endswith("url"):
            url = urlsplit(value)
            host = url.hostname or ""
            if url.port:
                host += f":{url.port}"
            return urlunsplit((url.scheme, host, url.path, "", ""))
        return value
    selected = {k: copy.deepcopy(cfg.get(k)) for k in
                ("seed", "paths", "retrieval", "orchestration", "llm")}
    selected["kb"] = {k: cfg.get("kb", {}).get(k) for k in ("chunk", "store")}
    return clean(selected)


def _input_file_metadata() -> dict:
    """Detect replacement/rebuild of local corpus files without hashing multi-GB indexes."""
    paths = list(get_path("indices").rglob("*"))
    store = get_config().get("kb", {}).get("store", {})
    paths.extend(get_path("processed") / str(store[k]) for k in
                 ("sqlite_file", "passages_file", "sections_file") if store.get(k))
    return {str(p): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(paths) if p.is_file()}


def _code_hash() -> str:
    files = [PROJECT_ROOT / "config.py", PROJECT_ROOT / "requirements.txt"]
    for package in ("eval", "orchestration", "llm", "retrieval", "kb"):
        files.extend(sorted((PROJECT_ROOT / package).rglob("*.py")))
    return _digest({str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files if p.is_file()})


@contextmanager
def _lock(path: Path):
    # TC2 Linux and the supported local macOS workflow both provide flock. A persistent
    # lock file is intentional: unlinking it could let another process lock a new inode.
    import fcntl
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("this review run is already active") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _validate_result(question: dict, result: dict) -> None:
    if (result.get("query_id") != question["query_id"].strip()
            or result.get("input_text") != " ".join(question["input_text"].split())):
        raise ValueError("result query identity differs from the eval question")
    reviews, papers, coverage = result["reviews"], result["papers"], result["coverage_report"]
    ids = [r["paper_id"] for r in reviews]
    if len(ids) != len(set(ids)) or any(r["verdict"] not in ("pass", "reject") for r in reviews):
        raise ValueError("duplicate paper or invalid review verdict")
    ranks = result.get("retrieval_ranks")
    if (not isinstance(ranks, dict) or set(ranks) != set(ids)
            or any(type(rank) is not int or rank < 1 for rank in ranks.values())):
        raise ValueError("missing or invalid retrieval ranks for the reviewed papers")
    expected = sorted((r for r in reviews if r["verdict"] == "pass"),
                      key=lambda r: passing_order_key(r["paper_id"], r["score"], ranks))
    if papers != expected or coverage["papers_reviewed"] != len(reviews) or coverage["papers_passed"] != len(papers):
        raise ValueError("incomplete review trace or inconsistent passing-paper ranking")
    if coverage.get("cache_hit"):
        raise ValueError("cached results cannot be used for review evaluation")
    if any(type(r["round"]) is not int or not 1 <= r["round"] <= coverage["rounds"] for r in reviews):
        raise ValueError("review round is outside the executed rounds")
    coverage_stats([coverage])  # validate before persisting a successful checkpoint


def _score(rows: list[dict]) -> dict:
    gold_total = reviewed = passed = unjudged = 0
    by_query = {}
    hits, gold = {}, {}
    stages: dict[str, list[float]] = {}
    for row in rows:
        q, result = row["question"], row["result"]
        qid, gold_ids = q["query_id"], set(q["gold_paper_ids"])
        trace = {r["paper_id"]: r for r in result["reviews"]}
        found = gold_ids & trace.keys()
        accepted = {r["paper_id"] for r in result["papers"]} & gold_ids
        gold_total += len(gold_ids)
        reviewed += len(found)
        passed += len(accepted)
        unjudged += len(trace.keys() - gold_ids)
        hits[qid] = [{"paper_id": r["paper_id"], "rank": i}
                     for i, r in enumerate(result["papers"], 1)]
        # This protocol deliberately scores source papers only, even if a future input
        # adds passage gold. Candidate-level labels require a separate annotation set.
        gold[qid] = {"gold_paper_ids": q["gold_paper_ids"]}
        by_query[qid] = {
            "source_passed": sorted(accepted), "source_rejected": sorted(found - accepted),
            "source_not_reviewed": sorted(gold_ids - found),
            "first_source_review_round": min((trace[p]["round"] for p in found), default=None),
            "first_source_pass_round": min((trace[p]["round"] for p in accepted), default=None),
        }
        for name, ms in result["latency_by_stage"].items():
            stages.setdefault(name, []).append(ms)
    metrics = compute_retrieval_metrics(hits, gold)
    # P@K on incomplete positive-only labels is easily mistaken for classifier precision.
    ranking = {k: v for k, v in metrics["per_granularity"]["paper"].items() if not k.startswith("P@")}
    for qid, ranks in metrics["per_query"].items():
        by_query[qid].update(ranks)
    return {
        "n_queries": len(rows), "known_source_papers": gold_total,
        "source_reviewed": reviewed, "source_passed": passed,
        "source_rejected": reviewed - passed, "source_not_reviewed": gold_total - reviewed,
        "source_review_rate": reviewed / gold_total if gold_total else None,
        "source_acceptance_rate": passed / gold_total if gold_total else None,
        "source_pass_rate_given_reviewed": passed / reviewed if reviewed else None,
        "unjudged_papers_reviewed": unjudged,
        "known_source_ranking": ranking,
        "coverage": coverage_stats([r["result"]["coverage_report"] for r in rows]),
        "latency_ms": {k: summarize(v) for k, v in sorted(stages.items())},
        "wall_ms": summarize([r["wall_ms"] for r in rows]),
        "per_query": by_query,
    }


def _report(manifest: dict, attempts: list[dict], selected: list[dict], error: dict | None) -> dict:
    complete = {a["question"]["query_id"]: a for a in attempts if a["status"] == "complete"}
    rows = [complete[q["query_id"]] for q in selected if q["query_id"] in complete]
    usage = {k: sum(a.get("usage", {}).get(k, 0) for a in attempts) for k in _USAGE_KEYS}
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return {
        "stage": "review", "format_version": 1, "tag": manifest["tag"],
        "status": "failed" if error else "complete" if len(rows) == len(selected) else "partial",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "selected_queries": len(selected), "completed_queries": len(rows),
        "failed_attempts": sum(a["status"] == "failed" for a in attempts),
        "interrupted_attempts": sum(a["status"] == "running" for a in attempts),
        "error": error, "usage": usage,
        "usage_scope": "Observed responses in completed/failed attempts, including JSON retries. "
                       "Interrupted attempts and transport failures may incur additional unreported usage; "
                       "raw best-effort usage is in metrics/llm_usage.jsonl. Not a billing total.",
        "protocol": {
            "labels": "Positive source papers only; all other candidates are unjudged, not negatives.",
            "review_precision_accuracy_f1": "Unavailable without candidate positive/negative judgments.",
            "coverage": "LLM-reported requirement coverage, not human relevance or answer quality.",
            "ranking": "Final passing papers sorted by descending review score, then best retrieval rank, "
                       "then paper_id (passing_order_key, D47; system order).",
            "cache": False, "timing": "No hidden warm-up. Queries started before the process finished its first "
                                      "query include component loading (cold starts).",
            "concurrency": "With workers > 1 (D48) several queries run at once in one process; their wall_ms and "
                           "stage latencies include waiting for each other's model and LLM calls, so per-query "
                           "latency comes from workers=1 runs. Usage is counted per query either way.",
            "failures": "Failed/incomplete queries are excluded from metrics and listed as incomplete, not misses.",
            "selection": "File order after split filtering, then limit; a prefix is a smoke test, not a representative sample.",
        },
        "provenance": manifest,
        "splits": {split: _score([r for r in rows if r["question"]["split"] == split])
                   for split in ("dev", "eval")},
        "incomplete_query_ids": [q["query_id"] for q in selected if q["query_id"] not in complete],
        "cold_start_query_ids": [a["question"]["query_id"] for a in attempts if a.get("cold_start")],
        "workers_used": sorted({a.get("workers", 1) for a in attempts}),
    }


def _run_pending(pending: list[dict], *, workers: int, run_dir: Path, manifest: dict,
                 attempts: list[dict], selected: list[dict], completed: set[str]) -> dict | None:
    """Run the pending questions, at most `workers` at once; return the first query error, if any.

    Attempt numbers, checkpoints and report.json are written under one lock, numbered in the
    order the questions start (file order). After a query error, or an interrupt, no new
    question starts; questions already running finish and are recorded. A question started
    before the process finished its first question is a cold start. Usage comes from
    llm.client.usage_scope, so questions running at once do not mix their counts (D48).
    """
    lock = threading.Lock()
    queue = iter(pending)
    state: dict = {"error": None, "stop": False, "warm": False}

    def start() -> tuple[dict, dict, Path] | None:
        with lock:
            if state["error"] is not None or state["stop"]:
                return None
            q = next(queue, None)
            if q is None:
                return None
            path = run_dir / "attempts" / f"{len(attempts) + 1:06d}.json"
            if path.exists():
                raise ValueError("attempt sequence is not contiguous")
            row = {"question": q, "status": "running", "cold_start": not state["warm"], "workers": workers,
                   "started_at": datetime.now(timezone.utc).isoformat()}
            _write(path, row)
            attempts.append(row)
            return q, row, path

    def worker() -> None:
        while (item := start()) is not None:
            q, row, path = item
            outcome = None
            started = time.perf_counter()
            with llm_client.usage_scope() as used:
                try:
                    result = asdict(run_query(q["input_text"], query_id=q["query_id"]))
                    _validate_result(q, result)
                    outcome = {"status": "complete", "result": result}
                except Exception as exc:
                    # Provider error strings can contain request details or credentials.
                    # Persist the class and query, never an unredacted provider response.
                    outcome = {"status": "failed", "error": {"query_id": q["query_id"], "type": type(exc).__name__}}
                    log.error("review evaluation stopped at %s (%s)", q["query_id"], type(exc).__name__)
                finally:
                    # Also on an interrupt: the attempt stays "running" but keeps its observed usage.
                    with lock:
                        row.update(outcome or {}, usage={k: used[k] for k in _USAGE_KEYS},
                                   wall_ms=round((time.perf_counter() - started) * 1000, 2))
                        _write(path, row)
                        if outcome is None:
                            state["stop"] = True
                        else:
                            if outcome["status"] == "complete":
                                completed.add(q["query_id"])
                            elif state["error"] is None:
                                state["error"] = outcome["error"]
                            state["warm"] = True
                            report = _report(manifest, attempts, selected, state["error"])
                            _write(run_dir / "report.json", report)
                            log.info("review: %d/%d complete; %d observed tokens", len(completed),
                                     len(selected), report["usage"]["total_tokens"])

    if workers == 1:
        worker()
        return state["error"]
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="review")
    try:
        for future in [pool.submit(worker) for _ in range(min(workers, len(pending)))]:
            future.result()
    finally:
        with lock:
            state["stop"] = True  # after an exception here, running questions finish but none start
        pool.shutdown(wait=True)
    return state["error"]


def run_review_eval(eval_set_path: str | Path, results_dir: str | Path, tag: str, *,
                    limit: int | None = None, split: str = "all", resume: bool = False,
                    workers: int = 1) -> dict:
    """Evaluate full Paper Search and atomically save report.json plus per-attempt traces.

    A query exception writes a failed checkpoint and returns status=failed; --resume retries
    only incomplete queries. Fingerprint/checkpoint errors raise before making API calls.
    `workers` questions run at once in this process (D48); it is not part of the fingerprint,
    so a run may resume with another value.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise ValueError("tag must be a safe filename component")
    if split not in ("all", "dev", "eval") or (limit is not None and (type(limit) is not int or limit <= 0)):
        raise ValueError("invalid split or nonpositive limit")
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be an integer from 1 to {MAX_WORKERS}")
    dataset = Path(eval_set_path)
    questions = load_eval_set(dataset)
    for q in questions:
        ids = q["gold_paper_ids"]
        if not q["query_id"].strip() or any(not isinstance(p, str) or not p.strip() for p in ids) or len(ids) != len(set(ids)):
            raise ValueError("empty query id or invalid/duplicate gold paper ids")
    selected = [q for q in questions if split == "all" or q["split"] == split][:limit]
    if not selected:
        raise ValueError("no questions selected")
    require_fresh_runtime()
    if llm_client._CLIENT is not None:
        raise RuntimeError("review evaluation requires a fresh LLM client in a dedicated process")
    cfg = get_config()
    if cfg["orchestration"]["review"].get("backend", "llm") != "llm":
        raise ValueError("review evaluation currently supports backend=llm only (Jev usage is not tracked)")
    run_dir = Path(results_dir).resolve() / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    with _lock(run_dir / ".lock"), config_overrides(cfg, {
            "orchestration.loop.cache": False, "paths.results": str(run_dir)}):
        runtime_environment = environment()
        identity = {
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "selected_query_ids": [q["query_id"] for q in selected], "split": split, "limit": limit,
            "effective_config": _safe_config(cfg), "indexes": index_manifests(), "code_sha256": _code_hash(),
            "input_files": _input_file_metadata(),
            "runtime": {k: runtime_environment.get(k) for k in ("python", "versions")},
        }
        metadata = run_dir / "manifest.json"
        fingerprint = _digest(identity)
        if metadata.exists():
            if not resume:
                raise FileExistsError("run already exists; use --resume or a new tag")
            manifest = json.loads(metadata.read_text(encoding="utf-8"))
            if manifest["fingerprint"] != fingerprint:
                raise ValueError("resume fingerprint differs (dataset, selection, config, indexes or code)")
        else:
            if resume:
                raise FileNotFoundError("cannot resume a run without manifest.json")
            manifest = {"tag": tag, "fingerprint": fingerprint, **identity,
                        "created_at": datetime.now(timezone.utc).isoformat(), "environment": runtime_environment}
            _write(metadata, manifest)
        attempts = []
        by_id = {q["query_id"]: q for q in selected}
        completed = set()
        for path in sorted((run_dir / "attempts").glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            qid = row["question"]["query_id"]
            if row["question"] != by_id.get(qid) or row["status"] not in ("running", "failed", "complete"):
                raise ValueError("checkpoint identity/status mismatch")
            if row["status"] == "complete":
                _validate_result(by_id[qid], row["result"])
                if qid in completed:
                    raise ValueError("duplicate completed query checkpoint")
                completed.add(qid)
            attempts.append(row)
        error = None
        try:
            pending = [q for q in selected if q["query_id"] not in completed]
            if pending:
                client = get_client()
                before, first_new = client.usage_summary(), len(attempts)
                error = _run_pending(pending, workers=workers, run_dir=run_dir, manifest=manifest,
                                     attempts=attempts, selected=selected, completed=completed)
                after = client.usage_summary()
                attributed = {k: sum(a["usage"][k] for a in attempts[first_new:]) for k in _USAGE_KEYS}
                if attributed != {k: after[k] - before[k] for k in _USAGE_KEYS}:
                    log.warning("LLM usage outside the per-query scopes: process %s, attempts %s",
                                {k: after[k] - before[k] for k in _USAGE_KEYS}, attributed)
            report = _report(manifest, attempts, selected, error)
            _write(run_dir / "report.json", report)
            return report
        finally:
            clear_runtime(release_models=True)
            llm_client._CLIENT = None


def format_summary(report: dict) -> str:
    usage = report["usage"]
    lines = [f"Review {report['tag']}: {report['status']}, "
             f"{report['completed_queries']}/{report['selected_queries']} queries",
             f"Observed API usage: {usage['calls']} calls, {usage['prompt_tokens']} input + "
             f"{usage['completion_tokens']} output tokens (not a billing total)"]
    for split, row in report["splits"].items():
        if row["n_queries"]:
            lines.append(f"{split}: source passed/rejected/not reviewed = {row['source_passed']}/"
                         f"{row['source_rejected']}/{row['source_not_reviewed']}; "
                         f"{row['unjudged_papers_reviewed']} other reviews remain unjudged")
    return "\n".join(lines)
