"""Answer-level evaluation of cited paper QA on finished Paper Search runs (Q4/Q5, D49).

For each question a review run completed (eval/review_eval.py), or each review row of an ablation
(eval/ablation.py, D46), the stored search result goes through `downstream.answer_from_search`,
the same evidence and generation code as the UI, and the answer is scored:
- against the eval set's source paper: was it among the supplied excerpts, and is it cited;
- RAGAS-style by an LLM judge (eval/answer_metrics.py): faithfulness, answer relevance,
  context relevance and context precision.
Reusing the stored searches pairs every answer with the search result the review report scored and
spends no search calls. An ablation source also gets paired comparisons of the answer metrics
(baseline -> A, baseline -> B, baseline -> AB and the one-switch edges).

Run in a dedicated process. Completed questions are immutable checkpoints; resume requires the same
source runs, selection, settings and code. The corpus and index files must still match the source
runs' manifests, because answers read passage text from the lookup store.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

from config import PROJECT_ROOT, get_config
from downstream import answer_from_search
from downstream.prompts import PROMPT_VERSION
from llm import client as llm_client
from llm.client import LLMClient, get_client
from orchestration.types import Criterion, FinalResult, QueryPlan, ReviewRecord, SubQuery
from retrieval.dense import get_encoder
from .answer_metrics import JUDGE_PROMPT_VERSION, judge_answer
from .latency import config_overrides, environment, summarize
from .retrieval_eval import index_manifests
from .review_eval import (MAX_WORKERS, _code_hash as _search_code_hash, _digest, _input_file_metadata,
                          _lock, _safe_config, _validate_result, _write)
from .runtime import clear_runtime, require_fresh_runtime

log = logging.getLogger(__name__)
_USAGE_KEYS = ("calls", "prompt_tokens", "completion_tokens")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
MAX_FAILURES = 5  # failed questions after which no new question starts in this invocation
JUDGED = ("faithfulness", "answer_relevance", "context_relevance", "context_precision")
LABELED = ("gold_in_sources", "gold_cited")


def _zero() -> dict:
    return {"generation": dict.fromkeys(_USAGE_KEYS, 0), "judge": dict.fromkeys(_USAGE_KEYS, 0)}


def _add(total: dict, usage: dict) -> None:
    for part in ("generation", "judge"):
        for key in _USAGE_KEYS:
            total[part][key] += usage[part][key]


def _code_hash() -> str:
    files = sorted((PROJECT_ROOT / "downstream").rglob("*.py"))
    return _digest({"search_and_eval": _search_code_hash(),
                    **{str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})


def _settings(cfg: dict) -> dict:
    settings = {"judge_model": None, "judge_temperature": 0.0, "judge_max_tokens": 2048,
                "relevance_questions": 3, **(cfg.get("eval", {}).get("answer") or {})}
    for key in ("judge_max_tokens", "relevance_questions"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"eval.answer.{key} must be a positive integer")
    if isinstance(settings["judge_temperature"], bool) or not isinstance(settings["judge_temperature"], (int, float)):
        raise ValueError("eval.answer.judge_temperature must be a number")
    if settings["judge_model"] is not None and (not isinstance(settings["judge_model"], str)
                                                or not settings["judge_model"].strip()):
        raise ValueError("eval.answer.judge_model must be null or a model name")
    return settings


def _final_result(data: dict) -> FinalResult:
    """Rebuild the FinalResult that review_eval stored with dataclasses.asdict."""
    plan = data["query_plan"]
    return FinalResult(
        query_id=data["query_id"], input_text=data["input_text"],
        query_plan=QueryPlan(plan["input_text"], [SubQuery(**s) for s in plan["sub_queries"]],
                             [Criterion(**c) for c in plan["criteria"]]),
        papers=[ReviewRecord(**p) for p in data["papers"]], coverage_report=data["coverage_report"],
        latency_by_stage=data["latency_by_stage"], reviews=[ReviewRecord(**r) for r in data["reviews"]],
        retrieval_ranks=data["retrieval_ranks"])


def _source_rows(source: Path) -> tuple[str, dict[str, Path], dict]:
    """("review_run" | "ablation", {row: review run directory}, the ablation plan or {})."""
    if (source / "rows").is_dir():
        plan = json.loads((source / "manifest.json").read_text(encoding="utf-8"))["plan"]
        if "review" not in plan["stages"]:
            raise ValueError("the ablation has no review cells to answer from")
        return "ablation", {row: source / "rows" / row / "review" for row in plan["rows"]}, plan
    if (source / "attempts").is_dir() and (source / "manifest.json").is_file():
        return "review_run", {"run": source}, {}
    raise ValueError(f"{source} is neither a review run nor an ablation run directory")


def _completed(directory: Path) -> dict[str, tuple[dict, dict]]:
    """{query_id: (question, search result)} of the run's validated complete attempts."""
    done: dict[str, tuple[dict, dict]] = {}
    for path in sorted((directory / "attempts").glob("*.json")):
        attempt = json.loads(path.read_text(encoding="utf-8"))
        if attempt.get("status") != "complete":
            continue
        question, result = attempt["question"], attempt["result"]
        _validate_result(question, result)
        if question["query_id"] in done:
            raise ValueError(f"{directory}: duplicate completed query {question['query_id']}")
        done[question["query_id"]] = (question, result)
    return done


def _evaluate(question: dict, result: dict, *, judge: LLMClient, encoder, settings: dict, workers: int) -> dict:
    usage = _zero()

    def measured(part: str, fn: Callable):
        with llm_client.usage_scope() as used:
            try:
                return fn()
            finally:
                usage[part] = dict(used)

    row: dict = {"question": question, "workers": workers}
    started = time.perf_counter()
    try:
        answer = measured("generation", lambda: answer_from_search(question["input_text"], _final_result(result)))
        generated = time.perf_counter()
        scores = measured("judge", lambda: judge_answer(
            judge, encoder, answer.input_text, answer.answer, answer.sources,
            n_questions=settings["relevance_questions"], max_tokens=settings["judge_max_tokens"]))
        gold = set(question["gold_paper_ids"])
        cited = {s.paper_id for s in answer.sources if s.number in answer.cited_source_numbers}
        row.update(status="complete",
                   answer={k: v for k, v in asdict(answer).items() if k != "search_result"},
                   task={"gold_in_sources": bool(gold & {s.paper_id for s in answer.sources}),
                         "gold_cited": bool(gold & cited)},
                   scores=scores, judge_ms=round((time.perf_counter() - generated) * 1000, 2))
    except Exception as exc:
        # Provider error strings can contain request details; keep the class and query only.
        row.update(status="failed", error={"query_id": question["query_id"], "type": type(exc).__name__})
        log.error("answer evaluation failed at %s (%s)", question["query_id"], type(exc).__name__)
    row.update(usage=usage, wall_ms=round((time.perf_counter() - started) * 1000, 2))
    return row


def _run_pending(pending: list[tuple[dict, dict]], workers: int, evaluate: Callable, record: Callable) -> None:
    """Evaluate up to `workers` questions at once; `record(row)` runs under one lock, True stops new starts."""
    lock = threading.Lock()
    queue = iter(pending)
    stop = threading.Event()

    def worker() -> None:
        while not stop.is_set():
            with lock:
                item = next(queue, None)
            if item is None:
                return
            row = evaluate(*item)
            with lock:
                if record(row):
                    stop.set()

    if workers == 1:
        worker()
        return
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="answer")
    try:
        for future in [pool.submit(worker) for _ in range(min(workers, len(pending)))]:
            future.result()
    finally:
        stop.set()  # after an interrupt, running questions finish and are recorded; none start
        pool.shutdown(wait=True)


def _values(row: dict) -> dict:
    return {**{m: row["scores"][m] for m in JUDGED}, **{m: float(row["task"][m]) for m in LABELED}}


def _mean(values: list) -> dict:
    defined = [v for v in values if v is not None]
    return {"n": len(defined), "mean": fmean(defined) if defined else None}


def _score(rows: list[dict]) -> dict:
    answers = [r["answer"] for r in rows]
    scores = [r["scores"] for r in rows]
    claims = sum(len(s["claims"]) for s in scores)
    supported = sum(v["supported"] for s in scores for v in s["claim_verdicts"])
    tokens = {part: [sum(r["usage"][part][k] for k in ("prompt_tokens", "completion_tokens")) for r in rows]
              for part in ("generation", "judge")}
    return {
        "n_queries": len(rows),
        "answered": sum(a["status"] == "answered" for a in answers),
        "insufficient_evidence": sum(a["status"] == "insufficient_evidence" for a in answers),
        "no_sources": sum(not a["sources"] for a in answers),
        **{m: _mean([float(r["task"][m]) for r in rows]) for m in LABELED},
        "faithfulness": {**_mean([s["faithfulness"] for s in scores]), "claims": claims, "supported_claims": supported,
                         "pooled": supported / claims if claims else None},
        "answer_relevance": {**_mean([s["answer_relevance"] for s in scores]),
                             "noncommittal": sum(s["noncommittal"] for s in scores)},
        "context_relevance": _mean([s["context_relevance"] for s in scores]),
        "context_precision": _mean([s["context_precision"] for s in scores]),
        "sources_per_answer": _mean([len(a["sources"]) for a in answers]),
        "generation_ms": summarize([a["latency_by_stage"]["generation_ms"] for a in answers if a["sources"]]),
        "tokens_per_query": {part: _mean(values) for part, values in tokens.items()},
        "per_query": {r["question"]["query_id"]: {"status": r["answer"]["status"], **_values(r)} for r in rows},
    }


def _compare(done: dict[str, dict[str, dict]], edges: list) -> list[dict]:
    """Paired answer-metric means on the questions both rows completed, per split."""
    comparisons = []
    for before, after in edges:
        splits = {}
        for split in ("dev", "eval"):
            paired = [q for q, r in done[before].items() if q in done[after] and r["question"]["split"] == split]
            metrics = {}
            for metric in (*JUDGED, *LABELED):
                pairs = [(_values(done[before][q])[metric], _values(done[after][q])[metric]) for q in paired]
                pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                metrics[metric] = {"n": len(pairs)}
                if pairs:
                    first, second = fmean(x for x, _ in pairs), fmean(y for _, y in pairs)
                    metrics[metric].update(before=first, after=second, delta=second - first)
            splits[split] = {"n_paired": len(paired), "metrics": metrics}
        comparisons.append({"before": before, "after": after, "splits": splits})
    return comparisons


def _report(manifest: dict, kind: str, plan: dict, state: dict, selection: list[str], stopped: bool) -> dict:
    usage = _zero()
    rows = {}
    for name, row in state.items():
        row_usage = _zero()
        for checkpoint in [*row["done"].values(), *row["failed"].values()]:
            _add(row_usage, checkpoint["usage"])
            _add(row_usage, checkpoint.get("earlier_usage") or _zero())
        _add(usage, row_usage)
        complete = [row["done"][q] for q in selection if q in row["done"]]
        rows[name] = {
            "source_dir": str(row["source_dir"]), "searched_queries": len(row["searched"]),
            "completed_queries": len(complete), "failed_query_ids": sorted(row["failed"]),
            "not_searched_query_ids": [q for q in selection if q not in row["searched"]],
            "usage": row_usage,
            "splits": {split: _score([r for r in complete if r["question"]["split"] == split])
                       for split in ("dev", "eval")},
        }
        if kind == "ablation":
            rows[name]["enabled_factors"] = plan["rows"][name]["enabled_factors"]
    finished = all(set(row["done"]) == set(row["searched"]) for row in state.values())
    report = {
        "stage": "answer", "format_version": 1, "tag": manifest["tag"], "source_kind": kind,
        "status": "failed" if stopped else "complete" if finished else "partial",
        "updated_at": datetime.now(timezone.utc).isoformat(), "selected_queries": len(selection),
        "usage": usage,
        "usage_scope": "Observed responses of complete and failed questions, including retries of failed ones, "
                       "split into answer generation and judging. Interrupted questions are not recorded; "
                       "raw best-effort usage is in metrics/llm_usage.jsonl. Not a billing total.",
        "protocol": {
            "answers": "downstream.answer_from_search on each stored search result: the UI's evidence and "
                       "generation code, one generation call, no search calls.",
            "labels": "gold_in_sources / gold_cited use the eval set's source paper (one per question).",
            "judge": "faithfulness, answer relevance, context relevance and precision are LLM-judge scores "
                     "(eval/answer_metrics.py), not human labels; a judge sharing the generator's model can "
                     "favor its own answers. No reference answers exist, so ROUGE/BERTScore/EM are not reported.",
            "undefined": "faithfulness is undefined for answers without claims and context scores for answers "
                         "without excerpts; each metric's n counts the defined values.",
            "concurrency": "With workers > 1 several questions run at once; generation_ms then includes waiting "
                           "for other questions' calls. Usage is counted per question either way.",
            "failures": "Failed questions are excluded from metrics and listed; --resume retries them.",
        },
        "provenance": manifest, "rows": rows,
        "workers_used": sorted({c.get("workers", 1) for row in state.values()
                                for c in [*row["done"].values(), *row["failed"].values()]}),
    }
    if kind == "ablation":
        edges = [tuple(e) for e in plan["edges"]]
        if "ab" in plan["rows"]:
            edges.insert(2, ("baseline", "ab"))
        report["comparisons"] = _compare({n: r["done"] for n, r in state.items()}, edges)
    return report


def _write_answers(path: Path, rows: list[dict]) -> None:
    """One line per completed question: the generated answer, its sources (no excerpt text) and scores."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for r in rows:
            q, a = r["question"], r["answer"]
            out.write(json.dumps({
                "query_id": q["query_id"], "split": q["split"], "input_text": q["input_text"],
                "gold_paper_ids": q["gold_paper_ids"], "status": a["status"], "answer": a["answer"],
                "cited_source_numbers": a["cited_source_numbers"],
                "sources": [{k: s[k] for k in ("number", "paper_id", "passage_id", "title", "year", "section", "url")}
                            for s in a["sources"]],
                **_values(r),
            }, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def run_answer_eval(source_dir: str | Path, results_dir: str | Path, tag: str, *, split: str = "all",
                    limit: int | None = None, resume: bool = False, workers: int = 1) -> dict:
    """Generate and score answers for a finished review run or review ablation; save report.json.

    `split` and `limit` select from the source's question order. Questions the source search did not
    complete are listed, not answered. Up to `workers` questions run at once; after MAX_FAILURES
    failed questions no new one starts and the report has status=failed. Fingerprint, source or
    checkpoint errors raise before any API call.
    """
    if not _SAFE_ID.fullmatch(tag):
        raise ValueError("tag must be a safe filename component")
    if split not in ("all", "dev", "eval") or (limit is not None and (type(limit) is not int or limit <= 0)):
        raise ValueError("invalid split or nonpositive limit")
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be an integer from 1 to {MAX_WORKERS}")
    source = Path(source_dir).resolve()
    kind, sources, plan = _source_rows(source)
    manifests, searched = {}, {}
    for name, directory in sources.items():
        if not (directory / "manifest.json").is_file():
            raise ValueError(f"{directory} has no review manifest (the search run has not started)")
        manifests[name] = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        searched[name] = _completed(directory)
    order = next(iter(manifests.values()))["selected_query_ids"]
    if any(m["selected_query_ids"] != order for m in manifests.values()):
        raise ValueError("the source rows selected different questions")
    questions: dict[str, dict] = {}
    for done in searched.values():
        for qid, (question, _) in done.items():
            if questions.setdefault(qid, question) != question:
                raise ValueError(f"question {qid} differs between source rows")
    selection = [q for q in order if q in questions and (split == "all" or questions[q]["split"] == split)][:limit]
    if not selection:
        raise ValueError("no completed search results selected")
    if bad := [q for q in selection if not _SAFE_ID.fullmatch(q)]:
        raise ValueError(f"query IDs are not safe file names: {bad[:3]}")
    require_fresh_runtime()
    if llm_client._CLIENT is not None:
        raise RuntimeError("answer evaluation requires a fresh LLM client in a dedicated process")
    indexes, files = index_manifests(), _input_file_metadata()
    for name, manifest in manifests.items():
        if manifest.get("indexes") != indexes or manifest.get("input_files") != files:
            raise ValueError(f"index or corpus files differ from source row {name}'s search run")
    cfg = get_config()
    settings = _settings(cfg)
    run_dir = Path(results_dir).resolve() / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    with _lock(run_dir / ".lock"), config_overrides(cfg, {"paths.results": str(run_dir)}):
        identity = {
            "source": {"kind": kind, "path": str(source),
                       "fingerprints": {n: m["fingerprint"] for n, m in manifests.items()},
                       "dataset_sha256": {n: m.get("dataset_sha256") for n, m in manifests.items()}},
            "selection": {"split": split, "limit": limit, "query_ids": selection},
            "searched": {n: [q for q in selection if q in done] for n, done in searched.items()},
            "settings": {"downstream": cfg.get("downstream"), "answer": settings, "llm": _safe_config(cfg)["llm"],
                         "dense_model": cfg["retrieval"]["dense"]["model_name"]},
            "prompt_versions": {"answer": PROMPT_VERSION, "judge": JUDGE_PROMPT_VERSION},
            "code_sha256": _code_hash(),
        }
        fingerprint = _digest(identity)
        metadata = run_dir / "manifest.json"
        if metadata.exists():
            if not resume:
                raise FileExistsError("answer run already exists; use --resume or a new tag")
            manifest = json.loads(metadata.read_text(encoding="utf-8"))
            if manifest["fingerprint"] != fingerprint:
                raise ValueError("resume fingerprint differs (sources, selection, settings or code)")
        else:
            if resume:
                raise FileNotFoundError("cannot resume an answer run without manifest.json")
            manifest = {"tag": tag, "fingerprint": fingerprint, **identity,
                        "created_at": datetime.now(timezone.utc).isoformat(), "environment": environment()}
            _write(metadata, manifest)
        state = {}
        for name in sources:
            row = {"source_dir": sources[name], "searched": set(identity["searched"][name]), "done": {}, "failed": {}}
            for path in sorted((run_dir / "rows" / name / "answers").glob("*.json")):
                checkpoint = json.loads(path.read_text(encoding="utf-8"))
                qid = checkpoint["question"]["query_id"]
                if (path.stem != qid or qid not in row["searched"] or checkpoint["question"] != questions[qid]
                        or checkpoint["status"] not in ("complete", "failed")):
                    raise ValueError(f"checkpoint {path} does not match this run")
                row["done" if checkpoint["status"] == "complete" else "failed"][qid] = checkpoint
            state[name] = row
        stopped = False
        try:
            pending = {n: [(questions[q], searched[n][q][1]) for q in selection
                           if q in state[n]["searched"] and q not in state[n]["done"]] for n in sources}
            if any(pending.values()):
                generator = get_client()
                judge = LLMClient(model=settings["judge_model"], temperature=settings["judge_temperature"])
                encoder = get_encoder()
                clients = (generator, judge)
                before = [c.usage_summary() for c in clients]
                attributed = dict.fromkeys(_USAGE_KEYS, 0)
                failures = 0
                for name in sources:
                    row = state[name]

                    def record(checkpoint: dict, row=row, name=name) -> bool:
                        nonlocal failures
                        qid = checkpoint["question"]["query_id"]
                        if qid in row["failed"]:  # a retry keeps the usage of the earlier failed attempts
                            earlier = row["failed"].pop(qid)
                            checkpoint["earlier_usage"] = _zero()
                            _add(checkpoint["earlier_usage"], earlier["usage"])
                            _add(checkpoint["earlier_usage"], earlier.get("earlier_usage") or _zero())
                        _write(run_dir / "rows" / name / "answers" / f"{qid}.json", checkpoint)
                        for part in ("generation", "judge"):
                            for key in _USAGE_KEYS:
                                attributed[key] += checkpoint["usage"][part][key]
                        if checkpoint["status"] == "complete":
                            row["done"][qid] = checkpoint
                        else:
                            row["failed"][qid] = checkpoint
                            failures += 1
                        log.info("answer %s: %d/%d complete", name, len(row["done"]), len(row["searched"]))
                        return failures >= MAX_FAILURES

                    _run_pending(pending[name], workers,
                                 lambda q, r: _evaluate(q, r, judge=judge, encoder=encoder,
                                                        settings=settings, workers=workers), record)
                    _write(run_dir / "report.json", _report(manifest, kind, plan, state, selection, False))
                    if failures >= MAX_FAILURES:
                        stopped = True
                        break
                observed = {k: sum(c.usage_summary()[k] - b[k] for c, b in zip(clients, before)) for k in _USAGE_KEYS}
                if attributed != observed:
                    log.warning("LLM usage outside the per-question scopes: process %s, questions %s",
                                observed, attributed)
            report = _report(manifest, kind, plan, state, selection, stopped)
            _write(run_dir / "report.json", report)
            for name in sources:
                done = [state[name]["done"][q] for q in selection if q in state[name]["done"]]
                _write_answers(run_dir / "rows" / name / "answers.jsonl", done)
            return report
        finally:
            clear_runtime(release_models=True)
            llm_client._CLIENT = None


def format_summary(report: dict) -> str:
    usage = report["usage"]
    tokens = {part: usage[part]["prompt_tokens"] + usage[part]["completion_tokens"] for part in usage}
    lines = [f"Answers {report['tag']}: {report['status']}; {report['selected_queries']} selected questions",
             f"Observed API usage: generation {tokens['generation']} tokens, judge {tokens['judge']} tokens "
             "(not a billing total)"]
    for name, row in report["rows"].items():
        for split, s in row["splits"].items():
            if not s["n_queries"]:
                continue
            values = ", ".join(f"{m} {s[m]['mean']:.3f}" for m in (*JUDGED, *LABELED) if s[m]["mean"] is not None)
            lines.append(f"{name}/{split}: n={s['n_queries']}; {values}")
        if row["failed_query_ids"]:
            lines.append(f"{name}: failed {len(row['failed_query_ids'])} (--resume retries them)")
    for edge in report.get("comparisons", []):
        for split, values in edge["splits"].items():
            faith = values["metrics"]["faithfulness"]
            if faith["n"]:
                lines.append(f"{split} {edge['before']} -> {edge['after']}: n={values['n_paired']}, "
                             f"faithfulness {faith['before']:.3f} -> {faith['after']:.3f} ({faith['delta']:+.3f})")
    return "\n".join(lines)
