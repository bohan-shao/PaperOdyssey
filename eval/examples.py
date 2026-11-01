"""The UI example inputs through the deployed system (Q2: "write five queries, get their results,
and measure the speed of the querying"; D51).

`run_examples()` answers each `ui.example_inputs` entry with `downstream.run()`, the code path of
the UI's Paper QA tab (Paper Search, then one cited answer), with the configured settings and the
search cache off, so every result and timing is computed in this run. Retrieval components are
loaded first and timed apart (`cold_start_s`); the inputs then run warm, one after another, in a
fresh dedicated process.

Per input the report keeps the passing papers (paper ID, title, review score), the coverage
report, the answer with its numbered sources (title, section, URL and the excerpt the model saw)
and cited numbers, the latency in ms of every search stage and of the answer step, and the LLM
usage of that input (`llm.client.usage_scope`). A failed input is recorded with its error and the
others still run. `scripts/run_examples.py` writes the report to
`data/results/metrics/examples_<date>.json`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime

from config import get_config
from downstream import DownstreamError
from downstream import run as answer
from llm import client as llm_client
from retrieval.pipeline import warm_up

from .latency import config_overrides, environment, summarize
from .retrieval_eval import index_manifests
from .review_eval import _safe_config
from .runtime import clear_runtime, require_fresh_runtime


def _example(text: str) -> dict:
    """One input through Paper Search and the cited answer, with its timings and LLM usage."""
    row: dict = {"query": text}
    started = time.perf_counter()
    with llm_client.usage_scope() as used:
        try:
            result = answer(text)
        except (ValueError, DownstreamError) as exc:
            cause = f" ({exc.__cause__})" if exc.__cause__ else ""
            result, row["status"], row["error"] = None, "failed", f"{type(exc).__name__}: {exc}{cause}"
    row["wall_ms"] = round((time.perf_counter() - started) * 1000)
    row["usage"] = dict(used)
    if result is None:
        return row
    search = result.search_result
    row.update({
        "status": result.status,
        "answer": result.answer,
        "cited_source_numbers": result.cited_source_numbers,
        "sources": [asdict(source) for source in result.sources],
        "papers": [{"paper_id": p.paper_id, "title": p.evidence.get("title"), "score": p.score}
                   for p in search.papers],
        "coverage_report": search.coverage_report,
        "latency_ms": {"search": search.latency_by_stage, "answer": result.latency_by_stage},
    })
    return row


def run_examples(inputs: Sequence[str] | None = None) -> dict:
    """Answer the example inputs (default `ui.example_inputs`) and return the report."""
    cfg = get_config()
    inputs = list(cfg.get("ui", {}).get("example_inputs", []) if inputs is None else inputs)
    if not inputs or any(not isinstance(text, str) or not text.strip() for text in inputs):
        raise ValueError("example inputs must be a nonempty list of nonempty strings")
    require_fresh_runtime()
    try:
        with config_overrides(cfg["orchestration"], {"loop.cache": False}):
            start = time.perf_counter()
            warm_up()
            cold_start_s = round(time.perf_counter() - start, 2)
            rows = [_example(text) for text in inputs]
            snapshot = _safe_config(cfg)
    finally:
        clear_runtime(release_models=True)
    done = [row for row in rows if row["status"] != "failed"]
    return {
        "format_version": 1,
        "created": datetime.now().isoformat(timespec="seconds"),
        "protocol": "downstream.run() per input (Paper Search + one cited answer), configured settings, "
                    "search cache off, retrieval components loaded first (cold_start_s), inputs run "
                    "warm one after another",
        "cold_start_s": cold_start_s,
        "summary": {
            "inputs": len(rows), "failed": len(rows) - len(done),
            "total_ms": summarize([row["latency_ms"]["answer"]["total_ms"] for row in done]),
            "search_ms": summarize([row["latency_ms"]["answer"]["search_ms"] for row in done]),
            "generation_ms": summarize([row["latency_ms"]["answer"]["generation_ms"] for row in done]),
            "tokens": summarize([row["usage"]["prompt_tokens"] + row["usage"]["completion_tokens"]
                                 for row in done]),
        },
        "examples": rows,
        "corpus": index_manifests(),
        "environment": environment(),
        "config_snapshot": snapshot,
    }


def format_summary(report: dict) -> str:
    """One line per input plus the totals (the JSON report has everything)."""
    lines = [f"Example inputs through Paper Search + cited answer (cold start {report['cold_start_s']} s)", ""]
    for number, row in enumerate(report["examples"], 1):
        tokens = row["usage"]["prompt_tokens"] + row["usage"]["completion_tokens"]
        if row["status"] == "failed":
            lines.append(f"{number}. FAILED after {row['wall_ms'] / 1000:.1f} s: {row['error']}")
        else:
            ms = row["latency_ms"]["answer"]
            lines.append(f"{number}. {ms['total_ms'] / 1000:.1f} s (search {ms['search_ms'] / 1000:.1f}, "
                         f"answer {ms['generation_ms'] / 1000:.1f}), {row['status']}, "
                         f"{len(row['papers'])} papers, {len(row['cited_source_numbers'])} cited, "
                         f"{tokens} tokens")
        lines.append(f"   {row['query']}")
    total = report["summary"]["total_ms"]
    if total["n"]:
        lines += ["", (f"total per input: mean {total['mean'] / 1000:.1f} s, p50 {total['p50'] / 1000:.1f} s, "
                       f"max {total['max'] / 1000:.1f} s over {total['n']} inputs")]
    return "\n".join(lines)
