"""Retrieval-core latency benchmark (Q2 latency, EVALUATION.md section 7).

Times `retrieval.pipeline.retrieve()` alone (no LLM stages) against the built indexes, in one
process, and returns one report (written to `data/results/metrics/latency_retrieval_<date>.json`,
or `latency_retrieval_production_<date>.json` with the production protocol, by
`scripts/benchmark_retrieval.py`):

- cold start: loading every component (`warm_up()`), reported apart from per-query numbers.
- configs: the four ablation rows (bm25, dense, hybrid, hybrid_rerank) plus hybrid_rerank with
  filters and with parent-doc mode; per stage (sparse / dense / fusion / rerank / parent /
  total) the n, mean, p50, p95 and max over all queries, all warm. `budget` checks the
  hybrid_rerank p95 against DESIGN.md section 4 (< 500 ms per retrieval path).
- throughput: queries/second of the default configuration with 1, 4 and 8 concurrent callers,
  plus the per-call latency they see (queueing included).
- ef_search_sweep: FAISS HNSW search time per query (the query is encoded once, outside the
  timing) and recall@k against exact search over all vectors, which needs no relevance labels.
- rerank_depth_sweep: cross-encoder time for the top 10/20/30/50 fused candidates, and the share
  of the top 10 that agrees with reranking all of them (a label-free cost/quality trade-off). The
  agreement is the raw reranker's top 10, before the final per-paper/per-parent caps.
- examples: the 5 UI example inputs with their top hits (Q2: "write 5 queries, produce results,
  measure latency").
- environment, config snapshot, corpus size and manifests of the actual indexes. The latest
  build report is supplementary, unverified context (EVALUATION.md section 9).

Queries are paper titles sampled from papers.sqlite with the configured seed (latency needs no
relevance labels). Configurations are switched by overriding the in-memory config
(`config_overrides`), never config.yaml, and are restored afterwards.

Protocols (D51): `legacy_50_v2` (the default) fixes the ranking switches and budgets in `BASE`
before loading anything, so runs stay comparable across config changes. `production` fixes
nothing: every row, the throughput, the examples and the sweeps run the configured settings
(query processing, diversity, 150/150/50 budgets), with each row changing only its own
switches, as in the retrieval stage of run_eval.py. Use `production` for the deployed
system's latency (Q2). HNSW searches with max(efSearch, k), so each protocol has its own
efSearch defaults (`EF_DEFAULTS`; production starts at its dense.top_k of 150) and every sweep
row records the value actually used.

Conventions: latencies in milliseconds; percentiles interpolate linearly between ranks (numpy's
default): p of sorted x at position (n - 1) * p / 100. Numbers depend on corpus size (BM25 and
HNSW search times grow with the index, the rerank cost per query does not), so rerun after the
full KB build before citing them.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import statistics
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from config import get_config, get_path
from kb import store as kb_store
from retrieval.pipeline import retrieve, warm_up
from retrieval.types import RetrievalFilters, ScoredPassage
from .runtime import clear_runtime, require_fresh_runtime

log = logging.getLogger(__name__)

# Versioned legacy experiment protocol. Model/backend/index identity comes from the config;
# all active ranking switches and budgets below are fixed before loading any components.
BASE = {"sparse.enabled": True, "dense.enabled": True, "rerank.enabled": True,
        "parent_doc.enabled": False, "parent_doc.max_chars": 6000, "filters.enabled": True,
        "query_processing.enabled": False, "diversity.enabled": False,
        "sparse.field_weighting.enabled": False, "parallel": True,
        "sparse.top_k": 50, "dense.top_k": 50, "fusion.top_k": 50, "rerank.top_k": 10,
        "fusion.rrf_k": 60, "fusion.weights": [1.0, 1.0],
        "sparse.bm25.k1": 0.9, "sparse.bm25.b": 0.4,
        "dense.hnsw.efSearch": 128, "dense.filter_exact_max": 50000,
        "rerank.batch_size": 64, "rerank.max_length": 512}
PROTOCOLS: dict[str, dict[str, object]] = {"legacy_50_v2": BASE, "production": {}}
# efSearch values swept when none are given. HNSW searches with max(efSearch, k), and production
# keeps dense.top_k 150, so values below 150 would all time the same search.
EF_DEFAULTS: dict[str, tuple[int, ...]] = {"legacy_50_v2": (32, 64, 128, 256), "production": (150, 200, 256, 512)}
FILTERS = RetrievalFilters(year_min=2020, categories=["cs.CL"])  # "recent NLP papers"
CONFIGS: dict[str, tuple[dict[str, object], RetrievalFilters | None]] = {
    "bm25": ({"dense.enabled": False, "rerank.enabled": False}, None),
    "dense": ({"sparse.enabled": False, "rerank.enabled": False}, None),
    "hybrid": ({"rerank.enabled": False}, None),
    "hybrid_rerank": ({}, None),
    "hybrid_rerank_filtered": ({}, FILTERS),
    "hybrid_rerank_parent": ({"parent_doc.enabled": True}, None),
}
BUDGET_MS = 500  # DESIGN.md section 4: parallel retrieval incl. rerank, per path
_UNMEASURED = 3  # queries run before each config's measurement (JIT / cache warm-up)


# ---------------------------------------------------------------- statistics


def percentile(values: Sequence[float], q: float) -> float:
    """q-th percentile (0-100), linear interpolation between ranks."""
    if not values:
        raise ValueError("percentile of no values")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(values: Sequence[float]) -> dict:
    """{"n", "mean", "p50", "p95", "max"}, rounded to 0.01; just {"n": 0} when empty."""
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": round(statistics.fmean(values), 2),
            "p50": round(percentile(values, 50), 2), "p95": round(percentile(values, 95), 2),
            "max": round(max(values), 2)}


def recall_at_k(approx_rows: Sequence[int], exact_rows: Sequence[int]) -> float:
    """Share of the exact top-k rows the approximate search also found (-1 = empty slot)."""
    exact = {int(r) for r in exact_rows if r >= 0}
    if not exact:
        return 1.0
    return len(exact & {int(r) for r in approx_rows if r >= 0}) / len(exact)


def overlap_at_k(ids: Sequence[str], reference: Sequence[str]) -> float:
    """Share of the reference list that also appears in ids (order ignored)."""
    if not reference:
        return 1.0
    return len(set(ids) & set(reference)) / len(reference)


# ---------------------------------------------------------------- inputs and config


def sample_title_queries(sqlite_path: str | Path, n: int, seed: int) -> list[str]:
    """n distinct paper titles (whitespace-normalized), sampled reproducibly."""
    titles = sorted({" ".join(p["title"].split()) for p in kb_store.iter_papers(sqlite_path)
                     if (p.get("title") or "").strip()})
    return random.Random(seed).sample(titles, min(n, len(titles)))


@contextmanager
def config_overrides(section: dict, overrides: dict[str, object]) -> Iterator[dict]:
    """Temporarily set dotted keys ("rerank.enabled") in a config section, restored on exit.

    The section dict is changed in place (retrieve() reads it on every call) and restored to an
    exact copy of its old content, even if the block raises.
    """
    saved = copy.deepcopy(section)
    try:
        for key, value in overrides.items():
            *parents, name = key.split(".")
            node = section
            for parent in parents:
                node = node.setdefault(parent, {})
            node[name] = value
        yield section
    finally:
        section.clear()
        section.update(saved)


# ---------------------------------------------------------------- measurements


def measure_config(section: dict, queries: Sequence[str], overrides: dict[str, object],
                   filters: RetrievalFilters | None = None, base: dict[str, object] = BASE) -> dict:
    """Per-stage latency of retrieve() for every query under one configuration."""
    stages: dict[str, list[float]] = {}
    hits = candidates = 0
    parent_counts = {"expanded": 0, "unchanged": 0, "no_parent": 0}
    with config_overrides(section, {**base, **overrides}):
        effective_config = copy.deepcopy(section)
        for q in queries[:_UNMEASURED]:
            retrieve(q, filters)
        for q in queries:
            result = retrieve(q, filters)
            for key, ms in result.latency_ms.items():
                stages.setdefault(key.removesuffix("_ms"), []).append(ms)
            hits += len(result.hits)
            candidates += result.total_candidates
            if section.get("parent_doc", {}).get("enabled", False):
                from retrieval.lookup import get_lookup_store

                children = get_lookup_store().get_passages(h.passage_id for h in result.hits)
                for hit in result.hits:
                    outcome = ("no_parent" if not hit.parent_section_id else "unchanged"
                               if hit.text == children[hit.passage_id]["text"] else "expanded")
                    parent_counts[outcome] += 1
    n = max(len(queries), 1)
    return {"overrides": overrides, "effective_config": effective_config,
            "filters": asdict(filters) if filters else None,
            "latency_ms": {stage: summarize(v) for stage, v in stages.items()},
            "parent_text_counts": parent_counts,
            "mean_hits": round(hits / n, 2), "mean_candidates": round(candidates / n, 2)}


def measure_throughput(section: dict, queries: Sequence[str], levels: Sequence[int],
                       base: dict[str, object] = BASE) -> list[dict]:
    """Queries/second of the default configuration with each number of concurrent callers."""
    rows = []
    with config_overrides(section, base):
        for level in levels:
            start = time.perf_counter()
            with ThreadPoolExecutor(level) as pool:
                totals = [r.latency_ms["total_ms"] for r in pool.map(retrieve, queries)]
            wall = time.perf_counter() - start
            rows.append({"concurrency": level, "queries": len(queries), "wall_s": round(wall, 3),
                         "qps": round(len(queries) / wall, 2), "latency_ms": summarize(totals)})
    return rows


def sweep_ef_search(query_vecs: object, index: object, values: Sequence[int], k: int) -> list[dict]:
    """HNSW search time and recall@k vs exact search for each efSearch value.

    FAISS searches with max(efSearch, k); each row records that value as `effective_ef_search`.

    Args:
        query_vecs: (n, dim) float32 query vectors (already encoded, so encoding is not timed).
        index: a FAISS IndexHNSWFlat; its stored vectors give the exact top-k.
    """
    import faiss

    exact = faiss.downcast_index(index.storage).search(query_vecs, k)[1]
    rows = []
    for ef in values:
        effective = max(ef, k)
        params = faiss.SearchParametersHNSW(efSearch=effective)
        times, recalls = [], []
        for i in range(len(query_vecs)):
            start = time.perf_counter()
            approx = index.search(query_vecs[i:i + 1], k, params=params)[1]
            times.append((time.perf_counter() - start) * 1000)
            recalls.append(recall_at_k(approx[0], exact[i]))
        rows.append({"ef_search": ef, "effective_ef_search": effective, "search_ms": summarize(times),
                     "recall_at_k": round(statistics.fmean(recalls), 4)})
    return rows


def sweep_rerank_depth(queries: Sequence[str], candidates: Sequence[Sequence[ScoredPassage]],
                       depths: Sequence[int],
                       rerank_fn: Callable[[str, list[ScoredPassage], int], list[ScoredPassage]],
                       top_k: int = 10) -> list[dict]:
    """Rerank time for the best `depth` fused candidates, and top-k agreement with the deepest.

    The agreement compares the reranker's raw top k. With diversity on (production), the deployed
    pipeline then applies `select_hits(..., final=True)` (per-paper and per-parent caps), so this
    is the raw reranker's agreement, not that of the final top k users see.

    Args:
        candidates: candidates[i] is the fused list (best first) for queries[i].
    """
    if not queries or len(queries) != len(candidates) or not depths or any(d <= 0 for d in depths) or top_k <= 0:
        raise ValueError("rerank sweep requires paired nonempty inputs and positive depths/top_k")
    times: dict[int, list[float]] = {d: [] for d in depths}
    tops: dict[int, list[list[str]]] = {d: [] for d in depths}
    for q, cands in zip(queries, candidates):
        for d in depths:
            start = time.perf_counter()
            top = rerank_fn(q, list(cands[:d]), top_k)
            times[d].append((time.perf_counter() - start) * 1000)
            tops[d].append([h.passage_id for h in top])
    reference = tops[max(depths)]
    return [{"depth": d, "actual_pairs": summarize([min(d, len(c)) for c in candidates]),
             "rerank_ms": summarize(times[d]),
             "top_k_agreement": round(statistics.fmean(overlap_at_k(ids, ref) for ids, ref
                                                       in zip(tops[d], reference)), 4)}
            for d in depths]


def example_results(section: dict, inputs: Sequence[str], top_n: int = 3,
                    base: dict[str, object] = BASE) -> list[dict]:
    """The default configuration's top hits and latency for each example input."""
    rows = []
    with config_overrides(section, base):
        for text in inputs:
            result = retrieve(text)
            rows.append({"query": text, "latency_ms": result.latency_ms, "top": [
                {"rank": h.rank, "score": round(h.score, 4), "title": h.title, "year": h.year,
                 "passage_id": h.passage_id} for h in result.hits[:top_n]]})
    return rows


def environment() -> dict:
    """Python / platform / GPU and the versions of the libraries on the retrieval path."""
    versions = {}
    for pkg in ("torch", "transformers", "sentence-transformers", "faiss-cpu", "pyserini", "numpy"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    gpu = None
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "gpu": gpu, "versions": versions}


# ---------------------------------------------------------------- the whole benchmark


def run_benchmark(n_queries: int = 100, concurrency: Sequence[int] = (1, 4, 8),
                  ef_values: Sequence[int] | None = None,
                  rerank_depths: Sequence[int] = (10, 20, 30, 50), protocol: str = "legacy_50_v2") -> dict:
    """Run one protocol (`PROTOCOLS`) with configured models/indexes in a fresh process.

    `ef_values` defaults to the protocol's `EF_DEFAULTS`.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown latency protocol {protocol!r}; choose from {sorted(PROTOCOLS)}")
    if ef_values is None:
        ef_values = EF_DEFAULTS[protocol]
    if n_queries <= 0 or any(not values or any(v <= 0 for v in values)
                             for values in (concurrency, ef_values, rerank_depths)):
        raise ValueError("benchmark query count, concurrency, ef values and depths must be positive")
    require_fresh_runtime()
    section = get_config()["retrieval"]
    requested = copy.deepcopy(section)
    base = PROTOCOLS[protocol]
    try:
        with config_overrides(section, base):
            report = _run_benchmark(n_queries, concurrency, ef_values, rerank_depths, protocol)
            report["requested_config_snapshot"] = requested
            return report
    finally:
        clear_runtime(release_models=True)


def _depth_candidates(section: dict, queries: Sequence[str], depth: int,
                      base: dict[str, object] = BASE) -> tuple[list, dict]:
    sparse_k = base.get("sparse.top_k", section["sparse"].get("top_k", 50))
    dense_k = base.get("dense.top_k", section["dense"].get("top_k", 50))
    overrides = {**base, "rerank.enabled": False, "fusion.top_k": depth,
                 "sparse.top_k": max(sparse_k, depth), "dense.top_k": max(dense_k, depth)}
    with config_overrides(section, overrides):
        return [retrieve(q, top_k=depth).hits for q in queries], copy.deepcopy(section)


def _run_benchmark(n_queries: int, concurrency: Sequence[int], ef_values: Sequence[int],
                   rerank_depths: Sequence[int], protocol: str = "legacy_50_v2") -> dict:
    """Run every measurement against the configured indexes and return the report."""
    from retrieval import dense, lookup, rerank, sparse

    cfg = get_config()
    section = cfg["retrieval"]
    seed = cfg.get("seed", 42)
    paper_db = get_path("processed") / cfg["kb"]["store"]["sqlite_file"]
    queries = sample_title_queries(paper_db, n_queries, seed)
    if not queries:
        raise RuntimeError("no paper titles in papers.sqlite to use as queries")

    log.info("cold start: loading indexes and models")
    start = time.perf_counter()
    base = PROTOCOLS[protocol]
    with config_overrides(section, {**base, "parent_doc.enabled": True}):
        warm_up()
    cold_start_s = round(time.perf_counter() - start, 2)

    configs = {}
    for name, (overrides, filters) in CONFIGS.items():
        log.info("config %s: %d queries", name, len(queries))
        configs[name] = measure_config(section, queries, overrides, filters, base)
    p95 = configs["hybrid_rerank"]["latency_ms"]["total"]["p95"]

    log.info("throughput at concurrency %s", list(concurrency))
    throughput = measure_throughput(section, queries, concurrency, base)

    log.info("efSearch sweep %s", list(ef_values))
    retriever = dense.get_dense_retriever()
    encoder = dense.get_encoder()
    import numpy as np

    query_vecs = np.vstack([encoder.encode_query(q) for q in queries])
    k = section["dense"].get("top_k", 50)
    ef_rows = sweep_ef_search(query_vecs, retriever.index, ef_values, k)

    log.info("rerank depth sweep %s", list(rerank_depths))
    fused, depth_config = _depth_candidates(section, queries, max(rerank_depths), base)
    top_k = section["rerank"].get("top_k", 10)
    depth_rows = sweep_rerank_depth(queries, fused, rerank_depths, rerank.rerank, top_k)

    examples = example_results(section, cfg.get("ui", {}).get("example_inputs", []), base=base)

    metrics_dir = get_path("results") / "metrics"
    builds = sorted(metrics_dir.glob("index_build_*.json"))
    return {
        "format_version": 2, "baseline": protocol, "baseline_overrides": copy.deepcopy(base),
        "created": datetime.now().isoformat(timespec="seconds"),
        "queries": {"source": "paper titles from papers.sqlite", "n": len(queries), "seed": seed,
                    "metadata_path": str(paper_db.resolve()), "first": queries[:5], "all": queries},
        "corpus": {"passages": retriever.index.ntotal,
                   "lookup_path": str(lookup.lookup_path().resolve()),
                   "indexes": {name: {"path": str(path.resolve()),
                                       "manifest": json.loads((path / "meta.json").read_text(encoding="utf-8"))}
                               for name, path in (("sparse", sparse.index_path()), ("dense", dense.index_path()))},
                   "index_build_report": str(builds[-1]) if builds else None,
                   "index_build": None,
                   "index_build_verified": False,
                   "index_build_note": "Latest build log is supplementary; it may describe different index paths."},
        "cold_start_s": cold_start_s,
        "startup_definition": "Previously unloaded retrieval components plus one warm-up request; "
                              "OS and model file caches may be warm. Not whole-process startup.",
        "configs": configs,
        "budget": {"limit_ms": BUDGET_MS, "config": "hybrid_rerank", "p95_ms": p95,
                   "within": p95 <= BUDGET_MS},
        "throughput": throughput,
        "ef_search_sweep": {"k": k, "rows": ef_rows},
        "rerank_depth_sweep": {"top_k": top_k, "candidate_config": depth_config, "rows": depth_rows},
        "examples": examples,
        "environment": environment(),
        "config_snapshot": copy.deepcopy(section),
    }


def format_summary(report: dict) -> str:
    """Plain-text tables of the headline numbers (the JSON report has everything)."""
    lines = [f"Retrieval latency benchmark ({report['baseline']}): {report['queries']['n']} queries, "
             f"{report['corpus']['passages']} passages, cold start {report['cold_start_s']} s", "",
             f"{'config':<24}{'total p50':>10}{'total p95':>10}  stage p50 (ms)"]
    for name, row in report["configs"].items():
        stats = row["latency_ms"]
        stages = ", ".join(f"{s} {v['p50']:g}" for s, v in stats.items() if s != "total")
        lines.append(f"{name:<24}{stats['total']['p50']:>10g}{stats['total']['p95']:>10g}  {stages}")
    b = report["budget"]
    lines += ["", f"budget: {b['config']} p95 {b['p95_ms']:g} ms vs {b['limit_ms']} ms -> "
                  f"{'within' if b['within'] else 'OVER'}", "",
              f"{'concurrency':<14}{'qps':>8}{'call p50':>10}{'call p95':>10}"]
    for row in report["throughput"]:
        lines.append(f"{row['concurrency']:<14}{row['qps']:>8g}{row['latency_ms']['p50']:>10g}"
                     f"{row['latency_ms']['p95']:>10g}")
    lines += ["", f"{'efSearch (used)':<16}{'search p50':>11}{'recall@' + str(report['ef_search_sweep']['k']):>11}"]
    for row in report["ef_search_sweep"]["rows"]:
        ef = f"{row['ef_search']} ({row.get('effective_ef_search', row['ef_search'])})"
        lines.append(f"{ef:<16}{row['search_ms']['p50']:>11g}{row['recall_at_k']:>11g}")
    lines += ["", f"{'rerank depth':<14}{'rerank p50':>11}{'top-' + str(report['rerank_depth_sweep']['top_k']) + ' agree':>13}"]
    for row in report["rerank_depth_sweep"]["rows"]:
        lines.append(f"{row['depth']:<14}{row['rerank_ms']['p50']:>11g}{row['top_k_agreement']:>13g}")
    return "\n".join(lines)
