"""Reproducible known-source retrieval probe, not a human-labeled relevance benchmark.

Sample complete papers, derive incomplete-title and abstract-term queries, and compare one
switch at a time through pipeline.retrieve(). Every query has one known source, not exhaustive
qrels. Report source recall/MRR and paper diversity separately; do not claim general relevance
gains from these proxy labels. Run in a dedicated process: model/index singletons are reset
between configurations. Inputs/outputs are resolved exclusively from the active config.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
import random
import re
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import get_config, get_path
from kb import store
from retrieval import lookup, pipeline, sparse

from .latency import BASE, config_overrides, summarize
from . import retrieval_metrics as metrics
from .runtime import clear_runtime as _clear_runtime, require_fresh_runtime

# Fixed generation vocabulary, independent of the retrieval settings being evaluated.
_QUERY_STOPWORDS = frozenset("a an and are as at be by for from in is it of on or that the this to was were with".split())


def probe_mode(hybrid: bool, reranking: bool) -> str:
    """Give every channel/reranking combination a distinct artifact label."""
    return ("hybrid" if hybrid else "sparse") + ("_rerank" if reranking else "")


def report_path(report: dict) -> Path:
    """Use a unique run ID so reruns never overwrite earlier reports."""
    return get_path("results") / "metrics" / f"ablation_retrieval_probe_{report['mode']}_{report['sample']['run_id']}.json"


def source_metrics(hits: list, source: str, k: int = 10) -> dict[str, float]:
    """Canonical paper-level metrics; diversity separately counts the first k passages."""
    records = [{"paper_id": h.paper_id, "passage_id": h.passage_id, "rank": h.rank} for h in hits]
    ordered = sorted(records, key=lambda h: h["rank"])
    return {f"source_recall@{k}": metrics.recall_at_k(records, {source}, k),
            f"source_mrr@{k}": metrics.mrr_at_k(records, {source}, k),
            f"source_ndcg@{k}": metrics.ndcg_at_k(records, {source}, k),
            f"distinct_papers_in_top_{k}_passages": float(len({h["paper_id"] for h in ordered[:k]}))}


def _queries(papers: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    stop = _QUERY_STOPWORDS
    chosen = rng.sample(papers, min(count, len(papers)))
    queries = []
    for i, paper in enumerate(chosen):
        title_words = re.findall(r"[a-zA-Z0-9]+", paper["title"])
        title_terms = [w for w in title_words if w.lower() not in stop]
        if i % 2:
            excluded = stop | {w.lower() for w in title_words}
            words = re.findall(r"[a-zA-Z0-9]+", paper.get("abstract") or "")
            terms = list(dict.fromkeys(w.lower() for w in words if w.lower() not in excluded and len(w) > 2))
            kind = "abstract_terms"
            terms = terms[:12]
        else:
            terms, kind = title_terms[::2], "partial_title"
        terms = terms or title_terms or title_words
        queries.append({"query_id": f"PROBE{i:04d}", "input_text": " ".join(terms),
                        "template_text": "Find papers on " + " ".join(terms),
                        "gold_paper_ids": [paper["paper_id"]], "gold_passage_ids": [],
                        "split": "dev", "label_kind": "known_source_proxy", "query_kind": kind})
    return queries


def prepare_sample(paper_count: int, query_count: int) -> dict:
    """Copy a seeded sample of whole papers into results, preserving original KB records."""
    if paper_count <= 0 or query_count <= 0:
        raise ValueError("paper_count and query_count must be positive")
    cfg = get_config()
    source = get_path("processed")
    names = cfg["kb"]["store"]
    for key in ("passages_file", "sections_file"):
        name = Path(names[key])
        if name.anchor or ".." in name.parts:
            raise ValueError(f"probe requires a relative KB {key} within paths.processed")
    papers = sorted((p for p in store.iter_papers(source / names["sqlite_file"])
                     if p.get("num_chunks", 0) > 0), key=lambda p: p["paper_id"])
    if not papers:
        raise ValueError("no processed papers available for the retrieval probe")
    sample = random.Random(cfg["seed"]).sample(papers, min(paper_count, len(papers)))
    ids = {p["paper_id"] for p in sample}
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    root = get_path("results") / "retrieval_probe" / f"seed{cfg['seed']}_papers{len(sample)}_{run_id}"
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    counts = {}
    indexed_papers: set[str] = set()
    for key in ("passages_file", "sections_file"):
        n = 0
        (processed / names[key]).parent.mkdir(parents=True, exist_ok=True)
        with (source / names[key]).open(encoding="utf-8") as src, \
                (processed / names[key]).open("w", encoding="utf-8", newline="\n") as dst:
            for line in src:
                paper_id = json.loads(line)["paper_id"]
                if paper_id in ids:
                    dst.write(line.rstrip("\r\n") + "\n")
                    n += 1
                    if key == "passages_file":
                        indexed_papers.add(paper_id)
        counts[key] = n
    if indexed_papers != ids:
        raise ValueError("sampled paper metadata and passage file disagree; rebuild or select the matching KB")
    queries = _queries(sample, query_count, cfg["seed"])
    (root / "queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queries), encoding="utf-8")
    with (processed / names["passages_file"]).open("rb") as stream:
        corpus_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"root": str(root), "run_id": run_id, "processed": str(processed), "paper_ids": sorted(ids),
            "num_papers": len(ids), "counts": counts, "queries": queries, "passages_sha256": corpus_hash}


def ablation_rows(hybrid: bool) -> tuple[dict[str, dict], dict[str, str]]:
    """Return effective experiments and explicit reasons for inapplicable rows."""
    rows = {"legacy": {}, "query_only": {"query_processing.enabled": True}}
    skipped = {}
    deeper = {"sparse.top_k": 150, "dense.top_k": 150}
    if hybrid:
        rows["depth_only"] = deeper
    else:
        skipped["depth_only"] = ("Sparse ranking has an unchanged top-50 prefix at a fixed candidate budget; "
                                 "compare depth_with_diversity against diversity_only instead.")
    rows["diversity_only"] = {"diversity.enabled": True}
    rows["depth_with_diversity"] = {**deeper, "diversity.enabled": True}
    rows["combined"] = {**rows["depth_with_diversity"], "query_processing.enabled": True}
    rows["sqlite"] = {**rows["combined"], "sparse.backend": "sqlite"}
    rows["sqlite_fields"] = {**rows["sqlite"], "sparse.field_weighting.enabled": True}
    return rows, skipped


def run_probe(paper_count: int, query_count: int, *, hybrid: bool = False,
              reranking: bool = False) -> dict:
    """Build sample indexes and compare legacy budgets, isolated switches and combined changes."""
    from retrieval import dense

    require_fresh_runtime()
    cfg = get_config()
    settings = cfg["eval"]["retrieval_probe"]
    sample = prepare_sample(paper_count, query_count)
    root = Path(sample["root"])
    names = cfg["kb"]["store"]
    passages = Path(sample["processed"]) / names["passages_file"]
    sections = Path(sample["processed"]) / names["sections_file"]
    report: dict = {"format_version": 2, "metric_version": "paper_ranking_v2",
              "label_kind": "known_source_proxy", "limitations":
              "Queries derive from indexed source text; incomplete qrels, not held-out relevance labels. "
               "Model proxies do not measure BGE/Qwen quality. Latency is local warm sequential latency.",
              "metric_scope": "Raw source terms; paired request-template results are reported separately.",
              "ranking_scope": "Sort all returned hits by rank, deduplicate papers, then apply metric k; "
                               "retrieval still returns at most 10 passages. Diversity uses a passage cutoff.",
              "setup_timing": "row_setup_ms includes index loading and one warm-up query. Models and OS caches "
                              "may be reused across rows; this is not comparable process cold-start latency.",
              "mode": probe_mode(hybrid, reranking),
              "seed": cfg["seed"], "sample": sample, "hybrid": hybrid, "reranking": reranking,
              "probe_settings": copy.deepcopy(settings),
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "processor": platform.processor(), "device": "cpu"},
              "build": {}, "rows": {}, "versions": {}}
    for package in ("numpy", "faiss-cpu", "rank-bm25", "sentence-transformers", "torch"):
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    overrides = {"paths.indices": str(root / "indices"), "paths.processed": sample["processed"],
                 "retrieval.lookup.db_file": "lookup.sqlite", "retrieval.parallel": False,
                 "retrieval.sparse.enabled": True, "retrieval.dense.enabled": hybrid,
                 "retrieval.rerank.enabled": reranking, "retrieval.parent_doc.enabled": False,
                 "retrieval.dense.model_name": settings["dense_model"],
                 "retrieval.dense.max_length": settings["dense_max_length"],
                 "retrieval.dense.batch_size": settings["batch_size"],
                 "retrieval.dense.device": "cpu", "retrieval.dense.query_format": "plain",
                 "retrieval.dense.query_instruction": None,
                 "retrieval.rerank.model_name": settings["rerank_model"],
                 "retrieval.rerank.device": "cpu",
                 "retrieval.rerank.batch_size": settings["batch_size"],
                 "retrieval.rerank.max_length": settings["rerank_max_length"]}
    old_torch_threads = old_faiss_threads = None
    with config_overrides(cfg, overrides):
        try:
            report["build"]["lookup"] = lookup.build_lookup(passages, sections, lookup.lookup_path())
            for backend in ("rankbm25", "sqlite"):
                report["build"][backend] = sparse.build_index(passages, root / "indices" / backend, backend=backend)
            if hybrid or reranking:
                import torch

                old_torch_threads = torch.get_num_threads()
                torch.set_num_threads(settings["cpu_threads"])
            if hybrid:
                import faiss

                old_faiss_threads = faiss.omp_get_max_threads()
                faiss.omp_set_num_threads(settings["cpu_threads"])
                cfg["retrieval"]["dense"]["faiss_dir"] = str(root / "indices/faiss")
                report["build"]["dense"] = dense.build_index(passages, dense.index_path())
            base = {**BASE, "dense.enabled": hybrid, "rerank.enabled": reranking,
                    "parallel": False, "sparse.backend": "rankbm25",
                    "rerank.batch_size": settings["batch_size"], "rerank.max_length": settings["rerank_max_length"]}
            rows, report["skipped_rows"] = ablation_rows(hybrid)
            report["comparisons"] = {"query_only": "legacy", "diversity_only": "legacy",
                                     "depth_with_diversity": "diversity_only", "combined": "depth_with_diversity",
                                     "sqlite": "combined", "sqlite_fields": "sqlite"}
            if hybrid:
                report["comparisons"]["depth_only"] = "legacy"
            for name, changes in rows.items():
                print(f"probe: {name}", flush=True)
                with config_overrides(cfg["retrieval"], {**base, **changes}):
                    backend = cfg["retrieval"]["sparse"]["backend"]
                    cfg["retrieval"]["sparse"]["index_dir"] = str(root / "indices" / backend)
                    _clear_runtime()
                    setup = time.perf_counter()
                    pipeline.retrieve(sample["queries"][0]["input_text"])
                    setup_ms = (time.perf_counter() - setup) * 1000
                    entries = []
                    for q in sample["queries"]:
                        for style, text in (("raw_terms", q["input_text"]), ("request_template", q["template_text"])):
                            result = pipeline.retrieve(text, top_k=10)
                            entries.append({"query_id": q["query_id"], "query_kind": q["query_kind"],
                                            "query_style": style,
                                            **source_metrics(result.hits, q["gold_paper_ids"][0]),
                                            "paper_ids": [h.paper_id for h in result.hits],
                                            "passage_ids": [h.passage_id for h in result.hits],
                                            "total_candidates": result.total_candidates,
                                            "latency_ms": result.latency_ms})
                    metric_names = list(source_metrics([], ""))
                    raw = [e for e in entries if e["query_style"] == "raw_terms"]
                    report["rows"][name] = {
                        "metrics": {m: statistics.fmean(e[m] for e in raw) for m in metric_names},
                        "by_query_style": {style: {m: statistics.fmean(e[m] for e in entries if e["query_style"] == style)
                                                   for m in metric_names} for style in ("raw_terms", "request_template")},
                        "by_query_kind": {kind: {m: statistics.fmean(e[m] for e in raw if e["query_kind"] == kind)
                                                 for m in metric_names} for kind in {e["query_kind"] for e in raw}},
                        "latency": {stage: summarize([e["latency_ms"][stage] for e in raw if stage in e["latency_ms"]])
                                    for stage in {s for e in raw for s in e["latency_ms"]}},
                        "row_setup_ms": setup_ms, "config": copy.deepcopy(cfg["retrieval"]), "per_query": entries}
                    # Persist after every row so interrupted experiments remain inspectable.
                    (root / "progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        finally:
            _clear_runtime(release_models=True)
            if old_torch_threads is not None:
                torch.set_num_threads(old_torch_threads)
            if old_faiss_threads is not None:
                faiss.omp_set_num_threads(old_faiss_threads)
    return report
