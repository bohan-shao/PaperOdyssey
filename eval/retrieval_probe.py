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
import math
import platform
import random
import re
import statistics
import time
from pathlib import Path

from config import get_config, get_path
from kb import store
from retrieval import lookup, pipeline, sparse

from .latency import config_overrides, summarize


def source_metrics(hits: list, source: str, k: int = 10) -> dict[str, float]:
    """Known-source metrics over distinct papers represented in the first k returned passages."""
    papers = list(dict.fromkeys(h.paper_id for h in hits[:k]))
    rank = papers.index(source) + 1 if source in papers else 0
    return {f"source_recall@{k}": float(bool(rank)), f"source_mrr@{k}": 1 / rank if rank else 0.0,
            f"source_ndcg@{k}": 1 / math.log2(rank + 1) if rank else 0.0,
            f"distinct_papers@{k}": float(len(papers))}


def _queries(papers: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    stop = set(get_config()["retrieval"].get("query_processing", {}).get("stopwords", []))
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
        queries.append({"query_id": f"PROBE{i:04d}", "input_text": "Find papers on " + " ".join(terms),
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
    papers = sorted((p for p in store.iter_papers(source / names["sqlite_file"])
                     if p.get("num_chunks", 0) > 0), key=lambda p: p["paper_id"])
    if not papers:
        raise ValueError("no processed papers available for the retrieval probe")
    sample = random.Random(cfg["seed"]).sample(papers, min(paper_count, len(papers)))
    ids = {p["paper_id"] for p in sample}
    root = get_path("results") / "retrieval_probe" / f"seed{cfg['seed']}_papers{len(sample)}"
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    counts = {}
    for key in ("passages_file", "sections_file"):
        n = 0
        with (source / names[key]).open(encoding="utf-8") as src, \
                (processed / names[key]).open("w", encoding="utf-8", newline="\n") as dst:
            for line in src:
                if json.loads(line)["paper_id"] in ids:
                    dst.write(line.rstrip("\r\n") + "\n")
                    n += 1
        counts[key] = n
    queries = _queries(sample, query_count, cfg["seed"])
    (root / "queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queries), encoding="utf-8")
    corpus_hash = hashlib.sha256((processed / names["passages_file"]).read_bytes()).hexdigest()
    return {"root": str(root), "processed": str(processed), "paper_ids": sorted(ids),
            "num_papers": len(ids), "counts": counts, "queries": queries, "passages_sha256": corpus_hash}


def _clear_runtime() -> None:
    """Release experiment-owned components between rows; never call in an online server."""
    if lookup._STORE is not None:
        lookup._STORE.close()
    lookup._STORE = None
    sparse._RETRIEVER = None
    pipeline._LOADED.clear()


def run_probe(paper_count: int, query_count: int, *, hybrid: bool = False,
              reranking: bool = False) -> dict:
    """Build sample indexes and compare legacy budgets, isolated switches and combined changes."""
    if pipeline._LOADED:
        raise RuntimeError("run the probe in a fresh dedicated process")
    cfg = get_config()
    settings = cfg["eval"]["retrieval_probe"]
    sample = prepare_sample(paper_count, query_count)
    root = Path(sample["root"])
    names = cfg["kb"]["store"]
    passages = Path(sample["processed"]) / names["passages_file"]
    sections = Path(sample["processed"]) / names["sections_file"]
    report: dict = {"label_kind": "known_source_proxy", "limitations":
              "Queries derive from indexed source text; incomplete qrels, not held-out relevance labels. "
              "Model proxies do not measure BGE/Qwen quality. Latency is local warm sequential latency.",
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
    with config_overrides(cfg, overrides):
        try:
            report["build"]["lookup"] = lookup.build_lookup(passages, sections, lookup.lookup_path())
            for backend in ("rankbm25", "sqlite"):
                report["build"][backend] = sparse.build_index(passages, root / "indices" / backend, backend=backend)
            if hybrid:
                import faiss
                import torch
                from retrieval import dense

                torch.set_num_threads(settings["cpu_threads"])
                faiss.omp_set_num_threads(settings["cpu_threads"])
                cfg["retrieval"]["dense"]["faiss_dir"] = str(root / "indices/faiss")
                report["build"]["dense"] = dense.build_index(passages, dense.index_path())
            base = {"sparse.top_k": 50, "dense.top_k": 50, "fusion.top_k": 50,
                    "query_processing.enabled": False, "diversity.enabled": False,
                    "sparse.backend": "rankbm25", "sparse.field_weighting.enabled": False}
            rows = {"legacy": {}, "query_only": {"query_processing.enabled": True},
                    "depth_only": {"sparse.top_k": 150, "dense.top_k": 150},
                    "diversity_only": {"diversity.enabled": True},
                    "combined": {"query_processing.enabled": True, "diversity.enabled": True,
                                 "sparse.top_k": 150, "dense.top_k": 150}}
            rows["sqlite"] = {**rows["combined"], "sparse.backend": "sqlite"}
            rows["sqlite_fields"] = {**rows["sqlite"], "sparse.field_weighting.enabled": True}
            for name, changes in rows.items():
                print(f"probe: {name}", flush=True)
                with config_overrides(cfg["retrieval"], {**base, **changes}):
                    backend = cfg["retrieval"]["sparse"]["backend"]
                    cfg["retrieval"]["sparse"]["index_dir"] = str(root / "indices" / backend)
                    _clear_runtime()
                    cold = time.perf_counter()
                    pipeline.retrieve(sample["queries"][0]["input_text"])
                    cold_ms = (time.perf_counter() - cold) * 1000
                    entries = []
                    for q in sample["queries"]:
                        result = pipeline.retrieve(q["input_text"], top_k=10)
                        entries.append({"query_id": q["query_id"], "query_kind": q["query_kind"],
                                        **source_metrics(result.hits, q["gold_paper_ids"][0]),
                                        "paper_ids": [h.paper_id for h in result.hits],
                                        "passage_ids": [h.passage_id for h in result.hits],
                                        "latency_ms": result.latency_ms})
                    metric_names = list(source_metrics([], ""))
                    report["rows"][name] = {
                        "metrics": {m: statistics.fmean(e[m] for e in entries) for m in metric_names},
                        "by_query_kind": {kind: {m: statistics.fmean(e[m] for e in entries if e["query_kind"] == kind)
                                                 for m in metric_names} for kind in {e["query_kind"] for e in entries}},
                        "latency": {stage: summarize([e["latency_ms"][stage] for e in entries if stage in e["latency_ms"]])
                                    for stage in {s for e in entries for s in e["latency_ms"]}},
                        "cold_start_ms": cold_ms, "config": copy.deepcopy(cfg["retrieval"]), "per_query": entries}
                    # Persist after every row so interrupted experiments remain inspectable.
                    (root / "progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        finally:
            _clear_runtime()
            if hybrid:
                from retrieval import dense

                dense._RETRIEVER = None
                dense._ENCODER = None
            from retrieval import rerank

            rerank._RERANKER = None
    return report
