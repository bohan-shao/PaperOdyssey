"""Retrieval evaluation on the eval set: the EVALUATION.md section 2 matrix (Q3).

Every question of the eval set (`data/eval/eval_set.jsonl`, SCHEMA.md section 7) is retrieved as
it is, `retrieve(input_text)` (no filters, the configured top_k), under each configuration:

    configuration   sparse   dense   rerank
    bm25            on       off     off
    dense           off      on      off
    hybrid_rrf      on       on      off
    hybrid_rerank   on       on      on      (the system default)

Only these three switches differ between the rows, and every row sets all three. Every other
retrieval setting (query views, diversity, candidate budgets, models, indexes) stays as
configured, so hybrid_rerank is exactly what Paper Search runs. Rows are switched by overriding
the in-memory config (`eval.latency.config_overrides`), never config.yaml, and the config is
restored afterwards.

`run_retrieval_eval` returns one report (written to `data/results/metrics/retrieval_<tag>.json`
by `scripts/run_eval.py --stage retrieval`):

- hits: one SCHEMA.md section 8 record per question and configuration, saved as
  `retrieval_results/<configuration>_<tag>.jsonl` as soon as the configuration finishes
  (`sub_query` is the input text, `latency_ms` the retrieve() total).
- metrics: `eval.retrieval_metrics.compute_retrieval_metrics` at paper level (gold_paper_ids),
  separately for the dev and eval splits (eval is the reported number, dev is for tuning), and
  within each split per `question_type` and per `question_form`. Every question has one gold
  paper, so Hit@K = R@K and the scores are a lower bound: another relevant paper earns nothing.
  The eval set has no passage-level gold, so the passage level reports n_queries 0.
- comparisons: bm25 -> hybrid_rrf, dense -> hybrid_rrf (fusion against one channel) and
  hybrid_rrf -> hybrid_rerank (the rerank ablation, EVALUATION.md section 3): per split, the
  questions whose gold paper enters the top 10 papers (failed -> succeeded, `fixed`) or leaves it
  (`broken`), with examples, and how often the gold paper moved up or down (a miss is below
  every position).
- per_query: each question's split, type and first gold-paper position in every configuration.
- per configuration: its effective retrieval config, the time of its first (warm-up) request,
  which loads the components no earlier configuration loaded, its wall time, the per-stage
  latency observed during the run (incidental: controlled latency numbers come from
  eval/latency.py), the mean number of returned passages and of distinct papers among them.
- the eval set's path and SHA-256, the index manifests, a config snapshot (seed, paths,
  retrieval) and dependency versions (EVALUATION.md section 9).

Run in a fresh dedicated process (eval/runtime.py). Retrieval failures are not caught: a broken
configuration stops the run instead of being scored as misses.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from config import PROJECT_ROOT, get_config
from retrieval import dense, lookup, sparse
from retrieval.pipeline import retrieve
from retrieval.types import RetrieveResult
from .latency import config_overrides, environment, summarize
from .retrieval_metrics import compute_retrieval_metrics, ranking
from .runtime import clear_runtime, require_fresh_runtime

log = logging.getLogger(__name__)

CONFIGS: dict[str, dict[str, bool]] = {
    "bm25": {"sparse.enabled": True, "dense.enabled": False, "rerank.enabled": False},
    "dense": {"sparse.enabled": False, "dense.enabled": True, "rerank.enabled": False},
    "hybrid_rrf": {"sparse.enabled": True, "dense.enabled": True, "rerank.enabled": False},
    "hybrid_rerank": {"sparse.enabled": True, "dense.enabled": True, "rerank.enabled": True},
}
COMPARISONS = (("bm25", "hybrid_rrf"), ("dense", "hybrid_rrf"), ("hybrid_rrf", "hybrid_rerank"))
SPLITS = ("dev", "eval")
K_LIST = [1, 3, 5, 10]
COMPARE_K = 10  # fixed / broken = the gold paper enters / leaves the top COMPARE_K papers
EXAMPLES = 5  # examples per direction and split in a comparison
_FIELDS = {"query_id": str, "input_text": str, "gold_paper_ids": list, "split": str}
_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


# ---------------------------------------------------------------- inputs and outputs


def load_eval_set(path: str | Path) -> list[dict]:
    """The eval-set records (SCHEMA.md section 7) in file order; blank lines are skipped.

    Raises:
        ValueError: a line that is not a JSON object, a missing or mistyped field, an empty input
            text or gold list, an unknown split or a repeated query_id (the message names the line).
    """
    questions: list[dict] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: not JSON ({exc})") from exc
            if not isinstance(q, dict):
                raise ValueError(f"{path}:{n}: not a JSON object")
            for key, kind in _FIELDS.items():
                if not isinstance(q.get(key), kind):
                    raise ValueError(f"{path}:{n}: {key} is missing or not a {kind.__name__}")
            # Only a question marked unanswerable (the challenge set's out_of_kb kind, D56) may have no gold.
            if not q["input_text"].strip() or (not q["gold_paper_ids"] and q.get("answerable", True) is not False):
                raise ValueError(f"{path}:{n}: empty input_text or gold_paper_ids")
            if q["split"] not in SPLITS:
                raise ValueError(f"{path}:{n}: split must be one of {SPLITS}, not {q['split']!r}")
            if q["query_id"] in seen:
                raise ValueError(f"{path}:{n}: repeated query_id {q['query_id']!r}")
            seen.add(q["query_id"])
            questions.append(q)
    return questions


def to_record(question: Mapping, stage: str, top_k: int, result: RetrieveResult) -> dict:
    """One SCHEMA.md section 8 record: a question's hits under one configuration."""
    return {"query_id": question["query_id"], "sub_query": question["input_text"], "stage": stage,
            "top_k": top_k,
            "hits": [{"passage_id": h.passage_id, "score": float(h.score), "rank": int(h.rank),
                      "paper_id": h.paper_id} for h in result.hits],
            "latency_ms": int(result.latency_ms.get("total_ms", 0))}


def write_records(path: Path, records: Sequence[Mapping]) -> None:
    """Write records as JSONL; the file is replaced only once it is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    os.replace(tmp, path)


def _relative(path: Path) -> str:
    """The path relative to the project root when it is inside it (links not followed)."""
    absolute = Path(os.path.abspath(path))
    return str(absolute.relative_to(PROJECT_ROOT)) if absolute.is_relative_to(PROJECT_ROOT) else str(absolute)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_manifests() -> dict:
    """Paths of the configured lookup store and indexes, with the indexes' meta.json (None if absent)."""
    out: dict = {"lookup": {"path": _relative(lookup.lookup_path())}}
    for name, path in (("sparse", sparse.index_path()), ("dense", dense.index_path())):
        try:
            manifest = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = None
        out[name] = {"path": _relative(path), "manifest": manifest}
    return out


# ---------------------------------------------------------------- retrieval and scoring


def run_config(section: dict, questions: Sequence[Mapping], name: str,
               progress_every: int = 100) -> tuple[list[dict], dict]:
    """retrieve() every question under the configuration `name` of CONFIGS.

    Returns:
        (the SCHEMA.md section 8 records in question order, the configuration's run statistics).
    """
    overrides = CONFIGS[name]
    stages: dict[str, list[float]] = {}
    records: list[dict] = []
    found = 0
    with config_overrides(section, overrides):
        effective = copy.deepcopy(section)
        top_k = section["rerank"].get("top_k", 10)
        start = time.perf_counter()
        retrieve("warm up")  # loads this configuration's indexes and models; not a question
        load_s = time.perf_counter() - start
        start = time.perf_counter()
        for i, question in enumerate(questions, start=1):
            result = retrieve(question["input_text"])
            for key, ms in result.latency_ms.items():
                stages.setdefault(key.removesuffix("_ms"), []).append(ms)
            record = to_record(question, name, top_k, result)
            records.append(record)
            found += any(h["paper_id"] in question["gold_paper_ids"] for h in record["hits"])
            if i % max(progress_every, 1) == 0 or i == len(questions):
                log.info("%s: %d/%d questions, %.0f s, gold paper returned for %.1f%%",
                         name, i, len(questions), time.perf_counter() - start, 100 * found / i)
        wall_s = time.perf_counter() - start
    n = max(len(records), 1)
    return records, {
        "overrides": dict(overrides), "effective_config": effective, "top_k": top_k,
        "load_s": round(load_s, 2), "wall_s": round(wall_s, 2),
        "latency_ms": {stage: summarize(values) for stage, values in stages.items()},
        "mean_hits": round(sum(len(r["hits"]) for r in records) / n, 2),
        "mean_distinct_papers": round(sum(len(ranking(r["hits"])) for r in records) / n, 2),
        "questions_without_hits": sum(not r["hits"] for r in records),
    }


def score_config(records: Sequence[Mapping], questions: Sequence[Mapping]) -> tuple[dict, dict[str, int | None]]:
    """Paper-level metrics of one configuration's hits, per split and per question type / form.

    Returns:
        ({"metrics": {split: compute_retrieval_metrics output without per_query},
          "by_question_type": {split: {type: paper-level metrics}}, "by_question_form": {...}},
         query_id -> first gold-paper position, None when the gold paper was not returned).
    """
    hits = {r["query_id"]: r["hits"] for r in records}
    scores: dict = {"metrics": {}, "by_question_type": {}, "by_question_form": {}}
    first_gold: dict[str, int | None] = {}
    for split in SPLITS:
        subset = [q for q in questions if q["split"] == split]
        metrics = compute_retrieval_metrics(hits, {q["query_id"]: q for q in subset}, K_LIST)
        first_gold.update((qid, v["first_gold_rank"]) for qid, v in metrics.pop("per_query").items())
        scores["metrics"][split] = metrics
        for field in ("question_type", "question_form"):
            groups: dict[str, dict[str, Mapping]] = {}
            for q in subset:
                groups.setdefault(str(q.get(field) or "unknown"), {})[q["query_id"]] = q
            scores[f"by_{field}"][split] = {
                group: compute_retrieval_metrics(hits, gold, K_LIST)["per_granularity"]["paper"]
                for group, gold in sorted(groups.items())}
    return scores, first_gold


def _position(rank: int | None) -> float:
    return math.inf if rank is None else rank


def compare(ranks: Mapping[str, Mapping[str, int | None]], questions: Sequence[Mapping],
            before: str, after: str, examples: int = EXAMPLES) -> dict:
    """Per split, the questions that `after` fixes or breaks relative to `before`.

    Args:
        ranks: configuration -> query_id -> first gold-paper position (None = not returned).

    `fixed` = the gold paper is in the top COMPARE_K papers of `after` but not of `before`
    (failed -> succeeded), `broken` = the reverse; `moved_up` / `moved_down` count questions whose
    gold paper is ranked better / worse (a miss is below every position). Examples keep the
    eval-set order.
    """
    out: dict = {"before": before, "after": after, "k": COMPARE_K, "splits": {}}
    for split in SPLITS:
        fixed: list[dict] = []
        broken: list[dict] = []
        n = up = down = 0
        for q in questions:
            if q["split"] != split:
                continue
            b, a = ranks[before][q["query_id"]], ranks[after][q["query_id"]]
            n += 1
            up += _position(a) < _position(b)
            down += _position(a) > _position(b)
            if (_position(b) <= COMPARE_K) != (_position(a) <= COMPARE_K):
                example = {"query_id": q["query_id"], "input_text": q["input_text"],
                           "rank_before": b, "rank_after": a}
                (fixed if _position(a) <= COMPARE_K else broken).append(example)
        out["splits"][split] = {"n_queries": n, "fixed": len(fixed), "broken": len(broken),
                                "moved_up": up, "moved_down": down,
                                "fixed_examples": fixed[:examples], "broken_examples": broken[:examples]}
    return out


# ---------------------------------------------------------------- the whole run


def run_retrieval_eval(eval_set_path: str | Path, results_dir: str | Path, tag: str,
                       configs: Sequence[str] = tuple(CONFIGS), limit: int | None = None,
                       progress_every: int = 100) -> dict:
    """Retrieve and score the eval set under each configuration; return the report.

    Args:
        eval_set_path: the eval set (SCHEMA.md section 7).
        results_dir: where the hits files `<configuration>_<tag>.jsonl` go (retrieval_results/).
        tag: the file-name suffix of this run's outputs (scripts/run_eval.py: the date).
        configs: names from CONFIGS, run in this order.
        limit: only the first `limit` questions (a quick check, not a reportable number).

    Raises:
        ValueError: unknown or repeated configuration names, a bad tag or limit, a broken eval set.
        RuntimeError: retrieval components were already loaded in this process, or a retrieval
            stage failed (retrieval.pipeline.retrieve).
    """
    if not configs or len(set(configs)) != len(configs) or any(c not in CONFIGS for c in configs):
        raise ValueError(f"configs must be distinct names from {list(CONFIGS)}, not {list(configs)}")
    if not _TAG.fullmatch(tag):
        raise ValueError(f"tag must be letters, digits, '.', '_' or '-', not {tag!r}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    eval_set_path, results_dir = Path(eval_set_path), Path(results_dir)
    questions = load_eval_set(eval_set_path)[:limit]
    if not questions:
        raise ValueError(f"no questions in {eval_set_path}")
    require_fresh_runtime()
    cfg = get_config()
    section = cfg["retrieval"]
    requested = copy.deepcopy(section)
    rows: dict[str, dict] = {}
    ranks: dict[str, dict[str, int | None]] = {}
    try:
        for name in configs:
            log.info("configuration %s: %d questions", name, len(questions))
            records, stats = run_config(section, questions, name, progress_every)
            path = results_dir / f"{name}_{tag}.jsonl"
            write_records(path, records)
            scores, ranks[name] = score_config(records, questions)
            rows[name] = {**stats, "results_file": _relative(path), **scores}
    finally:
        clear_runtime(release_models=True)
    return {
        "stage": "retrieval", "format_version": 1, "tag": tag,
        "created": datetime.now().isoformat(timespec="seconds"),
        "eval_set": {"path": _relative(eval_set_path), "sha256": _sha256(eval_set_path),
                     "n_questions": len(questions), "limit": limit,
                     "n_by_split": {s: sum(q["split"] == s for q in questions) for s in SPLITS}},
        "protocol": {
            "query": "the question's input_text as it is, no filters, the configured top_k passages",
            "level": "paper: gold_paper_ids; a paper takes its best passage's position",
            "metric_version": "paper_ranking_v2", "k_list": K_LIST, "mrr_ndcg_k": 10,
            "splits": "eval is the reported number, dev is for tuning (EVALUATION.md section 1)",
            "switches": "configurations differ only in sparse/dense/rerank.enabled; everything else as configured",
        },
        "configs": rows,
        "comparisons": [compare(ranks, questions, b, a) for b, a in COMPARISONS if b in ranks and a in ranks],
        "per_query": {q["query_id"]: {"split": q["split"], "question_type": q.get("question_type"),
                                      "first_gold_rank": {name: ranks[name][q["query_id"]] for name in ranks}}
                      for q in questions},
        "corpus": index_manifests(),
        "environment": environment(),
        "config_snapshot": {"seed": cfg.get("seed"), "paths": copy.deepcopy(cfg["paths"]), "retrieval": requested},
    }


def format_summary(report: dict) -> str:
    """Plain-text tables of the headline numbers per split (the JSON report has everything)."""
    names = ["Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10", "NDCG@10"]
    info = report["eval_set"]
    split_counts = ", ".join(f"{s} {n}" for s, n in info["n_by_split"].items())
    lines = [f"Retrieval evaluation: {info['n_questions']} questions ({split_counts}), paper level"
             + (f", first {info['limit']} only (not a reportable number)" if info.get("limit") else "")]
    for split in SPLITS:
        lines += ["", f"split {split} (n={info['n_by_split'][split]})",
                  f"{'configuration':<16}" + "".join(f"{n:>9}" for n in names) + f"{'p50 ms':>9}"]
        for name, row in report["configs"].items():
            metrics = row["metrics"][split]
            cells = "".join(f"{metrics[n]:>9.3f}" if metrics[n] is not None else f"{'-':>9}" for n in names)
            lines.append(f"{name:<16}{cells}{row['latency_ms'].get('total', {}).get('p50', '-'):>9}")
    if report["comparisons"]:
        lines += ["", f"top-{COMPARE_K} changes, eval split (fixed = failed -> succeeded):"]
        for c in report["comparisons"]:
            s = c["splits"]["eval"]
            lines.append(f"{c['before']} -> {c['after']}: fixed {s['fixed']}, broken {s['broken']}; "
                         f"gold paper up {s['moved_up']}, down {s['moved_down']} (of {s['n_queries']})")
    return "\n".join(lines)
