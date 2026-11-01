"""Controlled one-/two-factor ablation matrices over retrieval or Paper Search.

Selected factors are forced off in the baseline; all other configured settings stay fixed.
Cells run sequentially with isolated artifacts. Only jointly completed queries are compared.
No generation evaluation or model-provider determinism is implied by these experiments.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import random
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from config import get_config
from llm import client as llm_client
from .ablation_comparison import compare_rows, interaction
from .ablation_checks import validate_review_attempts
from .ablation_retrieval import _load_attempts, run_retrieval_cell
from .latency import config_overrides, environment
from .retrieval_eval import index_manifests, load_eval_set
from .review_eval import (_code_hash, _digest, _input_file_metadata, _lock, _safe_config,
                          _write, run_review_eval)
from .runtime import require_fresh_runtime

log = logging.getLogger(__name__)
FACTORS = {
    "rerank": {"key": "retrieval.rerank.enabled", "stages": ["retrieval", "review"]},
    "query_views": {"key": "retrieval.query_processing.enabled", "stages": ["retrieval", "review"]},
    "diversity": {"key": "retrieval.diversity.enabled", "stages": ["retrieval", "review"]},
    "field_weighting": {"key": "retrieval.sparse.field_weighting.enabled", "stages": ["retrieval", "review"]},
    "parent_doc": {"key": "retrieval.parent_doc.enabled", "stages": ["review"]},
    "hyde": {"key": "orchestration.hyde.enabled", "stages": ["review"]},
    "loop": {"key": "orchestration.loop.enabled", "stages": ["review"]},
    "agentic": {"key": "orchestration.agentic.enabled", "stages": ["review"]},  # D55: off = plain RAG
    "re_review": {"key": "orchestration.loop.re_review", "stages": ["review"]},  # D53 evidence merging
}
_UNSUPPORTED = {
    "semantic_chunking": "semantic chunking requires paired prebuilt corpus/lookup/sparse/dense indexes",
    "chunking": "semantic chunking requires paired prebuilt corpus/lookup/sparse/dense indexes",
    "filters": "filter ablation requires explicit constrained query scenarios; this runner passes no fixed filters",
    "jev": "Jev review usage/evaluation is not supported by the Paper Search evaluator",
}


def _value(cfg: dict, dotted: str):
    value = cfg
    for key in dotted.split("."):
        value = value[key]
    return value


def _selection(path: Path, split: str, limit: int | None) -> list[dict]:
    if split not in ("all", "dev", "eval"):
        raise ValueError("split must be all, dev or eval")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    questions = load_eval_set(path)
    for q in questions:
        gold = q["gold_paper_ids"]
        if (not q["query_id"].strip() or any(not isinstance(p, str) or not p.strip() for p in gold)
                or len(gold) != len(set(gold))):
            raise ValueError("invalid query ID or gold paper IDs")
    selected = [q for q in questions if split == "all" or q["split"] == split][:limit]
    if not selected:
        raise ValueError("no questions selected")
    return selected


def _plan(cfg: dict, factors: Sequence[str], stage: str, tag: str, questions: list[dict]) -> dict:
    if isinstance(factors, str) or not 1 <= len(factors) <= 2 or len(set(factors)) != len(factors):
        raise ValueError("choose one or two distinct ablation factors")
    for name in factors:
        if name not in FACTORS:
            raise ValueError(_UNSUPPORTED.get(name, f"unknown ablation factor {name!r}; choose from {list(FACTORS)}"))
    if stage not in ("retrieval", "review", "all"):
        raise ValueError("stage must be retrieval, review or all")
    if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise ValueError("tag must be a safe filename component")
    stages = ["retrieval", "review"] if stage == "all" else [stage]
    for name in factors:
        if any(s not in FACTORS[name]["stages"] for s in stages):
            raise ValueError(f"{name} supports only {FACTORS[name]['stages']}; specify a compatible --stage")
        try:
            value = _value(cfg, FACTORS[name]["key"])
        except KeyError as exc:
            raise ValueError(f"config is missing factor switch {FACTORS[name]['key']}") from exc
        if type(value) is not bool:
            raise ValueError(f"{FACTORS[name]['key']} must be a boolean")
    retrieval = cfg["retrieval"]
    if not (retrieval["sparse"].get("enabled", True) or retrieval["dense"].get("enabled", True)):
        raise ValueError("at least one retrieval channel must be enabled")
    if "review" in stages and cfg["orchestration"]["review"].get("backend", "llm") != "llm":
        raise ValueError("review ablations currently require review.backend=llm; Jev usage is unsupported")
    if "field_weighting" in factors:
        sparse = retrieval["sparse"]
        weights = sparse.get("field_weighting", {}).get("weights")
        if not sparse.get("enabled", True) or sparse.get("backend") != "sqlite":
            raise ValueError("field_weighting requires an enabled, prebuilt SQLite sparse backend")
        if (not isinstance(weights, list) or len(weights) != 3
                or any(isinstance(w, bool) or not isinstance(w, (float, int))
                       or not math.isfinite(w) or w <= 0 for w in weights)
                or len(set(weights)) == 1):
            raise ValueError("field_weighting requires three positive, nonuniform field weights")
    if "loop" in factors:
        rounds = cfg["orchestration"]["loop"].get("max_rounds", 3)
        if type(rounds) is not int or rounds < 2:
            raise ValueError("loop ablation requires max_rounds >= 2")
        if cfg["orchestration"].get("rewrite", {}).get("max_queries_per_round", 2) < 1:
            raise ValueError("loop ablation requires at least one rewrite query per round")
    if "agentic" in factors and {"hyde", "loop", "re_review"} & set(factors):
        raise ValueError("hyde, loop and re_review act only inside the agentic search; ablate them apart from agentic")
    if "re_review" in factors and "loop" not in factors and not cfg["orchestration"]["loop"].get("enabled", True):
        raise ValueError("re_review only acts in rounds 2+; enable the loop or ablate it with the loop factor")
    seed = cfg.get("seed")
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    enabled = {"baseline": [], "a": [factors[0]]}
    if len(factors) == 2:
        enabled.update(b=[factors[1]], ab=list(factors))
    rows = {}
    for name, active in enabled.items():
        overrides = {FACTORS[f]["key"]: f in active for f in factors}
        effective = copy.deepcopy(cfg)
        with config_overrides(effective, overrides):
            rows[name] = {"enabled_factors": active, "overrides": overrides,
                          "effective_config": _safe_config(effective)}
    edges = [("baseline", "a")]
    if len(factors) == 2:
        edges += [("baseline", "b"), ("a", "ab"), ("b", "ab")]
    return {
        "stage": "ablation", "format_version": 1, "status": "planned", "tag": tag,
        "factors": list(factors), "factor_definitions": {f: FACTORS[f] for f in factors},
        "stages": stages, "rows": rows, "edges": edges, "selected_queries": len(questions),
        "selected_query_ids": [q["query_id"] for q in questions], "seed": seed,
        "query_executions": len(questions) * len(rows) * len(stages),
        "review_query_executions": len(questions) * len(rows) if "review" in stages else 0,
        "protocol": {
            "controls": "Selected factors off in baseline; only named switches change. Other settings are fixed.",
            "comparison": "Hamming-distance-one edges; each metric uses the same jointly completed queries.",
            "selection": "File order after split filtering, then limit. Prefix limits are smoke checks.",
            "labels": "Positive source papers only; other candidates are unjudged, not negatives.",
            "randomness": "The configured seed is fixed and local RNGs are reset per cell. Remote LLM sampling "
                          "is not seeded; paired differences are observations, not deterministic causal effects.",
            "timing": "No hidden query warm-up. First new query per cell/process includes loading; "
                      "timings are descriptive and depend on OS caches, resume and provider load.",
            "usage": "Observed responses, including failed-attempt retries, not a billing total or token cap.",
            "scope": "Retrieval and Paper Search only; these are not QA generation scores or a complete Q5 evaluation.",
            "activation": "Enabled does not imply exercised on every query: HyDE can fall back, parent text "
                          "can remain unchanged and the loop can stop in one round. Only actual round counts "
                          "are tracked here; configured feature differences are not proof of successful activation.",
        },
    }


def _seed_local(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _review_observations(directory: Path, questions: list[dict]) -> tuple[dict, dict]:
    """Read complete traces and all observed attempt usage, including interrupted attempts."""
    validate_review_attempts(directory, questions)
    selected = {q["query_id"]: q for q in questions}
    rows = {}
    usage = {k: 0 for k in ("calls", "prompt_tokens", "completion_tokens")}
    for path in sorted((directory / "attempts").glob("*.json")):
        attempt = json.loads(path.read_text(encoding="utf-8"))
        for key in usage:
            usage[key] += attempt.get("usage", {}).get(key, 0)
        if attempt["status"] != "complete":
            continue
        qid = attempt["question"]["query_id"]
        q, result = selected[qid], attempt["result"]
        gold = set(q["gold_paper_ids"])
        ranked = [r["paper_id"] for r in result["papers"]]
        accepted = gold & set(ranked)
        cov = result["coverage_report"]
        rows[qid] = {
            "first_gold_rank": next((i for i, p in enumerate(ranked, 1) if p in gold), None),
            "source_success": bool(accepted), "source_recall": len(accepted) / len(gold) if gold else None,
            "wall_ms": attempt["wall_ms"], "cold_start": attempt.get("cold_start", False),
            "coverage_met": cov["covered_aspects"] == cov["total_aspects"] if cov["total_aspects"] else None,
            "rounds": cov["rounds"],
        }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return rows, usage


def _cell_summary(stage: str, report: dict, directory: Path, questions: list[dict]) -> dict:
    if stage == "review":
        observations, usage = _review_observations(directory, questions)
    else:
        observations = {qid: {
            "first_gold_rank": q["first_gold_rank"],
            "source_success": q["first_gold_rank"] is not None and q["first_gold_rank"] <= 10,
            "source_recall": None, "wall_ms": q["wall_ms"], "cold_start": q["cold_start"],
            "coverage_met": None, "rounds": None,
        } for qid, q in report.get("per_query", {}).items()}
        usage = report.get("usage", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    return {"status": report.get("status", "partial"), "completed_queries": len(observations),
            "report_file": str(directory / "report.json"), "usage": usage,
            "per_query": observations, "error": report.get("error")}


def _preflight_cells(directory: Path, plan: dict, questions: list[dict], cfg: dict,
                     fingerprint: str, identity: dict, content: str) -> None:
    """Validate every saved cell before allowing any new query in the matrix."""
    for row, spec in plan["rows"].items():
        for stage in plan["stages"]:
            cell_dir = directory / "rows" / row / stage
            path = cell_dir / "manifest.json"
            if not path.is_file():
                if any(cell_dir.rglob("*.json")):
                    raise ValueError(f"{row}/{stage} has checkpoints without a manifest")
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if stage == "retrieval":
                expected = _digest({"matrix": fingerprint, "row": row, "stage": stage})
                if (manifest.get("stage") != stage or manifest.get("format_version") != 1
                        or manifest.get("fingerprint") != expected or manifest.get("questions") != questions):
                    raise ValueError(f"{row}/{stage} checkpoint manifest differs")
                _load_attempts(cell_dir, questions, expected, cfg["retrieval"]["rerank"].get("top_k", 10))
            else:
                effective = copy.deepcopy(cfg)
                with config_overrides(effective, {**spec["overrides"], "orchestration.loop.cache": False,
                                                  "paths.results": str(cell_dir)}):
                    # Match the public review runner's manifest, including its selected JSONL.
                    expected = _digest({
                        "dataset_sha256": hashlib.sha256(content.encode()).hexdigest(),
                        "selected_query_ids": [q["query_id"] for q in questions], "split": "all", "limit": None,
                        "effective_config": _safe_config(effective), "indexes": identity["indexes"],
                        "code_sha256": identity["code_sha256"], "input_files": identity["input_files"],
                        "runtime": identity["runtime"],
                    })
                if manifest.get("fingerprint") != expected:
                    raise ValueError(f"{row}/{stage} checkpoint manifest differs")
                validate_review_attempts(cell_dir, questions)


def _aggregate(plan: dict, cells: dict, questions: list[dict], manifest: dict,
               error: dict | None, interrupted: bool) -> dict:
    report = copy.deepcopy(plan)
    expected = len(plan["rows"]) * len(plan["stages"])
    done = sum(c.get("status") == "complete" for row in cells.values() for c in row.values())
    report.update(status="partial" if interrupted else "failed" if error else
                  "complete" if done == expected else "partial", provenance=manifest,
                  completed_cells=done, selected_cells=expected, error=error,
                  updated_at=datetime.now(timezone.utc).isoformat())
    for row in report["rows"]:
        report["rows"][row]["results"] = cells.get(row, {})
    report["comparisons"] = {}
    report["interaction"] = {}
    # Ranking and source comparisons need gold papers; unanswerable challenge questions (D56) run in
    # every cell but are scored by eval.challenge, not here.
    ranked = [q for q in questions if q["gold_paper_ids"]]
    ranked_ids = {q["query_id"] for q in ranked}
    for stage in plan["stages"]:
        data = {r: {qid: o for qid, o in cells.get(r, {}).get(stage, {}).get("per_query", {}).items() if qid in ranked_ids}
                for r in plan["rows"]}
        report["comparisons"][stage] = compare_rows(ranked, data, plan["edges"], stage=stage)
        if len(plan["factors"]) == 2:
            report["interaction"][stage] = interaction(ranked, data, ("baseline", "a", "b", "ab"), stage=stage)
    report["usage"] = {k: sum(c["usage"][k] for row in cells.values() for c in row.values())
                       for k in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")}
    return report


def run_ablation(eval_set_path: str | Path, results_dir: str | Path, tag: str, *,
                 factors: Sequence[str], stage: str, split: str = "all", limit: int | None = None,
                 resume: bool = False, dry_run: bool = False, workers: int = 1) -> dict:
    """Plan/run a controlled matrix; resume matches full provenance and retains completed work.

    Dry-run only reads the dataset/config and prints a plan through the CLI: it does not
    import models, inspect indexes, make API calls, create directories or change config.
    Failed cells stop the matrix and return status=failed. Interrupts persist a partial report.
    `workers` questions run at once inside each review cell (D48); retrieval cells and the
    fingerprint are unaffected, so a matrix may resume with another value.
    """
    dataset, cfg = Path(eval_set_path).resolve(), get_config()
    questions = _selection(dataset, split, limit)
    plan = _plan(cfg, factors, stage, tag, questions)
    plan["selection"] = {"split": split, "limit": limit,
                         "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()}
    if dry_run:
        return plan
    require_fresh_runtime()
    if "review" in plan["stages"] and llm_client._CLIENT is not None:
        raise RuntimeError("review ablations require a fresh LLM client in a dedicated process")
    runtime = environment()
    identity = {"plan": plan, "code_sha256": _code_hash(), "indexes": index_manifests(),
                "input_files": _input_file_metadata(),
                "runtime": {k: runtime.get(k) for k in ("python", "versions")}}
    fingerprint = _digest(identity)
    directory = Path(results_dir).resolve() / tag
    directory.mkdir(parents=True, exist_ok=True)
    with _lock(directory / ".lock"):
        path = directory / "manifest.json"
        snapshot = directory / "selected_eval.jsonl"
        content = "".join(json.dumps(q, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for q in questions)
        if path.exists():
            if not resume:
                raise FileExistsError("ablation run exists; use --resume or a new tag")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") != fingerprint:
                raise ValueError("ablation resume fingerprint differs (data, selection, config, code or indexes/runtime)")
            if not snapshot.is_file() or snapshot.read_text(encoding="utf-8") != content:
                raise ValueError("ablation selected-query snapshot is missing or changed")
        else:
            if resume:
                raise FileNotFoundError("cannot resume an ablation without manifest.json")
            if any(p.name != ".lock" for p in directory.iterdir()):
                raise FileExistsError("ablation directory contains artifacts without its manifest")
            snapshot.write_text(content, encoding="utf-8")
            manifest = {"fingerprint": fingerprint, **identity, "environment": runtime,
                        "created_at": datetime.now(timezone.utc).isoformat()}
            _write(path, manifest)
        _preflight_cells(directory, plan, questions, cfg, fingerprint, identity, content)
        cells: dict = {}
        error = None
        interrupted = False
        for row, spec in plan["rows"].items():
            cells[row] = {}
            for cell_stage in plan["stages"]:
                cell_dir = directory / "rows" / row / cell_stage
                cell_report = {}
                try:
                    if (_input_file_metadata() != identity["input_files"] or _code_hash() != identity["code_sha256"]
                            or index_manifests() != identity["indexes"]):
                        raise ValueError("code or index/corpus metadata changed during the ablation run")
                    with config_overrides(cfg, spec["overrides"]):
                        _seed_local(cfg["seed"])
                        cell_resume = (cell_dir / "manifest.json").exists()
                        if cell_stage == "retrieval":
                            cell_report = run_retrieval_cell(questions, cell_dir,
                                _digest({"matrix": fingerprint, "row": row, "stage": cell_stage}), resume=cell_resume)
                        else:
                            cell_report = run_review_eval(snapshot, cell_dir.parent, cell_stage,
                                                          resume=cell_resume, workers=workers)
                except KeyboardInterrupt:
                    interrupted = True
                    if (cell_dir / "report.json").is_file():
                        cell_report = json.loads((cell_dir / "report.json").read_text(encoding="utf-8"))
                except Exception as exc:
                    error = {"row": row, "stage": cell_stage, "type": type(exc).__name__}
                    log.error("ablation stopped in %s/%s (%s)", row, cell_stage, type(exc).__name__)
                    if (cell_dir / "report.json").is_file():
                        cell_report = json.loads((cell_dir / "report.json").read_text(encoding="utf-8"))
                summary = _cell_summary(cell_stage, cell_report, cell_dir, questions)
                if error:
                    summary.update(status="failed", error=error)
                elif interrupted:
                    summary["status"] = "partial"
                elif summary["status"] != "complete":
                    error = {"row": row, "stage": cell_stage, "type": "IncompleteCell",
                             "detail": summary.get("error")}
                cells[row][cell_stage] = summary
                report = _aggregate(plan, cells, questions, manifest, error, interrupted)
                _write(directory / "report.json", report)
                log.info("ablation: %d/%d cells complete", report["completed_cells"], report["selected_cells"])
                if error or interrupted:
                    break
            if error or interrupted:
                break
        if interrupted:
            raise KeyboardInterrupt
        return report


def format_summary(report: dict) -> str:
    lines = [f"Ablation {report['tag']}: {report['status']}; factors {', '.join(report['factors'])}",
             f"Stages: {', '.join(report['stages'])}; {report['selected_queries']} identical questions per cell",
             f"Planned query executions: {report['query_executions']} "
             f"({report['review_query_executions']} full searches with possible paid API calls)"]
    for name, row in report["rows"].items():
        switches = ", ".join(f"{key}={str(value).lower()}" for key, value in row["overrides"].items())
        states = ", ".join(f"{s}:{v['status']}({v['completed_queries']})" for s, v in row.get("results", {}).items())
        lines.append(f"{name}: {switches}" + (f"; {states}" if states else ""))
    if "usage" in report:
        lines.append(f"Observed API usage across attempts: {report['usage']['total_tokens']} tokens")
    for stage, comparisons in report.get("comparisons", {}).items():
        for edge in comparisons:
            for split, values in edge["splits"].items():
                if not values["n_paired"]:
                    continue
                hit = values["metrics"]["Hit@10"]
                lines.append(f"{stage}/{split} {edge['before']} -> {edge['after']}: "
                             f"n={values['n_paired']}, Hit@10 {hit['before']:.3f} -> {hit['after']:.3f} "
                             f"(delta {hit['delta']:+.3f}), fixed={values['fixed']}, broken={values['broken']}")
    lines.append("Reports compare jointly completed questions per split; unknown candidates are not negative labels.")
    return "\n".join(lines)
