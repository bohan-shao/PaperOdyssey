"""Pure paired comparisons of completed ablation observations, with no API or file access.

Rows map query IDs to normalized completed observations. Missing queries are unpaired,
never retrieval misses. A rank is the first known source paper's position in the complete
returned paper ranking, or None when absent. Hit@K and MRR@10 follow retrieval_metrics;
NDCG and precision cannot be recovered from this summary and are deliberately omitted.
Optional metrics use only queries defined in every compared cell, with their own n.
"""

from __future__ import annotations

import math
from collections.abc import Callable

_SPLITS = ("dev", "eval")
_TIMING_NOTE = (
    "Wall times are descriptive, include flagged component cold starts, and may reuse OS/model "
    "caches; these comparisons are not a controlled latency benchmark."
)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _number(value: object, name: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        valid = math.isfinite(value) and value >= 0 and (maximum is None or value <= maximum)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and between zero and {maximum or 'infinity'}")
    return float(value)


def _validate(questions: list[dict], rows: dict[str, dict[str, dict]], stage: str) -> None:
    if stage not in ("retrieval", "review"):
        raise ValueError("stage must be retrieval or review")
    if not isinstance(questions, list) or not isinstance(rows, dict):
        raise ValueError("questions must be a list and rows a dictionary")
    ids: set[str] = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("each question must be a dictionary")
        qid = _text(question.get("query_id"), "query_id")
        if qid in ids:
            raise ValueError(f"duplicate question query_id {qid!r}")
        ids.add(qid)
        _text(question.get("input_text"), "input_text")
        if question.get("split") not in _SPLITS:
            raise ValueError(f"invalid split for {qid!r}")
        gold = question.get("gold_paper_ids")
        if not isinstance(gold, list) or not gold:
            raise ValueError(f"gold_paper_ids must be a nonempty list for {qid!r}")
        for pid in gold:
            _text(pid, "gold paper ID")
        if len(gold) != len(set(gold)):
            raise ValueError(f"duplicate gold paper ID for {qid!r}")
    for name, observations in rows.items():
        _text(name, "row name")
        if not isinstance(observations, dict):
            raise ValueError(f"row {name!r} must be a dictionary")
        for qid, observation in observations.items():
            if qid not in ids:
                raise ValueError(f"row {name!r} has unknown query ID {qid!r}")
            if not isinstance(observation, dict):
                raise ValueError(f"observation {name}/{qid} must be a dictionary")
            required = {"first_gold_rank", "source_success", "wall_ms", "cold_start"}
            if not required <= observation.keys():
                raise ValueError(f"observation {name}/{qid} is missing required fields")
            rank = observation["first_gold_rank"]
            if rank is not None and (type(rank) is not int or rank < 1):
                raise ValueError(f"first_gold_rank for {name}/{qid} must be a positive integer or None")
            for field in ("source_success", "cold_start"):
                if type(observation[field]) is not bool:
                    raise ValueError(f"{field} for {name}/{qid} must be a boolean")
            expected_success = rank is not None and (stage == "review" or rank <= 10)
            if observation["source_success"] != expected_success:
                raise ValueError(f"source_success and first_gold_rank disagree for {name}/{qid}")
            _number(observation["wall_ms"], "wall_ms")
            recall = observation.get("source_recall")
            if recall is not None:
                _number(recall, "source_recall", maximum=1.0)
                if stage == "review" and (recall > 0) != observation["source_success"]:
                    raise ValueError(f"source_recall and source_success disagree for {name}/{qid}")
            covered = observation.get("coverage_met")
            if covered is not None and type(covered) is not bool:
                raise ValueError("coverage_met must be a boolean or None")
            rounds = observation.get("rounds")
            if rounds is not None and (type(rounds) is not int or rounds < 1):
                raise ValueError("rounds must be a positive integer or None")


def _metrics(stage: str) -> dict[str, Callable[[dict], float | None]]:
    metrics: dict[str, Callable[[dict], float | None]] = {
        f"Hit@{k}": lambda row, k=k: float(row["first_gold_rank"] is not None
                                            and row["first_gold_rank"] <= k)
        for k in (1, 3, 5, 10)
    }
    metrics["MRR@10"] = lambda row: (1.0 / row["first_gold_rank"]
                                    if row["first_gold_rank"] is not None
                                    and row["first_gold_rank"] <= 10 else 0.0)
    metrics["wall_ms"] = lambda row: row["wall_ms"]
    if stage == "review":
        metrics.update({
            "source_success_rate": lambda row: float(row["source_success"]),
            "source_recall": lambda row: row.get("source_recall"),
            "coverage_met_rate": lambda row: row.get("coverage_met"),
            "mean_rounds": lambda row: row.get("rounds"),
        })
    return metrics


def _mean(values: list[float]) -> float | None:
    # Divide first so a finite set of large observations cannot overflow a sum.
    return math.fsum(value / len(values) for value in values) if values else None


def _example(question: dict, before: dict, after: dict) -> dict:
    return {"query_id": question["query_id"], "input_text": question["input_text"],
            "gold_paper_ids": list(question["gold_paper_ids"]),
            "rank_before": before["first_gold_rank"], "rank_after": after["first_gold_rank"]}


def _paired_split(questions: list[dict], rows: dict[str, dict[str, dict]], names: tuple[str, ...]) -> tuple[list[dict], dict]:
    paired = [q for q in questions if all(q["query_id"] in rows[name] for name in names)]
    paired_ids = {q["query_id"] for q in paired}
    return paired, {
        "n_selected": len(questions), "n_paired": len(paired),
        "paired_query_ids": [q["query_id"] for q in paired],
        "unpaired_query_ids": [q["query_id"] for q in questions if q["query_id"] not in paired_ids],
    }


def compare_rows(questions: list[dict], rows: dict[str, dict[str, dict]],
                 edges: list[tuple[str, str]], *, stage: str) -> list[dict]:
    """Compare each before/after edge on identical completed queries, separately per split.

    fixed/broken means Hit@10 changed for retrieval, or any known source was accepted for
    review. Review additionally reports top10_fixed/broken to keep ranking separate from
    source acceptance. All changed IDs are retained; examples are capped at five in input
    file order. Invalid observations, unknown query IDs, and invalid edges raise ValueError.
    """
    _validate(questions, rows, stage)
    if not isinstance(edges, list):
        raise ValueError("edges must be a list of before/after pairs")
    seen = set()
    for edge in edges:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            raise ValueError("each edge must contain two row names")
        before, after = edge
        if not isinstance(before, str) or not isinstance(after, str) or before == after:
            raise ValueError("each edge must contain distinct row names")
        if before not in rows or after not in rows or (before, after) in seen:
            raise ValueError("edge has unknown row names or is repeated")
        seen.add((before, after))
    reports = []
    for before, after in edges:
        splits = {}
        for split in _SPLITS:
            selected = [q for q in questions if q["split"] == split]
            paired, summary = _paired_split(selected, rows, (before, after))
            metrics = {}
            for name, metric in _metrics(stage).items():
                values = [(metric(rows[before][q["query_id"]]), metric(rows[after][q["query_id"]]))
                          for q in paired]
                values = [(left, right) for left, right in values if left is not None and right is not None]
                left, right = _mean([v[0] for v in values]), _mean([v[1] for v in values])
                metrics[name] = {"n": len(values), "before": left, "after": right,
                                 "delta": right - left if left is not None and right is not None else None}
            changes = {key: [] for key in ("fixed", "broken", "moved_up", "moved_down")}
            if stage == "review":
                changes.update(top10_fixed=[], top10_broken=[])
            for question in paired:
                qid = question["query_id"]
                left, right = rows[before][qid], rows[after][qid]
                if not left["source_success"] and right["source_success"]:
                    changes["fixed"].append(question)
                elif left["source_success"] and not right["source_success"]:
                    changes["broken"].append(question)
                rank_left, rank_right = left["first_gold_rank"], right["first_gold_rank"]
                if (rank_right or math.inf) < (rank_left or math.inf):
                    changes["moved_up"].append(question)
                elif (rank_right or math.inf) > (rank_left or math.inf):
                    changes["moved_down"].append(question)
                if stage == "review":
                    top_left = rank_left is not None and rank_left <= 10
                    top_right = rank_right is not None and rank_right <= 10
                    if top_right and not top_left:
                        changes["top10_fixed"].append(question)
                    elif top_left and not top_right:
                        changes["top10_broken"].append(question)
            summary.update(metrics=metrics, timing_note=_TIMING_NOTE,
                           cold_start_counts={name: sum(rows[name][q["query_id"]]["cold_start"] for q in paired)
                                              for name in (before, after)})
            for kind, changed in changes.items():
                summary[kind] = len(changed)
                summary[kind + "_query_ids"] = [q["query_id"] for q in changed]
                if kind not in ("moved_up", "moved_down"):
                    summary[kind + "_examples"] = [_example(q, rows[before][q["query_id"]],
                                                           rows[after][q["query_id"]]) for q in changed[:5]]
            splits[split] = summary
        reports.append({"before": before, "after": after, "stage": stage, "splits": splits})
    return reports


def interaction(questions: list[dict], rows: dict[str, dict[str, dict]],
                row_names: tuple[str, str, str, str], *, stage: str) -> dict:
    """Compute AB - A - B + baseline on the same four-cell completed subset per split.

    row_names is ordered (baseline, A, B, AB). This is a descriptive factorial contrast,
    not a significance test; stochastic LLM outcomes may differ across otherwise paired runs.
    """
    _validate(questions, rows, stage)
    if (not isinstance(row_names, (tuple, list)) or len(row_names) != 4
            or any(not isinstance(name, str) or name not in rows for name in row_names)
            or len(set(row_names)) != 4):
        raise ValueError("row_names must identify four distinct existing rows in baseline/A/B/AB order")
    roles = ("baseline", "A", "B", "AB")
    splits = {}
    for split in _SPLITS:
        paired, summary = _paired_split([q for q in questions if q["split"] == split], rows, tuple(row_names))
        metrics = {}
        selected_metrics = ("Hit@10", "MRR@10") + (("source_success_rate", "source_recall", "coverage_met_rate")
                                                   if stage == "review" else ())
        for name, metric in _metrics(stage).items():
            if name not in selected_metrics:
                continue
            values = [[metric(rows[row][q["query_id"]]) for row in row_names] for q in paired]
            values = [value for value in values if all(v is not None for v in value)]
            means = [_mean([value[i] for value in values]) for i in range(4)]
            metrics[name] = {"n": len(values), **dict(zip(roles, means)),
                             "interaction": means[3] - means[1] - means[2] + means[0] if values else None}
        summary["metrics"] = metrics
        splits[split] = summary
    return {"stage": stage, "rows": dict(zip(roles, row_names)),
            "formula": "AB - A - B + baseline", "splits": splits}
