"""Paper Search review metrics against explicit human labels (SCHEMA.md sections 5-7).

Review classification is conditional on a paper being reviewed and explicitly labeled for
the same query. A paper absent from human_labels is unknown, never a negative example. Call
review_accuracy separately per query; pool its confusion counts for a micro aggregate across
queries. Do not combine different queries into one paper_id -> label mapping.

Positive means relevant (human label 1) / verdict="pass". Accuracy = (TP+TN)/N,
precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN). A zero denominator
returns None (JSON null), not a perfect or zero score. Unreviewed labeled papers are reported
separately and are not classification false negatives; retrieval measures those misses.
The fixed dataset labels only its positive source papers. Classification scores on that
subset cannot establish false-positive performance or overall reviewer quality: inspect
label_classes and support, and retain this limitation when reporting any scores.

Coverage is the orchestration loop's self-reported aspect coverage, not human relevance or
answer faithfulness. It is computed from covered_aspects / total_aspects independently of
termination_reason: reaching min_papers does not imply coverage was met. Aggregation reports
both a macro mean of defined per-query ratios and the pooled aspect ratio, counts and rates
by final round, and termination reasons separately. Zero-aspect reports have undefined
coverage and are excluded from coverage rates, while remaining in other run statistics.

Both functions are pure and use no network, files, or optional dependencies. Hand-computed
examples and invalid-input cases are checked in tests/test_review_metrics.py.
"""

from __future__ import annotations

from collections import Counter


_TERMINATION_REASONS = {
    "coverage_met", "min_papers", "max_rounds", "loop_disabled", "rewrite_exhausted",
    "plain_rag",  # D55: the non-agentic baseline, no review
    "review_disabled",  # D61: one round without review, the sub-queries' rankings fused
}


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _ratio(numerator: int | float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def review_accuracy(records: list[dict], human_labels: dict[str, int]) -> dict:
    """Score one query's unique review verdicts only where human labels exist.

    Args:
        records: ReviewRecord dictionaries; paper_id and verdict are required. Other
            ReviewRecord fields are optional, since the metric only needs these two fields.
        human_labels: paper_id -> integer 0 or 1, based on explicit human majority labels.
            Missing IDs are unlabeled, not implicit zeroes. Booleans/floats are rejected.

    Returns:
        accuracy, precision, recall, f1 (float or None); n (scored records), n_records,
        n_unlabeled, n_labels_unreviewed; confusion {tp, tn, fp, fn}; support {positive,
        negative}; and label_classes (sorted human labels present among scored records).

    Raises:
        ValueError: invalid input containers, paper IDs, labels or verdicts, or duplicate
            review paper IDs. Even duplicate/unusable unlabeled records are rejected rather
            than silently accepted; one paper must not gain extra weight within a query.
    """
    if not isinstance(records, list) or not isinstance(human_labels, dict):
        raise ValueError("records must be a list and human_labels must be a dictionary")
    for paper_id, label in human_labels.items():
        _text(paper_id, "human_labels paper_id")
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError(f"human_labels[{paper_id!r}] must be integer 0 or 1")

    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    seen: set[str] = set()
    n_unlabeled = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"records[{index}] must be a dictionary")
        paper_id = _text(record.get("paper_id"), f"records[{index}].paper_id")
        verdict = record.get("verdict")
        if not isinstance(verdict, str) or verdict not in ("pass", "reject"):
            raise ValueError(f"records[{index}].verdict must be 'pass' or 'reject'")
        if paper_id in seen:
            raise ValueError(f"duplicate review paper_id {paper_id!r}; score each query separately")
        seen.add(paper_id)
        if paper_id not in human_labels:
            n_unlabeled += 1
            continue
        positive = human_labels[paper_id] == 1
        predicted = verdict == "pass"
        key = "tp" if positive and predicted else "fn" if positive else "fp" if predicted else "tn"
        confusion[key] += 1

    tp, tn, fp, fn = (confusion[key] for key in ("tp", "tn", "fp", "fn"))
    n = tp + tn + fp + fn
    support = {"positive": tp + fn, "negative": tn + fp}
    return {
        "accuracy": _ratio(tp + tn, n),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "n": n,
        "n_records": len(records),
        "n_unlabeled": n_unlabeled,
        "n_labels_unreviewed": len(human_labels.keys() - seen),
        "confusion": confusion,
        "support": support,
        "label_classes": [label for label, count in ((0, support["negative"]), (1, support["positive"])) if count],
    }


def coverage_stats(coverage_reports: list[dict]) -> dict:
    """Aggregate actual FinalResult.coverage_report fields (SCHEMA.md section 6).

    Required integer fields: total_aspects, covered_aspects, rounds (>=1), papers_reviewed,
    papers_passed. Counts are nonnegative, covered <= total, and passed <= reviewed.
    termination_reason must be one of the orchestration loop's five termination reasons, plain_rag (D55)
    or review_disabled (D61).
    Additional fields (cache_hit, review_backend, etc.) do not affect this aggregation.

    coverage_ratio pools aspect counts; mean_coverage averages per-query ratios with a
    positive total_aspects. coverage_met counts complete coverage among those queries;
    coverage_met_rate, unmet_rate and coverage_met_round_rates use n_coverage_defined as
    denominator. Distribution keys are string round numbers for stable JSON output; values
    are counts. coverage_met_round_distribution groups fully covered queries by their final
    round, not the first round coverage was reached (the reports only store final state).
    Termination reasons never determine coverage success, even for inconsistent reports.
    A zero denominator produces None. Empty input produces zero counts and empty distributions.

    Raises:
        ValueError: invalid containers, missing fields, mistyped/out-of-range counts, or
            invalid termination reasons. Invalid reports are not silently dropped.
    """
    if not isinstance(coverage_reports, list):
        raise ValueError("coverage_reports must be a list")
    rounds: Counter[int] = Counter()
    met_rounds: Counter[int] = Counter()
    reasons: Counter[str] = Counter()
    total_aspects = covered_aspects = reviewed = passed = total_rounds = 0
    n_defined = 0
    ratios: list[float] = []
    for index, report in enumerate(coverage_reports):
        if not isinstance(report, dict):
            raise ValueError(f"coverage_reports[{index}] must be a dictionary")
        counts = {
            key: _integer(report.get(key), f"coverage_reports[{index}].{key}", 1 if key == "rounds" else 0)
            for key in ("total_aspects", "covered_aspects", "rounds", "papers_reviewed", "papers_passed")
        }
        total, covered = counts["total_aspects"], counts["covered_aspects"]
        if covered > total:
            raise ValueError(f"coverage_reports[{index}]: covered_aspects exceeds total_aspects")
        if counts["papers_passed"] > counts["papers_reviewed"]:
            raise ValueError(f"coverage_reports[{index}]: papers_passed exceeds papers_reviewed")
        reason = report.get("termination_reason")
        if not isinstance(reason, str) or reason not in _TERMINATION_REASONS:
            raise ValueError(f"coverage_reports[{index}]: invalid termination_reason {reason!r}")
        round_no = counts["rounds"]
        rounds[round_no] += 1
        reasons[reason] += 1
        total_rounds += round_no
        total_aspects += total
        covered_aspects += covered
        reviewed += counts["papers_reviewed"]
        passed += counts["papers_passed"]
        if total:
            n_defined += 1
            ratios.append(covered / total)
            if covered == total:
                met_rounds[round_no] += 1

    n = len(coverage_reports)
    met = sum(met_rounds.values())
    return {
        "n": n,
        "n_coverage_defined": n_defined,
        "n_zero_aspects": n - n_defined,
        "total_aspects": total_aspects,
        "covered_aspects": covered_aspects,
        "coverage_ratio": _ratio(covered_aspects, total_aspects),
        "mean_coverage": _ratio(sum(ratios), n_defined),
        "coverage_met": met,
        "coverage_met_rate": _ratio(met, n_defined),
        "unmet_rate": _ratio(n_defined - met, n_defined),
        "round_distribution": {str(r): count for r, count in sorted(rounds.items())},
        "coverage_met_round_distribution": {str(r): count for r, count in sorted(met_rounds.items())},
        "coverage_met_round_rates": {str(r): _ratio(met_rounds[r], n_defined) for r in sorted(rounds)},
        "termination_reasons": dict(sorted(reasons.items())),
        "mean_rounds": _ratio(total_rounds, n),
        "mean_papers_reviewed": _ratio(reviewed, n),
        "mean_papers_passed": _ratio(passed, n),
    }
