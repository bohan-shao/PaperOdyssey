"""Inter-annotator agreement (IAA) computation (Q4 hard requirement: IAA >= 80%).

Unit of agreement: one (item, paper) pair, which every annotator labels relevant (the paper is in
their vote) or not relevant. An item's papers are its `paper_ids` (the papers shown to the
annotators) when given, else every paper some annotator voted for. In the eval set
(`eval/annotation/questions.py`) an item is one question with its one source paper, so a unit is
one question: "is this paper a good answer to this question?".

- `percentage_agreement`: the share of units on which all annotators gave the same label (the
  strict convention). The report also gives the mean pairwise agreement: the share of annotator
  pairs that agree on a unit, averaged over units (this is P-bar of Fleiss' kappa).
- `fleiss_kappa`: Fleiss' kappa over the same units, two categories (relevant / not relevant),
  kappa = (P-bar - P_e) / (1 - P_e). It corrects for chance agreement, so when nearly every label
  is the same (e.g. almost every question accepted) kappa can be low even at a high percentage
  agreement; it is undefined (NaN, null in the report) when every label is the same.
- `summarize`: both, with the counts; `compute_iaa_report`: the same, written to
  `data/results/metrics/iaa_<date>.json`.

Input: [{"query_id": str, "votes": [list[str], ...], "paper_ids": list[str] (optional)}], one
vote per annotator (that annotator's relevant paper ids), every item with the same number of
votes. Agreement needs every annotator's own vote, never the majority label.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

CONVENTION = ("unit = (item, paper); percentage_agreement = all annotators agree; "
              "pairwise_agreement = mean share of agreeing annotator pairs; "
              "Fleiss' kappa over relevant / not relevant")


def _units(annotations: list[dict]) -> list[tuple[int, int]]:
    """(relevant votes, annotators) per (item, paper) unit."""
    if not annotations:
        raise ValueError("no annotations")
    n = len(annotations[0]["votes"])
    if n < 2:
        raise ValueError("agreement needs at least 2 annotators")
    units = []
    for item in annotations:
        votes = [set(v) for v in item["votes"]]
        if len(votes) != n:
            raise ValueError(f"{item['query_id']}: {len(votes)} votes, other items have {n}")
        papers = item.get("paper_ids")
        if papers is None:
            papers = sorted(set().union(*votes))
        units += [(sum(p in v for v in votes), n) for p in papers]
    if not units:
        raise ValueError("no (item, paper) units: give paper_ids for items nobody voted relevant")
    return units


def _fleiss(counts: list[list[int]]) -> tuple[float, float]:
    """(P-bar, kappa) for counts[i][j] = annotators putting unit i in category j."""
    n = sum(counts[0])
    n_units = len(counts)
    p_bar = sum((sum(c * c for c in row) - n) / (n * (n - 1)) for row in counts) / n_units
    p_e = sum((sum(row[j] for row in counts) / (n_units * n)) ** 2 for j in range(len(counts[0])))
    kappa = math.nan if p_e == 1 else (p_bar - p_e) / (1 - p_e)
    return p_bar, kappa


def percentage_agreement(annotations: list[dict]) -> float:
    """Percentage agreement over items.

    Args:
        annotations: [{"query_id": str, "votes": [list[str], ...]}] (one entry per item,
            votes = each annotator's relevant paper IDs; optional "paper_ids", see the module docstring).

    Returns:
        Fraction of (item, paper) units on which all annotators agree (0–1).
    """
    units = _units(annotations)
    return sum(k in (0, n) for k, n in units) / len(units)


def fleiss_kappa(annotations: list[dict]) -> float:
    """Fleiss' kappa over 3 annotators.

    Args:
        annotations: same shape as percentage_agreement.

    Returns:
        Kappa value (−1..1); NaN when every label is the same.
    """
    return _fleiss([[k, n - k] for k, n in _units(annotations)])[1]


def summarize(annotations: list[dict]) -> dict:
    """Both metrics and the counts behind them, as one dict (the report without writing it)."""
    units = _units(annotations)
    pairwise, kappa = _fleiss([[k, n - k] for k, n in units])
    return {
        "percentage_agreement": sum(k in (0, n) for k, n in units) / len(units),
        "pairwise_agreement": pairwise,
        "fleiss_kappa": None if math.isnan(kappa) else kappa,
        "n_items": len(annotations),
        "n_units": len(units),
        "n_annotators": units[0][1],
        "relevant_share": sum(k for k, _ in units) / sum(n for _, n in units),
        "convention": CONVENTION,
    }


def compute_iaa_report(annotations: list[dict], out_path: str, extra: dict | None = None) -> dict:
    """Compute both metrics and write the report JSON. Returns the report dict.

    `extra` (e.g. the eval-set version) is added to the report.
    """
    report = {**summarize(annotations), **(extra or {})}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
