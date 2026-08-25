"""Inter-annotator agreement (IAA) computation (Q4 hard requirement: IAA >= 80%). TODO.

Implementation requirements (see eval/annotation/README.md and docs/SCHEMA.md §7):
- `percentage_agreement(annotations) -> float`: fraction of items where the 3 annotators
  agree on the majority paper-level relevance label (pairs/triples; report the convention used).
- `fleiss_kappa(annotations) -> float`: Fleiss' kappa over 3 annotators,
  item = query_id, category = relevant paper set agreement per item (or the graded
  passage-level labels for the subset; document which).
- Accept both raw votes and majority labels; output a report dict
  {percentage_agreement, fleiss_kappa, n_items, n_annotators}.
- Results go to `data/results/metrics/iaa_<date>.json` (with the eval_set version).

Constraints: use scikit-learn where convenient (already in requirements.txt);
the convention for "agreement" must be documented in the report (percentage agreement vs kappa).
"""

from __future__ import annotations

from collections.abc import Iterable


def percentage_agreement(annotations: list[dict]) -> float:
    """Percentage agreement over items.

    Args:
        annotations: [{"query_id": str, "votes": [list[str], ...]}] (one entry per item,
            votes = each annotator's relevant paper IDs).

    Returns:
        Fraction of items where annotators agree (0–1).
    """
    raise NotImplementedError  # TODO: implement, see this docstring


def fleiss_kappa(annotations: list[dict]) -> float:
    """Fleiss' kappa over 3 annotators.

    Args:
        annotations: same shape as percentage_agreement.

    Returns:
        Kappa value (−1..1).
    """
    raise NotImplementedError  # TODO: implement, see this docstring


def compute_iaa_report(annotations: list[dict], out_path: str) -> dict:
    """Compute both metrics and write the report JSON. Returns the report dict."""
    raise NotImplementedError  # TODO: implement, see this docstring
