"""Downstream task — reserved folder (coursework Q4/Q5, 40 points). Task type TBD.

This package is intentionally EMPTY for now: the downstream task type is not yet decided
(QA with cited answers / summarization / classification over papers). When decided:
- the task will CALL Paper Search (`orchestration.run_query()`) to obtain evidence (D24)
- the task logic + its types go here (run.py / types.py / README.md)
- only this package and its evaluation change; the framework stays untouched

Until then, the UI's Downstream tab is disabled/hidden (see ui/README.md).
"""

from __future__ import annotations
