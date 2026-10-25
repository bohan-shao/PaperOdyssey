"""Evaluation-set annotation tooling (Q4 hard requirement: >=1000 items, 3 annotators, IAA>=80%).

Contains:
- questions.py: LLM-drafted questions, marking sheets, majority vote -> eval_set.jsonl (D39)
- iaa.py: Fleiss' kappa + percentage agreement computation (3 annotators)

Annotation guidelines in annotation/guidelines.md; workflow and format in SCHEMA.md §7.
Annotation tooling starts with Excel/spreadsheet templates (see annotation/README.md); an optional
web page is a stretch goal and does not block the main line.
"""

from __future__ import annotations
