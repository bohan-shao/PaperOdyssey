"""The eval package: evaluation (Q3).

Submodules (active — the framework is decided):
- retrieval_metrics: Hit@K / P@K / R@K / MRR / NDCG (Paper Search, Q3)
- retrieval_eval:    the eval set through the EVALUATION.md section 2 matrix, scored per split
                     (`scripts/run_eval.py --stage retrieval`)
- review_metrics:    review verdict vs. human annotation (framework level)
- review_eval:       resumable Paper Search runs with complete review traces (`--stage review`)
- latency:           retrieval-core latency benchmark (Q2 latency, EVALUATION.md section 7)
- annotation/:       evaluation-set construction, mixed 2/3-annotator protocol and grouped IAA

Review scoring uses explicitly labeled query-paper pairs only. The current fixed set labels
positive source papers; unjudged papers are never inferred to be negatives.

Generation metrics (ROUGE/BERTScore) and RAGAS remain pending. Paper QA now returns
DownstreamResult (docs/SCHEMA.md section 10), including the answer, context, and citations
needed by those future metric runners (Q4).

Authoritative evaluation plan: docs/EVALUATION.md; results go to data/results/.
"""

from __future__ import annotations
