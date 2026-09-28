"""The eval package: evaluation (Q3).

Submodules (active — the framework is decided):
- retrieval_metrics: P@K / R@K / MRR / NDCG (Paper Search, Q3)
- review_metrics:    review verdict vs. human annotation (framework level)
- latency:           retrieval-core latency benchmark (Q2 latency, EVALUATION.md section 7)
- annotation/:       evaluation-set construction and annotation tooling (3 annotators, IAA>=80%)

Generation metrics (ROUGE/BERTScore) and RAGAS remain pending. Paper QA now returns
DownstreamResult (docs/SCHEMA.md section 10), including the answer, context, and citations
needed by those future metric runners (Q4).

Authoritative evaluation plan: docs/EVALUATION.md; results go to data/results/.
"""

from __future__ import annotations
