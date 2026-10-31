"""The eval package: evaluation (Q3).

Submodules (active — the framework is decided):
- retrieval_metrics: Hit@K / P@K / R@K / MRR / NDCG (Paper Search, Q3)
- retrieval_eval:    the eval set through the EVALUATION.md section 2 matrix, scored per split
                     (`scripts/run_eval.py --stage retrieval`)
- review_metrics:    review verdict vs. human annotation (framework level)
- review_eval:       resumable Paper Search runs with complete review traces (`--stage review`)
- ablation:          one/two-factor baseline/A/B/AB runs and paired single-factor comparisons
- answer_eval:       answers and scores the questions of a finished review run or ablation (D49)
- answer_audit:      human audit of sampled answers: correct / complete / supported (D54)
- answer_metrics:    RAGAS-style faithfulness / answer relevance / context relevance+precision
- latency:           retrieval-core latency benchmark (Q2 latency, EVALUATION.md section 7)
- annotation/:       evaluation-set construction, mixed 2/3-annotator protocol and grouped IAA

Review scoring uses explicitly labeled query-paper pairs only. The current fixed set labels
positive source papers; unjudged papers are never inferred to be negatives.

Answer metrics (Q4) score Paper QA's DownstreamResult (docs/SCHEMA.md section 10) with an LLM judge;
the eval set has no reference answers, so ROUGE/BERTScore are not computed.

Authoritative evaluation plan: docs/EVALUATION.md; results go to data/results/.
"""

from __future__ import annotations
