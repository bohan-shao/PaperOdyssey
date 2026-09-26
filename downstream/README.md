# downstream/ Cited paper question answering

`downstream.run(input_text, query_id=None) -> DownstreamResult` calls Paper Search once,
loads real passage text through `kb.store.load_passages`, and makes one answer-generation
call through `llm.client`. The existing search review/rewrite settings remain in effect.

```python
import json
from dataclasses import asdict
from downstream import run

result = run("How does low-rank adaptation reduce trainable parameters?")
print(result.answer)  # inline [1], [2] citations
print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
```

## Evidence and answers

- Select passing papers by descending review score, ties by paper ID. Defaults in
  `config.yaml`: 5 papers, 2 distinct passage IDs per paper, 2,400 characters per excerpt,
  1,024 output tokens. The stored child passages are used even with parent retrieval enabled;
  this first version does not pass complete parent sections to the answer model.
- Source numbers identify **passages**, so two passages from one paper may have different
  numbers. Each source contains title, year, section, paper/passage IDs, a validated arXiv
  link when available, and exactly the excerpt sent to the model.
- Missing, empty, or mismatched passages are omitted; review reasons and abstracts are not
  substitutes for original text. No usable sources means `insufficient_evidence` without
  calling the answer model. A missing/corrupt corpus file is an operational error.
- The prompt requests a concise answer in the question's language and requires inline
  citations. The model may report insufficient evidence even when excerpts exist.
- Numeric citation format and bounds are checked in code; malformed or uncited answers
  raise `DownstreamError`. There is no second model check or repair loop. These checks
  validate source identity, **not semantic support for every claim**.
- Blank input raises `ValueError`; search, corpus, and model failures raise
  `DownstreamError`, with the underlying exception chained for debugging.

## Output and measurement

The contract is in `docs/SCHEMA.md` §10. `sources` retains all supplied context;
`cited_source_numbers` records citations actually used. Both are kept for later evaluation.
`search_result` is the unmodified Paper Search result. Timing covers search, evidence loading,
generation, and total wall time; LLM token usage remains in the shared client usage log.
The UI offers a JSON download containing the answer, sources, search result, and timings.
Answer results are not cached or written automatically; Paper Search retains its own cache.

Evidence lookup scans the passage JSONL once per call, stopping when all selected IDs are
found. Large-corpus lookup latency must be measured when real retrieval is available.

## Verification and integration status

```bash
python -m unittest tests.test_downstream tests.test_ui -v
python -m unittest discover -s tests -v
python scripts/sanity_check.py
```

Tests cover real temporary passage files, citation/source mapping, budgets, no-evidence
behavior, failures, and Streamlit interactions. The integration test runs the real
orchestration and LLM SDK against controlled retrieval results and an HTTP mock transport.
It makes no external model calls. Python >=3.11 and the relevant common/LLM/UI/KB
dependencies from `requirements.txt` are required for the full test suite.

`retrieval.pipeline.retrieve()` is still a stub at this implementation's base commit.
Real indexed retrieval and the five configured example questions must be verified after
that module is integrated. This change does not claim live end-to-end readiness, generation
quality scores, or completion of the 1,000-item coursework evaluation.
