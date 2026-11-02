# downstream/ Cited paper question answering

`downstream.run(input_text, query_id=None) -> DownstreamResult` calls Paper Search once,
loads real passage text by ID from the retrieval lookup store (`retrieval.lookup`, D28), and
makes one answer-generation call through `llm.client`, re-asked once if the reply is invalid
(D50). The existing search review/rewrite settings remain in effect.
`answer_from_search(question, search)` is the part after the search; the answer evaluation
(`eval/answer_eval.py`, D49) calls it on stored search results.

```python
import json
from dataclasses import asdict
from downstream import run

result = run("How does low-rank adaptation reduce trainable parameters?")
print(result.answer)  # inline [1], [2] citations
print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
```

## Evidence and answers

- Select passing papers in Paper Search order (`passing_order_key`, D47): descending review score,
  ties by the paper's best retrieval rank, then paper ID. Defaults in
  `config.yaml`: 5 papers, 2 distinct passage IDs per paper, 2,400 characters per excerpt,
  1,024 output tokens. The stored child passages are used even with parent retrieval enabled;
  this first version does not pass complete parent sections to the answer model.
- Source numbers identify **passages**, so two passages from one paper may have different
  numbers. Each source contains title, year, section, paper/passage IDs, a validated arXiv
  link when available, and exactly the excerpt sent to the model.
- Missing, empty, or mismatched passages are omitted; review reasons and abstracts are not
  substitutes for original text. No usable sources means `insufficient_evidence` without
  calling the answer model. A missing or corrupt lookup store is an operational error.
- The prompt requests a concise answer in the question's language and requires inline
  citations. The model may report insufficient evidence even when excerpts exist.
- Generation goes through `llm.client.chat_json`, so a reply that is not valid JSON, or is
  cut off at the token limit, is re-asked there. Citations are then checked in code
  (`downstream/citations.py`, shared with the UI's source links and the answer judge): grouped
  forms such as `[1, 2]`, `[1-3]` or `[01]` are rewritten as `[1][2]`, `[1][2][3]`, `[1]`; a
  bracket whose numbers are not all supplied sources (the interval `[0, 1]`, a year range), a
  lone `[0]`, and index notation attached to a name (`X[0]`, `x^{[0]}`, `A[i][1]`) stay plain
  text. A lone `[n]` outside the sources, an answered response with no citation, or a bad
  status or shape is re-asked once with the reason (D50); a second invalid reply raises
  `DownstreamError`. These checks validate source identity, **not semantic support for every
  claim**.
- Blank input raises `ValueError`; search, evidence lookup, and model failures raise
  `DownstreamError`, with the underlying exception chained for debugging.

## Output and measurement

The contract is in `docs/SCHEMA.md` §10. `sources` retains all supplied context;
`cited_source_numbers` records citations actually used. Both are kept for later evaluation.
`search_result` is the unmodified Paper Search result. Timing covers search, evidence loading,
generation, and total wall time; LLM token usage remains in the shared client usage log.
The UI offers a JSON download containing the answer, sources, search result, and timings.
`run(..., progress=...)` passes the callback to Paper Search and adds an `answering` event
before the answer call (SCHEMA.md §6.1, D58), so the UI can show each step while it waits.
Answer results are not cached or written automatically; Paper Search retains its own cache.

Evidence is read with primary-key lookups in `lookup.sqlite`, which `scripts/build_indexes.py`
builds from the same `passages.jsonl` as the BM25/FAISS indexes, so the cost does not grow with
the corpus. The UI's search tab reads its excerpts from the same store on every rerun.

## Verification and integration status

```bash
python -m unittest tests.test_downstream tests.test_ui -v
python -m unittest discover -s tests -v
python scripts/sanity_check.py
```

Tests cover a real temporary lookup store, citation/source mapping, index notation, the
re-ask, budgets, no-evidence behavior, failures, and Streamlit interactions. The integration
test runs the real orchestration and LLM SDK against controlled retrieval results and an HTTP
mock transport. It makes no external model calls. Python >=3.11 and the relevant common/LLM/UI/KB
dependencies from `requirements.txt` are required for the full test suite.

The retrieval core (`retrieval.pipeline.retrieve()`) is now implemented, but a live
end-to-end run on the built indexes, including the five configured example questions, has not
been verified yet. Answer quality is measured by `scripts/run_eval.py --stage answer` (D49); no scores exist until
it runs on a finished search run.
