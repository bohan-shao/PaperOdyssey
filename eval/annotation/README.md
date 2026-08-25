# annotation/ Evaluation-set annotation

## Goal (coursework Q4 hard requirement)
- **>=1,000 items** annotated; **3 annotators** independently; majority vote finalizes
- **IAA >= 80%** (percentage agreement; also compute Fleiss' kappa, target kappa>=0.6)
- Format: `data/eval/eval_set.jsonl` (SCHEMA.md §7); deduplicated, type-balanced

## Annotation content (per item)
1. **Paper-level relevance** (required, 1000 items): gold_paper_ids
2. **Passage-level relevance** (subset of 100-200 items): gold_passage_ids
3. **gold answer** (for generation evaluation): reference answer

## Workflow
1. Draft 1000 inputs (5 types: definition / method comparison / survey / paper locating / technical detail, roughly balanced) -> deduplicate
2. **Pilot-annotate 50 items x 3 annotators** -> align criteria against the guidelines -> revise the guidelines
3. Formal annotation (in 10 batches of 100) -> compute IAA per batch -> batches below 80% go to disagreement resolution
4. Majority vote -> finalize eval_set.jsonl -> dev/eval split (10%/90%)

## Time budget (biggest manpower risk)
- 3 annotators x 1000 items: estimate 150-250 items per person per week (once tooled)
- Start 2 weeks early; if behind, prioritize 500 items + explain in the report

## Tooling
- Start: Excel/CSV template (columns: query_id, input_text, annotator, relevant_papers,
  passage_ids(subset), answer) -> script merges -> compute IAA
- Optional upgrade: simple annotation web page (Streamlit), stretch goal that does not block the main line
