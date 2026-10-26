# annotation/ Evaluation-set annotation

## Goal (coursework Q4 hard requirement)
- **>=1,000 items** annotated; **3 annotators** independently; majority vote finalizes
- **IAA >= 80%** (percentage agreement; also compute Fleiss' kappa, target kappa>=0.6)
- Format: `data/eval/eval_set.jsonl` (SCHEMA.md §7); deduplicated, type-balanced

## Annotation content (per item)
1. **Paper-level relevance** (required, 1000 items): gold_paper_ids
2. **Passage-level relevance** (subset of 100-200 items): gold_passage_ids
3. **gold answer** (for generation evaluation): reference answer

## Workflow (D39, agreed by the team on 2026-10-24)
For the annotators: [HOW_TO_MARK.md](HOW_TO_MARK.md) (what to do, with an example).

Questions are drafted by the LLM from one paper each and checked by people; that paper is the gold paper.

```bash
python scripts/eval_set.py draft --n 50       # pilot: 50 papers (needs an LLM key in .env)
python scripts/eval_set.py sheets             # data/eval/annotation/round1_part1_{A,B,C}.csv
#   each annotator fills `fits` (y/n) in their own sheet, alone (guidelines.md §6)
python scripts/eval_set.py build              # eval_set.jsonl + metrics/iaa_<date>.json + to_rewrite.csv
#   discuss the pilot's disagreements; adjust guidelines.md (or the prompt, before drafting more)
python scripts/eval_set.py draft --n 1100     # the rest (the first 50 stay; ~10% spare for rejected questions)
python scripts/eval_set.py sheets             # round1_part2_*.csv: only the questions not handed out yet
python scripts/eval_set.py build
#   one person writes `new_question` (and `new_type`) for the rows of to_rewrite.csv, then:
python scripts/eval_set.py rewrites           # adds them as round 2
python scripts/eval_set.py sheets --round 2   # everyone marks the new versions; build again
```

The annotators' names (default A B C) and the file names are set in `config.yaml` under
`eval.annotation`; change the names before the first `sheets` and keep them. `draft` samples from the
configured papers.sqlite with the config `seed` (`--papers` for another one, or a KB folder holding
it); rerun it after an interruption or a failed paper. `sheets` records which questions each part
holds in `round<R>_part<P>.json`, so a row deleted from a sheet is not handed out again. A blank
`fits` never erases a mark from another sheet; two different marks for one question are an error.

## Time budget (biggest manpower risk)
- Every annotator marks every question (needed for the agreement score): ~1,100 y/n marks each,
  reading a question and an abstract per row (a rough estimate, to be checked on the pilot)
- If behind, prioritize 500 items + explain in the report

## Tooling
- CSV sheets (open in Excel or Google Sheets, keep the file as CSV): query_id, round, question,
  paper_title, year, abstract, fits, note
- Optional upgrade: simple annotation web page (Streamlit), stretch goal that does not block the main line
