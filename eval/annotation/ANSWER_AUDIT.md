# How to check the answers (answer audit, D54)

For: Athul, yh, zk, Ernest and Rong. With the default 100 answers and two of us on each, everyone
gets 40 answers, about 2 hours.

## Why we do this

Our test questions have no written "right answers", so the automatic scores cannot tell us whether
an answer is actually correct. An AI judge scored every answer, but it is the same AI that wrote
them. **Your job is to check, as a person, whether each answer is right, complete and backed by
its sources.** Two of us check every answer separately; the report gives our agreement and our
scores next to the AI judge's.

## Your files

Athul sends you two files with your letter (A Athul, B yh, C zk, D Ernest, E Rong):

- `answer_audit_<letter>.html`: open it in a browser. For each answer it shows the question, the
  answer, and every paper excerpt the answering AI was given. The excerpts the answer cites are in
  bold; `[2]` in the answer means excerpt 2.
- `answer_audit_<letter>.csv`: open it in Excel or Google Sheets. One row per answer, with the
  same audit IDs (AQ001, ...) as the web page. You fill in three columns.

## What to do

For each audit ID: read the question, the answer, then the excerpts it cites. Then type `y`, `p`
(partly) or `n` in each column:

| Column | `y` | `p` (partly) | `n` |
|---|---|---|---|
| **correct** | Everything the answer says about the question is right, judged from the excerpts | Mostly right, but one statement is wrong or garbled | The main statement is wrong, off-topic, or it says there is no evidence although the excerpts answer the question |
| **complete** | It answers every part of the question | It misses a part, or answers only in general terms | It does not answer what was asked |
| **supported** | Every statement with a citation is backed by the excerpt it cites | Some cited statements go beyond their excerpt or cite the wrong one | Most cited statements are not backed by their excerpts |

- Judge only from the question, the answer and the excerpts. If an excerpt is cut off, you may
  open the paper link.
- An answer that says the evidence is insufficient: `correct` = `y` if the excerpts really do not
  answer the question, `n` if they do. `complete` = `n`. `supported` = `y` unless it cites something
  that does not fit.
- Use `note` for anything unusual (one short sentence is enough).
- Do not discuss answers with each other until everyone has sent their file back.

## Example

Question: *"How can we automatically construct diverse worked examples for chain-of-thought
prompting instead of hand-writing demonstrations for each task?"*

Answer: *"Auto-CoT clusters the questions of a dataset, picks one representative question per
cluster and lets the model write its reasoning chain with 'Let's think step by step' [1]. ..."*,
and excerpt [1] describes exactly these two steps.

→ correct `y`, complete `y`, supported `y`. If the answer also claimed "it beats manual CoT on every
benchmark" and no excerpt says that, supported would be `p`.

## When you are done

Save the CSV as CSV (UTF-8), keep the file name, and send it back to Athul. Do not change the
`audit_id`, `query_id` or `question` columns.
