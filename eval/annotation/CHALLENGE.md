# How to check the challenge questions (D56)

For: the two people named in `eval.challenge.annotators` (A Athul, B yh by default). About 40
questions, 30 minutes.

## Why we do this

Most of our test questions are found in one search round, so they cannot show what the search loop
adds. An AI drafted 40 harder questions of four kinds. **Your job is to check that each one is a
fair question of its kind.** Both of us check every question separately; only questions we both
accept are used.

## What to do

Open `challenge_<letter>.csv` in Excel or Google Sheets. For each row read the question, the kind,
and the paper title(s) and abstract(s) in the same row, then type `y` or `n` in **fits**:

| Kind | Type `y` when | Type `n` when |
|---|---|---|
| **comparison** | Answering it well needs both papers, and both fit the question | One paper alone answers it, or one does not fit |
| **two_part** | It asks two different things and this paper (by its abstract) addresses both | It asks one thing twice, or the paper clearly misses one part |
| **date** | The paper fits the question, and the time bound (shown in `time_bound`) is part of the question | The paper does not fit, or the bound is missing or garbled |
| **out_of_kb** | It is clearly about another field, not about AI, machine learning, NLP, data or computing | An AI or machine-learning paper could reasonably answer it |

General `n` reasons for any kind: names a model, method or dataset the paper introduces, copies the
title, states the results, or is not one clear sentence. Use `note` for anything unusual.

Do not discuss the questions with each other until both files are back. Save as CSV (UTF-8), keep
the file name, and send it to Athul. Do not change `challenge_id` or `question`.
