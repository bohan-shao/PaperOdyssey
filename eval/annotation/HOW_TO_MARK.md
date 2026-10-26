# How to mark the test questions

For: Athul, yh and zk. The first batch (50 rows) takes about 20-30 minutes.

## Why we do this

To test our paper search, we need questions where we know the right paper. An AI wrote one
question for each of 1,100 random papers in our knowledge base. **Your job is to check whether
each question really fits its paper.** The three of us check every question separately; the
project needs us to agree on at least 80% of them.

## Your files

Athul sends you your file. Do the test batch first; the rest comes after we have checked how
well we agree.

| Who | Test batch (50 rows, do this first) | The rest (1,050 rows, later) |
|---|---|---|
| Athul | `round1_part1_A.csv` | `round1_part2_A.csv` |
| yh | `round1_part1_B.csv` | `round1_part2_B.csv` |
| zk | `round1_part1_C.csv` | `round1_part2_C.csv` |

## What to do

1. Open your file in Excel or Google Sheets. Each row is one question.
2. Read the **question**.
3. Read the **paper_title** and **abstract** in the same row.
4. Ask yourself: *"If I typed this question into a paper search, would I be happy to get this paper?"*
5. In the **fits** column, type `y` (yes) or `n` (no).
6. When all rows are done, save the file (see below) and send it back to Athul.

## When to type `n`

- The paper does not really answer the question (it is about something else, or only mentions it on the side).
- The question is too vague: thousands of papers would fit just as well.
- The question asks about something the abstract does not talk about.
- The question gives the paper away: it copies the title, or uses the name the paper gives its own model or dataset.

Otherwise type `y`. Don't be too strict: a question does not have to mention everything in the paper.

## Example

This paper is **not** in your file; the questions below were written for this guide.

**Paper:** *Attention Is All You Need*. **Abstract (short):** proposes the Transformer, a network
for translation that uses only attention, with no recurrent or convolutional layers; it trains
faster and translates English to German and French better than earlier models.

| Question | fits | Why |
|---|---|---|
| How can a translation model drop recurrence and convolutions and rely only on attention to train faster? | `y` | exactly what the paper does |
| Looking for work on encoder-decoder translation models built entirely from attention layers. | `y` | a search request instead of a question is fine |
| How can neural networks be made better? | `n` | too vague |
| How can speech be recognised in noisy rooms? | `n` | the paper is not about this |
| Looking for the paper "Attention Is All You Need" on the Transformer. | `n` | copies the title and names the model |

## Rules

- **Mark alone.** Don't discuss rows or show your file to the others until all three of us are done.
- Only `y` or `n` in **fits**. If you are unsure, pick the closer one and write why in **note**.
- Don't change the other columns, don't delete or move rows, and don't rename the file.

## Saving the file

- **Excel:** File > Save As > "CSV UTF-8 (Comma delimited)". Keep the same file name.
- **Google Sheets:** File > Import > upload the file. When done: File > Download > Comma-separated values (.csv),
  then rename the downloaded file back to its original name (for example `round1_part1_B.csv`).
- **Numbers (Mac):** File > Export To > CSV, text encoding Unicode (UTF-8), same file name.

## What happens next

Athul runs a script that counts the votes. A question that at least 2 of us mark `y` goes into the
test set. If we agree on 80% or more, we mark the other 1,050 the same way. Questions most of us
mark `n` are rewritten by one of us later, and then all three mark the new version.

The detailed rules are in [guidelines.md](guidelines.md) §6; the commands Athul runs are in [README.md](README.md).
