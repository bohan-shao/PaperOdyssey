# How to mark the test questions

For: Athul, yh, zk, Ernest and Rong. Your file has about 420 rows; the test batch had 50.

## Why we do this

To test our paper search, we need questions where we know the right paper. An AI wrote one
question for each of 1,100 random papers in our knowledge base. **Your job is to check whether
each question really fits its paper.** Two of us check every question separately (the 50-question
test batch had three), and we need to agree on at least 80% of them.

## Your files

Athul sends you your file. The test batch is done (we agreed on 88%). In the main part each of the
1,050 questions goes to two of us, so everyone has 420 rows, and each pair of us shares 105 questions.

| Who | Test batch (50 rows, done) | Main part (420 rows) |
|---|---|---|
| Athul | `round1_part1_A.csv` | `round1_part2_A.csv` |
| yh | `round1_part1_B.csv` | `round1_part2_B.csv` |
| zk | `round1_part1_C.csv` | `round1_part2_C.csv` |
| Ernest | | `round1_part2_D.csv` |
| Rong | | `round1_part2_E.csv` |

When the two people on a question disagree (one `y`, one `n`), a third person gets that question
in a short extra file (`round1_part3_<letter>.csv`, a few rows each) and decides.

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
- The question gives the paper away: it is basically the whole title pasted in, or it uses a rare name
  only this paper has (the name it gives its own model or dataset, or a term it made up). Normal topic
  words that also appear in the title are fine: people search with those words too.

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

- **Mark alone.** Don't discuss rows or show your file to the others until everyone is done.
- Only `y` or `n` in **fits**. If you are unsure, pick the closer one and write why in **note**.
- Don't change the other columns, don't delete or move rows, and don't rename the file.

## Saving the file

- **Excel:** File > Save As > "CSV UTF-8 (Comma delimited)". Keep the same file name.
- **Google Sheets:** File > Import > upload the file. When done: File > Download > Comma-separated values (.csv),
  then rename the downloaded file back to its original name (for example `round1_part1_B.csv`).
- **Numbers (Mac):** File > Export To > CSV, text encoding Unicode (UTF-8), same file name.

## What happens next

Athul runs a script that counts the votes. A question both of its markers mark `y` goes into the
test set; when they disagree, the third person's mark decides. Questions marked `n` are rewritten
by one of us later and marked again.

The detailed rules are in [guidelines.md](guidelines.md) §6; the commands Athul runs are in [README.md](README.md).
