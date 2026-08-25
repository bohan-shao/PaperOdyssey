# Annotation guidelines (evaluation set)

> Align the criteria of the 3 annotators. Read this document before annotating; record any criterion disagreements in `data/eval/annotation/notes.md`.

## 1. Relevance definition (paper-level)

Given a user input (a research need), decide whether a paper **is relevant**:

| Level | Meaning | Example |
|---|---|---|
| Relevant (pass) | The paper substantially answers at least one aspect of the input (method/conclusion/background of direct value to the need) | The input asks about "LoRA variants"; the paper proposes AdaLoRA and reports comparison experiments |
| Borderline | Only tangentially related or mentioned in passing, not a main contribution | The paper mentions LoRA in one paragraph of related work |
| Not relevant (reject) | Unrelated to the input need | The input asks about efficient fine-tuning of large models; the paper is about image segmentation |

**Decision rules**:
- Judge only by the paper's **actual content** (abstract + body), never guess from the title
- If the abstract suffices, use it; when uncertain, you must check the body paragraphs
- Survey papers: if they systematically cover the input's aspect -> relevant
- Constraints such as time/category are extracted from the input (e.g. "after 2023") -> apply them the same way during annotation; a paper whose year does not match -> not relevant

## 2. Passage-level annotation (subset of 100-200 items)

- Within relevant papers, mark the most relevant passages that **directly support** the input need (up to 5 passages/item)
- Graded relevance (SCHEMA.md §7 graded): 2=directly answers the aspect; 1=partially relevant; 0=not relevant

## 3. Writing gold answers (subset or full set)

- Answer the input's core question in 2-4 sentences; may cite papers (writing the paper identifier is enough)
- Style: objective, information-dense; length 50-150 words

## 4. Agreement requirements and workflow

- 3 annotators annotate independently, **no discussion during annotation** (discussion only in the disagreement arbitration stage)
- After every 100-item batch: compute percentage agreement + Fleiss' kappa; batches below 80% enter arbitration
- Arbitration: the three annotators review the evidence for disputed items together (full text/paragraphs) -> reach agreement -> update the guidelines
- Final: majority vote finalizes

## 5. Common disagreements and rulings

| Disagreement type | Ruling principle |
|---|---|
| Borderline vs. relevant | Check whether it is a "substantial contribution" rather than a passing mention; if unclear, downgrade to not relevant (conservative) |
| Whether a survey is relevant | Relevant if it covers the input's aspect, even without new methods |
| Differing interpretation of time constraints | Take the input text literally ("last two years" = publication year >= current year - 2) |
