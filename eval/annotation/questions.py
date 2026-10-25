"""Evaluation-set questions: LLM-drafted search requests, checked by three annotators (Q4, D39).

The team's procedure (agreed on 2026-10-24):

1. `sample_papers`: N built papers (num_chunks > 0, abstract of at least 200 characters) in a
   fixed random order (`seed`), so a larger N extends a smaller one with the same first papers.
2. `draft_questions`: for each paper the LLM (`llm.client`) writes one search request a
   researcher could type, from the title and abstract only (`INSTRUCTIONS`); one line per question
   in drafts.jsonl. Resumable: query ids already in the file are skipped.
3. `write_sheets`: one CSV per annotator with the same rows (round<R>_part<P>_<annotator>.csv; a
   later call puts the questions without a sheet yet into the next part, so a pilot can come first).
   Each annotator marks alone, without discussing, whether the paper is a good answer to the
   question (`fits` y/n; guidelines.md §6).
4. `build_eval_set`: a question that most annotators accept enters eval_set.jsonl (SCHEMA.md §7)
   with its paper as the gold paper; agreement over every question version marked by all
   annotators goes to iaa_<date>.json; rejected questions go to to_rewrite.csv. A team member
   writes a new question there, `apply_rewrites` adds it to drafts.jsonl as the next round of the
   same query id, all annotators mark it again (`write_sheets` for that round), and the latest
   round of a query counts.

Each question has one gold paper, its source, so a retrieval run is scored by where it ranks that
paper (Hit@K, MRR; eval/retrieval_metrics.py). Another relevant paper in the top K earns nothing,
so the scores are a lower bound. The questions come from the LLM and are checked by people; the
eval set records which (`question_source`).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path

from kb import store

from . import iaa

log = logging.getLogger(__name__)

ANNOTATORS = ("A", "B", "C")
TYPES = ("definitional", "method-comparison", "survey", "paper-locating", "technical-detail")
MIN_ABSTRACT_CHARS = 200
PROMPT_VERSION = "q1"
SHEET_COLUMNS = ["query_id", "round", "question", "paper_title", "year", "abstract", "fits", "note"]
REWRITE_COLUMNS = ["query_id", "round", "paper_title", "abstract", "question", "new_question"]
_YES, _NO = {"y", "yes", "1", "true"}, {"n", "no", "0", "false"}

INSTRUCTIONS = f"""You help build an evaluation set for a search engine over arXiv papers in NLP, AI and machine learning.
The user gives the title and abstract of one paper. Write one search request that a researcher could type into the search engine, to which this paper is a clearly good answer.

Rules:
- One or two sentences, at most 40 words, plain English, in the researcher's own words.
- Ask about the paper's problem, idea, method or finding, specifically enough that this paper clearly fits.
- Write as someone who has not seen the paper: do not copy the title or long phrases of the abstract, and do not use names the paper introduces (its model, method, dataset or benchmark names), author names, arXiv ids or years.
- Do not refer to "this paper" or "the paper".

Also say which kind of request it is: definitional (what a concept is or how it works), method-comparison (how approaches compare), survey (an overview of an area), paper-locating (a specific study or result), technical-detail (how to do something specific).

Reply with JSON only: {{"question": "...", "type": "{'|'.join(TYPES)}"}}"""
_SCHEMA = '{"question": string, "type": one of ' + ", ".join(TYPES) + "}"


# ---------------------------------------------------------------- 1-2: sample and draft


def sample_papers(n: int, seed: int, sqlite_path: str | Path) -> list[dict]:
    """The first n built papers with an abstract, in a fixed random order."""
    if not Path(sqlite_path).exists():
        raise FileNotFoundError(f"no papers database at {sqlite_path}")
    papers = [p for p in store.iter_papers(sqlite_path)  # ordered by paper_id
              if p.get("num_chunks") and len((p.get("abstract") or "").strip()) >= MIN_ABSTRACT_CHARS]
    if n > len(papers):
        raise ValueError(f"{n} papers asked for, {len(papers)} built papers have an abstract")
    random.Random(seed).shuffle(papers)
    return papers[:n]


def _query_id(i: int) -> str:
    return f"Q{i:04d}"


def draft_question(paper: dict, client) -> dict:
    """{"question", "question_type"} for one paper (one chat_json call)."""
    messages = [{"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": json.dumps({"title": paper["title"], "abstract": paper["abstract"]},
                                                       ensure_ascii=False)}]
    reply = client.chat_json(messages, _SCHEMA)
    question = reply.get("question") if isinstance(reply, dict) else None
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{paper['paper_id']}: no question in the reply {reply!r}")
    qtype = str(reply.get("type", "")).strip().lower()
    return {"question": " ".join(question.split()), "question_type": qtype if qtype in TYPES else "other"}


def read_drafts(path: str | Path) -> list[dict]:
    """Every draft line (all rounds), in file order."""
    return store.read_records(path)


def latest_drafts(path: str | Path) -> dict[str, dict]:
    """query_id -> its latest round."""
    latest: dict[str, dict] = {}
    for rec in read_drafts(path):
        if rec["query_id"] not in latest or rec["round"] > latest[rec["query_id"]]["round"]:
            latest[rec["query_id"]] = rec
    return dict(sorted(latest.items()))


def draft_questions(papers: list[dict], drafts_path: str | Path, client=None, workers: int = 8,
                    progress: Callable[[int], None] | None = None) -> dict:
    """Draft one question per paper (query ids Q0001.. in sample order) into drafts.jsonl.

    Returns {"drafted", "skipped", "failed"}; failed papers are logged and drafted by a rerun.
    The first paper is drafted alone and its error is raised (a missing key or a wrong model fails
    once, not for every paper).
    """
    if client is None:
        from llm.client import get_client
        client = get_client()
    drafts_path = Path(drafts_path)
    done = {r["query_id"]: r["paper_id"] for r in read_drafts(drafts_path) if r["round"] == 1}
    todo = []
    for i, paper in enumerate(papers, start=1):
        qid = _query_id(i)
        if qid in done:
            if done[qid] != paper["paper_id"]:
                raise ValueError(f"{drafts_path} has {qid} for {done[qid]}, this sample has {paper['paper_id']}: "
                                 "it was made with another seed or paper set")
            continue
        todo.append((qid, paper))
    result = {"drafted": 0, "skipped": len(papers) - len(todo), "failed": 0}

    def one(item: tuple[str, dict]) -> tuple[str, dict, dict | Exception]:
        qid, paper = item
        try:
            return qid, paper, draft_question(paper, client)
        except Exception as exc:  # noqa: BLE001  one bad reply must not stop the other papers
            return qid, paper, exc

    if not todo:
        return result
    first = one(todo[0])
    if isinstance(first[2], Exception):
        raise first[2]
    drafts_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool, drafts_path.open("a", encoding="utf-8") as fh:
        for qid, paper, out in chain([first], pool.map(one, todo[1:])):  # written as they come, in order
            if isinstance(out, Exception):
                log.warning("%s (%s): %s", qid, paper["paper_id"], out)
                result["failed"] += 1
            else:
                rec = {"query_id": qid, "round": 1, "paper_id": paper["paper_id"], "arxiv_id": paper.get("arxiv_id"),
                       "title": paper["title"], "year": paper.get("year"), "abstract": paper["abstract"], **out,
                       "source": f"llm:{getattr(client, 'model', 'unknown')}", "prompt_version": PROMPT_VERSION}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                result["drafted"] += 1
            if progress:
                progress(1)
    return result


# ---------------------------------------------------------------- 3: marking sheets


def sheets_of(sheet_dir: str | Path, annotator: str) -> list[tuple[int, int, Path]]:
    """(round, part, path) of an annotator's marking sheets, round<R>_part<P>_<annotator>.csv."""
    pattern = re.compile(rf"round(\d+)_part(\d+)_{re.escape(annotator)}\.csv")
    found = [(int(m[1]), int(m[2]), p) for p in Path(sheet_dir).glob("round*.csv") if (m := pattern.fullmatch(p.name))]
    return sorted(found)


def write_sheets(drafts_path: str | Path, sheet_dir: str | Path, round_: int = 1,
                 annotators: tuple[str, ...] = ANNOTATORS) -> list[Path]:
    """Marking sheets, one per annotator with the same rows, for the questions whose latest round
    is `round_` and that are not in a sheet yet (the next part: a pilot first, the rest later).

    Never overwrites a sheet (it may hold marks).
    """
    earlier = [(part, path) for r, part, path in sheets_of(sheet_dir, annotators[0]) if r == round_]
    in_sheets = {key for _, path in earlier for key in read_sheet(path)}
    rows = [d for d in latest_drafts(drafts_path).values()
            if d["round"] == round_ and (d["query_id"], round_) not in in_sheets]
    if not rows:
        raise ValueError(f"no round-{round_} questions without a marking sheet in {drafts_path}")
    part = 1 + max((p for p, _ in earlier), default=0)
    paths = [Path(sheet_dir) / f"round{round_}_part{part}_{a}.csv" for a in annotators]
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise FileExistsError(f"not overwriting marking sheets: {existing}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as fh:  # BOM: Excel opens it as UTF-8
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(SHEET_COLUMNS)
            for d in rows:
                writer.writerow([d["query_id"], d["round"], d["question"], d["title"], d.get("year") or "",
                                 d["abstract"], "", ""])
    return paths


def read_sheet(path: str | Path) -> dict[tuple[str, int], bool | None]:
    """(query_id, round) -> True (fits) / False / None (not marked yet)."""
    marks: dict[tuple[str, int], bool | None] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        for line, row in enumerate(csv.DictReader(fh), start=2):
            value = (row.get("fits") or "").strip().lower()
            if value and value not in _YES | _NO:
                raise ValueError(f"{path}, row {line}: fits must be y or n, not {row.get('fits')!r}")
            marks[(row["query_id"], int(row["round"]))] = (value in _YES) if value else None
    return marks


# ---------------------------------------------------------------- 4: eval set, agreement, rewrites


def _split(query_id: str, seed: int, dev_ratio: float) -> str:
    """dev / eval by a hash of the query id: a query keeps its split when the set grows."""
    h = int(hashlib.sha256(f"{seed}:{query_id}".encode()).hexdigest(), 16) % 10_000
    return "dev" if h < dev_ratio * 10_000 else "eval"


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def build_eval_set(drafts_path: str | Path, sheet_dir: str | Path, eval_set_path: str | Path, iaa_path: str | Path,
                   rewrite_path: str | Path, annotators: tuple[str, ...] = ANNOTATORS, dev_ratio: float = 0.1,
                   seed: int = 42) -> dict:
    """Majority vote, agreement report and the list of questions to rewrite.

    Returns {"questions", "pending", "accepted", "rejected", "duplicates", "dev", "eval", "iaa"}.
    """
    marks: dict[str, dict] = {a: {} for a in annotators}
    for a in annotators:
        for _, _, path in sheets_of(sheet_dir, a):
            marks[a].update(read_sheet(path))

    def labels(d: dict) -> list[bool | None]:
        return [marks[a].get((d["query_id"], d["round"])) for a in annotators]

    marked = [(d, votes) for d in read_drafts(drafts_path) if None not in (votes := labels(d))]
    if not marked:
        raise ValueError("no question is marked by every annotator yet")

    latest = latest_drafts(drafts_path)
    records, rejected, seen = [], [], set()
    summary = {"questions": 0, "pending": 0, "accepted": 0, "rejected": 0, "duplicates": 0, "dev": 0, "eval": 0}
    for qid, d in latest.items():
        summary["questions"] += 1
        votes = labels(d)
        if None in votes:
            summary["pending"] += 1
        elif 2 * sum(votes) <= len(votes):
            summary["rejected"] += 1
            rejected.append(d)
        elif _norm(d["question"]) in seen:  # keep the first of two identical questions
            summary["duplicates"] += 1
        else:
            seen.add(_norm(d["question"]))
            split = _split(qid, seed, dev_ratio)
            summary["accepted"] += 1
            summary[split] += 1
            records.append({
                "query_id": qid, "input_text": d["question"], "gold_paper_ids": [d["paper_id"]],
                "gold_passage_ids": [], "gold_answer": "",
                "annotations": [{"annotator": a, "papers": [d["paper_id"]] if v else [], "answer": ""}
                                for a, v in zip(annotators, votes)],
                "majority_paper_ids": [d["paper_id"]], "split": split,
                "question_type": d["question_type"], "question_source": d["source"], "round": d["round"],
            })

    _write_rewrites(rejected, Path(rewrite_path), latest)  # first: it refuses to drop unapplied new questions
    eval_set_path = Path(eval_set_path)
    eval_set_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    eval_set_path.write_text(text, encoding="utf-8")
    summary["iaa"] = iaa.compute_iaa_report(
        [{"query_id": f"{d['query_id']}/r{d['round']}", "votes": [[d["paper_id"]] if v else [] for v in votes],
          "paper_ids": [d["paper_id"]]} for d, votes in marked],
        str(iaa_path),
        extra={"annotators": list(annotators), "eval_set": eval_set_path.name,
               "eval_set_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "accepted": summary["accepted"]})
    return summary


def _write_rewrites(rejected: list[dict], path: Path, latest: dict[str, dict]) -> None:
    """Write the rejected questions; refuses while the old list has new questions not applied yet."""
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as fh:
            waiting = [row["query_id"] for row in csv.DictReader(fh) if (row.get("new_question") or "").strip()
                       and latest.get(row["query_id"], {}).get("round") == int(row["round"])]
        if waiting:
            raise FileExistsError(f"{path} has new questions not applied yet ({', '.join(waiting[:5])}): "
                                  "run `rewrites` first")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(REWRITE_COLUMNS)
        for d in rejected:
            writer.writerow([d["query_id"], d["round"], d["title"], d["abstract"], d["question"], ""])


def apply_rewrites(rewrite_path: str | Path, drafts_path: str | Path) -> int:
    """Add each filled-in new_question to drafts.jsonl as the next round of its query; returns how many.

    A row whose query already has a later round (applied before) is skipped.
    """
    latest = latest_drafts(drafts_path)
    added = []
    with Path(rewrite_path).open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            new = " ".join((row.get("new_question") or "").split())
            if not new:
                continue
            d = latest.get(row["query_id"])
            if d is None:
                raise ValueError(f"{rewrite_path}: unknown query {row['query_id']}")
            if d["round"] != int(row["round"]):
                continue
            added.append({**d, "round": d["round"] + 1, "question": new, "source": "human",
                          "prompt_version": None})
    with Path(drafts_path).open("a", encoding="utf-8") as fh:
        fh.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in added)
    return len(added)
