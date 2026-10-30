"""Evaluation-set questions: LLM-drafted search requests, checked by people (Q4, D39, D42).

The team's procedure (agreed on 2026-10-24):

1. `sample_papers`: N built papers (num_chunks > 0, abstract of at least 200 characters), ordered
   by a hash of the seed and the paper id. A larger N extends a smaller one, and a paper added to or
   removed from the KB leaves the order of the others unchanged.
2. `draft_questions`: for each paper the LLM (`llm.client`) writes one search request a
   researcher could type, from the title and abstract only (`INSTRUCTIONS`); one line per question
   in drafts.jsonl, query ids Q0001.. in the order written. Resumable: papers already in the file
   are skipped, so a rerun drafts the ones that failed and a larger N adds new ones.
3. `write_sheets`: one CSV per annotator (round<R>_part<P>_<annotator>.csv) for the questions not
   handed out yet; round<R>_part<P>.json records which questions went to whom, so a pilot can come
   first and the rest later. Every question goes to all annotators, or to `per_question` of them
   balanced over the groups (D42: the pilot to 3, the rest to 2 of 5). Each annotator marks
   alone, without discussing, whether the paper is a good answer to the question (`fits` y/n;
   guidelines.md §6). An evenly split question goes to one more annotator (`write_tiebreak_sheets`).
4. `build_eval_set`: a question that most annotators accept enters eval_set.jsonl (SCHEMA.md §7)
   with its paper as the gold paper; agreement over every question version marked by all
   annotators goes to iaa_<date>.json; rejected questions, and a question identical to an earlier
   accepted one, go to to_rewrite.csv. A team member writes a new question there,
   `apply_rewrites` adds it to drafts.jsonl as the next round of the same query id, all
   annotators mark it again (`write_sheets` for that round), and the latest round of a query counts.

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
import re
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import combinations, islice
from pathlib import Path
from typing import TextIO

from kb import store

from . import iaa

log = logging.getLogger(__name__)

ANNOTATORS = ("A", "B", "C")
TYPES = ("definitional", "method-comparison", "survey", "paper-locating", "technical-detail")
MIN_ABSTRACT_CHARS = 200
MAX_FAILURES_IN_A_ROW = 10  # then the key, model or connection is wrong, not one paper
PROMPT_VERSION = "q3"  # pilots of 2026-10-26: q1 allowed two sentences and 40 words, q2 had no forms
SHEET_COLUMNS = ["query_id", "round", "question", "paper_title", "year", "abstract", "fits", "note"]
REWRITE_COLUMNS = ["query_id", "round", "reason", "paper_title", "abstract", "question", "new_question", "new_type"]
_YES, _NO = {"y", "yes", "1", "true"}, {"n", "no", "0", "false"}

INSTRUCTIONS = f"""You help build an evaluation set for a search engine over arXiv papers in NLP, AI and machine learning.
The user gives the title and abstract of one paper. Write one search request that a researcher could type into the search engine, to which this paper is a clearly good answer.
The user also gives the form to write: "question" (a question, e.g. starting with How, What or Which) or "request" (a search request, e.g. "Papers on ..." or "Looking for work that ...").

Rules:
- One sentence of 12 to 25 words that asks for one thing, in plain English and the researcher's own words.
- Ask about the paper's problem, idea or method, specifically enough that this paper clearly fits.
- Do not state the paper's results or conclusions: the researcher does not know them yet.
- Write as someone who has not seen the paper: do not copy the title or long phrases of the abstract, and do not use names the paper introduces (its model, method, dataset or benchmark names), author names, arXiv ids or years.
- Do not refer to "this paper" or "the paper".

Also say which kind of request it is: definitional (what a concept is or how it works), method-comparison (how approaches compare), survey (an overview of an area), paper-locating (a specific study or result), technical-detail (how to do something specific).

Reply with JSON only: {{"question": "...", "type": "{'|'.join(TYPES)}"}}"""
_SCHEMA = '{"question": string, "type": one of ' + ", ".join(TYPES) + "}"


def _hash(seed: int, key: str) -> int:
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), 16)


def _open_append(path: Path) -> TextIO:
    """Append handle; ends a torn last line (an interrupted write) first, so no record joins it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torn = False
    if path.exists() and path.stat().st_size:
        with path.open("rb") as fh:
            fh.seek(-1, 2)
            torn = fh.read(1) != b"\n"
    fh = path.open("a", encoding="utf-8")
    if torn:
        fh.write("\n")
    return fh


# ---------------------------------------------------------------- 1-2: sample and draft


def sample_papers(n: int, seed: int, sqlite_path: str | Path) -> list[dict]:
    """The first n built papers with an abstract, ordered by a hash of seed and paper id."""
    if not Path(sqlite_path).is_file():
        raise FileNotFoundError(f"no papers database at {sqlite_path}")
    papers = [p for p in store.iter_papers(sqlite_path)
              if p.get("num_chunks") and len((p.get("abstract") or "").strip()) >= MIN_ABSTRACT_CHARS]
    if n > len(papers):
        raise ValueError(f"{n} papers asked for, {len(papers)} built papers have an abstract")
    papers.sort(key=lambda p: _hash(seed, p["paper_id"]))
    return papers[:n]


def _norm_type(value: object) -> str:
    qtype = re.sub(r"[\s_]+", "-", str(value or "").strip().lower())
    return qtype if qtype in TYPES else "other"


def question_form(paper_id: str) -> str:
    """"request" for 40% of the papers (by a hash of the id), else "question", so not every item is a how-question."""
    return "request" if _hash(0, f"form:{paper_id}") % 10 < 4 else "question"


def draft_question(paper: dict, client) -> dict:
    """{"question", "question_type", "question_form"} for one paper (one chat_json call)."""
    form = question_form(paper["paper_id"])
    messages = [{"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": json.dumps({"title": paper["title"], "abstract": paper["abstract"],
                                                        "form": form}, ensure_ascii=False)}]
    reply = client.chat_json(messages, _SCHEMA)
    question = reply.get("question") if isinstance(reply, dict) else None
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{paper['paper_id']}: no question in the reply {reply!r}")
    return {"question": " ".join(question.split()), "question_type": _norm_type(reply.get("type")),
            "question_form": form}


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
    """Draft one question per paper not in drafts.jsonl yet; the next free query id goes to each.

    Returns {"drafted", "skipped", "failed"}. A failed paper is logged and drafted by a rerun;
    after MAX_FAILURES_IN_A_ROW failures in a row the run stops (RuntimeError).
    """
    if client is None:
        from llm.client import get_client
        client = get_client()
    drafts_path = Path(drafts_path)
    existing = read_drafts(drafts_path)
    drafted = {r["paper_id"] for r in existing}
    next_id = 1 + max((int(r["query_id"][1:]) for r in existing), default=0)
    todo = [p for p in papers if p["paper_id"] not in drafted]
    result = {"drafted": 0, "skipped": len(papers) - len(todo), "failed": 0}

    def one(paper: dict) -> dict | Exception:
        try:
            return draft_question(paper, client)
        except Exception as exc:  # noqa: BLE001  one bad reply must not stop the other papers
            return exc

    workers = max(1, workers)
    papers_left = iter(todo)
    in_flight: deque[tuple[dict, Future]] = deque()
    in_a_row = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, _open_append(drafts_path) as fh:

        def submit(count: int) -> None:  # at most 2 x workers calls under way, so a stop wastes few
            for paper in islice(papers_left, count):
                in_flight.append((paper, pool.submit(one, paper)))

        submit(2 * workers)
        while in_flight:  # written as they come, in sample order
            paper, future = in_flight.popleft()
            out = future.result()
            submit(1)
            if progress:
                progress(1)
            if isinstance(out, Exception):
                log.warning("%s: %s", paper["paper_id"], out)
                result["failed"] += 1
                in_a_row += 1
                if in_a_row >= MAX_FAILURES_IN_A_ROW:
                    for _, f in in_flight:
                        f.cancel()
                    raise RuntimeError(f"stopped after {in_a_row} failed LLM calls in a row "
                                       f"(check the key, model and connection); last error: {out}") from out
                continue
            in_a_row = 0
            rec = {"query_id": f"Q{next_id:04d}", "round": 1, "paper_id": paper["paper_id"],
                   "arxiv_id": paper.get("arxiv_id"), "title": paper["title"], "year": paper.get("year"),
                   "abstract": paper["abstract"], **out, "source": f"llm:{getattr(client, 'model', 'unknown')}",
                   "prompt_version": PROMPT_VERSION}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            next_id += 1
            result["drafted"] += 1
    return result


# ---------------------------------------------------------------- 3: marking sheets


def sheets_of(sheet_dir: str | Path, annotator: str) -> list[tuple[int, int, Path]]:
    """(round, part, path) of an annotator's marking sheets, round<R>_part<P>_<annotator>.csv."""
    pattern = re.compile(rf"round(\d+)_part(\d+)_{re.escape(annotator)}\.csv")
    found = [(int(m[1]), int(m[2]), p) for p in Path(sheet_dir).glob("round*.csv") if (m := pattern.fullmatch(p.name))]
    return sorted(found)


def _parts(sheet_dir: str | Path, round_: int) -> list[dict]:
    """The manifests of a round's parts: {"round", "part", "annotators", "query_ids", "assigned"
    (query_id -> annotators; absent in early manifests: everyone), "tiebreak" (optional)}."""
    parts = [json.loads(p.read_text("utf-8")) for p in Path(sheet_dir).glob(f"round{round_}_part*.json")]
    return sorted(parts, key=lambda m: m["part"])


def handed_out(sheet_dir: str | Path, round_: int) -> dict[str, tuple[list[str], list[str]]]:
    """query_id -> (its annotators, its tie-breakers) over the round's parts."""
    out: dict[str, tuple[list[str], list[str]]] = {}
    for m in _parts(sheet_dir, round_):
        assigned = m.get("assigned") or {qid: m["annotators"] for qid in m["query_ids"]}
        for qid, names in assigned.items():
            main, extra = out.setdefault(qid, ([], []))
            (extra if m.get("tiebreak") else main).extend(names)
    return out


def _write_part(sheet_dir: Path, round_: int, rows: list[dict], assigned: dict[str, list[str]],
                annotators: tuple[str, ...], meta: dict) -> list[Path]:
    """One sheet per annotator with their rows, and the part's manifest; never overwrites."""
    part = 1 + max((m["part"] for m in _parts(sheet_dir, round_)), default=0)
    per_annotator = {a: [d for d in rows if a in assigned[d["query_id"]]] for a in annotators}
    paths = {a: sheet_dir / f"round{round_}_part{part}_{a}.csv" for a in annotators if per_annotator[a]}
    manifest = sheet_dir / f"round{round_}_part{part}.json"
    existing = [str(p) for p in [*paths.values(), manifest] if p.exists()]
    if existing:
        raise FileExistsError(f"not overwriting marking sheets: {existing}")
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for a, path in paths.items():
        with path.open("w", encoding="utf-8-sig", newline="") as fh:  # BOM: Excel opens it as UTF-8
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(SHEET_COLUMNS)
            for d in per_annotator[a]:
                writer.writerow([d["query_id"], d["round"], d["question"], d["title"], d.get("year") or "",
                                 d["abstract"], "", ""])
    manifest.write_text(json.dumps({"round": round_, "part": part, "annotators": list(paths), **meta,
                                    "query_ids": [d["query_id"] for d in rows], "assigned": assigned},
                                   indent=1) + "\n", encoding="utf-8")
    return list(paths.values())


def write_sheets(drafts_path: str | Path, sheet_dir: str | Path, round_: int = 1,
                 annotators: tuple[str, ...] = ANNOTATORS, per_question: int | None = None) -> list[Path]:
    """Marking sheets for the questions whose latest round is `round_` and that no earlier part of
    the round handed out (its manifest says which).

    Every question goes to `per_question` annotators (default: all). With fewer than all, the
    questions go to each group of that size in turn (e.g. the 10 pairs of 5 people), so everyone
    gets the same number of rows and every pair shares the same number of questions. Each
    annotator's sheet holds only their rows. Never overwrites a sheet (it may hold marks).
    """
    k = per_question or len(annotators)
    if not 1 <= k <= len(annotators):
        raise ValueError(f"per_question must be between 1 and {len(annotators)}, not {k}")
    done = handed_out(sheet_dir, round_)
    rows = [d for d in latest_drafts(drafts_path).values() if d["round"] == round_ and d["query_id"] not in done]
    if not rows:
        raise ValueError(f"no round-{round_} questions without a marking sheet in {drafts_path}")
    groups = list(combinations(annotators, k))
    assigned = {d["query_id"]: list(groups[i % len(groups)]) for i, d in enumerate(rows)}
    return _write_part(Path(sheet_dir), round_, rows, assigned, annotators, {"per_question": k})


def write_tiebreak_sheets(drafts_path: str | Path, sheet_dir: str | Path, round_: int = 1,
                          annotators: tuple[str, ...] = ANNOTATORS) -> list[Path]:
    """Sheets for the questions of the round whose annotators are split evenly (one y, one n):
    each goes to one more annotator who has not seen it, the one with the fewest such rows so far.
    The tie-breaker's mark decides the vote; agreement is computed without it."""
    marks = {a: read_marks(sheet_dir, a) for a in annotators}
    done = handed_out(sheet_dir, round_)
    latest = latest_drafts(drafts_path)
    load = dict.fromkeys(annotators, 0)
    rows, assigned = [], {}
    for qid, (main, extra) in done.items():
        d = latest.get(qid)
        if d is None or d["round"] != round_ or extra:
            continue
        votes = [marks.get(a, {}).get((qid, round_)) for a in main]
        if None in votes or 2 * sum(votes) != len(votes):
            continue
        candidates = [a for a in annotators if a not in main]
        if not candidates:
            raise ValueError(f"{qid}: every annotator has already marked it")
        pick = min(candidates, key=lambda a: load[a])  # the first of the least loaded, in config order
        load[pick] += 1
        rows.append(d)
        assigned[qid] = [pick]
    if not rows:
        raise ValueError(f"no round-{round_} question is split evenly and waiting for a tie-breaker")
    rows.sort(key=lambda d: d["query_id"])
    return _write_part(Path(sheet_dir), round_, rows, assigned, annotators, {"tiebreak": True})


def read_sheet(path: str | Path) -> dict[tuple[str, int], bool | None]:
    """(query_id, round) -> True (fits) / False / None (not marked yet); blank rows are skipped."""
    marks: dict[tuple[str, int], bool | None] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"query_id", "round", "fits"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            qid = (row["query_id"] or "").strip()
            if not qid:
                continue
            try:
                round_ = int(row["round"])
            except (TypeError, ValueError):
                raise ValueError(f"{path}, row {line}: round must be a number, not {row['round']!r}") from None
            value = (row["fits"] or "").strip().lower()
            if value and value not in _YES | _NO:
                raise ValueError(f"{path}, row {line}: fits must be y or n, not {row['fits']!r}")
            marks[(qid, round_)] = (value in _YES) if value else None
    return marks


def read_marks(sheet_dir: str | Path, annotator: str) -> dict[tuple[str, int], bool]:
    """Every y/n an annotator gave, over all their sheets. A blank cell never erases a mark; two
    different marks for the same question are an error naming both files."""
    marks: dict[tuple[str, int], bool] = {}
    source: dict[tuple[str, int], str] = {}
    for _, _, path in sheets_of(sheet_dir, annotator):
        for key, value in read_sheet(path).items():
            if value is None:
                continue
            if key in marks and marks[key] != value:
                raise ValueError(f"{annotator} marked {key[0]} (round {key[1]}) both "
                                 f"{'y' if marks[key] else 'n'} in {source[key]} and {'y' if value else 'n'} in {path.name}")
            marks[key], source[key] = value, path.name
    return marks


# ---------------------------------------------------------------- 4: eval set, agreement, rewrites


def _split(query_id: str, seed: int, dev_ratio: float) -> str:
    """dev / eval by a hash of the query id: a query keeps its split when the set grows."""
    return "dev" if _hash(seed, query_id) % 10_000 < dev_ratio * 10_000 else "eval"


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def build_eval_set(drafts_path: str | Path, sheet_dir: str | Path, eval_set_path: str | Path, iaa_path: str | Path,
                   rewrite_path: str | Path, annotators: tuple[str, ...] = ANNOTATORS, dev_ratio: float = 0.1,
                   seed: int = 42) -> dict:
    """Majority vote, agreement report and the list of questions to rewrite.

    A question's votes are those of the annotators its sheets went to (`handed_out`), tie-breakers
    included. More y than n accepts it, more n rejects it, an even split waits for a tie-breaker
    (`write_tiebreak_sheets`). Agreement leaves the tie-breakers out (they saw only split
    questions) and is reported per number of annotators, as Fleiss' kappa needs the same number
    on every question: {"by_annotator_count": {"3": {...}, "2": {...}}, ...}.

    Returns {"questions", "pending", "tied", "accepted", "rejected", "duplicates", "dev", "eval", "iaa"}.
    """
    drafts = read_drafts(drafts_path)
    rounds = {d["round"] for d in drafts}
    done = {r: handed_out(sheet_dir, r) for r in rounds}
    names = set(annotators) | {a for r in done.values() for main, extra in r.values() for a in main + extra}
    marks = {a: read_marks(sheet_dir, a) for a in sorted(names)}

    def votes(d: dict) -> tuple[list[tuple[str, bool | None]], list[tuple[str, bool | None]]]:
        main, extra = done[d["round"]].get(d["query_id"], ([], []))
        key = (d["query_id"], d["round"])
        return [(a, marks[a].get(key)) for a in main], [(a, marks[a].get(key)) for a in extra]

    marked = [(d, [v for _, v in main]) for d in drafts if (main := votes(d)[0]) and None not in dict(main).values()]
    if not marked:
        raise ValueError("no question is marked by all its annotators yet")

    latest = latest_drafts(drafts_path)
    records, to_rewrite, seen = [], [], {}
    summary = {"questions": 0, "pending": 0, "tied": 0, "accepted": 0, "rejected": 0, "duplicates": 0,
               "dev": 0, "eval": 0}
    for qid, d in latest.items():
        summary["questions"] += 1
        main, extra = votes(d)
        values = [v for _, v in main + extra]
        if not main or None in values:
            summary["pending"] += 1
        elif 2 * sum(values) == len(values):
            summary["tied"] += 1
        elif 2 * sum(values) < len(values):
            summary["rejected"] += 1
            to_rewrite.append((d, "rejected"))
        elif _norm(d["question"]) in seen:  # the first of two identical questions stays
            summary["duplicates"] += 1
            to_rewrite.append((d, f"same question as {seen[_norm(d['question'])]}"))
        else:
            seen[_norm(d["question"])] = qid
            split = _split(qid, seed, dev_ratio)
            summary["accepted"] += 1
            summary[split] += 1
            records.append({
                "query_id": qid, "input_text": d["question"], "gold_paper_ids": [d["paper_id"]],
                "gold_passage_ids": [], "gold_answer": "",
                "annotations": [{"annotator": a, "papers": [d["paper_id"]] if v else [], "answer": ""} for a, v in main]
                + [{"annotator": a, "papers": [d["paper_id"]] if v else [], "answer": "", "tiebreak": True}
                   for a, v in extra],
                "majority_paper_ids": [d["paper_id"]], "split": split,
                "question_type": d["question_type"], "question_source": d["source"], "round": d["round"],
                "question_form": d.get("question_form"),
            })

    _write_rewrites(to_rewrite, Path(rewrite_path), latest)  # first: it refuses to drop unapplied new questions
    eval_set_path = Path(eval_set_path)
    eval_set_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    eval_set_path.write_text(text, encoding="utf-8")
    by_count: dict[str, list[dict]] = {}
    for d, values in marked:
        by_count.setdefault(str(len(values)), []).append(
            {"query_id": f"{d['query_id']}/r{d['round']}", "votes": [[d["paper_id"]] if v else [] for v in values],
             "paper_ids": [d["paper_id"]]})
    report = {"by_annotator_count": {n: iaa.summarize(items) for n, items in sorted(by_count.items(), reverse=True)},
              "annotators": sorted(names), "eval_set": eval_set_path.name,
              "eval_set_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "accepted": summary["accepted"]}
    iaa_path = Path(iaa_path)
    iaa_path.parent.mkdir(parents=True, exist_ok=True)
    iaa_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["iaa"] = report
    return summary


def _write_rewrites(to_rewrite: list[tuple[dict, str]], path: Path, latest: dict[str, dict]) -> None:
    """Write the questions to rewrite; refuses while the old list has new questions not applied yet."""
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
        for d, reason in to_rewrite:
            writer.writerow([d["query_id"], d["round"], reason, d["title"], d["abstract"], d["question"], "", ""])


def apply_rewrites(rewrite_path: str | Path, drafts_path: str | Path) -> int:
    """Add each filled-in new_question (with new_type, blank = "other") to drafts.jsonl as the
    next round of its query; returns how many. A row applied before is skipped."""
    latest = latest_drafts(drafts_path)
    added = []
    with Path(rewrite_path).open(encoding="utf-8-sig", newline="") as fh:
        for line, row in enumerate(csv.DictReader(fh), start=2):
            new = " ".join((row.get("new_question") or "").split())
            if not new:
                continue
            d = latest.get(row["query_id"])
            if d is None:
                raise ValueError(f"{rewrite_path}, row {line}: unknown query {row['query_id']!r}")
            if d["round"] != int(row["round"]):
                continue
            added.append({**d, "round": d["round"] + 1, "question": new, "question_type": _norm_type(row.get("new_type")),
                          "question_form": None, "source": "human", "prompt_version": None})
    with _open_append(Path(drafts_path)) as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in added)
    return len(added)
