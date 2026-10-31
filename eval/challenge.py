"""Challenge set: harder questions the agentic search is built for (Q3/Q5, D56).

Most main-set questions (eval_set.jsonl, one source paper each) finish in one search round, so the
loop and the evidence merging of D53 are rarely exercised. This set adds four kinds of question,
drafted by the LLM from KB papers that are not in the main set, and checked by people before use:

- comparison: contrasts two related papers; both are gold. The second paper is often found only
  by a follow-up retrieval.
- two_part:   asks two things one paper answers in different places. A first round often brings
  evidence for one part only (missing evidence), so a later round must bring more of that paper.
- date:       a topic question with an explicit year bound ("since 2021", "in 2018 or earlier")
  the gold paper meets. Tests metadata filtering and whether answers cite papers out of bounds.
- out_of_kb:  a research question from a field the KB does not cover. The right outcome is an
  honest "insufficient evidence" (no gold papers, `answerable` false).

Pipeline: `draft_challenge` (LLM) -> `write_sheets` (each annotator checks every item: fits y/n)
-> `build_challenge_set` (keeps the items every annotator accepted; agreement) ->
challenge_set.jsonl in the eval-set format, run like the main set with `run_eval.py --eval-set`
-> `score_challenge` (per kind and configuration, from a review ablation and its answer run).
Papers are taken in sha256("<seed>:<paper_id>") order; related pairs are the nearest TF-IDF
neighbours (title + abstract) within a cosine band, so they share a problem without being
near-duplicates.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path

from kb import store

from .annotation.iaa import _fleiss
from .annotation.questions import MIN_ABSTRACT_CHARS

KINDS = ("comparison", "two_part", "date", "out_of_kb")
PROMPT_VERSION = "c1"
OUT_OF_KB_FIELDS = ("organic chemistry", "marine biology", "clinical cardiology", "macroeconomics",
                    "constitutional law", "observational astronomy", "structural engineering", "archaeology",
                    "music theory", "soil science", "volcanology", "medieval history", "plant pathology",
                    "sports physiology")
SHEET_COLUMNS = ["challenge_id", "kind", "question", "time_bound", "paper_1_title", "paper_1_year",
                 "paper_1_abstract", "paper_2_title", "paper_2_year", "paper_2_abstract", "fits", "note"]
_YES, _NO = {"y", "yes"}, {"n", "no"}

_ROLE = "You help build hard test questions for a search engine over arXiv papers in NLP, AI and machine learning."
_RULES = ("Rules: one sentence in plain English, as a researcher who has not read the papers would ask it; "
          "describe ideas in your own words, without names the papers introduce (model, method, dataset or "
          "benchmark names), titles, author names or arXiv ids; do not state results; do not say 'this paper'.")
COMPARISON = (f"{_ROLE}\nThe user gives two related papers, A and B. Write one question of 15 to 30 words that "
              "needs BOTH papers to answer well: it contrasts how the two approach a problem they share. "
              f"{_RULES} No years. If the papers do not share a problem a researcher would compare them on, "
              'reply {"skip": true}.\nReply with JSON only: {"question": "...", "aspects": ["what A offers", '
              '"what B offers"]}')
TWO_PART = (f"{_ROLE}\nThe user gives one paper. Write one question of 15 to 30 words that asks two different "
            "things this paper answers in different parts, for example how its approach works and how it is "
            "evaluated or which setting it targets; join the two asks with 'and'. "
            f"{_RULES} No years.\nReply with JSON only: {{\"question\": \"...\", \"parts\": [\"first ask\", "
            "\"second ask\"]}")
DATE = (f"{_ROLE}\nThe user gives one paper and a time bound. Write one question of 12 to 25 words to which this "
        "paper is a clearly good answer and that contains the time bound word for word. "
        f"{_RULES}\nReply with JSON only: {{\"question\": \"...\"}}")
OUT_OF_KB = ("You help test whether a search engine over AI and machine-learning papers admits when it has no "
             "evidence. The user names a field. Write one research question of 12 to 25 words that a scientist "
             "in that field might ask a paper search engine. It must be about the field itself, not about "
             "computing, software, data, statistics, machine learning or AI. One sentence, no names of people "
             'or papers.\nReply with JSON only: {"question": "..."}')
_SCHEMAS = {"comparison": 'object {"question": string, "aspects": [string, string]} or {"skip": true}',
            "two_part": 'object {"question": string, "parts": [string, string]}',
            "date": 'object {"question": string}', "out_of_kb": 'object {"question": string}'}


def _hash(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def candidate_papers(sqlite_path: str | Path, exclude: set[str], seed: int) -> list[dict]:
    """Built papers with an abstract that are not gold in the main set, in hash order."""
    papers = [p for p in store.iter_papers(sqlite_path)
              if p.get("num_chunks") and len((p.get("abstract") or "").strip()) >= MIN_ABSTRACT_CHARS
              and p["paper_id"] not in exclude]
    return sorted(papers, key=lambda p: _hash(seed, p["paper_id"]))


def related_pairs(papers: Sequence[dict], *, pool: int = 4000, low: float = 0.3,
                  high: float = 0.75) -> Iterator[tuple[dict, dict]]:
    """(A, B) pairs in the order of A: B is A's most similar unused paper with low <= cosine <= high."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    pool_papers = list(papers[:pool])
    matrix = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(
        [f"{p['title']}. {p['abstract']}" for p in pool_papers])
    used: set[int] = set()
    for i in range(len(pool_papers)):
        if i in used:
            continue
        sims = (matrix[i] @ matrix.T).toarray().ravel()
        best = max((j for j in range(len(pool_papers)) if j != i and j not in used and low <= sims[j] <= high),
                   key=lambda j: sims[j], default=None)
        if best is not None:
            used.update((i, best))
            yield pool_papers[i], pool_papers[best]


def _brief(paper: Mapping) -> dict:
    return {"title": paper["title"], "abstract": paper["abstract"]}


def _ask(client, kind: str, system: str, payload: dict) -> dict:
    reply = client.chat_json([{"role": "system", "content": system},
                              {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], _SCHEMAS[kind])
    if not isinstance(reply, dict):
        raise ValueError(f"{kind}: the reply is not a JSON object: {reply!r}")
    return reply


def _question(reply: Mapping) -> str | None:
    question = reply.get("question")
    return " ".join(question.split()) if isinstance(question, str) and question.strip() else None


def _record(kind: str, question: str, papers: Sequence[Mapping], source: str, **extra) -> dict:
    return {"kind": kind, "question": question, "gold_paper_ids": [p["paper_id"] for p in papers],
            "papers": [{k: p.get(k) for k in ("paper_id", "title", "year", "abstract")} for p in papers],
            "answerable": kind != "out_of_kb", "question_source": source, "prompt_version": PROMPT_VERSION,
            **extra}


def time_bound(paper: Mapping, seed: int) -> dict:
    """"since <year>" (year_min) for half the papers by hash, else "in <year> or earlier" (year_max)."""
    year = int(paper["year"])
    if int(_hash(seed, "bound:" + paper["paper_id"]), 16) % 2:
        return {"phrase": f"since {year}", "year_min": year}
    return {"phrase": f"in {year} or earlier", "year_max": year}


def draft_challenge(papers: Sequence[dict], client, *, per_kind: int = 10, seed: int = 42, source: str = "llm",
                    pairs: Callable[[Sequence[dict]], Iterator[tuple[dict, dict]]] = related_pairs,
                    max_tries: int = 3) -> list[dict]:
    """`per_kind` questions of each kind, IDs C001... in KINDS order.

    `papers` come in hash order (`candidate_papers`); no paper is used twice. A comparison the LLM
    skips, or a date question without its bound, is replaced by the next candidate, up to
    `max_tries` * `per_kind` candidates per kind.
    """
    used: set[str] = set()
    records: list[dict] = []

    def free(paper: Mapping) -> bool:
        return paper["paper_id"] not in used

    def take(kind: str, candidates: Iterator, make: Callable[[object], dict | None]) -> None:
        made = 0
        for tries, candidate in enumerate(candidates):
            if made == per_kind or tries >= max_tries * per_kind:
                break
            record = make(candidate)
            if record is not None:
                records.append(record)
                used.update(record["gold_paper_ids"])
                made += 1
        if made < per_kind:
            raise ValueError(f"only {made} of {per_kind} {kind} questions could be drafted")

    def comparison(pair: tuple[dict, dict]) -> dict | None:
        a, b = pair
        if not (free(a) and free(b)):
            return None
        reply = _ask(client, "comparison", COMPARISON, {"paper_a": _brief(a), "paper_b": _brief(b)})
        question = _question(reply)
        if reply.get("skip") is True or question is None:
            return None
        return _record("comparison", question, [a, b], source, aspects=reply.get("aspects"))

    def two_part(paper: dict) -> dict | None:
        reply = _ask(client, "two_part", TWO_PART, _brief(paper))
        question = _question(reply)
        return None if question is None else _record("two_part", question, [paper], source, parts=reply.get("parts"))

    def date(paper: dict) -> dict | None:
        bound = time_bound(paper, seed)
        question = _question(_ask(client, "date", DATE, {**_brief(paper), "time_bound": bound["phrase"]}))
        if question is None or bound["phrase"].lower() not in question.lower():
            return None
        return _record("date", question, [paper], source, time_bound=bound)

    def out_of_kb(field: str) -> dict | None:
        question = _question(_ask(client, "out_of_kb", OUT_OF_KB, {"field": field}))
        return None if question is None else _record("out_of_kb", question, [], source, field=field)

    take("comparison", (pair for pair in pairs([p for p in papers if free(p)])), comparison)
    take("two_part", (p for p in papers if free(p)), two_part)
    take("date", (p for p in papers if free(p) and isinstance(p.get("year"), int) and 2017 <= p["year"] <= 2024), date)
    take("out_of_kb", iter(OUT_OF_KB_FIELDS), out_of_kb)
    for number, record in enumerate(records, 1):
        record["challenge_id"] = f"C{number:03d}"
    return [{"challenge_id": r.pop("challenge_id"), **r} for r in records]


def write_sheets(drafts: Sequence[Mapping], sheet_dir: str | Path, annotators: Sequence[str]) -> list[Path]:
    """One CSV per annotator with every item (fits blank). Refuses to overwrite a sheet."""
    sheet_dir = Path(sheet_dir)
    paths = [sheet_dir / f"challenge_{a}.csv" for a in annotators]
    if existing := [p.name for p in paths if p.exists()]:
        raise ValueError(f"sheets exist already: {existing}")
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:  # BOM: Excel reads UTF-8
            writer = csv.writer(handle)
            writer.writerow(SHEET_COLUMNS)
            for d in drafts:
                p = [*d["papers"], {}, {}]
                bound = (d.get("time_bound") or {}).get("phrase", "")
                writer.writerow([d["challenge_id"], d["kind"], d["question"], bound,
                                 p[0].get("title", ""), p[0].get("year", ""), p[0].get("abstract", ""),
                                 p[1].get("title", ""), p[1].get("year", ""), p[1].get("abstract", ""), "", ""])
    return paths


def read_sheet(path: Path, drafts: Mapping[str, Mapping]) -> dict[str, bool]:
    """{challenge_id: fits}; every item marked y or n, IDs and questions unchanged."""
    marks: dict[str, bool] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle), 2):
            cid = (row.get("challenge_id") or "").strip()
            if not cid and not any((v or "").strip() for v in row.values()):
                continue
            if cid not in drafts or cid in marks:
                raise ValueError(f"{path.name} row {line}: unknown or repeated challenge_id {cid!r}")
            if " ".join((row.get("question") or "").split()) != drafts[cid]["question"]:
                raise ValueError(f"{path.name} row {line}: the question of {cid} was changed")
            mark = (row.get("fits") or "").strip().lower()
            if mark not in _YES | _NO:
                raise ValueError(f"{path.name} row {line}: fits must be y or n, got {mark!r}")
            marks[cid] = mark in _YES
    if missing := sorted(set(drafts) - set(marks)):
        raise ValueError(f"{path.name}: {len(missing)} items not marked, first {missing[0]}")
    return marks


def build_challenge_set(drafts: Sequence[Mapping], sheet_dir: str | Path, annotators: Sequence[str],
                        out_path: str | Path) -> dict:
    """Write the items every annotator accepted as eval-set records; return the agreement report."""
    by_id = {d["challenge_id"]: d for d in drafts}
    marks = {a: read_sheet(Path(sheet_dir) / f"challenge_{a}.csv", by_id) for a in annotators}
    accepted = [d for d in drafts if all(marks[a][d["challenge_id"]] for a in annotators)]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for d in accepted:
            handle.write(json.dumps({
                "query_id": d["challenge_id"], "input_text": d["question"], "gold_paper_ids": d["gold_paper_ids"],
                "gold_passage_ids": [], "gold_answer": "", "split": "eval", "question_type": d["kind"],
                "question_form": "question", "question_source": d["question_source"], "round": 1,
                "annotations": [{"annotator": a, "fits": marks[a][d["challenge_id"]]} for a in annotators],
                "majority_paper_ids": d["gold_paper_ids"], "answerable": d["answerable"],
                "time_bound": d.get("time_bound"),
            }, ensure_ascii=False) + "\n")
    votes = [[marks[a][cid] for a in annotators] for cid in by_id]
    _, kappa = _fleiss([[sum(v), len(v) - sum(v)] for v in votes]) if len(annotators) > 1 else (None, None)
    return {
        "format_version": 1, "created": datetime.now().isoformat(timespec="seconds"),
        "annotators": list(annotators), "drafted": len(drafts), "accepted": len(accepted),
        "by_kind": {k: {"drafted": sum(d["kind"] == k for d in drafts), "accepted": sum(d["kind"] == k for d in accepted)}
                    for k in KINDS},
        "percentage_agreement": sum(len(set(v)) == 1 for v in votes) / len(votes),
        "fleiss_kappa": kappa,
        "rejected": sorted(cid for cid in by_id if cid not in {d["challenge_id"] for d in accepted}),
        "out_path": str(out_path),
    }


# ---------------------------------------------------------------- scoring


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _latest_attempts(review_dir: Path) -> dict[str, dict]:
    out = {}
    for path in sorted((review_dir / "attempts").glob("*.json")):
        attempt = json.loads(path.read_text(encoding="utf-8"))
        if attempt.get("status") == "complete":
            out[attempt["question"]["query_id"]] = attempt
    return out


def _in_bound(year: object, bound: Mapping) -> bool:
    if not isinstance(year, int):
        return False  # an undated source cannot be shown to meet the bound
    return bound.get("year_min", year) <= year <= bound.get("year_max", year)


def _question_scores(question: Mapping, attempt: Mapping | None, answer: Mapping | None) -> dict:
    gold = set(question["gold_paper_ids"])
    row: dict = {"kind": question["question_type"], "searched": attempt is not None, "answered_run": answer is not None}
    if attempt is not None:
        result = attempt["result"]
        passing = {p["paper_id"] for p in result["papers"]}
        usage = attempt.get("usage", {})
        row.update(found=len(gold & passing) / len(gold) if gold else None,
                   rounds=result["coverage_report"]["rounds"],
                   re_reviews=result["coverage_report"].get("re_reviews", 0),
                   search_ms=result["latency_by_stage"]["total_ms"],
                   search_tokens=usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
    if answer is not None:
        cited = [s for s in answer["sources"] if s["number"] in set(answer["cited_source_numbers"])]
        cited_ids = {s["paper_id"] for s in cited}
        row.update(status=answer["status"], cited=len(gold & cited_ids) / len(gold) if gold else None,
                   abstained=answer["status"] == "insufficient_evidence")
        if question.get("time_bound"):
            outside = [s for s in cited if not _in_bound(s.get("year"), question["time_bound"])]
            row.update(cited_outside_bound=len(outside), cited_sources=len(cited))
    return row


def _mean(values: Sequence[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _summary(rows: Sequence[Mapping]) -> dict:
    searched = [r for r in rows if r["searched"]]
    answered = [r for r in rows if r["answered_run"]]
    out = {"n": len(rows), "searched": len(searched), "answered_run": len(answered),
           "found": _mean([r["found"] for r in searched]),
           "all_found": _mean([float(r["found"] == 1) for r in searched if r["found"] is not None]),
           "cited": _mean([r["cited"] for r in answered]),
           "all_cited": _mean([float(r["cited"] == 1) for r in answered if r["cited"] is not None]),
           "abstained": _mean([float(r["abstained"]) for r in answered]),
           "rounds": _mean([r["rounds"] for r in searched]), "re_reviews": _mean([r["re_reviews"] for r in searched]),
           "search_ms": _mean([r["search_ms"] for r in searched]),
           "search_tokens": _mean([r["search_tokens"] for r in searched])}
    bounded = [r for r in answered if "cited_sources" in r]
    if bounded:
        out["answers_citing_outside_bound"] = _mean([float(r["cited_outside_bound"] > 0) for r in bounded])
        sources = sum(r["cited_sources"] for r in bounded)
        out["cited_sources_outside_bound"] = sum(r["cited_outside_bound"] for r in bounded) / sources if sources else None
    return out


def score_challenge(searches_dir: str | Path, answers_dir: str | Path | None = None) -> dict:
    """Per row and kind: gold found / cited, abstention, bound violations, rounds, latency, tokens.

    `searches_dir` is a review ablation over the challenge set (`ablation_runs/<tag>`); `answers_dir`
    its answer run (`answer_runs/<tag>`), or None for search-only scores. Questions come from the
    ablation's selected_eval.jsonl; incomplete questions are counted (`searched`, `answered_run`)
    and left out of the means.
    """
    searches_dir = Path(searches_dir)
    questions = read_jsonl(searches_dir / "selected_eval.jsonl")
    if not any("answerable" in q for q in questions):
        raise ValueError(f"{searches_dir} was not run on the challenge set (no 'answerable' field)")
    rows_dir = searches_dir / "rows"
    report: dict = {"format_version": 1, "created": datetime.now().isoformat(timespec="seconds"),
                    "searches": str(searches_dir), "answers": str(answers_dir) if answers_dir else None,
                    "n_questions": len(questions), "by_kind_n": {k: sum(q["question_type"] == k for q in questions)
                                                                 for k in KINDS},
                    "rows": {}}
    plan = searches_dir / "report.json"  # the ablation's row order: baseline, a, b, ab
    order = list(json.loads(plan.read_text(encoding="utf-8"))["rows"]) if plan.exists() else []
    row_dirs = sorted((p for p in rows_dir.iterdir() if p.is_dir()),
                      key=lambda p: (order.index(p.name) if p.name in order else len(order), p.name))
    for row_dir in row_dirs:
        attempts = _latest_attempts(row_dir / "review")
        answers = {}
        if answers_dir is not None:
            path = Path(answers_dir) / "rows" / row_dir.name / "answers.jsonl"
            answers = {answer["query_id"]: answer for answer in read_jsonl(path)}
        per_question = {q["query_id"]: _question_scores(q, attempts.get(q["query_id"]), answers.get(q["query_id"]))
                        for q in questions}
        report["rows"][row_dir.name] = {
            "by_kind": {k: _summary([r for r in per_question.values() if r["kind"] == k]) for k in KINDS},
            "all": _summary(list(per_question.values())),
            "per_question": per_question,
        }
    return report


def format_summary(report: Mapping) -> str:
    """One line per row and kind with the headline measure of that kind."""
    lines = [f"Challenge set: {report['n_questions']} questions {report['by_kind_n']}", ""]
    measures = {"comparison": ("all_found", "all_cited"), "two_part": ("found", "cited"),
                "date": ("found", "answers_citing_outside_bound"), "out_of_kb": ("abstained", None)}

    def show(value: object) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    for row, data in report["rows"].items():
        for kind, (first, second) in measures.items():
            s = data["by_kind"][kind]
            parts = [f"{first} {show(s.get(first))}"] + ([f"{second} {show(s.get(second))}"] if second else [])
            lines.append(f"{row:<9}{kind:<11}n={s['n']:<3}" + ", ".join(parts)
                         + f", rounds {show(s['rounds'])}, re-reviews {show(s['re_reviews'])}, "
                         f"search {show(None if s['search_ms'] is None else s['search_ms'] / 1000)} s")
    return "\n".join(lines)
