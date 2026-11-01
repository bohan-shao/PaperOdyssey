"""Human audit of Paper QA answers (Q4 answer quality, D54).

The eval set has no reference answers (`gold_answer` is empty), so the D49 answer metrics say
whether the source paper was supplied or cited and what an LLM judge thinks, not whether an answer
is right. This audit asks people. `write_sheets` samples the answers of one finished answer-run
row (`answer_runs/<tag>/rows/<row>/`), gives each to `per_answer` annotators, and writes per
annotator a marking sheet (CSV) and a reading packet (HTML: the question, the answer and every
source excerpt the answer model saw, cited ones marked). `score_sheets` reads the marked sheets.

Strata come from the LLM judge and are hidden from annotators: `perfect` (answered,
faithfulness 1), `mostly` (answered, 0.8 <= faithfulness < 1), `low` (answered, faithfulness
below 0.8 or no claims) and `no_answer` (insufficient evidence). Low strata are oversampled;
overall rates are weighted back by each stratum's share of the row (post-stratification), so they
estimate the whole row, and the per-stratum rates show whether the judge tracks people.

Marks per criterion (eval/annotation/ANSWER_AUDIT.md): y = 1, p (partly) = 0.5, n = 0.
- correct:   what the answer says about the question is right (checked against the excerpts)
- complete:  it answers every part of the question
- supported: every cited statement is backed by the excerpt it cites
Agreement: exact agreement and Fleiss' kappa (categories y/p/n) per criterion. Sampling and
assignment are deterministic in the seed: sha256("<seed>:<query_id>") orders each stratum, and
the annotator pairs cycle over the sample in a second hash order, as in D42.
"""

from __future__ import annotations

import csv
import hashlib
import html
import itertools
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from .annotation.iaa import _fleiss

CRITERIA = ("correct", "complete", "supported")
MARKS = {"y": 1.0, "yes": 1.0, "p": 0.5, "partly": 0.5, "n": 0.0, "no": 0.0}
STRATA = ("perfect", "mostly", "low", "no_answer")
COLUMNS = ["audit_id", "query_id", "question", *CRITERIA, "note"]
MANIFEST = "answer_audit.json"


def stratum(row: Mapping) -> str:
    """The judge stratum of one answers.jsonl row."""
    if row["status"] != "answered":
        return "no_answer"
    faithfulness = row["faithfulness"]
    if faithfulness is None or faithfulness < 0.8:
        return "low"
    return "perfect" if faithfulness == 1 else "mostly"


def _hash(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def sample(rows: Sequence[Mapping], sizes: Mapping[str, int], seed: int) -> list[tuple[str, Mapping]]:
    """(stratum, row) pairs: the first `sizes[s]` rows of each stratum in hash order.

    A stratum smaller than its size gives all its rows; the shortfall goes to the other strata
    in STRATA order while they have rows left, so the sample keeps its total size if it can.
    """
    by_stratum = {s: sorted((r for r in rows if stratum(r) == s), key=lambda r: _hash(seed, r["query_id"]))
                  for s in STRATA}
    take = {s: min(int(sizes.get(s, 0)), len(by_stratum[s])) for s in STRATA}
    spare = sum(int(sizes.get(s, 0)) for s in STRATA) - sum(take.values())
    for s in STRATA:
        extra = min(spare, len(by_stratum[s]) - take[s])
        take[s] += extra
        spare -= extra
    return [(s, row) for s in STRATA for row in by_stratum[s][:take[s]]]


def _packet(annotator: str, items: Sequence[dict]) -> str:
    """A self-contained HTML page with everything needed to mark one annotator's answers."""
    e = html.escape
    parts = [f"<!doctype html><meta charset='utf-8'><title>Answer audit {e(annotator)}</title>",
             ("<style>body{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px}"
              "section{border-top:2px solid #888;margin-top:28px}blockquote{margin:6px 0 12px 16px;"
              "padding-left:10px;border-left:3px solid #ccc;white-space:pre-wrap}.cited{font-weight:600}</style>"),
             f"<h1>Answer audit, annotator {e(annotator)}: {len(items)} answers</h1>",
             (f"<p>Mark each answer in <code>answer_audit_{e(annotator)}.csv</code> (same audit IDs): "
              "<b>correct</b>, <b>complete</b>, <b>supported</b> = y / p / n. Rules and examples: "
              "eval/annotation/ANSWER_AUDIT.md. Judge only from the question, the answer and these "
              "excerpts (open the paper link when an excerpt is cut off).</p>")]
    for item in items:
        cited = set(item["cited_source_numbers"])
        parts += [f"<section><h2>{e(item['audit_id'])}</h2>",
                  f"<p><b>Question:</b> {e(item['question'])}</p>",
                  f"<p><b>Answer</b> ({e(item['status'])}):</p><blockquote>{e(item['answer'])}</blockquote>",
                  "<p><b>Sources the answer model saw</b> (cited ones in bold):</p><ol>"]
        for source in item["sources"]:
            mark = " class='cited'" if source["number"] in cited else ""
            link = f" <a href='{e(source['url'])}'>paper</a>" if source.get("url") else ""
            parts.append(f"<li value='{source['number']}'{mark}>[{source['number']}] {e(source['title'])} "
                         f"({source.get('year') or 'n.d.'}), section: {e(source.get('section') or '-')}{link}"
                         f"<blockquote>{e(source['text'])}</blockquote></li>")
        parts.append("</ol></section>")
    return "\n".join(parts) + "\n"


def write_sheets(row_dir: str | Path, out_dir: str | Path, annotators: Sequence[str], sizes: Mapping[str, int],
                 *, per_answer: int = 2, split: str = "eval", seed: int = 42) -> dict:
    """Sample one answer-run row, write the sheets, packets and manifest; return the manifest.

    Refuses an `out_dir` that already holds a manifest, so marked sheets are never overwritten.
    """
    row_dir, out_dir = Path(row_dir), Path(out_dir)
    if len(set(annotators)) != len(annotators) or not 2 <= per_answer <= len(annotators):
        raise ValueError("need distinct annotators and 2 <= per_answer <= the number of annotators")
    if (out_dir / MANIFEST).exists():
        raise ValueError(f"{out_dir / MANIFEST} exists; score it or choose another folder")
    answers_file = row_dir / "answers.jsonl"
    rows = [json.loads(line) for line in answers_file.open(encoding="utf-8") if line.strip()]
    rows = [row for row in rows if row["split"] == split]
    chosen = sample(rows, sizes, seed)
    if not chosen:
        raise ValueError(f"no {split} answers to sample in {answers_file}")
    chosen.sort(key=lambda pair: _hash(seed, "audit:" + pair[1]["query_id"]))  # IDs do not reveal strata
    pairs = list(itertools.combinations(annotators, per_answer))
    items, packets = [], {a: [] for a in annotators}
    for number, (group, row) in enumerate(chosen, 1):
        checkpoint = json.loads((row_dir / "answers" / f"{row['query_id']}.json").read_text(encoding="utf-8"))
        answer = checkpoint["answer"]
        if answer["answer"] != row["answer"]:
            raise ValueError(f"checkpoint of {row['query_id']} differs from answers.jsonl")
        item = {"audit_id": f"AQ{number:03d}", "query_id": row["query_id"], "stratum": group,
                "annotators": list(pairs[(number - 1) % len(pairs)]),
                "judge": {k: row[k] for k in ("faithfulness", "answer_relevance", "context_relevance",
                                              "context_precision", "gold_in_sources", "gold_cited")}}
        items.append(item)
        for annotator in item["annotators"]:
            packets[annotator].append({"audit_id": item["audit_id"], "query_id": row["query_id"],
                                       "question": row["input_text"], "status": answer["status"],
                                       "answer": answer["answer"], "sources": answer["sources"],
                                       "cited_source_numbers": answer["cited_source_numbers"]})
    out_dir.mkdir(parents=True, exist_ok=True)
    for annotator, assigned in packets.items():
        with (out_dir / f"answer_audit_{annotator}.csv").open("w", encoding="utf-8-sig", newline="") as handle:  # BOM: Excel reads UTF-8
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            writer.writerows([p["audit_id"], p["query_id"], p["question"], "", "", "", ""] for p in assigned)
        (out_dir / f"answer_audit_{annotator}.html").write_text(_packet(annotator, assigned), encoding="utf-8")
    population = {s: sum(stratum(r) == s for r in rows) for s in STRATA}
    manifest = {
        "format_version": 1, "created": datetime.now().isoformat(timespec="seconds"),
        "source": {"row_dir": str(row_dir), "answers_sha256": hashlib.sha256(answers_file.read_bytes()).hexdigest()},
        "split": split, "seed": seed, "per_answer": per_answer, "annotators": list(annotators),
        "requested_sizes": {s: int(sizes.get(s, 0)) for s in STRATA}, "population": population,
        "sampled": {s: sum(i["stratum"] == s for i in items) for s in STRATA},
        "workload": {a: len(packets[a]) for a in annotators}, "items": items,
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _read_marks(out_dir: Path, manifest: Mapping) -> dict[str, dict[str, dict[str, float]]]:
    """{audit_id: {annotator: {criterion: value}}}; every assigned row must be fully marked."""
    marks: dict[str, dict[str, dict[str, float]]] = {item["audit_id"]: {} for item in manifest["items"]}
    expected = {a: {i["audit_id"] for i in manifest["items"] if a in i["annotators"]} for a in manifest["annotators"]}
    query_ids = {i["audit_id"]: i["query_id"] for i in manifest["items"]}
    for annotator in manifest["annotators"]:
        path = out_dir / f"answer_audit_{annotator}.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or list(reader.fieldnames)[:len(COLUMNS)] != COLUMNS:
                raise ValueError(f"{path.name}: columns must be {COLUMNS}")
            seen = set()
            for line, row in enumerate(reader, 2):
                audit_id = (row["audit_id"] or "").strip()
                if not audit_id and not any((v or "").strip() for v in row.values()):
                    continue
                if audit_id not in expected[annotator] or audit_id in seen:
                    raise ValueError(f"{path.name} row {line}: unexpected or repeated audit ID {audit_id!r}")
                if row["query_id"].strip() != query_ids[audit_id]:
                    raise ValueError(f"{path.name} row {line}: query_id changed for {audit_id}")
                values = {}
                for criterion in CRITERIA:
                    mark = (row[criterion] or "").strip().lower()
                    if mark not in MARKS:
                        raise ValueError(f"{path.name} row {line}: {criterion} must be y, p or n, got {mark!r}")
                    values[criterion] = MARKS[mark]
                seen.add(audit_id)
                marks[audit_id][annotator] = values
            if missing := sorted(expected[annotator] - seen):
                raise ValueError(f"{path.name}: {len(missing)} assigned answers not marked, first {missing[0]}")
    return marks


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for k in range(start, end + 1):
            ranks[order[k]] = (start + end) / 2 + 1  # ties share their mean rank
        start = end + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman's rank correlation with mean ranks for ties; None if either side is constant."""
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = _mean(rx), _mean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx, syy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else None


def score_sheets(out_dir: str | Path) -> dict:
    """Agreement, per-stratum and weighted human scores, and the judge comparison."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / MANIFEST).read_text(encoding="utf-8"))
    marks = _read_marks(out_dir, manifest)
    items = manifest["items"]
    population = manifest["population"]
    present = [s for s in STRATA if any(i["stratum"] == s for i in items)]
    weight_total = sum(population[s] for s in present)
    per_item = {i["audit_id"]: {c: _mean([m[c] for m in marks[i["audit_id"]].values()]) for c in CRITERIA}
                for i in items}
    criteria = {}
    for criterion in CRITERIA:
        values = [[m[criterion] for m in marks[i["audit_id"]].values()] for i in items]
        counts = [[sum(v == level for v in vs) for level in (1.0, 0.5, 0.0)] for vs in values]
        by_stratum = {s: _mean([per_item[i["audit_id"]][criterion] for i in items if i["stratum"] == s])
                      for s in present}
        _, kappa = _fleiss(counts)
        criteria[criterion] = {
            "exact_agreement": sum(len(set(vs)) == 1 for vs in values) / len(values),
            "fleiss_kappa": kappa,
            "by_stratum": by_stratum,
            "weighted_mean": sum(population[s] * by_stratum[s] for s in present) / weight_total,
            "sample_mean": _mean([per_item[i["audit_id"]][criterion] for i in items]),
        }
    judged = [i for i in items if i["judge"]["faithfulness"] is not None]
    return {
        "format_version": 1, "created": datetime.now().isoformat(timespec="seconds"),
        "protocol": "y=1, p=0.5, n=0 per criterion; an answer's score is the mean of its annotators; "
                    "weighted_mean weights each judge stratum by its share of the row (population); "
                    "agreement over each answer's annotators (Fleiss' kappa, categories y/p/n); "
                    "annotators judge from the question, the answer and its excerpts, so correct means "
                    "correct with respect to the evidence shown, not checked against the full papers",
        "source": manifest["source"], "split": manifest["split"], "n_answers": len(items),
        "per_answer": manifest["per_answer"], "population": population, "sampled": manifest["sampled"],
        "criteria": criteria,
        "judge_vs_human": {
            "judge_faithfulness_by_stratum": {s: _mean([i["judge"]["faithfulness"] for i in judged
                                                        if i["stratum"] == s]) for s in present},
            "spearman_faithfulness_vs_supported": spearman(
                [i["judge"]["faithfulness"] for i in judged], [per_item[i["audit_id"]]["supported"] for i in judged]),
            "n": len(judged),
            "scope": "the stratified sample (low faithfulness oversampled), not the whole row",
        },
        "per_answer_scores": {i["audit_id"]: {"query_id": i["query_id"], "stratum": i["stratum"],
                                              **per_item[i["audit_id"]], "judge": i["judge"]} for i in items},
    }


def format_summary(report: Mapping) -> str:
    """The headline numbers (the JSON report has everything)."""
    lines = [(f"Answer audit: {report['n_answers']} answers, {report['per_answer']} annotators each "
              f"(sampled {report['sampled']} of {report['population']})"), "",
             f"{'criterion':<11}{'weighted':>9}{'agree':>7}{'kappa':>7}  by stratum"]
    for name, c in report["criteria"].items():
        strata = ", ".join(f"{s} {v:.2f}" for s, v in c["by_stratum"].items())
        kappa = "n/a" if c["fleiss_kappa"] != c["fleiss_kappa"] else f"{c['fleiss_kappa']:.2f}"
        lines.append(f"{name:<11}{c['weighted_mean']:>9.3f}{c['exact_agreement']:>7.0%}{kappa:>7}  {strata}")
    judge = report["judge_vs_human"]
    rho = judge["spearman_faithfulness_vs_supported"]
    lines += ["", "LLM judge faithfulness by stratum: "
                  + ", ".join(f"{s} {v:.2f}" for s, v in judge["judge_faithfulness_by_stratum"].items() if v is not None),
              f"Spearman(judge faithfulness, human supported) = {'n/a' if rho is None else f'{rho:.2f}'} over {judge['n']} answers "
              "(stratified sample, not the whole row)"]
    return "\n".join(lines)
