"""Batch-review candidate papers against fixed and request-specific criteria.

Implementation requirements (see orchestration/README.md for details, D10):
- Fixed rubric items come from config `orchestration.review.rubric` (default 5 items):
  relevance to the requirement / coverage of input aspects / recency / source credibility / technical depth.
- `batch_review(candidates, criteria, input_text, round) -> list[ReviewRecord]`:
  - config.review.batch_size papers per batch (default 5); each paper is fed title + truncated abstract + top 2–3 truncated passages
    (token budget from config `review.max_tokens_per_paper`)
  - Requires a JSON array output (verdict/score/reasons/covered/missing), retry once on parse failure
- Deduplication: by paper_id (a paper hit by multiple paths in a round is reviewed once; a later round may re-review
  it with passages not seen before, D53); deduplication happens at enqueue time (handled by loop),
  this module only reviews the batch it is given.
- verdict="reject" papers do not enter the candidate set; covered/missing are aggregated into the coverage_report (used by loop).
- Record the review-stage latency.
- Reviewer backend (config `review.backend`): "llm" (default, the prompt above) or "jev"
  (TypeSafe Jev via llm/jev.py, one call per paper: pass/reject as a choice question, the
  1-5 score as a score question, each extracted criterion as a yes/no question). Jev cannot
  write text, so its `reasons` are a short summary of its answers; the probabilities are kept
  in `evidence["jev"]` for the Q5 comparison of the two reviewers.

Constraints: does not call retrieval directly; input comes from retrieval hits aggregated by loop.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from config import get_config
from llm.client import get_client
from llm.jev import JevError, choice, get_jev_client, noul, score
from llm.prompt import build_review_prompt

from .errors import OrchestrationError
from .types import Criterion, ReviewRecord

_SCHEMA_DESC = (
    "array with exactly one object per candidate: paper_id, verdict ('pass'|'reject'), "
    "integer score 1..5, reasons, covered, missing"
)


_BACKENDS = {"llm", "jev"}

# Jev score levels, lowest first; level i maps to review score i + 1.
_JEV_SCORE_LEVELS = [
    "Not useful for this requirement",
    "Marginally useful: touches the topic but misses what the user needs",
    "Partly useful: relevant to some of the requirement",
    "Useful: relevant and addresses most of the requirement",
    "Highly useful: directly addresses the requirement with solid technical content",
]


def _normalise_candidate(candidate: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    paper_id = candidate.get("paper_id")
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise OrchestrationError("review candidate is missing paper_id")

    # A portable approximation keeps the prompt bounded without adding a tokenizer dependency.
    char_budget = max(400, max_tokens * 4)
    title = str(candidate.get("title") or "").strip()
    abstract = str(candidate.get("abstract") or "").strip()
    passages: list[dict[str, str]] = []
    for index, passage in enumerate(candidate.get("top_passages") or []):
        if isinstance(passage, dict):
            passage_id = str(passage.get("passage_id") or f"{paper_id}_passage_{index + 1}")
            text = str(passage.get("text") or "")
        else:
            passage_id = f"{paper_id}_passage_{index + 1}"
            text = str(passage)
        passages.append({"passage_id": passage_id, "text": text.strip()})

    passages = passages[:3]
    title_budget = min(len(title), max(80, char_budget // 10))
    remaining = max(0, char_budget - title_budget)
    abstract_budget = min(len(abstract), remaining // 2)
    remaining -= abstract_budget
    per_passage = remaining // max(1, len(passages))
    return {
        "paper_id": paper_id.strip(),
        "title": title[:title_budget],
        "abstract": abstract[:abstract_budget],
        "top_passages": [
            {"passage_id": item["passage_id"], "text": item["text"][:per_passage]}
            for item in passages
        ],
        "year": candidate.get("year"),
        "categories": candidate.get("categories") or [],
    }


def _validate_reviews(
    raw: object,
    candidates: list[dict[str, Any]],
    criteria: list[Criterion],
    round_no: int,
) -> list[ReviewRecord]:
    if not isinstance(raw, list):
        raise ValueError("review response must be a JSON array")
    candidate_by_id = {candidate["paper_id"]: candidate for candidate in candidates}
    if len(raw) != len(candidate_by_id):
        raise ValueError("review response must contain exactly one item per candidate")

    extracted = {
        criterion.criterion for criterion in criteria if criterion.source == "extracted"
    }
    records: list[ReviewRecord] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each review must be an object")
        paper_id = item.get("paper_id")
        if paper_id not in candidate_by_id or paper_id in seen:
            raise ValueError("review contains an unknown or duplicate paper_id")
        seen.add(paper_id)
        verdict = item.get("verdict")
        score = item.get("score")
        if verdict not in {"pass", "reject"}:
            raise ValueError("verdict must be pass or reject")
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError("score must be an integer from 1 to 5")
        reasons = item.get("reasons", "")
        if not isinstance(reasons, str):
            raise ValueError("reasons must be a string")

        covered = item.get("covered", [])
        missing = item.get("missing", [])
        if not isinstance(covered, list) or not all(isinstance(value, str) for value in covered):
            raise ValueError("covered must be an array of strings")
        if not isinstance(missing, list) or not all(isinstance(value, str) for value in missing):
            raise ValueError("missing must be an array of strings")
        if not set(covered).issubset(extracted) or not set(missing).issubset(extracted):
            raise ValueError("covered and missing values must copy extracted criteria verbatim")

        evidence = _evidence(candidate_by_id[paper_id])
        records.append(
            ReviewRecord(
                paper_id=paper_id,
                verdict=verdict,
                score=score,
                reasons=reasons.strip(),
                covered=list(dict.fromkeys(covered)),
                missing=list(dict.fromkeys(missing)),
                round=round_no,
                evidence=evidence,
            )
        )
    return records


def _evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": candidate["title"],
        "abstract_snippet": candidate["abstract"],
        "top_passages": [passage["passage_id"] for passage in candidate["top_passages"]],
    }


def _jev_questions(rubric: list[str], aspects: list[str]) -> dict[str, dict[str, Any]]:
    dimensions = ", ".join(item.replace("_", " ") for item in rubric) or "relevance"
    questions = {
        "verdict": choice(
            "Should this paper be kept as a candidate for the user's research requirement? "
            f"Judge it on: {dimensions}.",
            {
                "pass": "Relevant to the requirement and worth showing to the user",
                "reject": "Off-topic, too weak, or does not help with the requirement",
            },
        ),
        "score": score(
            f"Overall, how useful is this paper for the user's research requirement? Consider: {dimensions}.",
            _JEV_SCORE_LEVELS,
        ),
    }
    for index, aspect in enumerate(aspects):
        questions[f"aspect_{index}"] = noul(
            f"Does this paper address this part of the requirement: {aspect}?",
            true="The paper directly addresses it",
            false="The paper does not address it",
        )
    return questions


def _jev_review(
    candidates: list[dict[str, Any]],
    rubric: list[str],
    criteria: list[Criterion],
    input_text: str,
    round_no: int,
    cfg: dict[str, Any],
) -> list[ReviewRecord]:
    """Review each paper with Jev: typed answers instead of a JSON reply to parse."""
    jev_cfg = cfg.get("jev", {})
    pass_threshold = float(jev_cfg.get("pass_threshold", 0.5))
    covered_threshold = float(jev_cfg.get("covered_threshold", 0.5))
    aspects = list(dict.fromkeys(c.criterion for c in criteria if c.source == "extracted"))
    questions = _jev_questions(rubric, aspects)
    client = get_jev_client()
    records: list[ReviewRecord] = []
    for candidate in candidates:
        state = {
            "research_requirement": input_text,
            "paper": {
                "title": candidate["title"],
                "abstract": candidate["abstract"],
                "year": candidate["year"],
                "categories": candidate["categories"],
                "key_passages": [passage["text"] for passage in candidate["top_passages"]],
            },
        }
        try:
            answers = client.ask(state, questions)
        except JevError as exc:
            raise OrchestrationError(f"Jev review failed for {candidate['paper_id']}: {exc}") from exc

        verdict_answer = answers["verdict"]
        probabilities = verdict_answer.get("probabilities") or {}
        p_pass = float(probabilities.get("pass", 1.0 if verdict_answer["choice"] == "pass" else 0.0))
        verdict = "pass" if p_pass >= pass_threshold else "reject"
        raw_score = float(answers["score"]["score"])
        review_score = min(5, max(1, int(raw_score + 0.5) + 1))
        aspect_probs = {aspect: float(answers[f"aspect_{i}"]["noul"]) for i, aspect in enumerate(aspects)}
        covered = [aspect for aspect, prob in aspect_probs.items() if prob >= covered_threshold]
        missing = [aspect for aspect, prob in aspect_probs.items() if prob < covered_threshold]

        reasons = f"Jev: {verdict} (pass probability {p_pass:.2f}), score {review_score}/5."
        if covered:
            reasons += " Covers: " + "; ".join(covered) + "."
        if missing:
            reasons += " Not covered: " + "; ".join(missing) + "."
        evidence = _evidence(candidate)
        evidence["jev"] = {
            "pass_probability": round(p_pass, 4),
            "verdict_confidence": verdict_answer.get("confidence"),
            "score_raw": round(raw_score, 4),
            "score_confidence": answers["score"].get("confidence"),
            "aspect_probabilities": {aspect: round(prob, 4) for aspect, prob in aspect_probs.items()},
        }
        records.append(
            ReviewRecord(
                paper_id=candidate["paper_id"],
                verdict=verdict,
                score=review_score,
                reasons=reasons,
                covered=covered,
                missing=missing,
                round=round_no,
                evidence=evidence,
            )
        )
    return records


def batch_review(
    candidates: list[dict],
    criteria: list[Criterion],
    input_text: str,
    round_no: int,
) -> list[ReviewRecord]:
    """Batch-review candidate papers.

    Args:
        candidates: [{paper_id, title, abstract, top_passages: [str], year, categories}].
        criteria: The merged list of fixed + extracted criteria.
        input_text: The user's original input (one of the review bases).
        round_no: The current round number.

    Returns:
        One ReviewRecord per paper (verdict/score/reasons/covered/missing).
    """
    if not candidates:
        return []
    if round_no < 1:
        raise ValueError("round_no must be at least 1")

    cfg = get_config()["orchestration"].get("review", {})
    backend = str(cfg.get("backend", "llm")).strip().lower()
    if backend not in _BACKENDS:
        raise OrchestrationError(f"unknown orchestration.review.backend {backend!r} (use 'llm' or 'jev')")
    batch_size = max(1, int(cfg.get("batch_size", 5)))
    max_tokens = max(100, int(cfg.get("max_tokens_per_paper", 1500)))
    # First-attempt output limit of one review call (D58); None keeps the client's `llm.max_tokens`.
    output_limit = int(cfg["max_output_tokens"]) if cfg.get("max_output_tokens") else None
    rubric = cfg.get("rubric", [])
    fixed = [Criterion(str(item), source="fixed") for item in rubric]
    merged: list[Criterion] = []
    seen_criteria: set[tuple[str, str]] = set()
    for criterion in [*fixed, *criteria]:
        key = (criterion.criterion, criterion.source)
        if key not in seen_criteria:
            seen_criteria.add(key)
            merged.append(criterion)

    normalised = [_normalise_candidate(candidate, max_tokens) for candidate in candidates]
    if backend == "jev":
        return _jev_review(normalised, [str(item) for item in rubric], criteria, input_text, round_no, cfg)
    client = get_client()
    records: list[ReviewRecord] = []
    for offset in range(0, len(normalised), batch_size):
        batch = normalised[offset : offset + batch_size]
        messages = build_review_prompt(
            batch,
            [asdict(criterion) for criterion in merged],
            input_text,
        )
        last_error: Exception | None = None
        for _ in range(2):
            try:
                raw = client.chat_json(messages, _SCHEMA_DESC, max_tokens=output_limit)
                records.extend(_validate_reviews(raw, batch, criteria, round_no))
                last_error = None
                break
            except Exception as exc:  # Retry schema and transport failures once at this boundary.
                last_error = exc
        if last_error is not None:
            raise OrchestrationError(
                f"paper review failed after retry: {last_error}"
            ) from last_error
    return records
