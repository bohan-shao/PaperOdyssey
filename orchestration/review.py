"""Batch-review candidate papers against fixed and request-specific criteria.

Implementation requirements (see orchestration/README.md for details, D10):
- Fixed rubric items come from config `orchestration.review.rubric` (default 5 items):
  relevance to the requirement / coverage of input aspects / recency / source credibility / technical depth.
- `batch_review(candidates, criteria, input_text, round) -> list[ReviewRecord]`:
  - config.review.batch_size papers per batch (default 5); each paper is fed title + truncated abstract + top 2–3 truncated passages
    (token budget from config `review.max_tokens_per_paper`)
  - Requires a JSON array output (verdict/score/reasons/covered/missing), retry once on parse failure
- Deduplication: by paper_id (a paper hit by multiple paths is reviewed only once); deduplication happens at enqueue time (handled by loop),
  this module only reviews the batch it is given.
- verdict="reject" papers do not enter the candidate set; covered/missing are aggregated into the coverage_report (used by loop).
- Record the review-stage latency.

Constraints: does not call retrieval directly; input comes from retrieval hits aggregated by loop.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from config import get_config
from llm.client import get_client
from llm.prompt import build_review_prompt

from .errors import OrchestrationError
from .types import Criterion, ReviewRecord

_SCHEMA_DESC = (
    "array with exactly one object per candidate: paper_id, verdict ('pass'|'reject'), "
    "integer score 1..5, reasons, covered, missing"
)


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

        candidate = candidate_by_id[paper_id]
        evidence = {
            "title": candidate["title"],
            "abstract_snippet": candidate["abstract"],
            "top_passages": [
                passage["passage_id"] for passage in candidate["top_passages"]
            ],
        }
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
    batch_size = max(1, int(cfg.get("batch_size", 5)))
    max_tokens = max(100, int(cfg.get("max_tokens_per_paper", 1500)))
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
                raw = client.chat_json(messages, _SCHEMA_DESC)
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
