"""Prompt templates (centrally managed; the report cites template versions). TODO.

Template list (used by the orchestration layer's own LLM calls):
- build_understanding_prompt(input_text)      → query understanding
- build_review_prompt(candidates, criteria, input_text) → review
- build_rewrite_prompt(missing, existing)     → feedback-based rewriting

Implementation requirements:
- All templates write the output convention into the prompt (JSON structure description), working with llm/client.chat_json.
- Template version: module-level constant TEMPLATE_VERSION, recorded in the report/result files.
- (Task-specific templates — answer generation / faithfulness self-check — are defined later with the downstream task.)

Constraint: template text lives in the prompts dictionary (module level); it must not be scattered across business files.
"""

from __future__ import annotations

TEMPLATE_VERSION = "0.1.0"


def build_understanding_prompt(input_text: str) -> list[dict]:
    """Build the query understanding messages. Returns an OpenAI message list."""
    raise NotImplementedError  # TODO: implement, see llm/README.md


def build_review_prompt(candidates: list[dict], criteria: list[dict], input_text: str) -> list[dict]:
    """Build the review messages. candidates/criteria fields are described in orchestration/README.md."""
    raise NotImplementedError  # TODO: implement, see llm/README.md


def build_rewrite_prompt(missing: list[str], existing: list[str]) -> list[dict]:
    """Build the feedback-based rewriting messages."""
    raise NotImplementedError  # TODO: implement, see llm/README.md

