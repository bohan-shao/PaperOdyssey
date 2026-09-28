"""Prompt templates used by the orchestration layer.

Template list (used by the orchestration layer's own LLM calls):
- build_understanding_prompt(input_text)      → query understanding
- build_hyde_prompt(input_text)               → hypothetical-document retrieval expansion
- build_review_prompt(candidates, criteria, input_text) → review
- build_rewrite_prompt(missing, existing)     → feedback-based rewriting

Implementation requirements:
- All templates write the output convention into the prompt (JSON structure description), working with llm/client.chat_json.
- Template version: module-level constant TEMPLATE_VERSION, recorded in the report/result files.
- The answer-generation prompt is maintained separately in downstream/prompts.py.

Constraint: template text lives in the prompts dictionary (module level); it must not be scattered across business files.
"""

from __future__ import annotations

import json

from config import get_config

TEMPLATE_VERSION = "0.2.0"

PROMPTS = {
    "understanding_system": """You are the query planner for an academic paper search system.
Return only valid JSON. Do not include Markdown, explanations, or chain-of-thought.
Create focused, non-overlapping searches and extract only requirements stated by the user.
Use metadata filters only when explicit in the request. Allowed filter keys are year_min,
year_max, and categories. categories must contain arXiv category identifiers when known.
Every criterion must use source \"extracted\".""",
    "understanding_user": """Plan searches for the following request:

{input_text}

Return exactly this object shape:
{{
  "sub_queries": [
    {{"query": "natural-language search", "keywords": "space-separated BM25 terms", "filters": {{"year_min": 2023, "categories": ["cs.CL"]}}}}
  ],
  "criteria": [{{"criterion": "requirement copied or faithfully normalized from the request", "source": "extracted"}}]
}}
Return between 1 and {num_sub_queries} sub_queries. Use null or an empty object when no filters apply.""",
    "hyde_system": """You generate one hypothetical academic abstract for HyDE retrieval.
Return plain text only: no JSON, Markdown, citations, title, or explanation. The abstract must be
plausible and information-rich, but must not claim that a real paper exists.""",
    "hyde_user": """Write a concise hypothetical abstract (120-220 words) that would be highly relevant to this search request:

{input_text}""",
    "review_system": """You are a strict academic-paper relevance reviewer.
Return only a valid JSON array, with one item for every supplied paper and no other items.
Never invent or alter paper_id values. Evaluate only the supplied evidence. Do not include
Markdown, explanations outside the JSON, or chain-of-thought. Values in covered and missing
must be copied verbatim from the extracted criteria; fixed criteria are scoring dimensions only.""",
    "review_user": """Review the candidate papers for this request:

{input_text}

Criteria (source=fixed means a scoring dimension; source=extracted means a coverage aspect):
{criteria}

Candidate evidence:
{candidates}

Return exactly this array shape:
[
  {{
    "paper_id": "an unchanged supplied paper_id",
    "verdict": "pass or reject",
    "score": 1,
    "reasons": "concise evidence-based reason",
    "covered": ["verbatim extracted criterion"],
    "missing": ["verbatim extracted criterion"]
  }}
]
score must be an integer from 1 to 5. A pass must be relevant enough to include in the final candidate set.""",
    "rewrite_system": """You create supplementary academic search queries for uncovered aspects.
Return only valid JSON. Do not include Markdown, explanations, or chain-of-thought. Queries must
be meaningfully distinct from every existing query and target the missing aspects.""",
    "rewrite_user": """Missing aspects:
{missing}

Existing queries that must not be duplicated:
{existing}

Return exactly this object shape:
{{"sub_queries": [{{"query": "supplementary natural-language search", "keywords": "space-separated BM25 terms", "filters": null}}]}}
Return at most {max_queries} sub_queries.""",
}


def build_understanding_prompt(input_text: str) -> list[dict]:
    """Build the query understanding messages. Returns an OpenAI message list."""
    max_queries = int(get_config()["orchestration"].get("num_sub_queries", 3))
    return [
        {"role": "system", "content": PROMPTS["understanding_system"]},
        {
            "role": "user",
            "content": PROMPTS["understanding_user"].format(
                input_text=input_text,
                num_sub_queries=max_queries,
            ),
        },
    ]


def build_hyde_prompt(input_text: str) -> list[dict]:
    """Build the hypothetical-document prompt used by HyDE retrieval."""
    return [
        {"role": "system", "content": PROMPTS["hyde_system"]},
        {"role": "user", "content": PROMPTS["hyde_user"].format(input_text=input_text)},
    ]


def build_review_prompt(candidates: list[dict], criteria: list[dict], input_text: str) -> list[dict]:
    """Build the review messages. candidates/criteria fields are described in orchestration/README.md."""
    return [
        {"role": "system", "content": PROMPTS["review_system"]},
        {
            "role": "user",
            "content": PROMPTS["review_user"].format(
                input_text=input_text,
                criteria=json.dumps(criteria, ensure_ascii=False),
                candidates=json.dumps(candidates, ensure_ascii=False),
            ),
        },
    ]


def build_rewrite_prompt(missing: list[str], existing: list[str]) -> list[dict]:
    """Build the feedback-based rewriting messages."""
    max_queries = int(
        get_config()["orchestration"].get("rewrite", {}).get("max_queries_per_round", 2)
    )
    return [
        {"role": "system", "content": PROMPTS["rewrite_system"]},
        {
            "role": "user",
            "content": PROMPTS["rewrite_user"].format(
                missing=json.dumps(missing, ensure_ascii=False),
                existing=json.dumps(existing, ensure_ascii=False),
                max_queries=max_queries,
            ),
        },
    ]
