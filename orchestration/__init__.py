"""The orchestration package: agentic orchestration layer (Q3/Q5 innovation, D9).

Submodules:
- types:               QueryPlan / ReviewRecord / FinalResult type contracts (SCHEMA.md §4–§6)
- query_understanding: LLM splits the user input into sub-queries + keywords + requirement criteria
- review:              LLM batched review of candidate papers against "fixed criteria + input-extracted criteria"
- rewrite:             Generates supplementary queries based on missing aspects from review
- loop:                asyncio parallel loop control (producer-consumer + termination conditions)

Constraints: retrieval is only invoked through retrieval.pipeline.retrieve(); all LLM calls go
through llm.client (this package must not instantiate a raw OpenAI client).
"""

from __future__ import annotations

from .errors import OrchestrationError
from .loop import run_query

__all__ = ["OrchestrationError", "run_query"]
