"""llm package: the LLM layer for the whole system.

Submodules:
- client.py:  the single LLM client (DeepSeek by default, OpenAI-compatible; retries + usage accounting)
- prompt.py:  prompt templates (query understanding / review / rewriting for the orchestration layer)

Every LLM call in the system goes through `llm.client` (orchestration's understanding/review/rewrite,
and later the downstream task). Do not instantiate a raw OpenAI client anywhere else.
"""

from __future__ import annotations

from .client import LLMClient, get_client

__all__ = ["LLMClient", "get_client"]
