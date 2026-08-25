"""OpenAI-compatible LLM client wrapper (the only LLM entry point in the whole system). TODO.

Implementation requirements (see llm/README.md for details):
- Configuration: LLM_BASE_URL / LLM_MODEL / DEEPSEEK_API_KEY from `.env` (OpenAI-compatible).
- `get_client()` returns a process-level singleton LLMClient.
- Methods:
  - chat(messages, temperature, max_tokens) -> str
  - chat_json(messages, schema_desc) -> dict | list
    (requires the model to output JSON; on parse failure retry per config llm.retries;
    schema_desc is only a hint for the prompt, no external JSON-schema dependency)
  - stream_chat(messages) -> Iterator[str] (UI streaming)
- Unified: tenacity retries (429/timeout/5xx, exponential backoff),
  token usage accounting (accumulated and written to data/results/metrics/llm_usage.jsonl).
- Constraint: other modules must not import the openai client directly; evaluation scripts also go through this module.

Raises:
    LLMError: When retries are exhausted or authentication fails (the business layer decides the fallback strategy).
"""

from __future__ import annotations

from collections.abc import Iterator


class LLMClient:
    """OpenAI-compatible LLM client."""

    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 1024) -> str:
        """Single-turn text generation.

        Args:
            messages: OpenAI message format [{role, content}, ...].
            temperature: Sampling temperature.
            max_tokens: Maximum output length.

        Returns:
            The model's output text.
        """
        raise NotImplementedError  # TODO: implement, see llm/README.md

    def chat_json(self, messages: list[dict], schema_desc: str) -> dict | list:
        """A chat call with enforced JSON output.

        Args:
            messages: The message list (the prompt already contains the JSON format requirements).
            schema_desc: A description of the expected JSON structure (used for retry hints).

        Returns:
            The parsed dict or list.

        Raises:
            LLMError: After repeated parse failures.
        """
        raise NotImplementedError  # TODO: implement, see llm/README.md

    def stream_chat(self, messages: list[dict], temperature: float = 0.2) -> Iterator[str]:
        """Streaming generation (for the UI).

        Yields:
            Incremental text chunks.
        """
        raise NotImplementedError  # TODO: implement, see llm/README.md


def get_client() -> LLMClient:
    """Return the process-level singleton client."""
    raise NotImplementedError  # TODO: implement, see llm/README.md
