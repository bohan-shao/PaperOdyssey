"""OpenAI-compatible LLM client wrapper (the only LLM entry point in the whole system).

- Configuration: `llm.*` in config.yaml (model / base_url / retries / max_tokens / temperature,
  model and base_url come from `LLM_MODEL` / `LLM_BASE_URL` in `.env`). API key from `.env`:
  `DEEPSEEK_API_KEY` for DeepSeek, `OPENAI_API_KEY` for api.openai.com, or `LLM_API_KEY`
  for any other OpenAI-compatible provider (also accepted as an override for the first two).
- `get_client()` returns a process-level singleton `LLMClient`; the SDK client is created
  lazily, so importing / instantiating never needs a key (sanity checks, offline tests).
- Methods:
  - `chat(messages, temperature, max_tokens) -> str`
  - `chat_json(messages, schema_desc) -> dict | list`: parses the reply as JSON (tolerates
    ```json fences and surrounding prose); on a parse failure it re-asks up to `llm.retries`
    times, showing the model its previous reply, the parse error and `schema_desc`; a reply cut
    off by the token limit is retried with a doubled `max_tokens` (up to 8192).
    No provider JSON mode is forced, because callers expect top-level arrays as well as objects.
  - `stream_chat(messages, temperature) -> Iterator[str]` (UI streaming)
- Transport retries (tenacity, exponential backoff): 429 / timeout / connection error / 5xx,
  `llm.retries` extra attempts. Other API errors (400/401/403/404) fail immediately.
  The SDK's own retry loop is disabled so there is exactly one retry policy.
- Usage accounting: every call appends {time, model, kind, prompt_tokens, completion_tokens,
  latency_ms} to `data/results/metrics/llm_usage.jsonl` (cost reporting); running totals via
  `usage_summary()`. Logging failures never fail a call.
- Constraint: other modules must not import the openai client directly; evaluation scripts
  also go through this module.

Raises:
    LLMError: When retries are exhausted, the reply is never valid JSON, the key is missing, or
        authentication fails (the business layer decides the fallback strategy).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import get_config, get_path

if TYPE_CHECKING:
    import openai

_PLACEHOLDER_KEYS = {"", "sk-xxxx"}
_MAX_TOKENS_CAP = 8192
_ECHO_LIMIT = 4000  # characters of a bad JSON reply shown back to the model on retry


class LLMError(RuntimeError):
    """Unrecoverable LLM failure (retries exhausted, invalid JSON, missing key, auth error)."""


def _openai() -> Any:
    """Import the SDK on first use, so `import llm` (and scripts/sanity_check.py) needs only pyyaml."""
    import openai

    return openai


def _transient() -> tuple[type[Exception], ...]:
    sdk = _openai()
    return (sdk.RateLimitError, sdk.APITimeoutError, sdk.APIConnectionError, sdk.InternalServerError)


def _env_key(base_url: str) -> tuple[str, str]:
    """(api_key, name of the env var it came from) for the configured provider."""
    url = base_url.lower()
    if "deepseek" in url:
        names = ["LLM_API_KEY", "DEEPSEEK_API_KEY"]
    elif "api.openai.com" in url:
        names = ["LLM_API_KEY", "OPENAI_API_KEY"]
    else:
        names = ["LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"]
    for name in names:
        value = os.environ.get(name, "").strip().strip('"').strip("'")
        if value not in _PLACEHOLDER_KEYS:
            return value, name
    return "", " / ".join(names)


def parse_json_reply(text: str) -> dict | list:
    """Parse a model reply as JSON: plain JSON, a ```json fenced block, or JSON inside prose.

    Raises:
        ValueError: no JSON object/array can be decoded from the reply.
    """
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0].strip()
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        # JSON embedded in prose: take the longest decodable object/array, so a citation such
        # as "[1]" in the prose never wins over the actual payload.
        value, best_len, i = None, 0, 0
        decoder = json.JSONDecoder()
        while i < len(body):
            if body[i] in "[{":
                try:
                    candidate, end = decoder.raw_decode(body, i)
                except json.JSONDecodeError:
                    pass
                else:
                    if end - i > best_len:
                        value, best_len = candidate, end - i
                    i = end
                    continue
            i += 1
        if value is None:
            raise ValueError("reply contains no JSON object or array") from None
    if not isinstance(value, dict | list):
        raise ValueError(f"expected a JSON object or array, got {type(value).__name__}")
    return value


class LLMClient:
    """OpenAI-compatible LLM client."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        retries: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
        usage_path: str | Path | None = None,
        http_client: Any = None,
        retry_wait: Callable[..., float] | None = None,
    ) -> None:
        cfg = get_config()["llm"]  # also loads .env into os.environ
        self.model = model or str(cfg["model"])
        self.base_url = base_url or str(cfg["base_url"])
        self.retries = max(0, int(cfg.get("retries", 3) if retries is None else retries))
        self.max_tokens = int(cfg.get("max_tokens", 1024) if max_tokens is None else max_tokens)
        self.temperature = float(cfg.get("temperature", 0.2) if temperature is None else temperature)
        self.timeout_s = float(cfg.get("timeout_s", 60) if timeout_s is None else timeout_s)
        self._api_key = api_key
        self._usage_path = Path(usage_path) if usage_path else get_path("results") / "metrics" / "llm_usage.jsonl"
        self._http_client = http_client
        self._retry_wait = retry_wait  # None: exponential 1-30 s, built on the first call
        self._sdk: openai.OpenAI | None = None
        self._lock = threading.Lock()
        self._totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    # ------------------------------------------------------------ internals

    def _client(self) -> openai.OpenAI:
        with self._lock:
            if self._sdk is None:
                key, source = (self._api_key, "api_key") if self._api_key else _env_key(self.base_url)
                if not key:
                    raise LLMError(
                        f"no API key for {self.base_url}: set {source} in .env (see .env.example)"
                    )
                self._sdk = _openai().OpenAI(
                    api_key=key,
                    base_url=self.base_url,
                    timeout=self.timeout_s,
                    max_retries=0,  # tenacity below is the single retry policy
                    http_client=self._http_client,
                )
            return self._sdk

    def _call(self, fn: Callable[[], Any]) -> Any:
        """Run one API call with transport retries; map terminal errors to LLMError."""
        from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

        sdk, transient = _openai(), _transient()
        try:
            for attempt in Retrying(
                retry=retry_if_exception_type(transient),
                stop=stop_after_attempt(self.retries + 1),
                wait=self._retry_wait or wait_exponential(multiplier=1, min=1, max=30),
                reraise=True,
            ):
                with attempt:
                    return fn()
        except sdk.AuthenticationError as exc:
            raise LLMError(f"authentication failed for {self.base_url} (check the API key): {exc}") from exc
        except transient as exc:
            raise LLMError(f"LLM call failed after {self.retries + 1} attempts: {exc}") from exc
        except sdk.APIError as exc:
            raise LLMError(f"LLM request rejected: {exc}") from exc
        raise LLMError("LLM call made no attempt")  # unreachable; keeps type checkers happy

    def _record(self, kind: str, usage: Any, latency_ms: int) -> None:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        with self._lock:
            self._totals["calls"] += 1
            self._totals["prompt_tokens"] += prompt
            self._totals["completion_tokens"] += completion
            try:
                self._usage_path.parent.mkdir(parents=True, exist_ok=True)
                with self._usage_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "model": self.model,
                        "kind": kind,
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "latency_ms": latency_ms,
                    }) + "\n")
            except OSError:
                pass  # usage logging is best-effort

    def _complete(self, messages: list[dict], temperature: float | None, max_tokens: int | None,
                  kind: str) -> tuple[str, str | None]:
        client = self._client()
        started = time.perf_counter()
        resp = self._call(lambda: client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        ))
        self._record(kind, resp.usage, round((time.perf_counter() - started) * 1000))
        if not resp.choices:
            raise LLMError("LLM returned no choices")
        choice = resp.choices[0]
        return choice.message.content or "", choice.finish_reason

    # ------------------------------------------------------------ public API

    def chat(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> str:
        """Single-turn text generation.

        Args:
            messages: OpenAI message format [{role, content}, ...].
            temperature: Sampling temperature (default: config `llm.temperature`).
            max_tokens: Maximum output length (default: config `llm.max_tokens`).

        Returns:
            The model's output text.
        """
        text, _ = self._complete(messages, temperature, max_tokens, "chat")
        return text

    def chat_json(self, messages: list[dict], schema_desc: str, max_tokens: int | None = None) -> dict | list:
        """A chat call with enforced JSON output.

        Args:
            messages: The message list (the prompt already contains the JSON format requirements).
            schema_desc: A description of the expected JSON structure (used for retry hints).
            max_tokens: Output limit of the first attempt (default: config `llm.max_tokens`).

        Returns:
            The parsed dict or list.

        Raises:
            LLMError: After repeated parse failures.
        """
        conversation = list(messages)
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        error = ""
        for _ in range(self.retries + 1):
            text, finish_reason = self._complete(conversation, None, max_tokens, "json")
            try:
                return parse_json_reply(text)
            except ValueError as exc:
                error = str(exc)
                if finish_reason == "length":
                    max_tokens = min(max_tokens * 2, _MAX_TOKENS_CAP)
                    error = f"reply was cut off at the token limit ({error})"
                conversation = [
                    *messages,
                    {"role": "assistant", "content": text[:_ECHO_LIMIT]},
                    {"role": "user", "content": (
                        f"Your previous reply could not be parsed as JSON: {error}. "
                        f"Reply again with only valid JSON (no Markdown, no prose) matching: {schema_desc}"
                    )},
                ]
        raise LLMError(f"no valid JSON after {self.retries + 1} attempts: {error}")

    def stream_chat(self, messages: list[dict], temperature: float | None = None) -> Iterator[str]:
        """Streaming generation (for the UI).

        Yields:
            Incremental text chunks.
        """
        client = self._client()
        started = time.perf_counter()
        stream = self._call(lambda: client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        ))
        usage = None
        try:
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                for choice in chunk.choices or []:
                    delta = getattr(choice.delta, "content", None)
                    if delta:
                        yield delta
        except _openai().APIError as exc:
            raise LLMError(f"LLM stream failed: {exc}") from exc
        finally:
            self._record("stream", usage, round((time.perf_counter() - started) * 1000))

    def usage_summary(self) -> dict[str, int]:
        """Running totals for this process: calls, prompt_tokens, completion_tokens."""
        with self._lock:
            return dict(self._totals)


_CLIENT: LLMClient | None = None
_CLIENT_LOCK = threading.Lock()


def get_client() -> LLMClient:
    """Return the process-level singleton client."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = LLMClient()
        return _CLIENT
