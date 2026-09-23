"""Client for TypeSafe's Jev, a "System One" model that returns typed decisions instead of text.

Jev answers typed questions about a piece of text (the "state") in a single pass:
- choice: pick one option from a map of options,
- score:  a position on an ordered scale (0 = lowest level, returned as a probability-weighted float),
- noul:   the probability (0 to 1) that a yes/no statement is true.
It cannot write text, so it is only used where the system makes decisions (the optional Jev
reviewer in orchestration/review.py). Everything that generates text stays on llm/client.py.

API (https://docs.typesafe.ai/api.md): POST {base_url}/v1/systemone with
{"state", "model", "questions"} and the header `Authorization: Bearer $TYPESAFE_API_KEY`.

- Configuration: config `llm.jev.*` (model / base_url / retries / timeout_s); the key is
  `TYPESAFE_API_KEY` in `.env`. The key is read at the first call, so importing this module
  (and sanity_check.py) never needs it.
- Retries: 429 (rate limit), 529 (overloaded), other 5xx, timeouts and connection errors are
  retried with exponential backoff, as the TypeSafe docs recommend; 401 and 422 fail at once.
- Usage: every call appends {time, model, kind: "jev", prompt_tokens, completion_tokens,
  latency_ms} to `data/results/metrics/llm_usage.jsonl`, the same file the LLM client uses,
  so one cost report covers both.

Raises:
    JevError: missing key, rejected request, malformed reply, or retries exhausted.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import get_config, get_path

_RETRY_STATUS = {429, 500, 502, 503, 504, 529}
_ENDPOINT = "/v1/systemone"


class JevError(RuntimeError):
    """Unrecoverable Jev failure (missing key, rejected request, bad reply, retries exhausted)."""


class _Transient(Exception):
    """A failure worth retrying: rate limit, overload, server error or network problem."""


# ------------------------------------------------------------------ question builders


def choice(instructions: str, options: Mapping[str, str | None]) -> dict[str, Any]:
    """A choice question: Jev picks one key of `options` (the values describe each option)."""
    if not 1 <= len(options) <= 255:
        raise ValueError("a choice question needs 1 to 255 options")
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def score(instructions: str, levels: list[str]) -> dict[str, Any]:
    """A score question: `levels` go from lowest to highest (2 to 10 levels)."""
    if not 2 <= len(levels) <= 10:
        raise ValueError("a score question needs 2 to 10 levels")
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def noul(instructions: str, true: str | None = None, false: str | None = None) -> dict[str, Any]:
    """A yes/no question; the answer is the probability that the statement is true."""
    question: dict[str, Any] = {"type": "noul", "instructions": instructions}
    criteria = {key: value for key, value in (("true", true), ("false", false)) if value}
    if criteria:
        question["criteria"] = criteria
    return question


# ------------------------------------------------------------------ client


class JevClient:
    """Client for Jev's /v1/systemone endpoint."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        retries: int | None = None,
        timeout_s: float | None = None,
        usage_path: str | Path | None = None,
        transport: httpx.BaseTransport | None = None,
        retry_wait: Callable[..., float] | None = None,
    ) -> None:
        cfg = get_config()["llm"].get("jev", {})  # get_config() also loads .env into os.environ
        self.model = model or str(cfg.get("model", "jev-latest"))
        self.base_url = (base_url or str(cfg.get("base_url", "https://api.typesafe.ai"))).rstrip("/")
        self.retries = max(0, int(cfg.get("retries", 3) if retries is None else retries))
        self.timeout_s = float(cfg.get("timeout_s", 30) if timeout_s is None else timeout_s)
        self._api_key = api_key
        self._usage_path = Path(usage_path) if usage_path else get_path("results") / "metrics" / "llm_usage.jsonl"
        self._transport = transport
        self._retry_wait = retry_wait or wait_exponential(multiplier=1, min=1, max=30)
        self._http: httpx.Client | None = None
        self._lock = threading.Lock()
        self._totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _client(self) -> httpx.Client:
        with self._lock:
            if self._http is None:
                key = (self._api_key or os.environ.get("TYPESAFE_API_KEY", "")).strip()
                if not key:
                    raise JevError("no Jev API key: set TYPESAFE_API_KEY in .env (see .env.example)")
                self._http = httpx.Client(
                    base_url=self.base_url,
                    timeout=self.timeout_s,
                    transport=self._transport,
                    headers={"Authorization": f"Bearer {key}"},
                )
            return self._http

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._client().post(_ENDPOINT, json=payload)
        except httpx.TransportError as exc:  # timeouts and connection errors
            raise _Transient(f"network error: {exc}") from exc
        if resp.status_code in _RETRY_STATUS:
            raise _Transient(f"HTTP {resp.status_code}: {resp.text[:300]}")
        if resp.status_code == 401:
            raise JevError("Jev rejected the API key (HTTP 401): check TYPESAFE_API_KEY")
        if resp.status_code >= 400:
            raise JevError(f"Jev rejected the request (HTTP {resp.status_code}): {resp.text[:500]}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise JevError(f"Jev returned a reply that is not JSON: {resp.text[:300]}") from exc
        if not isinstance(body, dict):
            raise JevError("Jev returned a reply that is not a JSON object")
        return body

    def _record(self, usage: Any, latency_ms: int, model: str) -> None:
        usage = usage if isinstance(usage, dict) else {}
        prompt = int(usage.get("input_tokens", 0) or 0)
        completion = int(usage.get("output_tokens", 0) or 0)
        with self._lock:
            self._totals["calls"] += 1
            self._totals["prompt_tokens"] += prompt
            self._totals["completion_tokens"] += completion
            try:
                self._usage_path.parent.mkdir(parents=True, exist_ok=True)
                with self._usage_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "model": model,
                        "kind": "jev",
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "latency_ms": latency_ms,
                    }) + "\n")
            except OSError:
                pass  # usage logging must never fail a call

    def ask(self, state: str | dict | list, questions: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Ask typed questions about `state`.

        Args:
            state: The text to judge (a string, or a JSON object/array of text values).
            questions: {key: question} built with `choice()`, `score()` or `noul()`.

        Returns:
            {key: answer} with the same keys. A choice answer has "choice", "probabilities" and
            "confidence"; a score answer has "score", "probabilities" and "confidence"; a noul
            answer has "noul" (probability of yes).

        Raises:
            JevError: see the module docstring.
        """
        if not questions:
            raise ValueError("ask() needs at least one question")
        payload = {"state": state, "model": self.model, "questions": dict(questions)}
        started = time.perf_counter()
        try:
            for attempt in Retrying(
                retry=retry_if_exception_type(_Transient),
                stop=stop_after_attempt(self.retries + 1),
                wait=self._retry_wait,
                reraise=True,
            ):
                with attempt:
                    body = self._post(payload)
        except _Transient as exc:
            raise JevError(f"Jev call failed after {self.retries + 1} attempts: {exc}") from exc

        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise JevError("Jev reply has no 'answers' object")
        for key, question in questions.items():
            kind = question.get("type")
            answer = answers.get(key)
            if not isinstance(answer, dict) or answer.get("type", kind) != kind:
                raise JevError(f"Jev reply has no {kind} answer for {key!r}")
            value = answer.get(kind)
            if kind == "choice":
                if value not in question.get("criteria", {}):
                    raise JevError(f"Jev chose an unknown option for {key!r}: {value!r}")
            elif isinstance(value, bool) or not isinstance(value, int | float):
                raise JevError(f"Jev returned a non-numeric {kind} for {key!r}: {value!r}")
        latency_ms = round((time.perf_counter() - started) * 1000)
        self._record(body.get("usage"), latency_ms, str(body.get("model") or self.model))
        return {key: answers[key] for key in questions}

    def usage_summary(self) -> dict[str, int]:
        """Running totals for this process: calls, prompt_tokens, completion_tokens."""
        with self._lock:
            return dict(self._totals)


_CLIENT: JevClient | None = None
_CLIENT_LOCK = threading.Lock()


def get_jev_client() -> JevClient:
    """Return the process-level singleton Jev client."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = JevClient()
        return _CLIENT
