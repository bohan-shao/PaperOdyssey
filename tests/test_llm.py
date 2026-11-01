"""Offline tests for llm/client.py: the real OpenAI SDK against a mocked HTTP transport.

Run: python -m unittest tests.test_llm
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx2

from llm import client as llm_client
from llm.client import LLMClient, LLMError, parse_json_reply


def completion(content: str, finish_reason: str = "stop") -> dict:
    return {
        "id": "cmpl-1", "object": "chat.completion", "created": 0, "model": "deepseek-chat",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


class Server:
    """Scripted fake of the chat-completions endpoint; records every request body."""

    def __init__(self, *responses: httpx2.Response) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        return self.responses.pop(0)


def ok(content: str, finish_reason: str = "stop") -> httpx2.Response:
    return httpx2.Response(200, json=completion(content, finish_reason))


class ClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.usage = Path(self.tmp.name) / "llm_usage.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make(self, server: Server, **kw) -> LLMClient:
        return LLMClient(
            model="deepseek-chat", base_url="http://mock.local/v1", api_key=kw.pop("api_key", "sk-test"),
            retries=kw.pop("retries", 2), usage_path=self.usage, retry_wait=lambda _state: 0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(server)), **kw,
        )

    def test_chat_sends_config_and_logs_usage(self) -> None:
        server = Server(ok("hello"))
        client = self.make(server, max_tokens=64, temperature=0.1)
        self.assertEqual(client.chat([{"role": "user", "content": "hi"}]), "hello")
        body = server.requests[0]
        self.assertEqual((body["model"], body["max_tokens"], body["temperature"]), ("deepseek-chat", 64, 0.1))
        log = json.loads(self.usage.read_text())
        self.assertEqual((log["kind"], log["prompt_tokens"], log["completion_tokens"]), ("chat", 11, 7))
        self.assertEqual(client.usage_summary(), {"calls": 1, "prompt_tokens": 11, "completion_tokens": 7})

    def test_chat_json_accepts_fenced_arrays_and_prose(self) -> None:
        server = Server(ok('```json\n[{"paper_id": "P1"}]\n```'), ok('Sure! {"a": 1} Hope this helps.'))
        client = self.make(server)
        self.assertEqual(client.chat_json([{"role": "user", "content": "x"}], "array"), [{"paper_id": "P1"}])
        self.assertEqual(client.chat_json([{"role": "user", "content": "x"}], "object"), {"a": 1})

    def test_chat_json_reasks_with_the_parse_error(self) -> None:
        server = Server(ok("not json at all"), ok('{"ok": true}'))
        client = self.make(server)
        self.assertEqual(client.chat_json([{"role": "user", "content": "plan"}], "object with ok"), {"ok": True})
        retry_messages = server.requests[1]["messages"]
        self.assertEqual(retry_messages[1], {"role": "assistant", "content": "not json at all"})
        self.assertIn("object with ok", retry_messages[2]["content"])

    def test_chat_json_truncated_reply_doubles_max_tokens(self) -> None:
        server = Server(ok('[{"paper_id": "P1", "reas', finish_reason="length"), ok("[]"))
        client = self.make(server, max_tokens=100)
        self.assertEqual(client.chat_json([{"role": "user", "content": "x"}], "array"), [])
        self.assertEqual([r["max_tokens"] for r in server.requests], [100, 200])

    def test_chat_json_never_takes_a_fragment_of_a_cut_off_reply(self) -> None:
        # Seen with the real model: a 5-paper review array cut off at the limit parsed to its
        # first object, so the doubled-limit retry never ran and the review failed.
        cut = '[{"paper_id": "P1", "verdict": "pass"}, {"paper_id": "P2", "reas'
        full = '[{"paper_id": "P1", "verdict": "pass"}, {"paper_id": "P2", "verdict": "reject"}]'
        server = Server(ok(cut, finish_reason="length"), ok(full))
        client = self.make(server, max_tokens=100)
        self.assertEqual(len(client.chat_json([{"role": "user", "content": "x"}], "array")), 2)
        self.assertEqual([r["max_tokens"] for r in server.requests], [100, 200])
        self.assertEqual(parse_json_reply(cut), {"paper_id": "P1", "verdict": "pass"})  # prose fallback alone
        with self.assertRaisesRegex(ValueError, "not complete JSON"):
            parse_json_reply(cut, complete=True)
        self.assertEqual(parse_json_reply("```json\n" + full + "\n```", complete=True)[1]["paper_id"], "P2")

    def test_chat_json_max_tokens_sets_the_first_attempt(self) -> None:
        server = Server(ok('{"answer": "cut', finish_reason="length"), ok('{"answer": "ok"}'))
        client = self.make(server, max_tokens=100)
        self.assertEqual(client.chat_json([{"role": "user", "content": "x"}], "object", max_tokens=300),
                         {"answer": "ok"})
        self.assertEqual([r["max_tokens"] for r in server.requests], [300, 600])

    def test_chat_json_gives_up_with_llm_error(self) -> None:
        server = Server(ok("nope"), ok("still nope"), ok("never"))
        with self.assertRaises(LLMError):
            self.make(server, retries=2).chat_json([{"role": "user", "content": "x"}], "object")
        self.assertEqual(len(server.requests), 3)

    def test_transient_errors_are_retried(self) -> None:
        server = Server(httpx2.Response(429, json={"error": {"message": "slow down"}}),
                        httpx2.Response(503, json={"error": {"message": "busy"}}), ok("done"))
        self.assertEqual(self.make(server).chat([{"role": "user", "content": "x"}]), "done")
        self.assertEqual(len(server.requests), 3)

    def test_auth_error_is_not_retried(self) -> None:
        server = Server(httpx2.Response(401, json={"error": {"message": "bad key"}}))
        with self.assertRaisesRegex(LLMError, "authentication failed"):
            self.make(server).chat([{"role": "user", "content": "x"}])
        self.assertEqual(len(server.requests), 1)

    def test_missing_key_fails_at_first_call_not_at_import(self) -> None:
        server = Server()
        env = {k: v for k, v in os.environ.items() if k not in ("LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY")}
        env["DEEPSEEK_API_KEY"] = "sk-xxxx"  # the .env.example placeholder counts as missing
        with patch.dict(os.environ, env, clear=True):
            client = LLMClient(base_url="https://api.deepseek.com", usage_path=self.usage)
            with self.assertRaisesRegex(LLMError, "DEEPSEEK_API_KEY"):
                client.chat([{"role": "user", "content": "x"}])
        self.assertEqual(server.requests, [])

    def test_stream_chat_yields_deltas_and_records_usage(self) -> None:
        def chunk(payload: dict) -> str:
            return "data: " + json.dumps({"id": "c", "object": "chat.completion.chunk", "created": 0,
                                          "model": "deepseek-chat", **payload}) + "\n\n"
        body = (chunk({"choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}]})
                + chunk({"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]})
                + chunk({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})
                + "data: [DONE]\n\n")
        server = Server(httpx2.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"}))
        client = self.make(server)
        self.assertEqual("".join(client.stream_chat([{"role": "user", "content": "x"}])), "Hello")
        self.assertTrue(server.requests[0]["stream"])
        self.assertEqual(json.loads(self.usage.read_text())["completion_tokens"], 2)

    def test_get_client_is_a_singleton(self) -> None:
        with patch.object(llm_client, "_CLIENT", None):
            self.assertIs(llm_client.get_client(), llm_client.get_client())

    def test_usage_scope_counts_each_thread_alone(self) -> None:
        """Parallel callers (review workers, D48) see only their own calls; the totals see all."""
        client = self.make(Server(*(ok("x") for _ in range(5))))
        barrier = threading.Barrier(2, timeout=10)
        counted = {}

        def caller(name: str, calls: int) -> None:
            with llm_client.usage_scope() as used:
                barrier.wait()  # both scopes are open at the same time
                for _ in range(calls):
                    client.chat([{"role": "user", "content": name}])
                barrier.wait()
            counted[name] = used

        threads = [threading.Thread(target=caller, args=("a", 2)), threading.Thread(target=caller, args=("b", 3))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(counted["a"], {"calls": 2, "prompt_tokens": 22, "completion_tokens": 14})
        self.assertEqual(counted["b"], {"calls": 3, "prompt_tokens": 33, "completion_tokens": 21})
        self.assertEqual(client.usage_summary(), {"calls": 5, "prompt_tokens": 55, "completion_tokens": 35})

    def test_usage_scope_follows_asyncio_tasks_and_to_thread(self) -> None:
        client = self.make(Server(*(ok("x") for _ in range(4))))

        def ask() -> str:
            return client.chat([{"role": "user", "content": "hi"}])

        async def search() -> None:  # the shape of orchestration.loop: tasks calling to_thread
            await asyncio.to_thread(ask)
            await asyncio.gather(*(asyncio.to_thread(ask) for _ in range(2)))

        with llm_client.usage_scope() as used:
            asyncio.run(search())
        ask()  # outside any scope: counted in the totals only
        self.assertEqual(used, {"calls": 3, "prompt_tokens": 33, "completion_tokens": 21})
        self.assertEqual(client.usage_summary()["calls"], 4)


class ParseJsonTest(unittest.TestCase):
    def test_rejects_scalars_and_garbage(self) -> None:
        for bad in ("42", '"text"', "no json here", "```\n```"):
            with self.assertRaises(ValueError):
                parse_json_reply(bad)

    def test_skips_brackets_that_are_not_json(self) -> None:
        self.assertEqual(parse_json_reply('See [1] above. {"k": [1, 2]}'), {"k": [1, 2]})


if __name__ == "__main__":
    unittest.main()
