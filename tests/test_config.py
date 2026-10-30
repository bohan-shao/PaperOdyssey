"""Offline tests for config.py: precedence of the process environment, `.env` and built-in defaults."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config

LLM_KEYS = ("LLM_MODEL", "LLM_BASE_URL")


class DotenvPrecedenceTest(unittest.TestCase):
    def _load(self, dotenv: str | None, environ: dict[str, str]) -> dict[str, str | None]:
        """Run `_load_dotenv` against a temporary project root and a clean environment."""
        with tempfile.TemporaryDirectory() as tmp:
            if dotenv is not None:
                (Path(tmp) / ".env").write_text(dotenv, encoding="utf-8")
            with patch.object(config, "PROJECT_ROOT", Path(tmp)), patch.dict(os.environ, environ, clear=True):
                config._load_dotenv()
                return {key: os.environ.get(key) for key in LLM_KEYS}

    def test_dotenv_overrides_defaults(self) -> None:
        got = self._load("LLM_MODEL=deepseek/deepseek-v4.1-flash\nLLM_BASE_URL=https://zenmux.ai/api/v1\n", {})
        self.assertEqual(got, {"LLM_MODEL": "deepseek/deepseek-v4.1-flash",
                               "LLM_BASE_URL": "https://zenmux.ai/api/v1"})

    def test_environment_overrides_dotenv(self) -> None:
        got = self._load("LLM_MODEL=from-dotenv\n", {"LLM_MODEL": "from-env"})
        self.assertEqual(got["LLM_MODEL"], "from-env")

    def test_defaults_fill_missing_or_blank_values(self) -> None:
        defaults = {key: config._ENV_DEFAULTS[key] for key in LLM_KEYS}
        self.assertEqual(self._load(None, {}), defaults)
        self.assertEqual(self._load("LLM_MODEL=\n", {"LLM_BASE_URL": " "}), defaults)


if __name__ == "__main__":
    unittest.main()
