"""Configuration loader (the single config entry point).

- `get_config(path=None) -> dict`: load config.yaml, expand `${VAR}` placeholders from the
  environment (`.env` + defaults for LLM_MODEL / LLM_BASE_URL), cache a process-level singleton.
- `get_path(key) -> Path`: resolve a `paths.*` key to an absolute path relative to the project root.
- Fail fast on missing required sections (so a broken config is caught at startup, not mid-run).

Business code must never open config.yaml directly — always go through this module.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")

# Sections that must exist; fail fast if any is missing.
REQUIRED_SECTIONS = ("paths", "kb", "retrieval", "orchestration", "llm", "eval", "ui")

# Defaults so the config is loadable on a fresh checkout without a .env file.
_ENV_DEFAULTS = {
    "LLM_MODEL": "deepseek-chat",
    "LLM_BASE_URL": "https://api.deepseek.com",
}


def _load_dotenv() -> None:
    """Minimal .env loader (KEY=VALUE lines; never overrides existing env vars)."""
    for key, value in _ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    dotenv_path = PROJECT_ROOT / ".env"
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _expand_env(value: object) -> object:
    """Recursively expand ${VAR} placeholders in the config structure."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@lru_cache(maxsize=1)
def get_config(path: str | Path | None = None) -> dict:
    """Load the global configuration (env expansion + process-level singleton cache).

    Args:
        path: Optional explicit config path (defaults to <project_root>/config.yaml).

    Returns:
        The fully expanded config dict.

    Raises:
        ValueError: The YAML is not a mapping, or a required section is missing.
        OSError: The config file cannot be read.
    """
    _load_dotenv()
    cfg_path = Path(path) if path is not None else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path} must be a YAML mapping, got {type(raw).__name__}")
    cfg = _expand_env(raw)
    if not isinstance(cfg, dict):
        raise ValueError(f"{cfg_path} must expand to a YAML mapping")
    missing = [k for k in REQUIRED_SECTIONS if k not in cfg]
    if missing:
        raise ValueError(f"{cfg_path} missing required sections: {missing}")
    return cfg


def get_path(key: str) -> Path:
    """Resolve a `paths.*` key to an absolute path (relative to the project root).

    Args:
        key: A key under the `paths:` section, e.g. "processed".

    Returns:
        Absolute path.

    Raises:
        KeyError: The key is not defined in config.yaml under `paths:`.
    """
    paths = get_config().get("paths", {})
    if key not in paths:
        raise KeyError(f"unknown path key: {key!r} (available: {sorted(paths)})")
    return PROJECT_ROOT / str(paths[key])
