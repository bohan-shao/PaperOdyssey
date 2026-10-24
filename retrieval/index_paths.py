"""Resolve index directories and reject destinations that could erase shared data."""

from __future__ import annotations

import json
from pathlib import Path

from config import PROJECT_ROOT, get_config


def _reject_links(path: Path) -> None:
    for parent in (path, *path.parents):
        if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
            raise ValueError(f"index destination cannot traverse a link: {path}")


def index_directory(value: str | Path, cfg: dict) -> Path:
    """Accept legacy indices/<name> or absolute paths; require a dedicated directory."""
    configured_root = PROJECT_ROOT / cfg["paths"]["indices"]
    root = configured_root.resolve()
    subdir = Path(value)
    if not subdir.is_absolute() and subdir.parts and subdir.parts[0].lower() == "indices":
        subdir = Path(*subdir.parts[1:])
    original = configured_root / subdir
    _reject_links(original.absolute())
    out = original.resolve()
    if out == root or out in root.parents or (not subdir.is_absolute() and not out.is_relative_to(root)):
        raise ValueError(f"index directory must be a dedicated subdirectory: {value!s}")
    return out


def validate_destination(value: str | Path, passages: str | Path, kind: str) -> Path:
    """Validate before building and before deleting; never replace shared or unknown content.

    Explicit absolute destinations are supported. Existing directories may contain only the
    artifacts owned by this index type. Readers must be closed before an offline rebuild.
    """
    cfg = get_config()
    original = Path(value).absolute()
    out = original.resolve()
    _reject_links(original)
    protected = [PROJECT_ROOT.resolve(), Path(passages).resolve()]
    protected.extend((PROJECT_ROOT / v).resolve() for v in cfg["paths"].values())
    protected.append((PROJECT_ROOT / cfg["paths"]["indices"]
                      / cfg["retrieval"]["lookup"]["db_file"]).resolve())
    if any(p == out or p.is_relative_to(out) for p in protected):
        raise ValueError(f"index destination contains shared data or input: {out}")
    allowed = {"meta.json", "index.faiss", "mapping.jsonl"} if kind == "dense" else {
        "meta.json", "lucene", "rankbm25.pkl", "bm25.sqlite"}
    if out.exists():
        if not out.is_dir() or any(p.name not in allowed or p.is_symlink()
                                   or getattr(p, "is_junction", lambda: False)() for p in out.iterdir()):
            raise ValueError(f"index destination contains unrecognized artifacts: {out}")
        if any(out.iterdir()):
            try:
                meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
                owned = ("model_name" in meta and "dim" in meta) if kind == "dense" else (
                    meta.get("backend") in ("pyserini", "rankbm25", "sqlite"))
            except (OSError, ValueError, TypeError, AttributeError):
                owned = False
            if not owned:
                raise ValueError(f"index destination lacks an ownership manifest: {out}")
    return out
