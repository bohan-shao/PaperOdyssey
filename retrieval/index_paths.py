"""Resolve index directories and reject destinations that could erase shared data."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from config import PROJECT_ROOT, get_config


def _is_link(path: Path) -> bool:
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def index_directory(value: str | Path, cfg: dict) -> Path:
    """Accept legacy indices/<name> or absolute paths; require a dedicated directory."""
    configured_root = PROJECT_ROOT / cfg["paths"]["indices"]
    root = Path(os.path.abspath(configured_root))
    subdir = Path(value)
    if not subdir.is_absolute() and subdir.parts and subdir.parts[0].lower() == "indices":
        subdir = Path(*subdir.parts[1:])
    # Check lexical containment, not the target of a legitimate read-only index link.
    # Keep the leaf spelling so a later builder can reject replacing the link itself.
    out = Path(os.path.abspath(root / subdir))
    if out == root or out in root.parents or (not subdir.is_absolute() and not out.is_relative_to(root)):
        raise ValueError(f"index directory must be a dedicated subdirectory: {value!s}")
    return out


def validate_destination(value: str | Path, passages: str | Path, kind: str) -> Path:
    """Validate before building and before replacing; never replace shared or unknown content.

    Explicit absolute destinations are supported. Existing directories may contain only the
    artifacts owned by this index type. Readers must be closed before an offline rebuild.
    """
    cfg = get_config()
    original = Path(os.path.abspath(value))
    if _is_link(original):
        raise ValueError(f"cannot replace a linked index directory; use its explicit target: {original}")
    out = original.resolve()
    # Ancestor aliases (including macOS /var -> /private/var) are safe after resolution.
    protected = [PROJECT_ROOT.resolve(), Path(passages).resolve()]
    protected.extend((PROJECT_ROOT / v).resolve() for v in cfg["paths"].values())
    protected.append((PROJECT_ROOT / cfg["paths"]["indices"]
                      / cfg["retrieval"]["lookup"]["db_file"]).resolve())
    if any(p == out or p.is_relative_to(out) for p in protected):
        raise ValueError(f"index destination contains shared data or input: {out}")
    allowed = {"meta.json", "index.faiss", "mapping.jsonl"} if kind == "dense" else {
        "meta.json", "lucene", "rankbm25.pkl", "bm25.sqlite"}
    if out.exists():
        if not out.is_dir() or any(p.name not in allowed or _is_link(p) for p in out.iterdir()):
            raise ValueError(f"index destination contains unrecognized artifacts: {out}")
        if any(out.iterdir()):
            try:
                meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
                owned = isinstance(meta, dict) and (("model_name" in meta and "dim" in meta) if kind == "dense" else (
                    meta.get("backend") in ("pyserini", "rankbm25", "sqlite")))
            except (OSError, ValueError, TypeError, AttributeError):
                owned = False
            if not owned:
                raise ValueError(f"index destination lacks an ownership manifest: {out}")
    return out


def publish_index(staged: Path, destination: Path) -> None:
    """Publish a validated offline build; restore the old index if the final rename fails."""
    if staged.resolve().parent != destination.resolve().parent or staged.resolve() == destination.resolve():
        raise ValueError("staged index must be a separate sibling directory")
    backup = None
    if destination.exists():
        backup = Path(tempfile.mkdtemp(prefix=destination.name + ".", suffix=".backup", dir=destination.parent))
        backup.rmdir()
        destination.rename(backup)
    try:
        staged.rename(destination)
    except BaseException:
        if backup is not None:
            backup.rename(destination)
        raise
    if backup is not None:
        try:
            shutil.rmtree(backup)
        except OSError:
            logging.getLogger(__name__).warning("New index published; could not remove old backup %s", backup)
