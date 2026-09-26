"""Dependency-light sanity checks (replaces a test suite; see README: Sanity checks).

Run: python scripts/sanity_check.py [--config config.yaml]

Checks, in order; exits non-zero on the first failure:
1. config.yaml loads and ${VAR} placeholders expand (config.get_config)
2. every package imports cleanly: kb, retrieval, orchestration, llm, downstream, eval, ui
3. schema round-trips (keeps SCHEMA.md and types.py in sync):
   - retrieval.types: ScoredPassage / RetrieveResult → dict → JSON → back, fields identical
   - orchestration.types: QueryPlan / ReviewRecord / FinalResult → dict → JSON → back, fields identical
   - downstream.types: EvidenceSource / DownstreamResult → dict → JSON → back, fields identical

Constraints:
- No heavy dependencies: no test framework, no model/vector libraries (pyserini/faiss/transformers...).
- Only the standard library + the project's own light config dependency (pyyaml, via config.py),
  so a fresh checkout can run it right after `pip install pyyaml`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Make the project root importable when running `python scripts/sanity_check.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PACKAGES = ["kb", "retrieval", "orchestration", "llm", "downstream", "eval", "ui"]

REQUIRED_SECTIONS = ("paths", "kb", "retrieval", "orchestration", "llm", "eval", "ui")


def check_config(config_path: str) -> None:
    """Check 1: config.yaml loads and ${VAR} placeholders expand."""
    import config

    cfg = config.get_config(config_path)
    for section in REQUIRED_SECTIONS:
        if section not in cfg:
            raise SystemExit(f"[FAIL] config.yaml missing section: {section}")
    llm = cfg.get("llm", {})
    print(f"[ok] config.yaml loads ({len(cfg)} sections); llm.model={llm.get('model')}")


def check_imports() -> None:
    """Check 2: every package imports cleanly."""
    for name in PACKAGES:
        importlib.import_module(name)
        print(f"[ok] import {name}")


def _roundtrip(cls_name: str, module: object, make: object) -> None:
    """Serialize a dataclass instance → JSON → back; fields must be identical."""
    cls = getattr(module, cls_name)
    instance = make(cls)
    d = asdict(instance)
    back = json.loads(json.dumps(d))
    if back != d:
        raise SystemExit(f"[FAIL] {cls_name} round-trip mismatch: {d!r} != {back!r}")
    print(f"[ok] round-trip {cls_name}")


def check_schema_roundtrips() -> None:
    """Check 3: schema types serialize and read back with identical fields."""
    rt = importlib.import_module("retrieval.types")
    ot = importlib.import_module("orchestration.types")
    dt = importlib.import_module("downstream.types")

    _roundtrip(
        "ScoredPassage", rt,
        lambda c: c(
            passage_id="P1_0001", paper_id="P1", text="t", score=0.9, rank=1,
            stage="hybrid", title="T", year=2024, categories=["cs.CL"],
        ),
    )
    _roundtrip("RetrieveResult", rt, lambda c: c(hits=[], total_candidates=0))
    _roundtrip("QueryPlan", ot, lambda c: c(input_text="q"))
    _roundtrip("ReviewRecord", ot, lambda c: c(paper_id="P1", verdict="pass", score=4))
    _roundtrip(
        "FinalResult", ot,
        lambda c: c(
            query_id="Q1", input_text="q", query_plan=ot.QueryPlan(input_text="q"),
            papers=[], coverage_report={"total_aspects": 0},
        ),
    )
    source = dt.EvidenceSource(1, "P1", "P1_0001", "Title", 2024, "Method", None, "Evidence")
    _roundtrip("EvidenceSource", dt, lambda c: source)
    _roundtrip(
        "DownstreamResult", dt,
        lambda c: c(
            query_id="Q1", input_text="q", answer="Supported [1].", status="answered",
            sources=[source], cited_source_numbers=[1],
            search_result=ot.FinalResult("Q1", "q", ot.QueryPlan("q"), [], {}),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity checks: config, imports, schema round-trips")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    try:
        check_config(args.config)
        check_imports()
        check_schema_roundtrips()
    except SystemExit as e:
        print(str(e))
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 — report any failure cleanly and exit non-zero
        print(f"[FAIL] {type(e).__name__}: {e}")
        sys.exit(1)

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
