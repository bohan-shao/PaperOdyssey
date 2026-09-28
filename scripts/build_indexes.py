"""Build retrieval indexes (lookup store + BM25 + FAISS).

Usage:
    python scripts/build_indexes.py --config config.yaml [--only lookup|sparse|dense|all]

Pipeline: kb.store passages/sections -> retrieval.lookup.build_lookup
    -> retrieval.sparse.build_index (optional) -> retrieval.dense.build_index (optional)
All three read the same passages file, so hit ids always resolve in the lookup store.
Constraints: rebuilding overwrites old indexes (delete first, then build); build time and scale
are merged into results/metrics/index_build_<date>.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT, get_config  # noqa: E402
from retrieval.lookup import build_lookup, lookup_path  # noqa: E402

log = logging.getLogger("build_indexes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lookup store + BM25 + FAISS indexes")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--only", choices=["lookup", "sparse", "dense", "all"], default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per HF Hub request otherwise

    cfg = get_config(args.config)
    paths = {k: PROJECT_ROOT / str(v) for k, v in cfg["paths"].items()}
    kb_store = cfg["kb"]["store"]
    passages = paths["processed"] / kb_store["passages_file"]
    sections = paths["processed"] / kb_store["sections_file"]
    if not passages.exists():
        sys.exit(f"{passages} not found; build the KB first (scripts/build_kb.py)")

    report: dict = {"passages_file": str(passages)}
    if args.only in ("lookup", "all"):
        db = lookup_path(cfg)
        log.info("building lookup store -> %s", db)
        report["lookup"] = build_lookup(passages, sections, db)
        log.info("lookup store: %s", report["lookup"])
    if args.only in ("sparse", "all"):
        from retrieval import sparse

        bm25_dir, backend = sparse.index_path(cfg), cfg["retrieval"]["sparse"]["backend"]
        log.info("building BM25 index (%s) -> %s", backend, bm25_dir)
        report["sparse"] = sparse.build_index(passages, bm25_dir, backend=backend)
        log.info("BM25 index: %s", report["sparse"])
    if args.only in ("dense", "all"):
        from retrieval import dense

        faiss_dir = dense.index_path(cfg)
        log.info("building FAISS index (%s) -> %s", cfg["retrieval"]["dense"]["model_name"], faiss_dir)
        report["dense"] = dense.build_index(passages, faiss_dir)
        log.info("FAISS index: %s", report["dense"])

    # Merge into today's report, so an --only run keeps the other indexes' numbers.
    out = paths["results"] / "metrics" / f"index_build_{date.today().isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        report = {**json.loads(out.read_text(encoding="utf-8")), **report}
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.info("build report -> %s", out)


if __name__ == "__main__":
    main()
