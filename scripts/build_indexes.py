"""Build retrieval indexes (BM25 + FAISS). TODO.

Usage:
    python scripts/build_indexes.py --config config.yaml [--only sparse|dense]

Pipeline: kb.store.iter_passages -> retrieval.sparse.build_index (optional)
    -> retrieval.dense.build_index (optional)
Constraints: rebuilding overwrites old indexes (delete first, then build); record index build time and scale.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 + FAISS indexes")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--only", choices=["sparse", "dense", "all"], default="all")
    args = parser.parse_args()
    raise NotImplementedError  # TODO: implement, see retrieval/README.md


if __name__ == "__main__":
    main()
