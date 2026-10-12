"""Merge KB shard builds (one per KB member) into one knowledge base (kb.merge.merge_shards).

Usage:
    python scripts/merge_kb_shards.py --out OUT_DIR [--qa-from K] SHARD_DIR [SHARD_DIR ...]

Each SHARD_DIR is a member's data/ folder (raw/, processed/, results/); OUT_DIR gets the same layout.
kb/merge.py says what is merged and when the merge is refused (nothing is written then).
To use the result, copy OUT_DIR's folders over the configured ones (config.yaml `paths.raw`,
`paths.processed`, `paths.results`; data/raw, data/processed, data/results by default), or point
those paths at OUT_DIR, before running build_indexes.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config
from kb.merge import MergeError, merge_shards


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge KB shard builds into one knowledge base")
    parser.add_argument("shards", nargs="+", type=Path, help="each member's data/ folder")
    parser.add_argument("--out", required=True, type=Path, help="output data/ folder (without processed outputs)")
    parser.add_argument("--qa-from", type=int, default=0, help="shard (position in the list) whose QA sample is kept")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    try:
        stats = merge_shards(args.shards, args.out, get_config(args.config), qa_from=args.qa_from)
    except MergeError as exc:
        sys.exit(f"not merged, nothing written: {exc}")
    pipe = stats["pipeline"]
    print(f"merged {len(args.shards)} shards: {pipe['papers_selected']} selected -> {pipe['pdfs_available']} PDFs -> "
          f"{pipe['papers_built']} built, {pipe['papers_failed']} failed; {stats['n_chunks']} passages")


if __name__ == "__main__":
    main()
