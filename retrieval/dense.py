"""BGE-large-en-v1.5 dense retrieval + FAISS HNSW. TODO.

Implementation requirements (see retrieval/README.md for details):
- Encoding: `encode_texts(texts, batch_size) -> np.ndarray`,
  model name from config `retrieval.dense.model_name` (default BAAI/bge-large-en-v1.5).
- **The query-side instruction prefix is mandatory**: `"Represent this sentence for searching relevant passages: "`
  (BGE retrieval convention; omitting it significantly hurts scores).
- Build index: `build_index(passages_path, faiss_dir)`:
  1) read passages → encode → 2) FAISS IndexHNSWFlat (M=32, efConstruction=200,
  efSearch left config-adjustable) → 3) save vectors to disk + passage_id mapping file (JSONL/parquet).
- Search: `search(query, filters, top_k)`: encode the query → FAISS search →
  cosine similarity (inner product requires normalization) → filter metadata → ScoredPassage(stage="dense").
- Encoding resources: fp16 + large batch available on A40; record total encoding time
  (report the index construction cost).

Constraints: vector dimension must match the model's output dimension (1024); record the
faiss index version in the results directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import RetrievalFilters, ScoredPassage

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode texts into vectors in batches.

    Args:
        texts: list of texts to encode.
        batch_size: encoding batch size.

    Returns:
        A float32 matrix of shape (n, dim).
    """
    raise NotImplementedError  # TODO: implement; see retrieval/README.md


def build_index(passages_path: str | Path, faiss_dir: str | Path) -> None:
    """Encode all passages and build the FAISS HNSW index."""
    raise NotImplementedError  # TODO: implement; see retrieval/README.md


def search(
    query: str,
    filters: RetrievalFilters | None,
    top_k: int,
    ef_search: int = 128,
) -> tuple[list[ScoredPassage], int]:
    """Dense retrieval.

    Returns:
        (hits, latency_ms).
    """
    raise NotImplementedError  # TODO: implement; see retrieval/README.md
