"""retrieval package: retrieval core (Assignment Q2/Q3, 40 points).

Submodules:
- types:     type contracts for ScoredPassage / RetrievalFilters / RetrieveResult
- sparse:    Pyserini (Lucene BM25) sparse retrieval
- dense:     BGE-large-en-v1.5 encoding + FAISS HNSW dense retrieval
- fusion:    RRF hybrid fusion
- rerank:    cross-encoder reranking (top-50 → top-10)
- pipeline:  retrieve() unified entry point (the only retrieval interface the
             orchestration layer may call)

Key constraint: cross-module data formats per docs/SCHEMA.md; the orchestration layer
must not bypass pipeline.
"""

from __future__ import annotations

from .pipeline import retrieve

__all__ = ["retrieve"]
