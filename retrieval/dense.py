"""Model-aware dense retrieval: sentence-transformers + FAISS HNSW over unit vectors (cosine).

- `Encoder`: sentence-transformers wrapper returning float32, L2-normalized rows (inner product
  = cosine); fp16 on CUDA (`retrieval.dense.fp16`). `encode_query` selects a BGE, Qwen3 or plain
  instruction format; the document token limit is recorded in the index manifest. One lock around the
  model, because HF fast tokenizers fail when two threads use them at once.
- `build_index(passages_path, faiss_dir)`: streams passages.jsonl in chunks, encodes them and
  adds them to an IndexHNSWFlat (M / efConstruction from `retrieval.dense.hnsw`), then swaps the
  new dir in (a rebuild overwrites). Layout: `index.faiss` (HNSW graph + the float32 vectors),
  `mapping.jsonl` (row -> passage_id, year, categories), `meta.json` (model, dim, HNSW params,
  faiss version). Encoding and HNSW insertion times are reported separately.
- `DenseRetriever(faiss_dir, lookup)`: refuses an index built with another model, then serves
  `search(query, filters, top_k) -> (hits, latency_ms)`; stage="dense", score = cosine.
- `search(...)` / `encode_texts(...)`: module-level entries on process singletons.

Filters (year range, categories; same semantics as sparse) are applied inside FAISS with an ID
selector, never by cutting the top-k afterwards:
- no filter: HNSW search, efSearch = max(efSearch, top_k);
- the filter keeps <= `retrieval.dense.filter_exact_max` passages: exact search over just those
  rows (flat inner product on the stored vectors). HNSW alone is unreliable here: its graph walk
  mostly meets excluded nodes and comes back with too few hits;
- larger selections: HNSW with the selector, falling back to exact search if it comes back short.

Constraints: the index dim must match the model's output dim (1024 for bge-large); passages and
queries must be encoded by the same model (checked on open).
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from config import get_config
from kb import store as kb_store

from .lookup import LookupStore, get_lookup_store
from .index_paths import index_directory, publish_index, validate_destination
from .sparse import category_token
from .types import RetrievalFilters, ScoredPassage

log = logging.getLogger(__name__)

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
_CHUNK = 16384  # passages encoded and added to the index per step while building


# ---------------------------------------------------------------- encoder


class Encoder:
    """sentence-transformers model producing unit-length float32 embeddings."""

    def __init__(self, model_name: str, batch_size: int = 64, fp16: bool = True,
                 device: str | None = None) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        cfg = get_config()["retrieval"]["dense"]
        # Direct callers (e.g. semantic chunking or diagnostics) may use a different model.
        if model_name != cfg.get("model_name"):
            cfg = {}
        device = device or cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=device)
        if cfg.get("max_length") is not None:
            requested = int(cfg["max_length"])
            if requested <= 0:
                raise ValueError("dense.max_length must be positive")
            transformer = self.model[0]
            architecture = getattr(getattr(transformer, "auto_model", None), "config", None)
            limits = [getattr(architecture, "max_position_embeddings", None),
                      getattr(self.model.tokenizer, "model_max_length", None)]
            limits = [n for n in limits if isinstance(n, int) and 0 < n < 1_000_000_000]
            ceiling = min(limits) if limits else self.model.max_seq_length
            self.model.max_seq_length = min(requested, ceiling)
            if requested > ceiling:
                log.warning("dense.max_length=%s capped at %s for %s", requested, ceiling, model_name)
        self.max_length = self.model.max_seq_length
        self.query_instruction = query_instruction(model_name, cfg)
        if fp16 and device.startswith("cuda"):
            self.model.half()
        dim_fn = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.name = model_name
        self.dim = int(dim_fn())
        self.batch_size = batch_size
        self._lock = threading.Lock()

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """(n, dim) float32, rows L2-normalized."""
        import numpy as np

        with self._lock:
            vecs = self.model.encode(texts, batch_size=batch_size or self.batch_size,
                                     convert_to_numpy=True, normalize_embeddings=True,
                                     show_progress_bar=False, prompt="")
        return np.ascontiguousarray(vecs, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """(1, dim): model-specific retrieval instruction, applied only on the query side."""
        return self.encode([self.query_instruction + query])


def query_instruction(model_name: str, cfg: dict) -> str:
    """Resolve BGE/Qwen3/plain input formats; custom local models can set an explicit prefix."""
    explicit = cfg.get("query_instruction")
    if explicit is not None:
        return str(explicit)
    family = cfg.get("query_format", "auto")
    if family == "auto":
        name = model_name.casefold()
        english_bge = any(f"bge-{size}-en" in name for size in ("small", "base", "large"))
        family = "qwen3" if "qwen3-embedding" in name else "bge" if english_bge else "plain"
    if family == "bge":
        return QUERY_INSTRUCTION
    if family == "qwen3":
        task = cfg.get("task_instruction", "Given a scientific question, retrieve relevant passages from research papers")
        return f"Instruct: {task}\nQuery: "
    if family == "plain":
        return ""
    raise ValueError(f"unknown dense query_format {family!r}")


_ENCODER: Encoder | None = None
_ENCODER_LOCK = threading.Lock()


def get_encoder() -> Encoder:
    """Return the process-level singleton encoder (model loaded on first use)."""
    global _ENCODER
    with _ENCODER_LOCK:
        if _ENCODER is None:
            cfg = get_config()["retrieval"]["dense"]
            _ENCODER = Encoder(cfg["model_name"], cfg.get("batch_size", 64), cfg.get("fp16", True))
        return _ENCODER


def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode texts into vectors in batches.

    Args:
        texts: list of texts to encode (passages; queries go through `Encoder.encode_query`).
        batch_size: encoding batch size.

    Returns:
        A float32 matrix of shape (n, dim), rows L2-normalized.
    """
    return get_encoder().encode(texts, batch_size)


# ---------------------------------------------------------------- build


def _chunks(records: Iterator[dict], size: int) -> Iterator[list[dict]]:
    while chunk := list(islice(records, size)):
        yield chunk


def build_index(passages_path: str | Path, faiss_dir: str | Path,
                encoder: Encoder | None = None) -> dict:
    """Encode all passages and build the FAISS HNSW index, replacing any existing one.

    Args:
        passages_path: data/processed/passages.jsonl.
        faiss_dir: data/indices/faiss.
        encoder: defaults to the configured model (`get_encoder()`).

    Returns:
        {"model_name", "dim", "num_vectors", "encode_s", "index_s", "size_bytes",
         "faiss_version", "build_s"}.

    Raises:
        RuntimeError: when there are no passages to index.
    """
    out = validate_destination(faiss_dir, passages_path, "dense")
    import faiss

    t0 = time.perf_counter()
    hnsw = get_config()["retrieval"]["dense"]["hnsw"]
    encoder = encoder or get_encoder()
    index = faiss.IndexHNSWFlat(encoder.dim, hnsw["M"], faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = hnsw["efConstruction"]

    out.parent.mkdir(parents=True, exist_ok=True)
    with Path(passages_path).open(encoding="utf-8") as fh:
        total = sum(1 for _ in fh)
    tmp = Path(tempfile.mkdtemp(prefix=out.name + ".", suffix=".tmp", dir=out.parent))
    tmp.chmod(0o755)  # mkdtemp makes it owner-only; the index is shared data
    encode_s = index_s = 0.0
    try:
        with (tmp / "mapping.jsonl").open("w", encoding="utf-8") as mapping:
            for chunk in _chunks(kb_store.iter_passages(passages_path), _CHUNK):
                t = time.perf_counter()
                vecs = encoder.encode([p["text"] for p in chunk])
                encode_s += time.perf_counter() - t
                start = index.ntotal
                t = time.perf_counter()
                index.add(vecs)
                index_s += time.perf_counter() - t
                for row, p in enumerate(chunk, start=start):
                    mapping.write(json.dumps({"row": row, "passage_id": p["passage_id"],
                                              "year": p.get("year"),
                                              "categories": p.get("categories") or []},
                                             ensure_ascii=False) + "\n")
                log.info("dense: %d/%d passages encoded (%.0f/s)", index.ntotal, total,
                         index.ntotal / max(encode_s, 1e-9))
        if index.ntotal == 0:
            raise RuntimeError(f"no passages in {passages_path}")
        faiss.write_index(index, str(tmp / "index.faiss"))
        meta = {
            "model_name": encoder.name, "dim": encoder.dim, "num_vectors": index.ntotal,
            "metric": "inner_product", "normalized": True,
            "document_prompt": "",
            "query_instruction": getattr(encoder, "query_instruction", QUERY_INSTRUCTION),
            "max_length": getattr(encoder, "max_length", None),
            "hnsw": {"M": hnsw["M"], "efConstruction": hnsw["efConstruction"]},
            "faiss_version": faiss.__version__,
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        validate_destination(out, passages_path, "dense")
        publish_index(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {
        "model_name": encoder.name,
        "dim": encoder.dim,
        "num_vectors": meta["num_vectors"],
        "encode_s": round(encode_s, 3),
        "index_s": round(index_s, 3),
        "size_bytes": sum(f.stat().st_size for f in out.iterdir()),
        "faiss_version": faiss.__version__,
        "build_s": round(time.perf_counter() - t0, 3),
    }


# ---------------------------------------------------------------- search


class DenseRetriever:
    """HNSW search over one index directory; hits are hydrated through the lookup store."""

    def __init__(self, faiss_dir: str | Path, lookup: LookupStore, encoder: Encoder | None = None,
                 ef_search: int | None = None, exact_max: int | None = None) -> None:
        import faiss
        import numpy as np

        cfg = get_config()["retrieval"]["dense"]
        faiss_dir = Path(faiss_dir)
        meta_path = faiss_dir / "meta.json"
        if not meta_path.exists():
            raise RuntimeError(f"FAISS index not found: {faiss_dir} "
                               "(run scripts/build_indexes.py --only dense)")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_name = encoder.name if encoder is not None else cfg["model_name"]
        if meta["model_name"] != model_name:
            raise RuntimeError(f"FAISS index {faiss_dir} was built with {meta['model_name']!r} but "
                               f"queries would be encoded with {model_name!r}; rebuild the dense index")
        self._encoder = encoder or get_encoder()
        # Query instructions are experiment provenance, not part of document encoding.
        if meta.get("document_prompt", "") != "":
            raise RuntimeError("FAISS document prompt differs from the encoder; rebuild the dense index")
        if meta.get("max_length") != getattr(self._encoder, "max_length", None):
            # Pre-manifest BGE indexes used the model's 512-token limit.
            legacy_bge = ("max_length" not in meta and "bge-" in model_name.casefold()
                          and getattr(self._encoder, "max_length", None) == 512)
            if not legacy_bge:
                raise RuntimeError("FAISS document max_length differs from the encoder; rebuild the dense index")
        if self._encoder.dim != meta["dim"]:
            raise RuntimeError(f"FAISS index dim {meta['dim']} != encoder dim {self._encoder.dim}; "
                               "rebuild the dense index")

        self._faiss = faiss
        self._index = faiss.read_index(str(faiss_dir / "index.faiss"))
        self._flat = faiss.downcast_index(self._index.storage)  # the raw vectors, for exact search
        ids: list[str] = []
        years: list[int] = []
        by_category: dict[str, list[int]] = {}
        with (faiss_dir / "mapping.jsonl").open(encoding="utf-8") as fh:
            for row, line in enumerate(fh):
                rec = json.loads(line)
                if rec["row"] != row:
                    raise RuntimeError("FAISS mapping row order is invalid; rebuild the dense index")
                ids.append(rec["passage_id"])
                years.append(rec.get("year") or 0)
                for c in rec.get("categories") or []:
                    by_category.setdefault(category_token(c), []).append(rec["row"])
        if len(ids) != self._index.ntotal:
            raise RuntimeError(f"mapping.jsonl has {len(ids)} rows but the index has "
                               f"{self._index.ntotal} vectors; rebuild the dense index")
        self._ids = ids
        self._years = np.array(years)
        self._by_category = {c: np.array(rows) for c, rows in by_category.items()}
        self._ef = cfg["hnsw"]["efSearch"] if ef_search is None else ef_search
        self._exact_max = cfg.get("filter_exact_max", 50000) if exact_max is None else exact_max
        self._lookup = lookup
        self._encoder.encode_query("warm up")  # first (CUDA) call is slow; keep it out of timings

    @property
    def index(self) -> object:
        """The loaded FAISS HNSW index, for read-only use (e.g. the efSearch latency sweep)."""
        return self._index

    def _mask(self, filters: RetrievalFilters | None) -> np.ndarray | None:
        """Boolean mask of the rows the filters keep; None = no filtering."""
        import numpy as np

        if filters is None:
            return None
        mask = None
        if filters.year_min is not None or filters.year_max is not None:
            mask = np.ones(len(self._ids), dtype=bool)
            if filters.year_min is not None:
                mask &= self._years >= filters.year_min
            if filters.year_max is not None:
                mask &= self._years <= filters.year_max
        if filters.categories:
            in_any = np.zeros(len(self._ids), dtype=bool)
            for c in filters.categories:
                rows = self._by_category.get(category_token(c))
                if rows is not None:
                    in_any[rows] = True
            mask = in_any if mask is None else mask & in_any
        return mask

    def _search_rows(self, query: str, filters: RetrievalFilters | None, k: int,
                     ef: int) -> list[tuple[str, float]]:
        import numpy as np

        faiss = self._faiss
        q = self._encoder.encode_query(query)
        mask = self._mask(filters)
        if mask is None:
            scores, rows = self._index.search(q, k, params=faiss.SearchParametersHNSW(efSearch=max(ef, k)))
        else:
            kept = int(mask.sum())
            if kept == 0:
                return []
            bits = np.packbits(mask, bitorder="little")  # must outlive the searches below
            selector = faiss.IDSelectorBitmap(len(bits), faiss.swig_ptr(bits))  # n = bitmap bytes
            rows = None
            if kept > self._exact_max:
                scores, rows = self._index.search(
                    q, k, params=faiss.SearchParametersHNSW(efSearch=max(ef, k), sel=selector))
            if rows is None or (rows[0] >= 0).sum() < min(k, kept):
                scores, rows = self._flat.search(q, k, params=faiss.SearchParameters(sel=selector))
        return [(self._ids[r], float(s)) for s, r in zip(scores[0], rows[0]) if r >= 0]

    def search(self, query: str, filters: RetrievalFilters | None, top_k: int,
               ef_search: int | None = None) -> tuple[list[ScoredPassage], int]:
        """Dense top_k for one query (filters restrict candidates, never scores)."""
        t0 = time.perf_counter()
        ranked = []
        if query.strip() and top_k > 0:
            ranked = self._search_rows(query, filters, top_k, self._ef if ef_search is None else ef_search)
        hits = self._lookup.hydrate(ranked, stage="dense")
        return hits, round((time.perf_counter() - t0) * 1000)


def index_path(cfg: dict | None = None) -> Path:
    """Resolve under paths.indices; accept legacy indices/faiss and explicit absolute paths."""
    cfg = cfg or get_config()
    return index_directory(cfg["retrieval"]["dense"]["faiss_dir"], cfg)


_RETRIEVER: DenseRetriever | None = None
_RETRIEVER_LOCK = threading.Lock()


def get_dense_retriever() -> DenseRetriever:
    """Return the process-level singleton retriever (index + model loaded on first use)."""
    global _RETRIEVER
    with _RETRIEVER_LOCK:
        if _RETRIEVER is None:
            _RETRIEVER = DenseRetriever(index_path(), get_lookup_store())
        return _RETRIEVER


def search(
    query: str,
    filters: RetrievalFilters | None,
    top_k: int,
    ef_search: int | None = None,
) -> tuple[list[ScoredPassage], int]:
    """Dense retrieval.

    Returns:
        (hits, latency_ms).
    """
    return get_dense_retriever().search(query, filters, top_k, ef_search)
