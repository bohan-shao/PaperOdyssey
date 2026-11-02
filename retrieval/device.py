"""Torch device of the retrieval models, and the lock that keeps Apple GPU calls apart (D58).

`retrieval.dense.device` and `retrieval.rerank.device` set to null pick CUDA, then Apple MPS, then
CPU. Metal cannot run two models' work at once from one process's threads: the sub-queries of a
search encode and rerank in parallel, and overlapping MPS calls abort the process. So every model
on MPS takes one shared lock; on CUDA and CPU each model keeps its own.
"""

from __future__ import annotations

import threading

_MPS_LOCK = threading.Lock()


def torch_device(requested: str | None) -> str:
    """The configured device, or the fastest one available: cuda, then mps, then cpu."""
    if requested:
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)  # absent before torch 1.12
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def model_lock(device: str) -> threading.Lock:
    """The lock a model's calls run under: one lock shared by every model on MPS, else a new one."""
    return _MPS_LOCK if device.startswith("mps") else threading.Lock()
