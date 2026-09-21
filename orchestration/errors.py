"""Errors raised by the orchestration layer."""

from __future__ import annotations


class OrchestrationError(RuntimeError):
    """An unrecoverable failure while running the Paper Search workflow."""

