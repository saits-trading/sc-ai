"""Shared typed envelopes for bus and RAG pipeline."""

from dataclasses import dataclass
from typing import Any


@dataclass
class QueryEnvelope:
    query: str
    org_id: str       # Clerk org — namespaces Redis vector index
    top_k: int = 5
    metadata: dict[str, Any] | None = None


@dataclass
class RetrievalResult:
    chunks: list[str]
    scores: list[float]
    source_ids: list[str]
