"""Retrieval-augmented generation chain.

Pipeline: embed query → vector search → context assembly → LLM completion.
All storage is Redis-Stack (no external vector-db).
"""
from __future__ import annotations

import os
from typing import Iterator

import httpx

from lcore.vector import search
from lproto.schemas import QueryEnvelope, RetrievalResult

EMBED_URL    = os.environ["EMBED_URL"]     # e.g. https://api.openai.com/v1/embeddings
EMBED_MODEL  = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
EMBED_API_KEY = os.environ["EMBED_API_KEY"]

COMPLETION_URL = os.environ["COMPLETION_URL"]
COMPLETION_MODEL = os.environ.get("COMPLETION_MODEL", "gpt-4o-mini")
COMPLETION_API_KEY = os.environ.get("COMPLETION_API_KEY", EMBED_API_KEY)


def _embed(text: str) -> list[float]:
    r = httpx.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {EMBED_API_KEY}"},
        json={"input": text, "model": EMBED_MODEL},
        timeout=10,
    ).raise_for_status().json()
    return r["data"][0]["embedding"]


def retrieve(envelope: QueryEnvelope) -> RetrievalResult:
    """Embed the query and run KNN search against the tenant's vector index.

    Returns ranked chunks and scores; tenant scope enforced by tenant_id prefix.
    """
    vec     = _embed(envelope.query)
    hits    = search(envelope.tenant_id, vec, top_k=envelope.top_k)
    return RetrievalResult(
        chunks    = [h[2] for h in hits],
        scores    = [h[1] for h in hits],
        source_ids= [h[0] for h in hits],
    )


def generate(query: str, context_chunks: list[str]) -> Iterator[str]:
    """Stream a completion given retrieved context chunks.

    Yields response tokens as they arrive.
    """
    system = (
        "You are a helpful assistant. Answer only from the provided context. "
        "If the context does not contain the answer, say so."
    )
    context_text = "\n---\n".join(context_chunks)
    messages = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": f"Context:\n{context_text}\n\nQuestion: {query}"},
    ]
    with httpx.stream(
        "POST",
        COMPLETION_URL,
        headers={"Authorization": f"Bearer {COMPLETION_API_KEY}"},
        json={"model": COMPLETION_MODEL, "messages": messages, "stream": True},
        timeout=60,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("data: ") and not line.endswith("[DONE]"):
                chunk = line[6:]
                try:
                    import json
                    delta = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    pass


def rag(envelope: QueryEnvelope) -> Iterator[str]:
    """Full RAG pipeline: retrieve → generate (streaming).

    Retrieves context from the tenant-scoped vector index, then streams
    an LLM completion grounded in that context.
    """
    result = retrieve(envelope)
    yield from generate(envelope.query, result.chunks)
