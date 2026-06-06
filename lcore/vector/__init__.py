"""Redis-Stack HNSW vector index.

Per-tenant index: FT.CREATE idx:{tenant_id}:vectors ON HASH PREFIX 1 vec:{tenant_id}:
Each document stored as HASH with:
  - vec   : FLOAT32 blob (embedding bytes)
  - text  : source chunk
  - src_id: origin document id

ALLES-REDIS: no external vector-db; Redis-Stack handles KV + vector search.
"""
from __future__ import annotations

import os
import struct

import redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query


REDIS_URL   = os.environ["REDIS_URL"]
VECTOR_DIM  = int(os.environ.get("VECTOR_DIM", "1536"))   # e.g. 1536 for text-embedding-3-small
HNSW_M      = int(os.environ.get("HNSW_M", "16"))
HNSW_EF     = int(os.environ.get("HNSW_EF_CONSTRUCTION", "200"))

_client: redis.Redis | None = None


def _redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(REDIS_URL, decode_responses=False)
    return _client


def ensure_index(tenant_id: str) -> None:
    """Create HNSW index for tenant if it does not exist."""
    idx = f"idx:{tenant_id}:vectors"
    try:
        _redis().ft(idx).info()
    except Exception:
        _redis().ft(idx).create_index(
            fields=[
                TextField("text"),
                TagField("src_id"),
                VectorField(
                    "vec",
                    "HNSW",
                    {"TYPE": "FLOAT32", "DIM": VECTOR_DIM, "DISTANCE_METRIC": "COSINE",
                     "M": HNSW_M, "EF_CONSTRUCTION": HNSW_EF},
                ),
            ],
            definition=IndexDefinition(
                prefix=[f"vec:{tenant_id}:"],
                index_type=IndexType.HASH,
            ),
        )


def upsert(tenant_id: str, doc_id: str, text: str, embedding: list[float]) -> None:
    """Store a chunk and its embedding under the tenant's vector index."""
    ensure_index(tenant_id)
    key  = f"vec:{tenant_id}:{doc_id}"
    blob = struct.pack(f"{len(embedding)}f", *embedding)
    _redis().hset(key, mapping={"text": text, "src_id": doc_id, "vec": blob})


def search(
    tenant_id: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[tuple[str, float, str]]:
    """KNN vector search scoped to a single tenant.

    Returns list of (doc_id, score, text) sorted by ascending distance.
    """
    ensure_index(tenant_id)
    idx  = f"idx:{tenant_id}:vectors"
    blob = struct.pack(f"{len(query_embedding)}f", *query_embedding)
    q    = (
        Query(f"*=>[KNN {top_k} @vec $vec AS __score]")
        .sort_by("__score")
        .paging(0, top_k)
        .dialect(2)
        .return_fields("src_id", "__score", "text")
    )
    results = _redis().ft(idx).search(q, query_params={"vec": blob})
    return [
        (doc.src_id, float(doc.__score), doc.text)
        for doc in results.docs
    ]
