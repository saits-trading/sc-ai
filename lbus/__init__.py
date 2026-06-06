"""lbus — message bus layer.

Pub/sub routing and event fanout over Redis Streams.
No business logic here; lbus only routes typed envelopes (lproto).

Stream key pattern: stream:{tenant_id}:{event_type}
Consumer group:     cg:{tenant_id}:{consumer_name}
"""
from __future__ import annotations

import os
import time

import redis

from lproto.schemas import QueryEnvelope, RetrievalResult

REDIS_URL      = os.environ["REDIS_URL"]
STREAM_MAXLEN  = int(os.environ.get("STREAM_MAXLEN", "10000"))
READ_BLOCK_MS  = int(os.environ.get("BUS_READ_BLOCK_MS", "5000"))

_client: redis.Redis | None = None


def _redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(REDIS_URL, decode_responses=True)
    return _client


def _stream_key(tenant_id: str, event_type: str) -> str:
    return f"stream:{tenant_id}:{event_type}"


def publish_query(envelope: QueryEnvelope) -> str:
    """Publish a query envelope to the tenant's query stream.

    Returns the Redis stream message id.
    """
    key = _stream_key(envelope.org_id, "query")
    return _redis().xadd(
        key,
        {"query": envelope.query, "top_k": str(envelope.top_k)},
        maxlen=STREAM_MAXLEN,
        approximate=True,
    )


def publish_result(tenant_id: str, result: RetrievalResult, msg_id: str = "") -> str:
    """Publish a retrieval result back onto the tenant's result stream."""
    key = _stream_key(tenant_id, "result")
    payload = {
        "chunks":     "\n---\n".join(result.chunks),
        "scores":     ",".join(str(s) for s in result.scores),
        "source_ids": ",".join(result.source_ids),
        "ref_msg_id": msg_id,
    }
    return _redis().xadd(key, payload, maxlen=STREAM_MAXLEN, approximate=True)


def ensure_consumer_group(tenant_id: str, event_type: str, group: str) -> None:
    key = _stream_key(tenant_id, event_type)
    try:
        _redis().xgroup_create(key, group, id="$", mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def read_queries(
    tenant_id: str,
    consumer: str,
    group: str = "lcore",
    count: int = 10,
    block_ms: int = READ_BLOCK_MS,
) -> list[tuple[str, dict]]:
    """Blocking read from the tenant's query stream.

    Returns list of (msg_id, fields) for pending messages.
    Caller must ACK via ack_message() after processing.
    """
    ensure_consumer_group(tenant_id, "query", group)
    key = _stream_key(tenant_id, "query")
    entries = _redis().xreadgroup(group, consumer, {key: ">"}, count=count, block=block_ms)
    if not entries:
        return []
    return [(msg_id, fields) for _, messages in entries for msg_id, fields in messages]


def ack_message(tenant_id: str, event_type: str, group: str, msg_id: str) -> None:
    _redis().xack(_stream_key(tenant_id, event_type), group, msg_id)
