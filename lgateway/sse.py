"""SSE endpoint: serves lbus trading events to saitsCloud frontend.

GET /lbus/sse?event_type=trading_event

Auth: x-tenant-id-verified from sc-gateway (Clerk + DPoP).
Response: text/event-stream — browser EventSource compatible.

Events are read from lbus stream stream:{tenant_id}:{event_type} via
XREAD (blocking, cursor-based). Each event is forwarded as:
  data: {"topic": "...", "data": {...}}\n\n

The stream is populated by lbus.bridge.run_bridge() which consumes
the trading bot's /logs/sse endpoint.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

import redis.asyncio as aioredis
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from lbus import _stream_key, REDIS_URL

logger = logging.getLogger("sc-ai.lgateway.sse")

router = APIRouter()

_READ_BLOCK_MS = int(os.environ.get("SSE_READ_BLOCK_MS", "5000"))
_MAX_BACKLOG = int(os.environ.get("SSE_MAX_BACKLOG", "100"))


@router.get("/lbus/sse")
async def lbus_sse(
    request: Request,
    event_type: str = "trading_event",
    x_tenant_id_verified: str = Header(..., alias="x-tenant-id-verified"),
):
    """SSE stream of lbus events for the authenticated tenant.

    Reads from stream:{tenant_id}:{event_type} via async XREAD.
    Starts from newest messages — does not replay history.
    Browser EventSource auto-reconnects; last-event-id is NOT used
    (fire-and-forget, no guaranteed delivery).

    Args:
        event_type: lbus stream type (default: trading_event)
        x-tenant-id-verified: injected by sc-gateway, never client-supplied

    Returns:
        text/event-stream with JSON data frames
    """
    tenant_id = x_tenant_id_verified
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Missing tenant identity")

    stream_key = _stream_key(tenant_id, event_type)

    async def event_generator():
        r = aioredis.from_url(
            REDIS_URL, decode_responses=True,
            socket_keepalive=True, health_check_interval=30,
        )
        # Start from $ — only new events after connection
        last_id = "$"
        try:
            while True:
                if await request.is_disconnected():
                    return
                try:
                    entries = await r.xread(
                        {stream_key: last_id},
                        count=_MAX_BACKLOG,
                        block=_READ_BLOCK_MS,
                    )
                except Exception as exc:
                    logger.debug("lbus/sse redis error: %s", exc)
                    yield f"data: {json.dumps({'type': 'reconnecting'})}\n\n"
                    await asyncio.sleep(2)
                    continue

                if not entries:
                    # Timeout — send keepalive ping
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    continue

                for _key, messages in entries:
                    for msg_id, fields in messages:
                        last_id = msg_id
                        try:
                            data = json.loads(fields.get("data", "{}"))
                            payload = {
                                "topic": fields.get("topic", ""),
                                "data": data,
                                "lbus_id": msg_id,
                            }
                            yield f"data: {json.dumps(payload, default=str)}\n\n"
                        except Exception:
                            continue
        finally:
            try:
                await r.aclose()
            except Exception:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
