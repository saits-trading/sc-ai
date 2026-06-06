"""pub-sse-bridge: trading-bot SSE → lbus stream fanout.

Connects to the trading bot's /logs/sse endpoint (EventSource) and forwards
events into lbus streams keyed by stream:{tenant_id}:trading_event.

This lets saitsCloud tenants subscribe to real-time trading events via
GET /lbus/sse without sc-ai needing direct access to the bot's Redis.

Config (env):
  BOT_SSE_URL       — full URL of bot's /logs/sse endpoint (required)
  BOT_SSE_TOKEN     — HMAC token for bot's token= auth query param (required)
  BOT_TENANT_ID     — Clerk org / lbus tenant for routing (required)
  BRIDGE_DEDUP_TTL  — Redis SETEX TTL for seen message-id dedup (default 300s)
  BRIDGE_RECONNECT_DELAY — seconds before reconnect on disconnect (default 5)

Lifecycle: call `run_bridge()` from the app lifespan as an asyncio.Task.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import AsyncIterator

import httpx
import redis.asyncio as aioredis

from lbus import _stream_key, REDIS_URL, STREAM_MAXLEN

logger = logging.getLogger("sc-ai.lbus.bridge")

_BOT_SSE_URL = os.environ.get("BOT_SSE_URL", "")
_BOT_SSE_TOKEN = os.environ.get("BOT_SSE_TOKEN", "")
_BOT_TENANT_ID = os.environ.get("BOT_TENANT_ID", "")
_DEDUP_TTL = int(os.environ.get("BRIDGE_DEDUP_TTL", "300"))
_RECONNECT_DELAY = float(os.environ.get("BRIDGE_RECONNECT_DELAY", "5"))

_STREAM_TYPE = "trading_event"


async def _sse_lines(url: str, token: str) -> AsyncIterator[str]:
    """Yield raw SSE lines from the bot endpoint. Reconnects on any error."""
    full_url = f"{url}?token={token}" if token else url
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("GET", full_url, headers={"Accept": "text/event-stream"}) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                yield line


async def _publish_event(r: aioredis.Redis, tenant_id: str, event: dict) -> None:
    """Write one trading event to the lbus stream and set dedup key."""
    msg_id = str(event.get("id") or event.get("exec_id") or "")
    if msg_id:
        dedup_key = f"bridge:dedup:{tenant_id}:{msg_id}"
        if await r.exists(dedup_key):
            return
        await r.setex(dedup_key, _DEDUP_TTL, "1")

    stream_key = _stream_key(tenant_id, _STREAM_TYPE)
    payload = {
        "topic": event.get("topic", ""),
        "data": json.dumps(event.get("data", event), default=str),
        "ts": str(time.time()),
    }
    await r.xadd(stream_key, payload, maxlen=STREAM_MAXLEN, approximate=True)
    logger.debug("bridge → %s: %s", stream_key, payload.get("topic"))


async def run_bridge() -> None:
    """Long-running bridge task. Reconnects on disconnect or parse error.

    Start as: asyncio.create_task(run_bridge())
    """
    if not _BOT_SSE_URL or not _BOT_TENANT_ID:
        logger.warning("pub-sse-bridge disabled: BOT_SSE_URL or BOT_TENANT_ID not set")
        return

    r = aioredis.from_url(REDIS_URL, decode_responses=True, socket_keepalive=True)
    logger.info("pub-sse-bridge starting: %s → tenant=%s", _BOT_SSE_URL, _BOT_TENANT_ID)

    while True:
        try:
            async for line in _sse_lines(_BOT_SSE_URL, _BOT_SSE_TOKEN):
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                # Skip reconnecting heartbeats
                if event.get("type") == "reconnecting":
                    continue
                await _publish_event(r, _BOT_TENANT_ID, event)
        except Exception as exc:
            logger.info("pub-sse-bridge disconnected: %s — reconnecting in %ss", exc, _RECONNECT_DELAY)
            await asyncio.sleep(_RECONNECT_DELAY)

    await r.aclose()
