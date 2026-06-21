"""
WebSocket handler for real-time job status updates.

Architecture:
  Worker ──publish──▶ Redis channel "job:{job_id}"
                              │
                     FastAPI subscribes via aioredis
                              │
                     WebSocket client ◀──forward──

Clients connect to ``/ws/jobs``, then send a JSON subscribe message:
  {"subscribe": ["job_id_1", "job_id_2", ...]}

The server subscribes to the corresponding Redis channels and forwards
any status payloads to the client as they arrive.

Fallback:  if the client can't open a WebSocket, React Query polling on
           GET /api/jobs?ids=... every 3 seconds covers the same data.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from app.config import settings

logger = logging.getLogger(__name__)


async def handle_jobs_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("WebSocket connected from %s", websocket.client)

    subscriptions: set[str] = set()
    redis_conn: Optional[object] = None
    pubsub = None

    try:
        import redis.asyncio as aioredis  # type: ignore[import]

        redis_conn = aioredis.from_url(settings.redis_url, decode_responses=True)
        pubsub = redis_conn.pubsub()

        async def redis_listener() -> None:
            """Forward Redis pub/sub messages to the WebSocket client.

            Uses get_message() polling instead of pubsub.listen() because listen()
            internally loops `while self.subscribed` — before the client sends its
            first subscribe message the loop exits immediately, triggering
            FIRST_COMPLETED and closing the connection.
            """
            while True:
                try:
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=5.0
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug("Redis pubsub read error: %s", exc)
                    await asyncio.sleep(0.1)
                    continue
                if message is None:
                    continue
                if message.get("type") == "message":
                    try:
                        await websocket.send_text(message["data"])
                    except WebSocketDisconnect:
                        return
                    except Exception as exc:
                        logger.debug("WebSocket send error: %s", exc)
                        return

        async def client_listener() -> None:
            """
            Process incoming messages from the client.
            Expected format: {"subscribe": ["id1", "id2"]}
                         or: {"unsubscribe": ["id1"]}
                         or: "ping"
            """
            nonlocal subscriptions

            while True:
                try:
                    raw = await websocket.receive_text()
                except WebSocketDisconnect:
                    return

                if raw.strip() == "ping":
                    await websocket.send_text("pong")
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if "subscribe" in msg:
                    new_ids = [str(i) for i in msg["subscribe"] if i not in subscriptions]
                    if new_ids:
                        channels = [f"job:{i}" for i in new_ids]
                        await pubsub.subscribe(*channels)
                        subscriptions.update(new_ids)
                        logger.debug("WS subscribed to %s", channels)

                elif "unsubscribe" in msg:
                    remove_ids = [str(i) for i in msg["unsubscribe"] if i in subscriptions]
                    if remove_ids:
                        channels = [f"job:{i}" for i in remove_ids]
                        await pubsub.unsubscribe(*channels)
                        subscriptions.difference_update(remove_ids)

        redis_task = asyncio.create_task(redis_listener())
        client_task = asyncio.create_task(client_listener())

        done, pending = await asyncio.wait(
            [redis_task, client_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except ImportError:
        logger.warning(
            "redis package not found — WebSocket will accept connections but won't push updates. "
            "Install redis: pip install redis"
        )
        # Keep the connection alive so the client's fallback polling kicks in
        try:
            while True:
                text = await websocket.receive_text()
                if text.strip() == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            pass
    except Exception as exc:
        logger.error("WebSocket error: %s", exc, exc_info=True)
    finally:
        if pubsub is not None:
            try:
                if subscriptions:
                    await pubsub.unsubscribe(*[f"job:{i}" for i in subscriptions])
                await pubsub.aclose()
            except Exception:
                pass
        if redis_conn is not None:
            try:
                await redis_conn.aclose()
            except Exception:
                pass
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("WebSocket cleanup complete")
