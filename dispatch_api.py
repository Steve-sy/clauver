"""
Reusable LiveKit dispatch helper for Clauver.

This module creates explicit agent dispatches via the LiveKit Python API.
It is intended to be shared by scripts, MCP bridge code, and other internal tooling.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from livekit import api

from mcp_bridge.lib.config import get_boss_name

DEFAULT_AGENT_NAME = "clauver-general"
DEFAULT_MODE = "message_delivery"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def get_default_agent_name() -> str:
    return os.getenv("CLAUVER_AGENT_NAME", DEFAULT_AGENT_NAME)


def build_room_name(prefix: str = "clauver-dispatch") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def build_dispatch_metadata(
    *,
    phone_number: str,
    task: str,
    target_name: str | None = None,
    boss: str | None = None,
    source: str = "dispatch_api",
    mode: str = DEFAULT_MODE,
    request_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    if boss is None:
        boss = get_boss_name()
    if request_id is None:
        request_id = str(uuid.uuid4())

    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()

    return {
        "version": "1",
        "mode": mode,
        "source": source,
        "request_id": request_id,
        "created_at": created_at,
        "phone_number": phone_number,
        "boss": boss,
        "target_name": target_name,
        "task": task,
    }


async def create_dispatch(
    *,
    phone_number: str,
    task: str,
    target_name: str | None = None,
    boss: str | None = None,
    agent_name: str | None = None,
    room_name: str | None = None,
    source: str = "dispatch_api",
    mode: str = DEFAULT_MODE,
) -> dict[str, Any]:
    require_env("LIVEKIT_URL")
    require_env("LIVEKIT_API_KEY")
    require_env("LIVEKIT_API_SECRET")

    resolved_agent_name = agent_name or get_default_agent_name()
    resolved_room_name = room_name or build_room_name()

    metadata = build_dispatch_metadata(
        phone_number=phone_number,
        task=task,
        target_name=target_name,
        boss=boss,
        source=source,
        mode=mode,
    )

    lkapi = api.LiveKitAPI()

    try:
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=resolved_agent_name,
                room=resolved_room_name,
                metadata=json.dumps(metadata),
            )
        )

        return {
            "dispatch_id": getattr(dispatch, "id", None),
            "room": getattr(dispatch, "room", resolved_room_name),
            "agent_name": getattr(dispatch, "agent_name", resolved_agent_name),
            "request_id": metadata["request_id"],
            "metadata": metadata,
        }
    finally:
        await lkapi.aclose()