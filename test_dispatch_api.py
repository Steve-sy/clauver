# test_dispatch_api.py

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from livekit import api


load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


async def main() -> None:
    require_env("LIVEKIT_URL")
    require_env("LIVEKIT_API_KEY")
    require_env("LIVEKIT_API_SECRET")

    agent_name = os.getenv("CLAUVER_AGENT_NAME", "clauver-general")
    room_name = os.getenv("CLAUVER_TEST_ROOM", f"clauver-test-{uuid.uuid4().hex[:8]}")

    metadata = {
        "version": "1",
        "mode": "message_delivery",
        "source": "test_dispatch_api.py",
        "request_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phone_number": os.getenv("CLAUVER_TEST_PHONE_NUMBER", "+61000000000"),
        "boss": os.getenv("CLAUVER_TEST_BOSS", "Max"),
        "target_name": os.getenv("CLAUVER_TEST_TARGET_NAME", "Patrick"),
        "task": os.getenv(
            "CLAUVER_TEST_TASK",
            "Tell Patrick that Max is sick today and will not be coming to work."
        ),
    }

    lkapi = api.LiveKitAPI()

    try:
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=agent_name,
                room=room_name,
                metadata=json.dumps(metadata),
            )
        )

        print("\n=== Dispatch created ===")
        print(f"dispatch_id: {getattr(dispatch, 'id', None)}")
        print(f"room:        {getattr(dispatch, 'room', None)}")
        print(f"agent_name:  {getattr(dispatch, 'agent_name', None)}")
        print(f"metadata:    {getattr(dispatch, 'metadata', None)}")

        dispatches = await lkapi.agent_dispatch.list_dispatch(room_name=room_name)

        print("\n=== Dispatches in room ===")
        print(f"count: {len(dispatches)}")
        for i, item in enumerate(dispatches, start=1):
            print(
                f"{i}. id={getattr(item, 'id', None)} "
                f"agent_name={getattr(item, 'agent_name', None)} "
                f"room={getattr(item, 'room', None)}"
            )

        print("\n=== Sent metadata ===")
        print(json.dumps(metadata, indent=2))

    finally:
        await lkapi.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)