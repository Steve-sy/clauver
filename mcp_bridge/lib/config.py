"""
Configuration helpers for the Clauver MCP bridge.
"""

from __future__ import annotations

import os


DEFAULT_AGENT_NAME = "clauver-general"


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def ensure_livekit_env() -> None:
    get_required_env("LIVEKIT_URL")
    get_required_env("LIVEKIT_API_KEY")
    get_required_env("LIVEKIT_API_SECRET")


def get_agent_name() -> str:
    return os.getenv("CLAUVER_AGENT_NAME", DEFAULT_AGENT_NAME)


def get_boss_name() -> str:
    return os.getenv("BOSS_NAME", "Boss")