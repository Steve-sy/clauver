"""
Canonical metadata helpers for Clauver MCP dispatches.
"""

from __future__ import annotations

from typing import Any

from dispatch_api import build_dispatch_metadata
from mcp_bridge.lib.validate import (
    validate_boss,
    validate_phone_number,
    validate_target_name,
    validate_task,
)


def build_canonical_metadata(
    *,
    phone_number: str,
    task: str,
    target_name: str | None = None,
    boss: str | None = None,
    source: str = "mcp_bridge",
) -> dict[str, Any]:
    validated_phone_number = validate_phone_number(phone_number)
    validated_task = validate_task(task)
    validated_target_name = validate_target_name(target_name)
    validated_boss = validate_boss(boss)

    return build_dispatch_metadata(
        phone_number=validated_phone_number,
        task=validated_task,
        target_name=validated_target_name,
        boss=validated_boss,
        source=source,
    )