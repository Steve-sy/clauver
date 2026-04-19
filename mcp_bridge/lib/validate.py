"""
Validation helpers for Clauver MCP inputs.
"""

from __future__ import annotations

import re
import unicodedata

from mcp_bridge.lib.config import get_boss_name


E164_PATTERN = re.compile(r"^\+[1-9]\d{1,14}$")

VAGUE_TASKS = {
    "call",
    "book",
    "schedule",
    "appointment",
    "make a call",
    "phone",
    "message",
    "send message",
    "tell them",
    "let them know",
    "contact them",
    "follow up",
    "help",
    "do it",
    "sort it out",
    "handle it",
    "talk to them",
    "speak to them",
    "check with them",
    "find out",
    "ask them",
    "test",
    "testing",
    "hello",
    "hi",
}


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()


def validate_phone_number(phone_number: str) -> str:
    if not isinstance(phone_number, str):
        raise ValueError("phone_number must be a string")

    normalized = _normalize_text(phone_number)

    if not normalized:
        raise ValueError("phone_number is required")

    if not E164_PATTERN.fullmatch(normalized):
        raise ValueError(
            "phone_number must be in strict E.164 format, e.g. +61412345678"
        )

    return normalized


def validate_task(task: str) -> str:
    if not isinstance(task, str):
        raise ValueError("task must be a string")

    normalized = _normalize_text(task)

    if not normalized:
        raise ValueError("task is required")

    if len(normalized) < 8:
        raise ValueError("task is too short; provide the exact message or goal")

    lowered = normalized.lower()
    if lowered in VAGUE_TASKS:
        raise ValueError(
            "task is too vague; provide the exact message or concrete goal"
        )

    if len(lowered.split()) <= 2 and lowered in {
        "call him",
        "call her",
        "call them",
        "message him",
        "message her",
        "message them",
        "tell him",
        "tell her",
        "tell them",
    }:
        raise ValueError(
            "task is too vague; provide the exact message or concrete goal"
        )

    return normalized


def validate_target_name(target_name: str | None) -> str | None:
    if target_name is None:
        return None

    if not isinstance(target_name, str):
        raise ValueError("target_name must be a string if provided")

    normalized = _normalize_text(target_name)

    if not normalized:
        return None

    if len(normalized) > 80:
        raise ValueError("target_name is too long")

    return normalized


def validate_boss(boss: str | None) -> str:
    if boss is None:
        return get_boss_name()

    if not isinstance(boss, str):
        raise ValueError("boss must be a string if provided")

    normalized = _normalize_text(boss)

    if not normalized:
        return get_boss_name()

    if len(normalized) > 80:
        raise ValueError("boss is too long")

    return normalized