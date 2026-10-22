"""Instruction -> Plan. Uses Llama 3.1 when configured, else the rule-based mock."""

from __future__ import annotations

from ..config import settings
from ..schemas import Plan
from . import llama, mock


def make_plan(instruction: str, base: Plan | None = None) -> tuple[Plan, str | None]:
    """Returns the plan and, if the live planner failed, the reason it fell back."""
    if settings.planner_live:
        try:
            return llama.plan(instruction, base), None
        except Exception as exc:  # network, auth, or invalid output: keep the demo alive
            fallback = mock.plan(instruction, base)
            fallback.source = "mock planner (fallback)"
            return fallback, f"Llama planner failed ({exc}); used the rule-based planner instead."
    return mock.plan(instruction, base), None
