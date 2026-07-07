"""Guard telemetry — in-process event buffer and mode overrides.

Strategies are pure functions with no DB access, so guard fires are buffered
here and drained by the selfplay observer (src/eval/selfplay.py) right after
each decision, where run_id/hand_id/decision_index are known. The buffer only
ever holds the current decision's events: the strategy clears it at the top
of choose_action, and the observer drains it after every action.

Env-var mode overrides let you A/B a guard without editing code:

    POKER_GUARD_DISABLE=id1,id2    disable listed guards entirely
    POKER_GUARD_SHADOW=id1,id2     force listed guards to shadow mode
    POKER_GUARD_ACTIVATE=id1,id2   force listed guards active (overrides
                                   a default-shadow registration)

Each accepts "*" to mean all guards. Precedence: DISABLE > SHADOW >
ACTIVATE > registration default (conservative: on conflict a guard
shadows rather than activates).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardEvent:
    """One guard fire (applied or shadow) for the current decision."""

    guard_id: str
    phase: str  # "pre" or "post"
    precedence: int
    shadow: bool
    applied: bool
    original_action: str  # "__pending__" for pre-guards (no core proposal yet)
    original_amount: int | None
    final_action: str
    final_amount: int | None
    reason: str
    street: str | None
    pot: int | None
    call_price: int | None
    available_actions: str  # comma-joined


_pending: list[GuardEvent] = []


def record_event(event: GuardEvent) -> None:
    _pending.append(event)


def drain_events() -> list[GuardEvent]:
    """Return buffered events and clear the buffer."""
    events = list(_pending)
    _pending.clear()
    return events


def clear_events() -> None:
    _pending.clear()


def _env_matches(name: str, guard_id: str) -> bool:
    raw = os.environ.get(name, "")
    ids = {part.strip() for part in raw.split(",") if part.strip()}
    return "*" in ids or guard_id in ids


def guard_mode(guard_id: str, *, shadow_default: bool) -> str:
    """Effective mode for a guard: "active", "shadow", or "off".

    Env overrides win over the registration-time shadow flag.
    Precedence: DISABLE > SHADOW > ACTIVATE > registration default.
    """
    if _env_matches("POKER_GUARD_DISABLE", guard_id):
        return "off"
    if _env_matches("POKER_GUARD_SHADOW", guard_id):
        return "shadow"
    if _env_matches("POKER_GUARD_ACTIVATE", guard_id):
        return "active"
    return "shadow" if shadow_default else "active"
