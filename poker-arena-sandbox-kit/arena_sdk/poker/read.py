"""Read the `table` — the bits every poker bot needs but that aren't a single
field. Position especially: the table has NO `button`/`dealer`/`position` field,
so you derive it. These helpers work identically on the local `table` and the
live server `table`.

    from arena_sdk.poker.read import hero, is_button, to_call, pot_odds
"""
from __future__ import annotations

from typing import Optional


def hero(table: dict) -> dict:
    """Your seat dict (the one whose seatNumber == selfSeatNumber)."""
    me = table.get("selfSeatNumber")
    return next((s for s in (table.get("seats") or []) if s.get("seatNumber") == me), {})


def hole_cards(table: dict) -> list:
    """Your two hole cards (they live under YOUR seat, not at table['holeCards'])."""
    return list(hero(table).get("holeCards") or [])


def button_seat(table: dict) -> Optional[int]:
    """Seat number of the button — **heads-up only** (the Arena sandbox is HU).
    Heads-up the button posts the small blind, so we return the seat that posted
    it in `recentEvents`. With more than 2 seats the button is NOT the small-blind
    poster, so this returns None rather than mislead. None too if the blind posts
    aren't in the events yet."""
    if len([s for s in (table.get("seats") or [])]) > 2:
        return None
    sb = table.get("smallBlindChips")
    for ev in table.get("recentEvents") or []:
        s = ev.get("summary") or {}
        if ev.get("type") == "BlindPosted" and s.get("amount") == sb:
            return s.get("seatNumber")
    return None


def is_button(table: dict) -> bool:
    """Are YOU the button (heads-up: the small blind, in position postflop)?
    Heads-up only — always False with more than 2 seats (see `button_seat`)."""
    btn = button_seat(table)
    return btn is not None and btn == table.get("selfSeatNumber")


def to_call(table: dict) -> int:
    """Chips you must add to call right now (0 = checking is free)."""
    return int((table.get("allowedActions") or {}).get("callChips") or 0)


def pot_odds(table: dict) -> float:
    """Break-even equity to call: call / (pot + call). 0.0 when checking is free."""
    call = to_call(table)
    if call <= 0:
        return 0.0
    pot = int(table.get("potChips") or 0)
    return call / (pot + call)


def can(table: dict, action: str) -> bool:
    """Is `action` ('fold'|'check'|'call'|'bet'|'raise') legal right now?"""
    return action in ((table.get("allowedActions") or {}).get("availableActions") or [])
