"""Heads-up ``hutight001`` with a sole-nuts-only all-in policy."""

from __future__ import annotations

from poker_bot.guards.nut_all_in import veto_non_nut_all_in
from poker_bot.strategies.hutight001 import choose_action as _base_choose_action

ActionDecision = tuple[str | None, int | None, str]


def _resolve_my_seat(table: dict, my_seat: dict | None) -> dict | None:
    if my_seat is not None:
        return my_seat
    seat_number = table.get("actingSeatNumber") or table.get("selfSeatNumber")
    if seat_number is None:
        return None
    seat = next(
        (
            candidate
            for candidate in table.get("seats", [])
            if candidate.get("seatNumber") == seat_number
        ),
        None,
    )
    if seat is not None:
        return seat
    return {
        "seatNumber": seat_number,
        "holeCards": table.get("holeCards", table.get("hero_cards", [])),
        "currentBetChips": table.get("currentBetChips", table.get("bet", 0)),
        "stackChips": table.get("stackChips", table.get("stack", 0)),
        "bet": table.get("bet", 0),
    }


def choose_action(table: dict, my_seat: dict | None = None) -> ActionDecision:
    """Use ``hutight001`` unless its final decision would stack off non-nuts."""
    resolved_seat = _resolve_my_seat(table, my_seat)
    decision = _base_choose_action(table, resolved_seat)
    if not resolved_seat:
        return decision
    return veto_non_nut_all_in(table, resolved_seat, decision)


def act(table: dict) -> ActionDecision:
    """Strategy entry point matching the legacy strategy's table-only API."""
    return choose_action(table)
