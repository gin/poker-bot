import os
import sys
from poker_bot.strategies.nnbase import choose_action as nn_choose_action


def choose_action(
    table: dict, my_seat: dict | None
) -> tuple[str | None, int | None, str]:
    """Strategy interface compatible with benchmark runner."""
    if my_seat is None:
        return None, None, "No seat found"

    action, amount, msg = nn_choose_action(table, my_seat)
    return action, int(amount) if amount else 0, msg


def act(table: dict) -> dict:
    my_seat_num = table.get("actingSeatNumber") or table.get("selfSeatNumber")
    if my_seat_num is None:
        return {
            "action": "fold",
            "amount": 0,
            "message": "No acting seat found, folding",
        }
    seats = table.get("seats", [])
    my_seat = next((s for s in seats if s.get("seatNumber") == my_seat_num), {})
    if not my_seat:
        my_seat = {
            "seatNumber": my_seat_num,
            "holeCards": table.get("holeCards", table.get("hero_cards", [])),
            "currentBetChips": table.get("currentBetChips", table.get("bet", 0)),
            "stackChips": table.get("stackChips", table.get("stack", 0)),
            "bet": table.get("bet", 0),
        }
    action, amount, msg = choose_action(table, my_seat)
    return {
        "action": action,
        "amount": amount,
        "message": msg,
    }
