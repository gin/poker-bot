"""Balanced survival strategy with modest tight-aggressive corrections.

This strategy uses survival_lookahead as its blueprint, then adds a preflop
discipline layer so it does not profile as loose-passive.
"""

from __future__ import annotations

from poker_bot.strategies import survival_lookahead, survival_lookup
from poker_bot.strategies.adaptive import (
    BIG_BLIND,
    board_texture,
    capped,
    has_top_pair_or_better,
    made_hand_rank,
    pot_odds,
    preflop_score,
)

ActionDecision = tuple[str | None, int | None, str]


def active_seat_numbers(table):
    return [
        int(seat.get("seatNumber"))
        for seat in table.get("seats", [])
        if not seat.get("folded", False)
        and not seat.get("hasFolded", False)
        and seat.get("seatNumber") is not None
    ]


def position_bucket(table, my_seat):
    seats = active_seat_numbers(table)
    player_count = max(2, len(seats))
    button = int(table.get("buttonSeatNumber") or seats[0] if seats else 1)
    seat_number = int(my_seat.get("seatNumber") or button)
    offset = (seat_number - button) % player_count

    if player_count <= 3:
        return "short"
    if offset == 0 or offset == player_count - 1:
        return "late"
    if offset in {1, 2}:
        return "blind"
    if offset == 3:
        return "early"
    return "middle"


def preflop_thresholds(table, my_seat):
    active = survival_lookup.active_players(table)
    position = position_bucket(table, my_seat)
    if active <= 3 or position == "short":
        return 56, 46
    if position == "late":
        return 62, 49
    if position == "middle":
        return 68, 52
    if position == "blind":
        return 70, 50
    return 74, 56


def balanced_raise_amount(table, allowed, score):
    minimum = allowed.get("minRaiseTo")
    if minimum is None:
        return None
    if score < 74:
        return int(minimum)
    pot = int(table.get("potChips") or 0)
    current_bet = int(table.get("currentBet") or 0)
    fraction = 0.45 if score >= 90 else 0.25
    target = max(int(minimum), current_bet + int(max(pot, BIG_BLIND) * fraction))
    return capped(target, allowed)


def balanced_bet_amount(table, allowed, strong=False):
    minimum = int(allowed.get("minBet") or BIG_BLIND)
    pot = int(table.get("potChips") or 0)
    fraction = 0.50 if strong else 0.34
    return capped(max(minimum, int(max(pot, BIG_BLIND) * fraction)), allowed)


def balanced_preflop_action(table, my_seat):
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    hole_cards = my_seat.get("holeCards", [])
    score = preflop_score(hole_cards)
    pot = int(table.get("potChips") or 0) + sum(
        int(seat.get("currentBetChips") or 0) for seat in table.get("seats", [])
    )
    call_amount = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    required = pot_odds(call_amount, pot)
    raise_threshold, call_threshold = preflop_thresholds(table, my_seat)
    blind_size = max(int(allowed.get("minBet") or 0), BIG_BLIND)
    facing_raise = (
        int(table.get("currentBet") or 0) > blind_size or call_amount > blind_size
    )

    if "raise" in available and score >= raise_threshold:
        amount = balanced_raise_amount(table, allowed, score)
        return "raise", amount, f"balanced value/open raise score {score}"

    if "call" in available:
        if facing_raise:
            if score >= raise_threshold + 2 or (
                score >= call_threshold + 8 and required <= 0.14
            ):
                return "call", call_amount, f"balanced defend score {score}"
        elif score >= call_threshold and required <= 0.38:
            return "call", call_amount, f"balanced selective call score {score}"

        if "check" in available:
            return "check", None, f"balanced preflop check score {score}"
        return "fold", None, f"balanced preflop fold score {score}"

    if "bet" in available and score >= call_threshold + 6:
        amount = balanced_bet_amount(table, allowed, strong=score >= raise_threshold)
        return "bet", amount, f"balanced preflop bet score {score}"

    if "check" in available:
        return "check", None, f"balanced preflop check score {score}"
    if "fold" in available:
        return "fold", None, f"balanced preflop fold score {score}"
    return None


def balanced_postflop_adjustment(table, my_seat, blueprint):
    action, _amount, _message = blueprint
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if action != "check" or "bet" not in available:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards) if board_cards else {"wet": False}
    active = survival_lookup.active_players(table)

    if rank >= 2 or top_pair:
        amount = balanced_bet_amount(table, allowed, strong=rank >= 3)
        return "bet", amount, f"balanced value pressure rank {rank}"

    if (
        active <= 2
        and not texture.get("wet", False)
        and preflop_score(hole_cards) >= 70
    ):
        amount = balanced_bet_amount(table, allowed)
        return "bet", amount, "balanced dry-board continuation"

    return None


def choose_action(table, my_seat) -> ActionDecision:
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if not available:
        return None, None, "No legal actions available"
    if not my_seat:
        if "fold" in available:
            return "fold", None, "Fallback: seat not found"
        return None, None, "No matching seat found"

    if table.get("street", "Preflop") == "Preflop":
        preflop = balanced_preflop_action(table, my_seat)
        if preflop is not None:
            return preflop

    blueprint = survival_lookahead.choose_action(table, my_seat)
    postflop = balanced_postflop_adjustment(table, my_seat, blueprint)
    if postflop is not None:
        return postflop

    action, amount, message = blueprint
    return action, amount, f"balanced blueprint: {message}"
