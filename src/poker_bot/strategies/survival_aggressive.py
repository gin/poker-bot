"""Tighter aggressive survival variant for meta testing."""

from __future__ import annotations

from poker_bot.strategies import survival_balanced, survival_lookahead, survival_lookup
from poker_bot.strategies.adaptive import (
    board_texture,
    has_top_pair_or_better,
    made_hand_rank,
    pot_odds,
    preflop_score,
)

ActionDecision = tuple[str | None, int | None, str]


def aggressive_thresholds(table, my_seat):
    active = survival_lookup.active_players(table)
    position = survival_balanced.position_bucket(table, my_seat)
    if active <= 3 or position == "short":
        return 54, 52
    if position == "late":
        return 60, 58
    if position == "middle":
        return 66, 64
    if position == "blind":
        return 68, 64
    return 72, 68


def effective_preflop_pot(table):
    return int(table.get("potChips") or 0) + sum(
        int(seat.get("currentBetChips") or 0) for seat in table.get("seats", [])
    )


def aggressive_preflop_action(table, my_seat):
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    hole_cards = my_seat.get("holeCards", [])
    score = preflop_score(hole_cards)
    call_amount = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    required = pot_odds(call_amount, effective_preflop_pot(table))
    raise_threshold, call_threshold = aggressive_thresholds(table, my_seat)
    blind_size = max(int(allowed.get("minBet") or 0), survival_balanced.BIG_BLIND)
    facing_raise = (
        int(table.get("currentBet") or 0) > blind_size or call_amount > blind_size
    )

    if "raise" in available and score >= raise_threshold:
        amount = survival_balanced.balanced_raise_amount(table, allowed, score)
        return "raise", amount, f"aggressive open/value raise score {score}"

    if "call" in available:
        if facing_raise and (
            score >= raise_threshold + 4
            or (score >= call_threshold + 6 and required <= 0.12)
        ):
            return "call", call_amount, f"aggressive defend score {score}"
        if "check" in available:
            return "check", None, f"aggressive check score {score}"
        return "fold", None, f"aggressive fold score {score}"

    if "bet" in available and score >= call_threshold:
        amount = survival_balanced.balanced_bet_amount(
            table,
            allowed,
            strong=score >= raise_threshold,
        )
        return "bet", amount, f"aggressive preflop bet score {score}"

    if "check" in available:
        return "check", None, f"aggressive check score {score}"
    if "fold" in available:
        return "fold", None, f"aggressive fold score {score}"
    return None


def aggressive_postflop_adjustment(table, my_seat, blueprint):
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
        amount = survival_balanced.balanced_bet_amount(table, allowed, strong=rank >= 3)
        return "bet", amount, f"aggressive value pressure rank {rank}"

    if (
        active <= 2
        and not texture.get("wet", False)
        and preflop_score(hole_cards) >= 62
    ):
        amount = survival_balanced.balanced_bet_amount(table, allowed)
        return "bet", amount, "aggressive dry-board continuation"

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
        preflop = aggressive_preflop_action(table, my_seat)
        if preflop is not None:
            return preflop

    blueprint = survival_lookahead.choose_action(table, my_seat)
    postflop = aggressive_postflop_adjustment(table, my_seat, blueprint)
    if postflop is not None:
        return postflop

    action, amount, message = blueprint
    return action, amount, f"aggressive blueprint: {message}"
