"""Auto-researched strategy based on survival_balanced_pp_pd_pr_counter.

The counter strategy is already strong against patch1-style opponents. This
iteration keeps that baseline intact, preserves the proven six-max branch, and
adds narrower short-handed pressure where counter remains conservative.
"""

from __future__ import annotations

from poker_bot.strategies import survival_balanced_pp_pd_pr_counter as counter
from poker_bot.strategies.adaptive import (
    BIG_BLIND,
    board_texture,
    has_top_pair_or_better,
    made_hand_rank,
    pot_odds,
    preflop_score,
)

ActionDecision = tuple[str | None, int | None, str]


def research_preflop_pressure(table, my_seat, base):
    if table.get("street", "Preflop") != "Preflop" or not counter.short_handed(table):
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "raise" not in available or not counter.unopened_preflop(table, allowed):
        return None

    hole_cards = my_seat.get("holeCards", [])
    score = preflop_score(hole_cards)
    position = counter.patch1.position_bucket(table, my_seat)
    threshold = 49 if position in {"short", "late"} else 54
    if action in {"call", "check", "fold"} and score >= threshold:
        amount = counter.patch1.balanced_raise_amount(table, allowed, max(score, 56))
        return "raise", amount, f"auto research widened open score {score}"
    return None


def research_probe_pressure(table, my_seat, base):
    if table.get("street", "Preflop") == "Preflop" or not counter.short_handed(table):
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if action != "check" or "bet" not in available:
        return None
    if not counter.patch1.no_one_has_bet(allowed, table):
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    texture = board_texture(board_cards) if board_cards else {"wet": False}
    if texture.get("wet", False):
        return None

    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    score = preflop_score(hole_cards)
    if rank >= 2 and counter.patch1.fragile_rank_two(hole_cards, board_cards, rank):
        return None
    if rank >= 1 or top_pair or score >= 49:
        amount = counter.pressure_bet_amount(
            table,
            allowed,
            0.31 if rank or top_pair else 0.22,
        )
        return "bet", amount, f"auto research thin dry-board probe rank {rank}"
    return None


def research_bluff_catch(table, my_seat, base):
    if table.get("street", "Preflop") == "Preflop" or not counter.short_handed(table):
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    price = counter.call_amount(allowed)
    if action != "fold" or "call" not in available or price <= 0:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = int(table.get("potChips") or 0)
    required = pot_odds(price, pot)
    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    draw = counter.patch1.has_good_draw(hole_cards, board_cards)
    stack = int(my_seat.get("stackChips") or 0)
    cheap_stack_price = stack <= 0 or price <= max(BIG_BLIND, int(stack * 0.12))

    if (rank == 1 or top_pair or draw) and required <= 0.20 and cheap_stack_price:
        return "call", price, f"auto research cheap bluff-catch rank {rank}"
    return None


def research_short_handed_action(table, my_seat, base):
    for adjustment in (
        research_preflop_pressure,
        research_bluff_catch,
        research_probe_pressure,
    ):
        action = adjustment(table, my_seat, base)
        if action is not None:
            return action
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

    base = counter.choose_action(table, my_seat)
    if counter.seated_players(table) < 4:
        research = research_short_handed_action(table, my_seat, base)
        if research is not None:
            return research

    action, amount, message = base
    return action, amount, f"0:{message}"
