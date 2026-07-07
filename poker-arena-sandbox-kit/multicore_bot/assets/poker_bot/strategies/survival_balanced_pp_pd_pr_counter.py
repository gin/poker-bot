"""Counter-strategy derived from survival_balanced_pp_pd_pr_patch1.

The base patch1 strategy is solid but conservative in two places that are
easy to exploit in self-play: short-handed dry boards where it checks too much,
and multiway pots where its heads-up-oriented pressure leaks chips. This
strategy keeps patch1 as the default line, adds selective heads-up pressure,
and delegates crowded tables to the 6-max survival profile.
"""

from __future__ import annotations

from poker_bot.strategies import survival_balanced_pp_pd_pr_patch1 as patch1
from poker_bot.strategies import survival_sixmax
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


def active_players(table):
    return max(1, len(patch1.active_seat_numbers(table)))


def seated_players(table):
    return sum(1 for seat in table.get("seats", []) if seat.get("agentId"))


def call_amount(allowed):
    return int(allowed.get("callAmount") or allowed.get("callChips") or 0)


def min_raise_to(allowed):
    if allowed.get("minRaiseTo") is not None:
        return int(allowed["minRaiseTo"])
    raise_range = allowed.get("raiseRange") or {}
    value = raise_range.get("min")
    return int(value) if value is not None else None


def min_bet(allowed):
    if allowed.get("minBet") is not None:
        return int(allowed["minBet"])
    bet_range = allowed.get("betRange") or {}
    return int(bet_range.get("min") or BIG_BLIND)


def pressure_bet_amount(table, allowed, fraction=0.28):
    pot = int(table.get("potChips") or 0)
    target = max(min_bet(allowed), int(max(pot, BIG_BLIND) * fraction))
    return capped(target, allowed)


def pressure_raise_amount(table, allowed, fraction=0.48):
    minimum = min_raise_to(allowed)
    if minimum is None:
        return None
    pot = int(table.get("potChips") or 0)
    current_bet = int(table.get("currentBet") or 0)
    target = current_bet + int(max(pot, BIG_BLIND) * fraction)
    return capped(max(minimum, target), allowed)


def short_handed(table):
    return active_players(table) <= 3


def unopened_preflop(table, allowed):
    blind_size = max(int(allowed.get("minBet") or 0), BIG_BLIND)
    return (
        int(table.get("currentBet") or 0) <= blind_size
        and call_amount(allowed) <= blind_size
    )


def heads_up_preflop_pressure(table, my_seat, base):
    if table.get("street", "Preflop") != "Preflop" or not short_handed(table):
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "raise" not in available or not unopened_preflop(table, allowed):
        return None

    hole_cards = my_seat.get("holeCards", [])
    score = preflop_score(hole_cards)
    position = patch1.position_bucket(table, my_seat)
    threshold = 51 if position in {"short", "late"} else 56
    if action in {"call", "check", "fold"} and score >= threshold:
        amount = patch1.balanced_raise_amount(table, allowed, max(score, 56))
        return "raise", amount, f"counter short-handed open pressure score {score}"
    return None


def heads_up_dry_board_pressure(table, my_seat, base):
    if table.get("street", "Preflop") == "Preflop" or not short_handed(table):
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if action != "check" or "bet" not in available:
        return None
    if not patch1.no_one_has_bet(allowed, table):
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    texture = board_texture(board_cards) if board_cards else {"wet": False}
    if texture.get("wet", False):
        return None

    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    score = preflop_score(hole_cards)
    paired = texture.get("paired", False)
    if rank >= 2 and patch1.fragile_rank_two(hole_cards, board_cards, rank):
        return None
    if rank >= 1 or top_pair or score >= (56 if paired else 52):
        amount = pressure_bet_amount(table, allowed, 0.34 if rank or top_pair else 0.26)
        return "bet", amount, f"counter dry-board pressure rank {rank}"
    return None


def heads_up_raise_pressure(table, my_seat, base):
    if table.get("street", "Preflop") == "Preflop" or not short_handed(table):
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    price = call_amount(allowed)
    if price <= 0 or "raise" not in available or action not in {"call", "fold"}:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = int(table.get("potChips") or 0)
    required = pot_odds(price, pot)
    texture = board_texture(board_cards) if board_cards else {"wet": False}
    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    draw = patch1.has_good_draw(hole_cards, board_cards)

    if rank >= 2 and not patch1.fragile_rank_two(hole_cards, board_cards, rank):
        amount = pressure_raise_amount(table, allowed, 0.62)
        return "raise", amount, f"counter value/protection raise rank {rank}"

    if (
        draw
        and not texture.get("paired", False)
        and required <= 0.20
        and patch1.stable_mix_percent(
            hole_cards,
            board_cards,
            table.get("street", "Flop"),
            pot,
        )
        < 55
    ):
        amount = pressure_raise_amount(table, allowed, 0.48)
        return "raise", amount, "counter semi-bluff raise"

    if top_pair and not texture.get("wet", False) and required <= 0.20:
        amount = pressure_raise_amount(table, allowed, 0.42)
        return "raise", amount, "counter dry-board top-pair raise"

    return None


def heads_up_counter_action(table, my_seat, base):
    for adjustment in (
        heads_up_preflop_pressure,
        heads_up_raise_pressure,
        heads_up_dry_board_pressure,
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

    if seated_players(table) >= 4:
        action, amount, message = survival_sixmax.choose_action(table, my_seat)
        return action, amount, f"counter six-max: {message}"

    base = patch1.choose_action(table, my_seat)
    counter = heads_up_counter_action(table, my_seat, base)
    if counter is not None:
        return counter

    action, amount, message = base
    return action, amount, f"counter patch1 baseline: {message}"
