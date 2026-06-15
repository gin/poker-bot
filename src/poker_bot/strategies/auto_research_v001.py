"""Auto-research candidate v001.

This candidate keeps the current champion as the default policy and only
adjusts six-player spots where the baseline is intentionally conservative:
premium preflop open pressure and cheap postflop continues with showdown value
or strong draws.
"""

from __future__ import annotations

from poker_bot.strategies import auto_research as champion
from poker_bot.strategies.adaptive import (
    board_texture,
    has_top_pair_or_better,
    made_hand_rank,
    pot_odds,
    preflop_score,
)
from poker_bot.strategies.survival_sixmax import hand_class

ActionDecision = tuple[str | None, int | None, str]

PREMIUM_OPEN_CLASSES = {
    "AA",
    "KK",
    "QQ",
    "JJ",
    "TT",
    "AKs",
    "AKo",
    "AQs",
}


def call_amount(allowed):
    return int(allowed.get("callAmount") or allowed.get("callChips") or 0)


def blind_size(allowed):
    return max(1, int(allowed.get("minBet") or 0))


def cap_amount(amount, allowed):
    return max(0, min(int(amount), int(allowed.get("maxCommit") or amount)))


def active_opponents(table, my_seat):
    my_id = my_seat.get("agentId")
    return sum(
        1
        for seat in table.get("seats", [])
        if seat.get("agentId") != my_id
        and not seat.get("folded", False)
        and not seat.get("hasFolded", False)
    )


def seated_players(table):
    return sum(1 for seat in table.get("seats", []) if seat.get("agentId"))


def no_large_preflop_raise(table, allowed):
    blind = blind_size(allowed)
    return int(table.get("currentBet") or 0) <= blind and call_amount(allowed) <= blind


def profile_value(profile, name):
    value = getattr(profile, name, None)
    if value is not None:
        return value
    if isinstance(profile, dict):
        return profile.get(name)
    return None


def profile_call_frequency(profile):
    value = profile_value(profile, "call_frequency")
    if value is not None:
        return float(value)
    calls = int(profile_value(profile, "calls") or 0)
    bets = int(profile_value(profile, "bets") or 0)
    raises = int(profile_value(profile, "raises") or 0)
    folds = int(profile_value(profile, "folds") or 0)
    actions = calls + bets + raises + folds
    if actions <= 0:
        return 0.0
    return calls / actions


def high_calling_table(table):
    profiles = list((table.get("opponentProfiles") or {}).values())
    observed = [
        profile
        for profile in profiles
        if int(profile_value(profile, "hands_seen") or 0) >= 25
    ]
    if len(observed) < 3:
        return False
    average_call_frequency = sum(
        profile_call_frequency(profile) for profile in observed
    )
    average_call_frequency /= len(observed)
    return average_call_frequency >= 0.28


def raise_to_amount(table, allowed, multiplier):
    minimum = allowed.get("minRaiseTo")
    if minimum is None:
        return None
    blind = blind_size(allowed)
    current_bet = int(table.get("currentBet") or 0)
    target = max(int(minimum), current_bet, int(blind * multiplier))
    return cap_amount(target, allowed)


def preflop_premium_pressure(table, my_seat, base):
    if table.get("street", "Preflop") != "Preflop":
        return None

    action, _amount, _message = base
    if action == "raise":
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if (
        "raise" not in available
        or not no_large_preflop_raise(table, allowed)
        or not high_calling_table(table)
    ):
        return None

    hole_cards = my_seat.get("holeCards", [])
    hand = hand_class(hole_cards)
    score = preflop_score(hole_cards)
    opponents = active_opponents(table, my_seat)
    if opponents < 3:
        return None

    if hand in PREMIUM_OPEN_CLASSES or score >= 82:
        amount = raise_to_amount(table, allowed, 4.0 if score >= 96 else 3.0)
        return "raise", amount, f"v001 six-max premium open pressure {hand}/{score}"
    return None


def cheap_postflop_continue(table, my_seat, base):
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    price = call_amount(allowed)
    if action != "fold" or "call" not in available or price <= 0:
        return None

    pot = int(table.get("potChips") or 0)
    required = pot_odds(price, pot)
    stack = int(my_seat.get("stackChips") or 0)
    blind = blind_size(allowed)
    if required > 0.16 or price > max(blind, int(stack * 0.08)):
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    opponents = active_opponents(table, my_seat)
    rank = made_hand_rank(hole_cards, board_cards) if board_cards else 0
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    draw = champion.counter.patch1.has_good_draw(hole_cards, board_cards)
    texture = board_texture(board_cards) if board_cards else {"wet": False}

    if rank >= 2 and not champion.counter.patch1.fragile_rank_two(
        hole_cards,
        board_cards,
        rank,
    ):
        return "call", price, f"v001 cheap continue made rank {rank}"
    if opponents <= 3 and (top_pair or rank == 1):
        return "call", price, f"v001 cheap bluff catch rank {rank}"
    if draw and not texture.get("paired", False) and required <= 0.12:
        return "call", price, "v001 cheap draw continue"
    return None


def sixmax_adjustment(table, my_seat, base):
    if seated_players(table) < 4:
        return None
    for adjustment in (
        preflop_premium_pressure,
        cheap_postflop_continue,
    ):
        decision = adjustment(table, my_seat, base)
        if decision is not None:
            return decision
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

    base = champion.choose_action(table, my_seat)
    adjusted = sixmax_adjustment(table, my_seat, base)
    if adjusted is not None:
        return adjusted

    action, amount, message = base
    return action, amount, f"1:{message}"
