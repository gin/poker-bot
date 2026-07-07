"""Auto-research candidate v003.

This candidate keeps v002 as the champion baseline and adds a conservative
range-modeling layer with deterministic mixed frequencies. The new layer only
targets six-max tables with strong profile evidence: high-calling tables keep
v002's premium pressure, while high fold-to-bet tables can face a small mixed
dry-board probe when hero has a preflop range edge.
"""

from __future__ import annotations

from poker_bot.mixing import choose_weighted
from poker_bot.range_model import class_strength, combo_class, estimate_action_range
from poker_bot.range_model.preflop import position_label
from poker_bot.strategies import auto_research as original_champion
from poker_bot.strategies import auto_research_v002 as champion
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


def _range_value(allowed, range_name, key):
    value_range = allowed.get(range_name) or {}
    if not isinstance(value_range, dict):
        return None
    value = value_range.get(key)
    if value is None:
        return None
    return int(value)


def blind_size(allowed, table=None):
    if table is not None and table.get("bigBlindChips") is not None:
        return max(1, int(table["bigBlindChips"]))
    minimum = allowed.get("minBet")
    if minimum is None:
        minimum = _range_value(allowed, "betRange", "min")
    return max(1, int(minimum or 0))


def min_bet_to(allowed, table=None):
    minimum = allowed.get("minBet")
    if minimum is None:
        minimum = _range_value(allowed, "betRange", "min")
    if minimum is None:
        minimum = blind_size(allowed, table)
    return int(minimum)


def min_raise_to(allowed):
    minimum = allowed.get("minRaiseTo")
    if minimum is None:
        minimum = _range_value(allowed, "raiseRange", "min")
    if minimum is None:
        return None
    return int(minimum)


def cap_amount(amount, allowed, range_name=None):
    caps = [allowed.get("maxCommit")]
    if range_name is not None:
        caps.append(_range_value(allowed, range_name, "max"))
    caps = [int(cap) for cap in caps if cap is not None]
    if not caps:
        caps = [int(amount)]
    return max(0, min(int(amount), min(caps)))


def live_seat(seat):
    if not seat.get("agentId"):
        return False
    status = str(seat.get("status") or "").lower()
    if status in {"folded", "settled", "empty", "busted", "sitting_out"}:
        return False
    return not seat.get("folded", False) and not seat.get("hasFolded", False)


def active_opponents(table, my_seat):
    my_id = my_seat.get("agentId")
    return sum(
        1
        for seat in table.get("seats", [])
        if seat.get("agentId") != my_id and live_seat(seat)
    )


def active_opponent_seats(table, my_seat):
    my_id = my_seat.get("agentId")
    return [
        seat
        for seat in table.get("seats", [])
        if seat.get("agentId") != my_id and live_seat(seat)
    ]


def seated_players(table):
    return sum(1 for seat in table.get("seats", []) if seat.get("agentId"))


def no_large_preflop_raise(table, allowed):
    blind = blind_size(allowed, table)
    return int(table.get("currentBet") or 0) <= blind and call_amount(allowed) <= blind


def no_one_has_bet(table, allowed):
    return int(table.get("currentBet") or 0) == 0 and call_amount(allowed) == 0


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


def profile_fold_to_bet_frequency(profile):
    value = profile_value(profile, "fold_to_bet_frequency")
    if value is not None:
        return float(value)
    folds = int(profile_value(profile, "fold_to_bet") or 0)
    opportunities = int(profile_value(profile, "opportunities_to_fold_to_bet") or 0)
    if opportunities <= 0:
        return 0.0
    return folds / opportunities


def observed_profiles(table, minimum_hands=25, active_only=False):
    raw_profiles = table.get("opponentProfiles") or {}
    profiles = raw_profiles.values()
    if active_only:
        active_ids = [
            seat.get("agentId") for seat in table.get("seats", []) if live_seat(seat)
        ]
        active_profiles = [
            raw_profiles[agent_id]
            for agent_id in active_ids
            if agent_id in raw_profiles
        ]
        if active_profiles:
            profiles = active_profiles

    return [
        profile
        for profile in profiles
        if int(profile_value(profile, "hands_seen") or 0) >= minimum_hands
    ]


def high_calling_table(table):
    observed = observed_profiles(table)
    if len(observed) < 3:
        return False
    call_frequencies = [profile_call_frequency(profile) for profile in observed]
    high_callers = sum(1 for frequency in call_frequencies if frequency >= 0.28)
    if high_callers < 4:
        return False
    average_call_frequency = sum(call_frequencies)
    average_call_frequency /= len(observed)
    return average_call_frequency >= 0.28


def high_fold_to_bet_table(table):
    observed = observed_profiles(table, active_only=True)
    if len(observed) < 3:
        return False
    fold_frequencies = [profile_fold_to_bet_frequency(profile) for profile in observed]
    high_folders = sum(1 for frequency in fold_frequencies if frequency >= 0.62)
    required_high_folders = 3 if len(observed) == 3 else 4
    if high_folders < required_high_folders:
        return False
    average_fold_frequency = sum(fold_frequencies)
    average_fold_frequency /= len(observed)
    return average_fold_frequency >= 0.64


def raise_to_amount(table, allowed, multiplier):
    minimum = min_raise_to(allowed)
    if minimum is None:
        return None
    blind = blind_size(allowed, table)
    current_bet = int(table.get("currentBet") or 0)
    target = max(int(minimum), current_bet, int(blind * multiplier))
    return cap_amount(target, allowed, "raiseRange")


def bet_amount(table, allowed, fraction):
    minimum = min_bet_to(allowed, table)
    pot = int(table.get("potChips") or 0)
    target = max(minimum, int(max(pot, blind_size(allowed, table)) * fraction))
    return cap_amount(target, allowed, "betRange")


def range_situation_for_seat(table, seat, allowed):
    blind = blind_size(allowed, table)
    current_bet = int(table.get("currentBet") or 0)
    seat_bet = int(seat.get("currentBetChips") or 0)
    if current_bet > blind or seat_bet > blind:
        return "defend"
    return "open"


def range_action_for_seat(table, seat, allowed):
    blind = blind_size(allowed, table)
    seat_bet = int(seat.get("currentBetChips") or 0)
    current_bet = int(table.get("currentBet") or 0)
    if seat_bet > blind:
        return "raise"
    if seat_bet > 0 or current_bet > 0:
        return "call"
    return "check"


def average_opponent_range_strength(table, my_seat):
    allowed = table.get("allowedActions", {})
    known_cards = list(my_seat.get("holeCards", [])) + list(table.get("boardCards", []))
    total_weight = 0.0
    weighted_strength = 0.0
    for seat in active_opponent_seats(table, my_seat):
        estimated = estimate_action_range(
            position=position_label(table, seat),
            situation=range_situation_for_seat(table, seat, allowed),
            action=range_action_for_seat(table, seat, allowed),
            known_cards=known_cards,
            amount=seat.get("currentBetChips"),
            pot=table.get("potChips"),
        )
        for combo, weight in estimated.weights.items():
            weighted_strength += weight * class_strength(combo_class(combo))
            total_weight += weight
    if total_weight <= 0:
        return 0.0
    return weighted_strength / total_weight


def hero_preflop_range_strength(my_seat):
    try:
        return class_strength(combo_class(my_seat.get("holeCards", [])))
    except ValueError:
        return 0.0


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
        return "raise", amount, f"v003 six-max premium open pressure {hand}/{score}"
    return None


def range_mixed_dry_probe(table, my_seat, base):
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if action != "check" or "bet" not in available:
        return None
    if not no_one_has_bet(table, allowed) or not high_fold_to_bet_table(table):
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    texture = board_texture(board_cards) if board_cards else {"wet": False}
    if texture.get("wet", False) or texture.get("paired", False):
        return None

    rank = made_hand_rank(hole_cards, board_cards) if board_cards else 0
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    if rank > 0 or top_pair:
        return None

    opponents = active_opponents(table, my_seat)
    if opponents > 3:
        return None

    hero_strength = hero_preflop_range_strength(my_seat)
    opponent_strength = average_opponent_range_strength(table, my_seat)
    edge = hero_strength - opponent_strength
    if edge < 0.03 or preflop_score(hole_cards) < 70:
        return None

    chosen = choose_weighted(
        (("bet", 0.24), ("check", 0.76)),
        "v003-range-dry-probe",
        table,
        my_seat,
        strategy="auto_research_v003",
        extra=(round(edge, 2), opponents),
    )
    if chosen != "bet":
        return None

    amount = bet_amount(table, allowed, 0.26)
    return (
        "bet",
        amount,
        f"v003 mixed range probe edge {edge:.2f} vs high-fold table",
    )


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
    blind = blind_size(allowed, table)
    if required > 0.16 or price > max(blind, int(stack * 0.08)):
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    opponents = active_opponents(table, my_seat)
    rank = made_hand_rank(hole_cards, board_cards) if board_cards else 0
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    draw = original_champion.counter.patch1.has_good_draw(hole_cards, board_cards)
    texture = board_texture(board_cards) if board_cards else {"wet": False}

    if rank >= 2 and not original_champion.counter.patch1.fragile_rank_two(
        hole_cards,
        board_cards,
        rank,
    ):
        return "call", price, f"v003 cheap continue made rank {rank}"
    if opponents <= 3 and (top_pair or rank == 1):
        return "call", price, f"v003 cheap bluff catch rank {rank}"
    if draw and not texture.get("paired", False) and required <= 0.12:
        return "call", price, "v003 cheap draw continue"
    return None


def sixmax_adjustment(table, my_seat, base):
    if seated_players(table) < 4:
        return None
    for adjustment in (
        preflop_premium_pressure,
        range_mixed_dry_probe,
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
    return action, amount, f"3:{message}"
