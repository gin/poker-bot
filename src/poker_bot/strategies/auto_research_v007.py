"""Auto-research candidate v007.

This candidate keeps the current v005 line as the baseline and adds a narrow
range/CFR/mixing layer for close postflop bluff-catch spots. The intent is to
use the new reusable primitives without reopening the broader local-search
variance that made v006 trail the tweaked v005 in raw benchmarks.
"""

from __future__ import annotations

from functools import lru_cache

from poker_bot.cfr.kuhn import train_kuhn
from poker_bot.mixing import resolve_distribution
from poker_bot.range_model import class_strength, combo_class, estimate_action_range
from poker_bot.range_model.preflop import position_label
from poker_bot.strategies import auto_research_v005 as champion
from poker_bot.strategies.adaptive import (
    RANK_VALUES,
    has_top_pair_or_better,
    made_hand_rank,
    pot_odds,
)

ActionDecision = tuple[str | None, int | None, str]

SIMPLE_PRESSURE_MIN_PRICE = 0.30
SIMPLE_PRESSURE_MAX_PRICE = 0.36
PAIRED_BOARD_MIN_FOLD_PRICE = 0.35
VALUE_HEAVY_MAX_AVG_CALL = 0.08
VALUE_HEAVY_MIN_AGGRESSION = 0.48
VALUE_HEAVY_MAX_FOLD_TO_BET = 0.56


def _clamp(value, low, high):
    return max(low, min(high, value))


def call_amount(allowed):
    return int(allowed.get("callAmount") or allowed.get("callChips") or 0)


def _range_action_for_seat(table, seat, allowed):
    blind = int(allowed.get("minBet") or 2)
    current_bet = int(table.get("currentBet") or 0)
    seat_bet = int(seat.get("currentBetChips") or 0)
    if seat_bet > blind:
        return "raise"
    if current_bet > 0 or seat_bet > 0:
        return "call"
    return "check"


def _range_situation_for_seat(table, seat, allowed):
    blind = int(allowed.get("minBet") or 2)
    current_bet = int(table.get("currentBet") or 0)
    seat_bet = int(seat.get("currentBetChips") or 0)
    if current_bet > blind or seat_bet > blind:
        return "defend"
    return "open"


def live_opponent_seats(table, my_seat):
    hero_id = (my_seat or {}).get("agentId")
    hero_seat = (my_seat or {}).get("seatNumber")
    return [
        seat
        for seat in table.get("seats", [])
        if seat.get("agentId") != hero_id
        and seat.get("seatNumber") != hero_seat
        and not seat.get("folded", False)
        and not seat.get("hasFolded", False)
    ]


def active_opponents(table, my_seat):
    return max(1, len(live_opponent_seats(table, my_seat)))


def profile_value(profile, name):
    value = getattr(profile, name, None)
    if value is not None:
        return value
    if isinstance(profile, dict):
        return profile.get(name)
    return None


def profile_frequencies(table):
    rows = []
    for profile in (table.get("opponentProfiles") or {}).values():
        hands = int(profile_value(profile, "hands_seen") or 0)
        if hands < 20:
            continue
        rows.append(
            {
                "call": float(profile_value(profile, "call_frequency") or 0.0),
                "aggression": float(
                    profile_value(profile, "aggression_frequency") or 0.0
                ),
                "fold_to_bet": float(
                    profile_value(profile, "fold_to_bet_frequency") or 0.0
                ),
            }
        )
    return tuple(rows)


def table_frequency_summary(table):
    rows = profile_frequencies(table)
    if not rows:
        return {"samples": 0, "call": 0.0, "aggression": 0.0, "fold_to_bet": 0.0}
    return {
        "samples": len(rows),
        "call": sum(row["call"] for row in rows) / len(rows),
        "aggression": sum(row["aggression"] for row in rows) / len(rows),
        "fold_to_bet": sum(row["fold_to_bet"] for row in rows) / len(rows),
    }


def value_heavy_profile(table):
    summary = table_frequency_summary(table)
    return (
        summary["samples"] >= 3
        and summary["call"] <= VALUE_HEAVY_MAX_AVG_CALL
        and summary["aggression"] >= VALUE_HEAVY_MIN_AGGRESSION
        and summary["fold_to_bet"] <= VALUE_HEAVY_MAX_FOLD_TO_BET
    )


def hero_range_strength(my_seat):
    try:
        return class_strength(combo_class(my_seat.get("holeCards", [])))
    except ValueError:
        return 0.0


def average_opponent_range_strength(table, my_seat):
    allowed = table.get("allowedActions", {})
    known_cards = [
        *list(my_seat.get("holeCards", [])),
        *list(table.get("boardCards", [])),
    ]
    weighted_strength = 0.0
    total_weight = 0.0
    for seat in live_opponent_seats(table, my_seat):
        estimated = estimate_action_range(
            position=position_label(table, seat),
            situation=_range_situation_for_seat(table, seat, allowed),
            action=_range_action_for_seat(table, seat, allowed),
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


@lru_cache(maxsize=1)
def _kuhn_strategy():
    return train_kuhn(400).strategy


def cfr_call_prior(kicker):
    """Map Hold'em top-pair kicker buckets onto Kuhn facing-bet call priors."""
    card = "K" if kicker >= RANK_VALUES["Q"] else "Q"
    row = _kuhn_strategy().get(f"{card}|b", {})
    return _clamp(float(row.get("call", 0.5)), 0.0, 1.0)


def cfr_medium_pair_call_prior():
    row = _kuhn_strategy().get("Q|b", {})
    return _clamp(float(row.get("call", 0.38)), 0.0, 1.0)


def simple_profile_river_bluff_catch(table, my_seat, base) -> ActionDecision | None:
    if table.get("street", "Preflop") == "Preflop":
        return None
    if table.get("street") != "River":
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    price = call_amount(allowed)
    if action != "fold" or "call" not in available or price <= 0:
        return None
    if not value_heavy_profile(table):
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3 or made_hand_rank(hole_cards, board_cards) != 1:
        return None

    pot = int(table.get("potChips") or 0)
    required = pot_odds(price, pot)
    if required < SIMPLE_PRESSURE_MIN_PRICE or required > SIMPLE_PRESSURE_MAX_PRICE:
        return None

    hero_strength = hero_range_strength(my_seat)
    opponent_strength = average_opponent_range_strength(table, my_seat)
    range_edge = hero_strength - opponent_strength
    summary = table_frequency_summary(table)
    call_probability = cfr_medium_pair_call_prior()
    call_probability += max(0.0, summary["aggression"] - 0.55) * 0.35
    call_probability += max(0.0, 0.36 - required) * 1.25
    call_probability += _clamp(range_edge, -0.08, 0.10) * 0.25
    if has_top_pair_or_better(hole_cards, board_cards):
        call_probability += 0.08
    call_probability = _clamp(call_probability, 0.38, 0.74)

    decision = resolve_distribution(
        (("call", call_probability), ("fold", 1.0 - call_probability)),
        "v007-range-cfr-simple-river-bluff-catch",
        table,
        my_seat,
        strategy="auto_research_v007",
        extra=(
            round(required, 2),
            round(range_edge, 2),
            round(summary["call"], 2),
            round(summary["aggression"], 2),
            has_top_pair_or_better(hole_cards, board_cards),
        ),
    )
    if decision.selected != "call":
        return None

    return (
        "call",
        price,
        (
            "v007 mixed range/CFR simple-profile river bluff catch: "
            f"dist {decision.summary()} roll {decision.roll:.2f}, "
            f"required {required:.0%}, range edge {range_edge:+.2f}, "
            f"call {summary['call']:.0%}, "
            f"agg {summary['aggression']:.0%}"
        ),
    )


def paired_board_range_fold(table, my_seat, base) -> ActionDecision | None:
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    price = call_amount(allowed)
    if action != "call" or "fold" not in available or price <= 0:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if not champion.fragile_rank_two_on_paired_board(hole_cards, board_cards):
        return None

    pot = int(table.get("potChips") or 0)
    required = pot_odds(price, pot)
    if required < PAIRED_BOARD_MIN_FOLD_PRICE:
        return None

    hero_strength = hero_range_strength(my_seat)
    opponent_strength = average_opponent_range_strength(table, my_seat)
    range_edge = hero_strength - opponent_strength
    call_prior = cfr_medium_pair_call_prior()
    fold_probability = 0.76
    fold_probability += (required - PAIRED_BOARD_MIN_FOLD_PRICE) * 1.8
    fold_probability -= max(0.0, range_edge) * 0.25
    fold_probability += max(0.0, 0.45 - call_prior) * 0.18
    fold_probability = _clamp(fold_probability, 0.55, 0.90)

    decision = resolve_distribution(
        (("fold", fold_probability), ("call", 1.0 - fold_probability)),
        "v007-range-cfr-paired-board-fold",
        table,
        my_seat,
        strategy="auto_research_v007",
        extra=(round(required, 2), round(range_edge, 2), table.get("street")),
    )
    if decision.selected != "fold":
        return None

    return (
        "fold",
        None,
        (
            "v007 mixed range/CFR paired-board fold: "
            f"dist {decision.summary()} roll {decision.roll:.2f}, "
            f"required {required:.0%}, range edge {range_edge:+.2f}"
        ),
    )


def sixmax_adjustment(table, my_seat, base) -> ActionDecision | None:
    for adjustment in (
        paired_board_range_fold,
        simple_profile_river_bluff_catch,
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
    return action, amount, f"7:{message}"
