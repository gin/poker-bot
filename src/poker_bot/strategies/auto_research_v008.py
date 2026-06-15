"""Auto-research candidate v008.

Builds on v007 with two improvements:

1. **Preflop loose-call policy** (new in v008): live telemetry showed the
   inherited adaptive.py baseline folding too many playable hands preflop —
   specifically marginal holdings (score 38–50) at pot odds where they are
   +EV to call in six-max.  A position-aware cap (BTN/CO ≤28%, BB ≤25%,
   OOP ≤20%) rescues those hands while a multiway guard (≤3 opponents) and
   a stack-commit cap (≤15% of stack) prevent the wider range from leaking
   into clearly –EV situations.

2. **Postflop bluff-catch / paired-board fold layer** (carried from v007):
   a narrow range/CFR/mixing layer for close river bluff-catch spots and
   fragile rank-two hands on paired boards.
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
    preflop_score,
)

ActionDecision = tuple[str | None, int | None, str]

# Preflop loose-call thresholds (v008 fix for over-folding)
PREFLOP_LOOSE_CALL_MIN_SCORE = 38  # below this, still fold (true junk)
PREFLOP_LOOSE_CALL_SCORE_THRESHOLD = 50  # at/above this, base already calls
PREFLOP_LOOSE_CALL_MAX_PRICE_IP = 0.28  # BTN / CO (in-position)
PREFLOP_LOOSE_CALL_MAX_PRICE_BB = 0.25  # BB (closing action / discount)
PREFLOP_LOOSE_CALL_MAX_PRICE_OOP = 0.20  # all other seats (out-of-position)
PREFLOP_LOOSE_CALL_MAX_OPPONENTS = 3  # disable when 4+ opponents remain

# Postflop bluff-catch / paired-board parameters (carried from v007)
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


def preflop_loose_call(table, my_seat, base) -> ActionDecision | None:
    """Rescue marginal preflop hands that the base strategy folds too eagerly.

    The adaptive baseline folds any hand scoring < 46 when pot odds exceed 12%.
    In six-max, hands scoring 38-49 (e.g. T7s, J9o, A3o, Q8o, 98o) are
    profitable calls at reasonable pot odds, particularly in position.  This
    policy intercepts those fold decisions and converts them to calls when the
    pot odds are within a position-adjusted cap.
    """
    if table.get("street", "Preflop") != "Preflop":
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    price = call_amount(allowed)
    if action != "fold" or "call" not in available or price <= 0:
        return None

    hole_cards = my_seat.get("holeCards", [])
    score = preflop_score(hole_cards)
    if (
        score < PREFLOP_LOOSE_CALL_MIN_SCORE
        or score >= PREFLOP_LOOSE_CALL_SCORE_THRESHOLD
    ):
        return None

    # Don't widen into overly multiway pots — marginal hands play poorly vs many.
    n_opponents = active_opponents(table, my_seat)
    if n_opponents > PREFLOP_LOOSE_CALL_MAX_OPPONENTS:
        return None

    pot = int(table.get("potChips") or 0)
    required = pot_odds(price, pot)

    # Position-aware cap: in-position gets a wider range.
    pos = position_label(table, my_seat)
    if pos in {"BTN", "CO"}:
        max_price = PREFLOP_LOOSE_CALL_MAX_PRICE_IP
    elif pos == "BB":
        max_price = PREFLOP_LOOSE_CALL_MAX_PRICE_BB
    else:  # SB, UTG, MP — out-of-position
        max_price = PREFLOP_LOOSE_CALL_MAX_PRICE_OOP

    if required > max_price:
        return None

    # Stack-depth guard: avoid calling off >15% of remaining stack preflop
    # with a marginal holding (would put us in an SPR-awkward spot postflop).
    stack = int(my_seat.get("stackChips") or 0)
    if stack > 0 and price > stack * 0.15:
        return None

    return (
        "call",
        price,
        (
            f"v008 preflop loose call: score {score} pos {pos} "
            f"required {required:.0%} cap {max_price:.0%}"
        ),
    )


def sixmax_adjustment(table, my_seat, base) -> ActionDecision | None:
    for adjustment in (
        preflop_loose_call,
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
    return action, amount, f"8:{message}"  # noqa: E501
