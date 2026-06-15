"""Auto-research candidate v004.

This candidate keeps v003 as the champion baseline and adds a postflop
pot-control guard for the leak observed in live telemetry: weak non-top-pair
hands betting wet multiway boards as thin value, then folding to pressure.
"""

from __future__ import annotations

from poker_bot.mixing import resolve_distribution
from poker_bot.strategies import auto_research_v003 as champion
from poker_bot.strategies.adaptive import (
    board_texture,
    has_top_pair_or_better,
    made_hand_rank,
    pot_odds,
)

ActionDecision = tuple[str | None, int | None, str]

FOLD_TO_BET_PRIOR = 0.38
CALL_FREQUENCY_PRIOR = 0.24
PROFILE_PRIOR_WEIGHT = 20


def _clamp(value, low, high):
    return max(low, min(high, value))


def _beta_mean(successes, trials, prior_mean, prior_weight=PROFILE_PRIOR_WEIGHT):
    trials = max(0, int(trials or 0))
    successes = _clamp(float(successes or 0), 0.0, trials)
    return (successes + prior_mean * prior_weight) / (trials + prior_weight)


def _profile_value(profile, name):
    return champion.profile_value(profile, name)


def bayesian_fold_to_bet_frequency(profile):
    explicit = _profile_value(profile, "fold_to_bet_frequency")
    if explicit is not None:
        return _clamp(float(explicit), 0.0, 1.0)
    return _beta_mean(
        _profile_value(profile, "fold_to_bet"),
        _profile_value(profile, "opportunities_to_fold_to_bet"),
        FOLD_TO_BET_PRIOR,
    )


def bayesian_call_frequency(profile):
    explicit = _profile_value(profile, "call_frequency")
    if explicit is not None:
        return _clamp(float(explicit), 0.0, 1.0)
    calls = int(_profile_value(profile, "calls") or 0)
    bets = int(_profile_value(profile, "bets") or 0)
    raises = int(_profile_value(profile, "raises") or 0)
    folds = int(_profile_value(profile, "folds") or 0)
    return _beta_mean(calls, calls + bets + raises + folds, CALL_FREQUENCY_PRIOR)


def _average(values, default):
    values = tuple(values)
    if not values:
        return default
    return sum(values) / len(values)


def bayesian_pressure_summary(table, my_seat):
    opponents = champion.active_opponents(table, my_seat)
    profiles = champion.observed_profiles(
        table,
        minimum_hands=12,
        active_only=True,
    )
    profile_confidence = min(1.0, len(profiles) / max(1, opponents))
    fold_to_bet = _average(
        (bayesian_fold_to_bet_frequency(profile) for profile in profiles),
        FOLD_TO_BET_PRIOR,
    )
    call_frequency = _average(
        (bayesian_call_frequency(profile) for profile in profiles),
        CALL_FREQUENCY_PRIOR,
    )
    hero_strength = champion.hero_preflop_range_strength(my_seat)
    opponent_strength = champion.average_opponent_range_strength(table, my_seat)
    range_disadvantage = opponent_strength - hero_strength
    return {
        "opponents": opponents,
        "profile_count": len(profiles),
        "profile_confidence": profile_confidence,
        "fold_to_bet": fold_to_bet,
        "call_frequency": call_frequency,
        "hero_strength": hero_strength,
        "opponent_strength": opponent_strength,
        "range_disadvantage": range_disadvantage,
    }


def weak_pair_check_probability(summary):
    confidence = summary["profile_confidence"]
    probability = 0.24
    probability += _clamp(summary["range_disadvantage"], -0.14, 0.16) * 0.65
    probability += (
        confidence * (summary["call_frequency"] - CALL_FREQUENCY_PRIOR) * 0.85
    )
    probability -= confidence * (summary["fold_to_bet"] - FOLD_TO_BET_PRIOR) * 0.70
    probability += max(0, summary["opponents"] - 3) * 0.018
    if confidence < 0.25:
        return _clamp(probability, 0.08, 0.30)
    return _clamp(probability, 0.12, 0.84)


def distribution_extra(summary):
    return (
        round(summary["range_disadvantage"], 2),
        round(summary["fold_to_bet"], 2),
        round(summary["call_frequency"], 2),
        summary["opponents"],
    )


def distribution_message(prefix, decision, summary):
    return (
        f"{prefix}: dist {decision.summary()} roll {decision.roll:.2f}, "
        f"range gap {summary['range_disadvantage']:+.2f}, "
        f"fold {summary['fold_to_bet']:.0%}, call {summary['call_frequency']:.0%}"
    )


def weak_pair_wet_board_pot_control(table, my_seat, base) -> ActionDecision | None:
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if action != "bet" or "check" not in available:
        return None
    if not champion.no_one_has_bet(table, allowed):
        return None
    if champion.seated_players(table) < 4:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None

    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards)
    draw = champion.original_champion.counter.patch1.has_good_draw(
        hole_cards,
        board_cards,
    )
    if rank == 1 and not top_pair and texture.get("wet", False) and not draw:
        summary = bayesian_pressure_summary(table, my_seat)
        check_probability = weak_pair_check_probability(summary)
        decision = resolve_distribution(
            (("check", check_probability), ("bet", 1.0 - check_probability)),
            "v004-weak-pair-wet-board-pot-control",
            table,
            my_seat,
            strategy="auto_research_v004",
            extra=distribution_extra(summary),
        )
        if decision.selected != "check":
            return None
        return (
            "check",
            None,
            distribution_message(
                "v004 mixed pot control weak non-top pair wet board",
                decision,
                summary,
            ),
        )
    return None


def threshold_pressure_call_probability(
    summary,
    required,
    rank,
    top_pair,
    draw,
    texture,
):
    probability = 0.10
    if required <= 0.08:
        probability += 0.42
    elif required <= 0.14:
        probability += 0.30
    elif required <= 0.19:
        probability += 0.18
    if rank == 1 or top_pair:
        probability += 0.22
    if draw:
        probability += 0.18
    if texture.get("wet", False) and not draw:
        probability -= 0.12
    probability -= _clamp(summary["range_disadvantage"], -0.12, 0.18) * 0.45
    probability += (
        summary["profile_confidence"]
        * (summary["fold_to_bet"] - FOLD_TO_BET_PRIOR)
        * 0.20
    )
    return _clamp(probability, 0.0, 0.72)


def mixed_threshold_pressure_response(table, my_seat, base) -> ActionDecision | None:
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    price = champion.call_amount(allowed)
    if action != "fold" or "call" not in available or price <= 0:
        return None

    pot = int(table.get("potChips") or 0)
    required = pot_odds(price, pot)
    if required > 0.20 or champion.active_opponents(table, my_seat) > 2:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None

    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards)
    draw = champion.original_champion.counter.patch1.has_good_draw(
        hole_cards,
        board_cards,
    )
    if rank == 0 and not top_pair and not draw:
        return None

    summary = bayesian_pressure_summary(table, my_seat)
    call_probability = threshold_pressure_call_probability(
        summary,
        required,
        rank,
        top_pair,
        draw,
        texture,
    )
    decision = resolve_distribution(
        (("call", call_probability), ("fold", 1.0 - call_probability)),
        "v004-threshold-pressure-response",
        table,
        my_seat,
        strategy="auto_research_v004",
        extra=(*distribution_extra(summary), round(required, 2), rank, draw),
    )
    if decision.selected != "call":
        return None
    return (
        "call",
        price,
        distribution_message(
            f"v004 mixed threshold-pressure defense required {required:.0%}",
            decision,
            summary,
        ),
    )


def sixmax_adjustment(table, my_seat, base) -> ActionDecision | None:
    for adjustment in (
        weak_pair_wet_board_pot_control,
        mixed_threshold_pressure_response,
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
    return action, amount, f"4:{message}"
