"""
s5base
cut from s4v002
"""

from __future__ import annotations

from functools import lru_cache

from poker_bot.cfr.kuhn import train_kuhn
from poker_bot.hand_eval import evaluate_hand
from poker_bot.mixing import resolve_distribution
from poker_bot.opponents import OpponentProfile, profile_from_mapping
from poker_bot.range_model import class_strength, combo_class, estimate_action_range
from poker_bot.range_model.preflop import position_label
from poker_bot.strategies import auto_research as original_champion
from poker_bot.strategies.auto_research_v003 import (
    active_opponents,
    average_opponent_range_strength,
    hero_preflop_range_strength,
    observed_profiles,
    profile_value,
)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1: Constants
# ════════════════════════════════════════════════════════════════════════════════

BIG_BLIND = 2
RANK_VALUES = {rank: index for index, rank in enumerate("23456789TJQKA", start=2)}
FOLD_TO_BET_PRIOR = 0.38
CALL_FREQUENCY_PRIOR = 0.24
PROFILE_PRIOR_WEIGHT = 20

ActionDecision = tuple[str | None, int | None, str]

PREMIUMS = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
# Premium opening classes for six-max pressure (from auto_research_v003).
# Referenced by preflop_premium_pressure; only reachable once opponent
# profiles exist (high_calling_table), so it stayed latent until the
# promotion gate ran with track_opponents enabled.
PREMIUM_OPEN_CLASSES = {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs"}
DEFEND_RANGES_BB = {
    "AA",
    "KK",
    "QQ",
    "JJ",
    "TT",
    "99",
    "88",
    "AKs",
    "AKo",
    "AQs",
    "AJs",
    "KQs",
}

# Preflop loose-call thresholds (v008 fix for over-folding)
PREFLOP_LOOSE_CALL_MIN_SCORE = 38  # below this, still fold (true junk)
PREFLOP_LOOSE_CALL_SCORE_THRESHOLD = 50  # at/above this, base already calls
PREFLOP_LOOSE_CALL_MAX_PRICE_IP = 0.28  # BTN / CO (in-position)
PREFLOP_LOOSE_CALL_MAX_PRICE_BB = 0.25  # BB (closing action / discount)
PREFLOP_LOOSE_CALL_MAX_PRICE_OOP = 0.20  # all other seats (out-of-position)
PREFLOP_LOOSE_CALL_MAX_OPPONENTS = 3  # disable when 4+ opponents remain

# Position-aware preflop blind-defense widening (v008 telemetry fix). Live arena
# data (auto_research_v008) showed the base auto_research / profiled-counter
# chain folding genuinely playable hands — AQo (66), KQ (68/76), ATo (64), KJ,
# suited aces, small pairs — because its profiled fold threshold
# (~48 + (n-2)*6, up to ~70 six-handed) is calibrated for full-ring nit play.
# From the blinds we close (BB) or near-close (SB) the action at a discount, so
# cheap defends with real equity are mandatory. Restricted to the blinds:
# widening late position regressed the multi-way benchmark (BTN/CO calls bloat
# pots vs the counter/adaptive field). Early seats keep the base's tight range.
# A price cap (position-specific) and stack-fraction cap bound the risk.
PREFLOP_DEFENSE_MIN_SCORE = 50  # decent broadways / pairs / suited aces
PREFLOP_DEFENSE_MAX_PRICE = {
    "BB": 0.40,  # closes the action: best odds, widest defend
    "SB": 0.24,  # still OOP postflop: tighter than BB
}
PREFLOP_DEFENSE_MAX_STACK_FRACTION = 0.12  # never call off >12% stack


# Postflop bluff-catch / paired-board parameters (carried from v007)
SIMPLE_PRESSURE_MIN_PRICE = 0.30
SIMPLE_PRESSURE_MAX_PRICE = 0.36
PAIRED_BOARD_MIN_FOLD_PRICE = 0.35
VALUE_HEAVY_MAX_AVG_CALL = 0.08
VALUE_HEAVY_MIN_AGGRESSION = 0.48
VALUE_HEAVY_MAX_FOLD_TO_BET = 0.56

# Above this observed call frequency, suppress weak-pair wet-board pot-control
# checks and keep the base bet. Empirically (champion-gate ablation) checking
# weak pairs only helps vs very-low-frequency callers (simple, ~0.02); against
# sticky callers (~0.30) it forfeits EV. The cutoff sits between the two fields.
WEAK_PAIR_POT_CONTROL_MAX_CALL_FREQ = 0.15

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2: Card / Hand Utilities
# ════════════════════════════════════════════════════════════════════════════════


def card_values(cards):
    return [RANK_VALUES.get(card[0], 0) for card in cards]


def made_hand_rank(hole_cards, board_cards):
    if len(board_cards) < 3:
        return 0
    board_rank = evaluate_hand(board_cards) if len(board_cards) >= 5 else (0,)
    full_rank = evaluate_hand(list(hole_cards) + list(board_cards))
    category = full_rank[0]
    if len(board_cards) >= 5 and full_rank == board_rank:
        return 0
    return category


def rank_counts(cards):
    counts = {}
    for value in card_values(cards):
        counts[value] = counts.get(value, 0) + 1
    return counts


def hole_pair_rank(hole_cards):
    values = card_values(hole_cards)
    if len(values) == 2 and values[0] == values[1]:
        return values[0]
    return None


def _clamp(value, low, high):
    return max(low, min(high, value))


def _rank_index(card):
    ranks = "23456789TJQKA"
    return ranks.index(card[0].upper()) if card and card[0].upper() in ranks else -1


def hand_class(hole_cards):
    """Convert ['Ah', 'Kd'] -> 'AKo', ['Ah', 'Ks'] -> 'AKs', ['Ah', 'As'] -> 'AA'"""
    if len(hole_cards) != 2:
        return ""
    first, second = hole_cards
    r1, r2 = first[0].upper(), second[0].upper()
    s1, s2 = first[-1].upper(), second[-1].upper()
    if _rank_index(first) < _rank_index(second):
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return r1 + r2
    return f"{r1}{r2}{'s' if s1 == s2 else 'o'}"


def preflop_score(hole_cards):
    """Numerical preflop hand strength score (higher = better)."""
    if len(hole_cards) != 2:
        return 0
    first, second = card_values(hole_cards)
    high = max(first, second)
    low = min(first, second)
    score = high * 3 + low
    if first == second:
        score += 34 + high * 2
    if hole_cards[0][1] == hole_cards[1][1]:
        score += 8
    if abs(first - second) == 1:
        score += 5
    if high >= 12 and low >= 10:
        score += 12
    return score


def hero_preflop_range_strength(my_seat):
    try:
        return class_strength(combo_class(my_seat.get("holeCards", [])))
    except ValueError:
        return 0.0


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3: Board / Hand Evaluation
# ════════════════════════════════════════════════════════════════════════════════


def pot_odds(call_amount, pot):
    if call_amount <= 0:
        return 0.0
    return call_amount / (pot + call_amount)


def effective_pot(table):
    """Total chips in the pot including bets not yet swept from this round.

    The arena populates ``potChips`` with everything wagered, but the selfplay
    simulator reports ``potChips`` as the pot from *completed* rounds only —
    current-street bets (including posted blinds preflop) live on the seats as
    ``currentBetChips`` and are not yet included. Pot-odds logic that reads
    ``potChips`` directly therefore sees pot=0 preflop in selfplay and computes
    a 100% price, which makes every blind defend look unprofitable. Summing the
    live bets recovers the true pot in both environments.
    """
    pot = int(table.get("potChips") or 0)
    live_bets = sum(
        int(seat.get("currentBetChips") or 0) for seat in table.get("seats", [])
    )
    return pot + live_bets


def board_texture(board_cards):
    suits = [card[1] for card in board_cards]
    values = sorted(set(card_values(board_cards)))
    max_suit_count = max((suits.count(suit) for suit in set(suits)), default=0)
    connected = any(
        values[index + 2] - values[index] <= 4 for index in range(len(values) - 2)
    )
    paired = len(values) < len(board_cards)
    return {
        "wet": max_suit_count >= 3 or connected,
        "paired": paired,
        "high": any(value >= 12 for value in values),
    }


def has_top_pair_or_better(hole_cards, board_cards):
    if not board_cards:
        return False
    board_high = max(card_values(board_cards))
    hole_values = card_values(hole_cards)
    all_values = card_values(list(hole_cards) + list(board_cards))
    return any(
        value == board_high and all_values.count(value) >= 2 for value in hole_values
    )


def top_pair_kicker_value(hole_cards, board_cards):
    if len(hole_cards) != 2 or not board_cards:
        return None
    board_values = card_values(board_cards)
    board_high = max(board_values)
    hole_values = card_values(hole_cards)
    if board_values.count(board_high) != 1 or hole_values.count(board_high) != 1:
        return None
    if board_high not in hole_values:
        return None
    return max(value for value in hole_values if value != board_high)


def has_top_pair_good_kicker(hole_cards, board_cards):
    kicker = top_pair_kicker_value(hole_cards, board_cards)
    return kicker is not None and kicker >= RANK_VALUES["Q"]


def top_pair_defense_price_cap(
    hole_cards, board_cards, *, street="Flop", active_opponents=1
):
    kicker = top_pair_kicker_value(hole_cards, board_cards)
    if kicker is None:
        return 0.24

    cap = 0.30
    if kicker == RANK_VALUES["A"]:
        cap += 0.06
    elif kicker >= RANK_VALUES["Q"]:
        cap += 0.03

    if street == "Flop":
        cap += 0.01
    elif street == "Turn":
        cap += 0.02
    elif street == "River":
        cap -= 0.02

    active_opponents = max(1, int(active_opponents or 1))
    if active_opponents <= 2:
        cap += 0.02
    elif active_opponents >= 4:
        cap -= 0.02

    texture = board_texture(board_cards)
    if texture.get("wet", False):
        cap -= 0.03
    if texture.get("paired", False):
        cap -= 0.04

    return max(0.24, min(cap, 0.40))


def strong_top_pair_defense(table, my_seat, base) -> ActionDecision | None:
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    price = call_amount(allowed)
    if action != "fold" or "call" not in available or price <= 0:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None
    if made_hand_rank(hole_cards, board_cards) != 1:
        return None
    if not has_top_pair_good_kicker(hole_cards, board_cards):
        return None

    pot = int(table.get("potChips") or 0)
    required = pot_odds(price, pot)
    cap = top_pair_defense_price_cap(
        hole_cards,
        board_cards,
        street=table.get("street", "Flop"),
        active_opponents=active_opponents(table, my_seat),
    )
    if required > cap:
        return None

    return (
        "call",
        price,
        f"strong top-pair defense required {required:.0%} cap {cap:.0%}",
    )


def has_overpair_to_board(hole_cards, board_cards):
    pair_rank = hole_pair_rank(hole_cards)
    if pair_rank is None or not board_cards:
        return False
    return pair_rank > max(card_values(board_cards), default=0)


def paired_board_ranks(board_cards):
    return {value for value, count in rank_counts(board_cards).items() if count >= 2}


def board_has_two_pair(board_cards):
    return len(paired_board_ranks(board_cards)) >= 2


def board_has_pair(board_cards):
    return bool(paired_board_ranks(board_cards))


def board_dominated_two_pair(hole_cards, board_cards, rank):
    if rank != 2 or not board_has_two_pair(board_cards):
        return False
    hole_values = set(card_values(hole_cards))
    return not hole_values.intersection(paired_board_ranks(board_cards))


def paired_board_pot_control(table, my_seat, base) -> ActionDecision | None:
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None

    fragile_rank_two = fragile_rank_two_on_paired_board(hole_cards, board_cards)
    non_nut_full_house = non_nut_trips_board_full_house(hole_cards, board_cards)
    if not fragile_rank_two and not non_nut_full_house:
        return None

    if action == "bet" and "check" in available and no_one_has_bet(table, allowed):
        return (
            "check",
            None,
            "pot control: fragile paired-board value hand",
        )

    if action == "raise" and "call" in available:
        price = call_amount(allowed)
        pot = int(table.get("potChips") or 0)
        required = pot_odds(price, pot)
        stack = int(my_seat.get("stackChips") or 0)
        texture = board_texture(board_cards)
        if fragile_rank_two and (required > 0.42 or price > max(stack, 1)):
            if "fold" in available:
                return (
                    "fold",
                    None,
                    f"folded fragile paired-board hand at {required:.0%} price",
                )
            return None
        descriptor = "non-nut full house" if non_nut_full_house else "fragile two pair"
        wet_suffix = " wet" if texture.get("wet", False) else ""
        return (
            "call",
            price,
            f"capped paired-board aggression with {descriptor}{wet_suffix}",
        )

    return None


def paired_board_rank_two(hole_cards, board_cards, rank):
    if rank != 2:
        return False
    return bool(paired_board_ranks(board_cards))


def _beta_mean(successes, trials, prior_mean, prior_weight=PROFILE_PRIOR_WEIGHT):
    trials = max(0, int(trials or 0))
    successes = _clamp(float(successes or 0), 0.0, trials)
    return (successes + prior_mean * prior_weight) / (trials + prior_weight)


def _average(values, default):
    values = tuple(values)
    if not values:
        return default
    return sum(values) / len(values)


def bayesian_fold_to_bet_frequency(profile):
    explicit = profile_value(profile, "fold_to_bet_frequency")
    if explicit is not None:
        return _clamp(float(explicit), 0.0, 1.0)
    return _beta_mean(
        profile_value(profile, "fold_to_bet"),
        profile_value(profile, "opportunities_to_fold_to_bet"),
        FOLD_TO_BET_PRIOR,
    )


def bayesian_call_frequency(profile):
    explicit = profile_value(profile, "call_frequency")
    if explicit is not None:
        return _clamp(float(explicit), 0.0, 1.0)
    calls = int(profile_value(profile, "calls") or 0)
    bets = int(profile_value(profile, "bets") or 0)
    raises = int(profile_value(profile, "raises") or 0)
    folds = int(profile_value(profile, "folds") or 0)
    return _beta_mean(calls, calls + bets + raises + folds, CALL_FREQUENCY_PRIOR)


def bayesian_pressure_summary(table, my_seat):
    opponents = active_opponents(table, my_seat)
    profiles = observed_profiles(
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
    hero_strength = hero_preflop_range_strength(my_seat)
    opponent_strength = average_opponent_range_strength(table, my_seat)
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


def weak_pair_wet_board_pot_control(table, my_seat, base) -> ActionDecision | None:
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if action != "bet" or "check" not in available:
        return None
    if not no_one_has_bet(table, allowed):
        return None
    if seated_players(table) < 4:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None

    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards)
    draw = original_champion.counter.patch1.has_good_draw(
        hole_cards,
        board_cards,
    )
    if rank == 1 and not top_pair and texture.get("wet", False) and not draw:
        summary = bayesian_pressure_summary(table, my_seat)
        # Opponent-aware gate (v008 telemetry / champion-gate tuning). Checking
        # weak non-top pair for pot control only helps against low-frequency
        # callers (simple-style: call_freq ~0), where the check loses no value.
        # Against sticky, high-frequency callers (threshold_pressure /
        # counter_adaptive, call_freq ~0.3) this check forfeits value and fold
        # equity and measurably lowered EV vs the champion field, so we keep the
        # base bet there. Only suppress the bet when the read is trustworthy and
        # the field is passive.
        if (
            summary["profile_confidence"] >= 0.5
            and summary["call_frequency"] >= WEAK_PAIR_POT_CONTROL_MAX_CALL_FREQ
        ):
            return None
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
                "mixed pot control weak non-top pair wet board",
                decision,
                summary,
            ),
        )
    return None


def fragile_rank_two(hole_cards, board_cards, rank):
    return board_dominated_two_pair(hole_cards, board_cards, rank) or (
        paired_board_rank_two(hole_cards, board_cards, rank)
        and not has_top_pair_or_better(hole_cards, board_cards)
    )


def fragile_rank_two_on_paired_board(hole_cards, board_cards):
    rank = made_hand_rank(hole_cards, board_cards)
    if rank != 2 or not board_has_pair(board_cards):
        return False
    return not has_top_pair_or_better(hole_cards, board_cards)


def trips_board_ranks(board_cards):
    return {value for value, count in rank_counts(board_cards).items() if count >= 3}


def non_nut_trips_board_full_house(hole_cards, board_cards):
    if len(board_cards) < 4:
        return False
    trip_ranks = trips_board_ranks(board_cards)
    if not trip_ranks:
        return False
    full_rank = evaluate_hand(list(hole_cards) + list(board_cards))
    if full_rank[0] != 6:
        return False
    triple_rank, pair_rank = full_rank[1], full_rank[2]
    if triple_rank not in trip_ranks:
        return False
    return pair_rank < RANK_VALUES["A"]


def mixed_threshold_pressure_response(table, my_seat, base) -> ActionDecision | None:
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
    if required > 0.20 or active_opponents(table, my_seat) > 2:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None

    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards)
    draw = original_champion.counter.patch1.has_good_draw(
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
            f"mixed threshold-pressure defense required {required:.0%}",
            decision,
            summary,
        ),
    )


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
        f"mixed range probe edge {edge:.2f} vs high-fold table",
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
        return "call", price, f"cheap continue made rank {rank}"
    if opponents <= 3 and (top_pair or rank == 1):
        return "call", price, f"cheap bluff catch rank {rank}"
    if draw and not texture.get("paired", False) and required <= 0.12:
        return "call", price, "cheap draw continue"
    return None


def has_flush_draw(hole_cards, board_cards):
    if len(board_cards) not in {3, 4}:
        return False
    suits = [card[1] for card in list(hole_cards) + list(board_cards)]
    hole_suits = {card[1] for card in hole_cards}
    return any(suits.count(suit) >= 4 and suit in hole_suits for suit in hole_suits)


def has_open_ended_straight_draw(hole_cards, board_cards):
    if len(board_cards) not in {3, 4}:
        return False
    values = set(card_values(list(hole_cards) + list(board_cards)))
    if 14 in values:
        values.add(1)
    for low in range(1, 11):
        window = set(range(low, low + 4))
        if window.issubset(values):
            return True
    return False


def has_good_draw(hole_cards, board_cards):
    return has_flush_draw(hole_cards, board_cards) or has_open_ended_straight_draw(
        hole_cards, board_cards
    )


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4: Table / Seat Utilities
# ════════════════════════════════════════════════════════════════════════════════

# Seat statuses that mean the player is NOT contesting the hand. The arena
# sends `status` and no `folded` flags; the local simulator sends `folded`
# flags and no `status`. Checking only folded flags made every liveness
# count wrong in live play (telemetry showed active == seated on every
# arena turn/river decision). Blocklist rather than status == "Active" so
# unknown statuses (e.g. AllIn) stay counted as live. Self-contained (no
# hand_utils import) so the file stays bundle-able.
_DEAD_SEAT_STATUSES = frozenset(
    {"Folded", "SittingOut", "Waiting", "Out", "Busted", "Eliminated"}
)


def seat_is_live(seat):
    """True if the seat is still contesting the hand (arena or simulator)."""
    if seat.get("folded", False) or seat.get("hasFolded", False):
        return False
    return seat.get("status") not in _DEAD_SEAT_STATUSES


def active_seat_numbers(table):
    return [
        int(seat.get("seatNumber"))
        for seat in table.get("seats", [])
        if seat_is_live(seat) and seat.get("seatNumber") is not None
    ]


def active_players(table):
    return max(1, len(active_seat_numbers(table)))


def seated_players(table):
    return sum(1 for seat in table.get("seats", []) if seat.get("agentId"))


def active_opponents(table, my_seat):
    my_id = (my_seat or {}).get("agentId")
    return sum(
        1
        for seat in table.get("seats", [])
        if seat.get("agentId") != my_id and seat_is_live(seat)
    )


def live_opponent_seats(table, my_seat):
    hero_id = (my_seat or {}).get("agentId")
    hero_seat = (my_seat or {}).get("seatNumber")
    return [
        seat
        for seat in table.get("seats", [])
        if seat.get("agentId") != hero_id
        and seat.get("seatNumber") != hero_seat
        and seat_is_live(seat)
    ]


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


def max_commit(allowed, default=0):
    if allowed.get("maxCommit") is not None:
        return int(allowed["maxCommit"])
    raise_range = allowed.get("raiseRange") or {}
    bet_range = allowed.get("betRange") or {}
    return int(raise_range.get("max") or bet_range.get("max") or default)


def capped(amount, allowed):
    return max(0, min(int(amount), int(allowed.get("maxCommit", amount))))


def blind_size(allowed, table=None):
    if table is not None and table.get("bigBlindChips") is not None:
        return max(1, int(table["bigBlindChips"]))
    minimum = allowed.get("minBet")
    if minimum is None:
        bet_range = allowed.get("betRange") or {}
        minimum = bet_range.get("min")
    return max(1, int(minimum or 0))


def no_one_has_bet(allowed, table):
    return call_amount(allowed) == 0 and int(table.get("currentBet") or 0) == 0


def no_large_preflop_raise(table, allowed):
    blind = blind_size(allowed, table)
    return int(table.get("currentBet") or 0) <= blind and call_amount(allowed) <= blind


def unopened_preflop(table, allowed):
    blind_size_val = max(int(allowed.get("minBet") or 0), BIG_BLIND)
    return (
        int(table.get("currentBet") or 0) <= blind_size_val
        and call_amount(allowed) <= blind_size_val
    )


def short_handed(table):
    return active_players(table) <= 3


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


def covered_by_larger_stack(table, my_seat):
    my_stack = int(my_seat.get("stackChips") or 0)
    for seat in table.get("seats", []):
        if seat.get("agentId") == my_seat.get("agentId"):
            continue
        if not seat_is_live(seat):
            continue
        if int(seat.get("stackChips") or 0) > my_stack:
            return True
    return False


def stack_total(seat):
    return int(seat.get("stackChips") or 0) + int(seat.get("currentBetChips") or 0)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5: Bet / Raise Amount Helpers
# ════════════════════════════════════════════════════════════════════════════════


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


def probe_bet_amount(table, allowed, active):
    minimum = int(allowed.get("minBet") or BIG_BLIND)
    pot = int(table.get("potChips") or 0)
    fraction = 0.32 if active >= 4 else 0.38
    return capped(max(minimum, int(max(pot, BIG_BLIND) * fraction)), allowed)


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


def raise_to_amount(table, allowed, target, all_in=False):
    minimum = min_raise_to(allowed)
    if minimum is None:
        return None
    cap_val = max_commit(allowed, minimum)
    if all_in:
        return cap_val
    return capped(max(minimum, int(target)), allowed)


def bet_amount_frac(table, allowed, fraction):
    pot = int(table.get("potChips") or 0)
    minimum = min_bet(allowed)
    return capped(max(minimum, int(max(pot, BIG_BLIND) * fraction)), allowed)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6: Opponent Profile Helpers
# ════════════════════════════════════════════════════════════════════════════════


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
    folds_val = int(profile_value(profile, "fold_to_bet") or 0)
    opportunities = int(profile_value(profile, "opportunities_to_fold_to_bet") or 0)
    if opportunities <= 0:
        return 0.0
    return folds_val / opportunities


def observed_profiles(table, minimum_hands=25, active_only=False):
    raw = table.get("opponentProfiles") or {}
    profiles = list(raw.values())
    if active_only:
        active_ids = [
            seat.get("agentId")
            for seat in table.get("seats", [])
            if seat_is_live(seat)
        ]
        active_profiles = [raw[a_id] for a_id in active_ids if a_id in raw]
        if active_profiles:
            profiles = active_profiles
    return [
        p for p in profiles if int(profile_value(p, "hands_seen") or 0) >= minimum_hands
    ]


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


def high_calling_table(table):
    observed = observed_profiles(table)
    if len(observed) < 3:
        return False
    call_freqs = [profile_call_frequency(p) for p in observed]
    high_callers = sum(1 for f in call_freqs if f >= 0.28)
    if high_callers < 4:
        return False
    avg_call = sum(call_freqs) / len(observed)
    return avg_call >= 0.28


def high_fold_to_bet_table(table):
    observed = observed_profiles(table, active_only=True)
    if len(observed) < 3:
        return False
    fold_freqs = [profile_fold_to_bet_frequency(p) for p in observed]
    high_folders = sum(1 for f in fold_freqs if f >= 0.62)
    required_high = 3 if len(observed) == 3 else 4
    if high_folders < required_high:
        return False
    avg_fold = sum(fold_freqs) / len(observed)
    return avg_fold >= 0.64


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7: EXACT PREFLOP STRATEGY (Copied from auto_research_v007 dependencies)
# ════════════════════════════════════════════════════════════════════════════════


def table_profiles(table, my_seat):
    """Exact copy from profiled_counter_adaptive.py"""
    raw_profiles = table.get("opponentProfiles", {})
    my_id = (my_seat or {}).get("agentId")
    profiles = []
    for seat in table.get("seats", []):
        agent_id = seat.get("agentId")
        if not agent_id or agent_id == my_id:
            continue
        raw = raw_profiles.get(agent_id)
        if isinstance(raw, OpponentProfile):
            profiles.append(raw)
        elif isinstance(raw, dict):
            profiles.append(profile_from_mapping(agent_id, raw))
    return profiles


def table_tendencies(profiles):
    """Exact copy from profiled_counter_adaptive.py"""
    labels = [profile.label() for profile in profiles]
    return {
        "has_bluffer": "bluffer" in labels or "loose_aggressive" in labels,
        "has_station": "calling_station" in labels,
        "all_patient": bool(labels)
        and all(label in {"patient_methodical", "unknown"} for label in labels),
        "has_aggressive": any(
            label in {"loose_aggressive", "tight_aggressive", "bluffer"}
            for label in labels
        ),
    }


def pressure_seats(table, my_seat):
    """Exact copy from survival_sixmax.py"""
    my_bet = int(my_seat.get("currentBetChips") or 0)
    my_id = my_seat.get("agentId")
    return [
        seat
        for seat in table.get("seats", [])
        if seat.get("agentId") != my_id
        and seat_is_live(seat)
        and int(seat.get("currentBetChips") or 0) > my_bet
    ]


def aggressive_profile_ids(table, my_seat):
    """Exact copy from survival_sixmax.py"""
    aggressive_ids = set()
    for profile in table_profiles(table, my_seat):
        if profile.label() in {"bluffer", "loose_aggressive"}:
            aggressive_ids.add(profile.agent_id)
    return aggressive_ids


def bully_context(table, my_seat):
    """Exact copy from survival_sixmax.py"""
    pressured_by = pressure_seats(table, my_seat)
    if not pressured_by:
        return None

    hero_total = max(stack_total(my_seat), 1)
    aggressive_ids = aggressive_profile_ids(table, my_seat)
    large_pressure = [
        seat
        for seat in pressured_by
        if stack_total(seat) >= hero_total * 1.2
        or seat.get("agentId") in aggressive_ids
    ]
    if not large_pressure:
        return None

    largest = max(large_pressure, key=stack_total)
    return {
        "seat": largest,
        "known_aggressive": largest.get("agentId") in aggressive_ids,
        "stack_ratio": stack_total(largest) / hero_total,
    }


def preflop_premium_pressure(table, my_seat, base):
    """
    From v003
    """
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
        return "raise", amount, f"premium open pressure {hand}/{score}"
    return None


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 8: EXACT POSTFLOP LOGIC FROM auto_research_v007
# ════════════════════════════════════════════════════════════════════════════════

SIMPLE_PRESSURE_MIN_PRICE = 0.30
SIMPLE_PRESSURE_MAX_PRICE = 0.36
PAIRED_BOARD_MIN_FOLD_PRICE = 0.35
VALUE_HEAVY_MAX_AVG_CALL = 0.08
VALUE_HEAVY_MIN_AGGRESSION = 0.48
VALUE_HEAVY_MAX_FOLD_TO_BET = 0.56


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


def active_opponents_v007(table, my_seat):
    return max(1, len(live_opponent_seats(table, my_seat)))


def profile_value_v007(profile, name):
    value = getattr(profile, name, None)
    if value is not None:
        return value
    if isinstance(profile, dict):
        return profile.get(name)
    return None


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
        strategy="flattened_v001",
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
    if not fragile_rank_two_on_paired_board(hole_cards, board_cards):
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
        strategy="flattened_v001",
        extra=(round(required, 2), round(range_edge, 2), table.get("street")),
    )
    if decision.selected != "fold":
        return None

    return (
        "fold",
        None,
        (
            "mixed range/CFR paired-board fold: "
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
            f"preflop loose call: score {score} pos {pos} "
            f"required {required:.0%} cap {max_price:.0%}"
        ),
    )


def preflop_blind_defense(table, my_seat, base) -> ActionDecision | None:
    """Defend / call wider in position than the base's full-ring fold threshold.

    The base auto_research / profiled-counter chain uses a full-ring-calibrated
    fold threshold (~48 + (n-2)*6) that over-folds at six-max. Live arena
    telemetry (auto_research_v008) showed it folding AQo, KQ, ATo, KJ, suited
    aces and small pairs for tiny prices. This rescue converts those folds to
    calls from the blinds (closing / near-closing the action) and from late
    position (BTN/CO, where equity realizes in position). Early seats are not
    widened. A price cap (position-specific) and stack-fraction cap keep it from
    bloating pots or committing stack with marginal holdings.
    """
    if table.get("street", "Preflop") != "Preflop":
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    price = call_amount(allowed)
    if action != "fold" or "call" not in available or price <= 0:
        return None

    pos = position_label(table, my_seat)
    max_price = PREFLOP_DEFENSE_MAX_PRICE.get(pos)
    if max_price is None:  # UTG / MP: keep the base's tight range
        return None

    hole_cards = my_seat.get("holeCards", [])
    score = preflop_score(hole_cards)
    if score < PREFLOP_DEFENSE_MIN_SCORE:
        return None

    pot = effective_pot(table)
    required = pot_odds(price, pot)
    if required > max_price:
        return None

    # Never call off a meaningful chunk of stack with a defend-range hand.
    stack = int(my_seat.get("stackChips") or 0)
    if stack > 0 and price > stack * PREFLOP_DEFENSE_MAX_STACK_FRACTION:
        return None

    return (
        "call",
        price,
        (
            f"preflop defense: score {score} pos {pos} "
            f"required {required:.0%} cap {max_price:.0%}"
        ),
    )


# from combining v005, v007: good
def spr_commitment_lock(table, my_seat, base) -> ActionDecision | None:
    """Rescue two-pair-or-better from folding when SPR is low (pot-committed)."""
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    price = call_amount(allowed)

    # We only rescue folds
    if action != "fold" or "call" not in available or price <= 0:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None

    opponents = active_opponents(table, my_seat)
    hand_rank = made_hand_rank(hole_cards, board_cards)

    # Restrict the rescue to genuinely strong made hands (two pair or better):
    # forcing thin top-pair calls vs value-heavy opponents is -EV and only picks
    # up reverse-implied coolers. One-pair bluff-catchers are excluded outright.
    if hand_rank < 2:
        return None

    # "Pot-committed" in the strict sense: SPR below threshold means the call
    # risks little relative to the pot, where two pair or better is a clear
    # call. Wider multi-way pots warrant a tighter bar than short-handed. This
    # rule is net-positive at high sample (every cell vs flattened >= 0, with
    # strict gains short-handed) and never regresses the baseline materially.
    spr_threshold = 1.5 if opponents >= 4 else 3.0

    # Calculate SPR (Stack-to-Pot Ratio)
    stack = int(my_seat.get("stackChips") or 0)
    pot = int(table.get("potChips") or 0)
    spr = stack / max(1, pot)

    # If SPR is below threshold, we are pot-committed. Don't fold.
    if spr < spr_threshold:
        return (
            "call",
            price,
            (
                f"spr commitment lock: spr {spr:.2f} < {spr_threshold}, "
                f"calling with strong hand vs {opponents} opps"
            ),
        )

    return None


def ip_dry_board_cbet_exploit(table, my_seat, base) -> ActionDecision | None:
    """Exploitatively c-bet small on dry boards when IP and opponent folds often."""
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])

    # We only intercept when the base strategy wants to check
    if action != "check" or "bet" not in available:
        return None

    # Restrict to a genuine 2-handed table. active_opponents-based routing
    # would also fire this in a multi-way pot that narrowed to heads-up,
    # applying HU-only logic against a full-ring range.
    if seated_players(table) > 2:
        return None

    # 1. Structural: Must be In Position (BTN, CO, MP)
    pos = position_label(table, my_seat)
    if pos not in {"BTN", "CO", "MP"}:
        return None

    # 2. Board: Must be dry (no flush draws, no straight draws)
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None
    texture = board_texture(board_cards)
    if texture.get("wet", False):
        return None

    # 3. Opponent Exploit: They fold to bet too often (> 65%)
    # Tightened from 55% to prevent leaking against competent bots that defend well.
    summary = bayesian_pressure_summary(table, my_seat)
    # Require a trustworthy read: point estimates blend with priors, so an
    # early/thin sample can spuriously cross the threshold against bots that
    # actually float and check-raise back.
    if summary["profile_confidence"] < 0.50:
        return None
    if summary["fold_to_bet"] < 0.65:
        return None

    # EXECUTE: Bet small (33% pot) for maximum fold equity with minimal risk
    pot = int(table.get("potChips") or 0)
    bet_size = max(2, int(pot * 0.33))  # Minimum 2 chips (1 BB)

    return (
        "bet",
        bet_size,
        "v001 IP dry board exploit: high fold-to-bet, small c-bet",
    )


def high_wtsd_thin_value_bet(table, my_seat, base) -> ActionDecision | None:
    """Extract thin value from calling stations by betting medium-strength hands."""
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])

    # We only intercept when the base strategy wants to check
    if action != "check" or "bet" not in available:
        return None

    # Restrict to a genuine 2-handed table (see ip_dry_board_cbet_exploit).
    if seated_players(table) > 2:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None

    # Must have exactly one pair (2nd pair, 3rd pair, or weak top pair).
    # We don't want to bluff a calling station (rank 0), and strong hands
    # (rank >= 2) are usually already bet by the base strategy.
    if made_hand_rank(hole_cards, board_cards) != 1:
        return None

    # Opponent Exploit: They call too often (call frequency / WTSD > 70%)
    # AND they are passive (fold_to_bet > 50%), ensuring they aren't aggressive check-raisers.
    summary = bayesian_pressure_summary(table, my_seat)
    # Require a trustworthy read before betting thin for value.
    if summary["profile_confidence"] < 0.50:
        return None
    if summary["call_frequency"] < 0.70 or summary["fold_to_bet"] < 0.50:
        return None

    # EXECUTE: Bet small (50% pot) for thin value.
    # Calling stations will call this with worse, and we avoid bloating the pot.
    pot = int(table.get("potChips") or 0)
    bet_size = max(2, int(pot * 0.50))

    return (
        "bet",
        bet_size,
        "v001 thin value exploit: high call freq, betting 1-pair for value HU",
    )


def royal_flush_check_call_guard(table, my_seat, blueprint) -> ActionDecision | None:
    """Prevent folding and aggressive actions when royal flush is possible.

    Preflop: if hero holds AKs, never fold/raise — only check or call.
    Postflop: if a royal flush is still possible given hole + board cards,
    never fold or raise — only check or call. If no royal flush chance, exit guard.
    """
    action, _amount, _message = blueprint

    hole_cards = my_seat.get("holeCards", [])
    if len(hole_cards) != 2:
        return None

    street = table.get("street", "Preflop")

    # Preflop: AKs should never fold or raise — only check or call
    if street == "Preflop":
        hand = hand_class(hole_cards)
        if hand != "AKs":
            return None
        # AKs — check if possible, else call (never fold, never raise)
        allowed = table.get("allowedActions", {})
        available = allowed.get("availableActions", [])
        if "check" in available and no_one_has_bet(allowed, table):
            return "check", None, "royal flush guard: AKs preflop, checking"
        if "call" in available:
            price = call_amount(allowed)
            return "call", price, "royal flush guard: AKs preflop, calling"
        return None

    # Postflop: check if royal flush is still possible
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None

    # Reuse the existing royal_flush_possible logic
    from poker_bot.strategies.royal_flush import royal_flush_possible

    if not royal_flush_possible(hole_cards, board_cards):
        return None  # no royal flush chance — exit guard

    # Royal flush possible — never fold/raise, only check or call
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "check" in available and no_one_has_bet(allowed, table):
        return "check", None, "royal flush guard: royal flush possible, checking"
    if "call" in available:
        price = call_amount(allowed)
        return "call", price, "royal flush guard: royal flush possible, calling"

    return None


def preflop_premium_3bet_shove(table, my_seat, blueprint) -> ActionDecision | None:
    """Shove premium hands after 3+ preflop raise-backs.

    In prolonged preflop wars (3+ raise-backs), premium hands (AA/KK/QQ,
    AKs/KQs) should go all-in instead of capping or calling.
    """
    action, _amount, _message = blueprint
    if table.get("street") != "Preflop":
        return None
    if action not in ("raise", "bet"):
        return None

    # Count how many times hero has raised this hand
    history = table.get("actionHistory") or table.get("action_history") or []
    my_id = (my_seat or {}).get("agentId")
    raise_count = sum(
        1
        for h in history
        if h.get("agentId") == my_id
        and h.get("action") in ("raise", "bet")
        and h.get("street") == "Preflop"
    )

    if raise_count < 3:
        return None

    # Check if hero holds a premium hand worth shoving
    hole_cards = my_seat.get("holeCards", [])
    if len(hole_cards) != 2:
        return None
    hand = hand_class(hole_cards)
    premium_shove_hands = {"AA", "KK", "QQ", "AKs", "KQs"}
    if hand not in premium_shove_hands:
        return None

    # Go all-in with premium hands
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "all-in" in available:
        hero_stack = int(my_seat.get("stackChips") or 0)
        return (
            "all-in",
            hero_stack,
            f"preflop war premium shove: {hand} after {raise_count} raise-backs",
        )
    if "raise" in available:
        hero_stack = int(my_seat.get("stackChips") or 0)
        return (
            "raise",
            hero_stack,
            f"preflop war premium shove: {hand} after {raise_count} raise-backs (raise=stack)",
        )

    return None


def sixmax_adjustment(table, my_seat, base) -> ActionDecision | None:
    # 0. Universal mathematical truth: Pot commitment overrides everything (all table sizes)
    decision = spr_commitment_lock(table, my_seat, base)
    if decision is not None:
        return decision

    # 0.5. Royal flush pre-decision guard: check/call only when royal flush is possible
    from poker_bot.neural.guardrails import royal_flush_predecision_guard

    pre_guard = royal_flush_predecision_guard(table, my_seat)
    if pre_guard and pre_guard.fired:
        action, amount, reason = (
            pre_guard.final_action,
            pre_guard.final_action,  # placeholder, refined below
            pre_guard.reason,
        )
        allowed = table.get("allowedActions") or {}
        if action == "call":
            amount = allowed.get("callAmount") or allowed.get("callChips") or 0
        elif action == "check":
            amount = None
        else:
            amount = None
        return action, amount, reason

    # 0.6. Preflop premium shove: AA/KK/QQ/AKs/KQs go all-in after 3+ raise-backs
    decision = preflop_premium_3bet_shove(table, my_seat, base)
    if decision is not None:
        return decision

    # 1. Tightly-gated exploits. Each self-guards on table size (true 2-handed)
    # and profile confidence, so they only fire where they are +EV and stay
    # silent everywhere else, preserving the superset guarantee below.
    for exploit in (
        ip_dry_board_cbet_exploit,
        high_wtsd_thin_value_bet,
    ):
        if (decision := exploit(table, my_seat, base)) is not None:
            return decision

    # 2. Universal fallbacks: reproduce flattened's exact cascade order so that
    # v2 never gives back EV in spots the specialized branches did not handle.
    for adjustment in (
        preflop_loose_call,  # from v008
        preflop_blind_defense,  # v008 telemetry fix: six-max blind over-folding
        preflop_premium_pressure,  # from v003
        paired_board_pot_control,  # from v005
        paired_board_range_fold,  # from v007
        weak_pair_wet_board_pot_control,  # from v004 (was dropped in v2)
        strong_top_pair_defense,  # from v005
        mixed_threshold_pressure_response,  # from v004 (was dropped in v2)
        simple_profile_river_bluff_catch,  # from v007
        range_mixed_dry_probe,  # from v003
        cheap_postflop_continue,  # from v003
    ):
        if (decision := adjustment(table, my_seat, base)) is not None:
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

    base = original_champion.choose_action(table, my_seat)
    adjusted = sixmax_adjustment(table, my_seat, base)
    if adjusted is not None:
        return adjusted

    action, amount, message = base
    return action, amount, f"{message}"
