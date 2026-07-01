"""
Season 3, base
Cut from s2base
Optimize for heads up (1 opponent table)

    Opponent                   s3base  s3v009  Delta  Gate
    ─────────────────────────  ──────  ──────  ─────  ───────
    simple                     +38.8   +38.8   +0.0   ✅ PASS
    all_in_everytime           +482.2  +482.2  +0.0   ✅ PASS
    adaptive                   +16.3   +16.3   +0.0   ✅ PASS
    profiled_counter_adaptive  +14.7   +14.7   +0.0   ✅ PASS
    threshold_pressure         +15.0   +15.0   +0.0   ✅ PASS
    anti_threshold             +54.4   +54.4   +0.0   ✅ PASS
    royal_flush                +20.4   +20.4   +0.0   ✅ PASS
    royal_adaptive             +20.5   +20.5   +0.0   ✅ PASS
    survival_balanced          +29.8   +29.8   +0.0   ✅ PASS
    survival_aggressive        +31.0   +31.0   +0.0   ✅ PASS
    auto_research_v005         +29.6   +29.6   +0.0   ✅ PASS
    auto_research_v008         +29.6   +29.6   +0.0   ✅ PASS
    flattened_v2               +26.9   +26.9   +0.0   ✅ PASS
    s2baseog                   +24.3   +24.3   +0.0   ✅ PASS
    s2v002                     +25.6   +25.6   +0.0   ✅ PASS
    s2v004                     +24.3   +24.3   +0.0   ✅ PASS
    s2v008                     +24.1   +24.1   +0.0   ✅ PASS
    s2v009                     +7.0    +7.0    +0.0   ✅ PASS
    s2v014                     +9.5    +9.5    +0.0   ✅ PASS

 Changes made to s3base.py and simulator.py

 1. Added board_assisted_two_pair_guard(): prevents over-valuing two pair on
    a paired board when one pair rank is fully on the board. Fires only vs
    tight opponents (VPIP% < 25%).

 2. Added river_two_pair_raise_guard(): prevents value-raising with two pair
    on the river vs tight opponents.

 3. Added preflop_min_raise_war_cap(): caps preflop raises after 3+ raise-backs
    in the same hand. Converts raise → call/check.

 4. Changed counter short-handed open pressure bet sizing from max(score, 56)
    to score. Bet size is now proportional to preflop score.

 5. Modified simulator.py to populate actionHistory in the table dict before
    each strategy call. Enables actionHistory-based guards.

 6. Added river_one_pair_over_call(): folds one pair on the river when facing
    a bet > 50% of pot. One pair is a bluff-catcher and should fold to large
    river bets where it's likely behind a stronger hand.

    The guard is inert against the 18 benchmark opponents (no river one-pair
    decisions triggered in benchmark conditions), so benchmark deltas are +0.0.
    Validated by tests/hu/test_river_one_pair_over_call.py (9 tests).

    Reference: PLAN_FIX_RIVER_ONE_PAIR.md
"""

from __future__ import annotations

from functools import lru_cache

from poker_bot.cfr.kuhn import train_kuhn
from poker_bot.hand_eval import best_hand_rank_without, evaluate_hand
from poker_bot.mixing import choose_weighted, resolve_distribution
from poker_bot.opponents import OpponentProfile, profile_from_mapping
from poker_bot.range_model import (
    BayesianRangeTracker,
    average_summary,
    class_strength,
    combo_class,
    estimate_action_range,
)

# Fixed position label implementation that uses all seated players instead of dynamic active players
BUTTON_POSITIONS = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["BTN", "SB", "BB", "CO"],
    5: ["BTN", "SB", "BB", "MP", "CO"],
    6: ["BTN", "SB", "BB", "UTG", "MP", "CO"],
}
CANONICAL_6MAX = {1: "BTN", 2: "SB", 3: "BB", 4: "UTG", 5: "MP", 6: "CO"}


def seated_seat_numbers(table):
    return sorted(
        seat.get("seatNumber")
        for seat in table.get("seats", [])
        if seat.get("seatNumber") is not None and seat.get("agentId") is not None
    )


def position_label(table, my_seat):
    seat_number = (my_seat or {}).get("seatNumber")
    seats = seated_seat_numbers(table)
    button = table.get("buttonSeatNumber")
    if button in seats and seat_number in seats:
        ordered = seats[seats.index(button) :] + seats[: seats.index(button)]
        labels = BUTTON_POSITIONS.get(len(ordered), BUTTON_POSITIONS[6])
        return labels[ordered.index(seat_number)]
    return CANONICAL_6MAX.get(seat_number, "MP")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1: Constants
# ════════════════════════════════════════════════════════════════════════════════

BIG_BLIND = 2
RANK_VALUES = {rank: index for index, rank in enumerate("23456789TJQKA", start=2)}
FOLD_TO_BET_PRIOR = 0.38
CALL_FREQUENCY_PRIOR = 0.24
PROFILE_PRIOR_WEIGHT = 20

ActionDecision = tuple[str | None, int | None, str]
_RANGE_TRACKER = BayesianRangeTracker()

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
SET_MINING_MAX_PRICE = 0.118  # 8:1 implied odds required for set-mining
SET_MINING_TIGHT_OPPONENT_VPIP = 0.18  # fold small pairs vs patient/nitty opponents
SET_MINING_MIN_SPR = 8.0  # need deep enough stacks to realize implied odds
SMALL_PAIR_MULTIWAY_MIN_PLAYERS = 3  # 3+ way pots = too much competition to set-mine
SMALL_PAIR_MULTIWAY_MAX_PRICE = 0.05  # in multi-way, only call if price is very cheap
MEDIUM_HAND_MULTIWAY_MIN_PLAYERS = 3  # 3+ way pots require tighter medium-hand defense
MEDIUM_HAND_MULTIWAY_MAX_PRICE = (
    0.20  # tighter than 0.35 single-way (top pair value drops)
)

# Above this observed call frequency, suppress weak-pair wet-board pot-control
# checks and keep the base bet. Empirically (champion-gate ablation) checking
# weak pairs only helps vs very-low-frequency callers (simple, ~0.02); against
# sticky callers (~0.30) it forfeits EV. The cutoff sits between the two fields.
WEAK_PAIR_POT_CONTROL_MAX_CALL_FREQ = 0.15

# Sliver-shove defense: when call/pot is below this floor, the call is +EV
# vs any plausible range (any two cards clear ~10% equity). Used by
# `sliver_shove_guard` to override rank-0 folds at river. Currently
# restricted to the river (no future streets → equity realization = 100%);
# expand to turn/flop only after data confirms we don't bleed EV via
# reverse-implied odds or low equity realization on earlier streets.
SLIVER_SHOVE_POT_ODDS_FLOOR = 0.10

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


def private_made_hand(hole_cards, board_cards):
    """Return (category, hole_cards_used) for hero's best hand.

    hole_cards_used:
        0 → the best 5-card hand is exactly the board (no private value),
        1 → one hole card contributed to the best hand,
        2 → both hole cards contributed.

    Works for any board length >= 3.
    """
    if len(hole_cards) != 2 or len(board_cards) < 3:
        return 0, 0
    pool = list(hole_cards) + list(board_cards)
    full_rank = evaluate_hand(pool)
    if len(board_cards) >= 5:
        board_rank = evaluate_hand(board_cards)
        if full_rank == board_rank:
            return 0, 0
    used = 0
    h1, h2 = hole_cards
    if full_rank[0] > best_hand_rank_without(pool, [h1])[0]:
        used += 1
    if full_rank[0] > best_hand_rank_without(pool, [h2])[0]:
        used += 1
    return full_rank[0], used


def private_made_hand_rank(hole_cards, board_cards):
    """Category of hero's hand, or 0 if the hand is board-made."""
    category, used = private_made_hand(hole_cards, board_cards)
    return category if used > 0 else 0


# Approximate postflop value used to detect "weaker than preflop" hands.
_PRIVATE_CATEGORY_VALUE = {
    0: 0,
    1: 12,
    2: 24,
    3: 36,
    4: 48,
    5: 60,
    6: 72,
    7: 84,
    8: 96,
}


def relative_hand_drop(hole_cards, board_cards, *, drop_threshold=15):
    """True when hero's actual value is meaningfully below preflop_score.

    Combines category and private-card count into a coarse postflop score and
    compares it against ``preflop_score``.  Useful for spotting hands that
    got *weaker* than they were preflop (e.g. KQ on A♥A♦A♣ 5 4).
    """
    if len(board_cards) < 3:
        return False
    category, used = private_made_hand(hole_cards, board_cards)
    post_value = _PRIVATE_CATEGORY_VALUE.get(category, 0) + used * 6
    return preflop_score(hole_cards) - post_value >= drop_threshold


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1b: Research adjustments (inlined from auto_research.py)
# ════════════════════════════════════════════════════════════════════════════════


def research_preflop_pressure(table, my_seat, base):
    if table.get("street", "Preflop") != "Preflop" or not short_handed(table):
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "raise" not in available or not counter_unopened_preflop(table, allowed):
        return None

    hole_cards = my_seat.get("holeCards", [])
    score = preflop_score(hole_cards)
    position = position_bucket(table, my_seat)
    threshold = 49 if position in {"short", "late"} else 54
    if action in {"call", "check", "fold"} and score >= threshold:
        amount = balanced_raise_amount(table, allowed, max(score, 56))
        return "raise", amount, f"auto research widened open score {score}"
    return None


def research_probe_pressure(table, my_seat, base):
    if table.get("street", "Preflop") == "Preflop" or not short_handed(table):
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if action != "check" or "bet" not in available:
        return None
    if not no_one_has_bet(allowed, table):
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    texture = board_texture(board_cards) if board_cards else {"wet": False}
    if texture.get("wet", False):
        return None

    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    score = preflop_score(hole_cards)
    if rank >= 2 and fragile_rank_two(hole_cards, board_cards, rank):
        return None
    if texture.get("paired", False) and rank == 1 and not top_pair:
        return None
    # High-card boards dominate bottom pair with weak kickers. Don't probe-bet
    # one-pair hands that don't use a top board card.
    if texture.get("high", False) and rank == 1 and not top_pair:
        return None
    if rank >= 1 or top_pair or score >= 49:
        amount = pressure_bet_amount(
            table,
            allowed,
            0.31 if rank or top_pair else 0.22,
        )
        return "bet", amount, f"auto research thin dry-board probe rank {rank}"
    return None


def research_bluff_catch(table, my_seat, base):
    if table.get("street", "Preflop") == "Preflop" or not short_handed(table):
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    price = call_amount(allowed)
    if action != "fold" or "call" not in available or price <= 0:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = int(table.get("potChips") or 0)
    required = pot_odds(price, pot)
    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    draw = has_good_draw(hole_cards, board_cards)
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


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1c: Inlined counter + patch1 logic
# (from survival_balanced_pp_pd_pr_counter.py and survival_balanced_pp_pd_pr_patch1.py)
# ════════════════════════════════════════════════════════════════════════════════


def counter_unopened_preflop(table, allowed):
    blind_size = max(int(allowed.get("minBet") or 0), BIG_BLIND)
    return (
        int(table.get("currentBet") or 0) <= blind_size
        and call_amount(allowed) <= blind_size
    )


def preflop_thresholds(table, my_seat):
    active = lookup_active_players(table)
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


def postflop_raise_amount(table, allowed, strong=False, semi_bluff=False):
    minimum = allowed.get("minRaiseTo")
    if minimum is None:
        return None
    pot = int(table.get("potChips") or 0)
    current_bet = int(table.get("currentBet") or 0)
    if strong:
        fraction = 0.85
    elif semi_bluff:
        fraction = 0.55
    else:
        fraction = 0.65
    target = current_bet + int(max(pot, BIG_BLIND) * fraction)
    return capped(max(int(minimum), target), allowed)


def stable_mix_percent(hole_cards, board_cards, street, pot):
    key = "|".join([street, str(pot), *sorted(hole_cards), *sorted(board_cards)])
    return sum((index + 1) * ord(char) for index, char in enumerate(key)) % 100


def opponent_barrels_streets(table, my_seat, opponent_id=None) -> bool:
    """Detect if an opponent has bet/raised on every postflop street.

    A triple barrel (bet flop, turn, and river) is a strong signal. Against
    tight/passive players this is usually a very strong value hand. Against
    known bluffers it may be a bluff, but the bot should still be cautious.
    """
    history = table.get("actionHistory") or table.get("action_history") or []
    if not history:
        return False

    my_id = (my_seat or {}).get("agentId")
    target_id = opponent_id or my_id

    # Find the active opponent who is not the hero.
    if opponent_id is None:
        for seat in table.get("seats", []):
            seat_id = seat.get("agentId")
            if seat_id and seat_id != my_id and not seat.get("folded", False):
                target_id = seat_id
                break

    if not target_id:
        return False

    streets = {"Flop", "Turn", "River"}
    bet_actions = {"bet", "raise", "all-in"}
    seen_streets = set()

    for event in history:
        agent_id = event.get("agentId") or event.get("agent_id")
        street = event.get("street")
        action = str(event.get("action", "")).lower()

        if agent_id != target_id or street not in streets:
            continue

        if action in bet_actions:
            seen_streets.add(street)

    return len(seen_streets) == 3


def opponent_barrels_current_street(table, my_seat, opponent_id=None) -> bool:
    """Detect if an opponent has bet/raised on the current street.

    This is used for single-street aggression detection (e.g., turn barrel).
    """
    history = table.get("actionHistory") or table.get("action_history") or []
    if not history:
        return False

    my_id = (my_seat or {}).get("agentId")
    target_id = opponent_id or my_id

    # Find the active opponent who is not the hero.
    if opponent_id is None:
        for seat in table.get("seats", []):
            seat_id = seat.get("agentId")
            if seat_id and seat_id != my_id and not seat.get("folded", False):
                target_id = seat_id
                break

    if not target_id:
        return False

    current_street = table.get("street", "")
    bet_actions = {"bet", "raise", "all-in"}

    for event in history:
        agent_id = event.get("agentId") or event.get("agent_id")
        street = event.get("street")
        action = str(event.get("action", "")).lower()

        if agent_id == target_id and street == current_street and action in bet_actions:
            return True

    return False


def has_preflop_advantage(table, my_seat):
    score = preflop_score(my_seat.get("holeCards", []))
    position = position_bucket(table, my_seat)
    if position in {"late", "short"}:
        return score >= 62
    if position == "middle":
        return score >= 68
    return score >= 72


def balanced_preflop_action(table, my_seat):
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    hole_cards = my_seat.get("holeCards", [])
    score = preflop_score(hole_cards)
    pot = int(table.get("potChips") or 0) + sum(
        int(seat.get("currentBetChips") or 0) for seat in table.get("seats", [])
    )
    call_amount_value = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    required = pot_odds(call_amount_value, pot)
    raise_threshold, call_threshold = preflop_thresholds(table, my_seat)
    blind_size = max(int(allowed.get("minBet") or 0), BIG_BLIND)
    facing_raise = (
        int(table.get("currentBet") or 0) > blind_size or call_amount_value > blind_size
    )

    if "raise" in available and score >= raise_threshold:
        amount = balanced_raise_amount(table, allowed, score)
        return "raise", amount, f"balanced value/open raise score {score}"

    if "call" in available:
        stack = int(my_seat.get("stackChips") or 1)
        raise_price = call_amount_value / max(stack, 1)
        if facing_raise:
            # Shove-heavy opponents punish speculative calls. Fold true junk at
            # very large stack-commitment prices, but keep playable short-handed
            # defending ranges for the rest of the field.
            if raise_price > 0.50 and score < 50:
                return "fold", None, f"balanced preflop fold vs shove score {score}"
            if score >= call_threshold + 6 or (
                score >= call_threshold + 2 and required <= 0.30
            ):
                return "call", call_amount_value, f"balanced defend score {score}"
        elif score >= call_threshold and required <= 0.38:
            return "call", call_amount_value, f"balanced selective call score {score}"

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
    active = lookup_active_players(table)

    if rank >= 2 and fragile_rank_two(hole_cards, board_cards, rank):
        return None

    # Paired boards punish thin value with medium-strength hands. Against any
    # opponent, middle pair with a weak kicker on a paired board is
    # vulnerable to sets and two pair. Don't probe-bet for thin value into
    # a board that dominates us.
    texture = board_texture(board_cards) if board_cards else {"wet": False}
    paired = texture.get("paired", False)
    if paired and rank == 1 and not top_pair:
        return None
    # High-card boards (A/K/Q/J) dominate bottom pair and weak two pair.
    # Don't probe-bet one-pair hands that don't use a top board card.
    if texture.get("high", False) and rank == 1 and not top_pair:
        return None

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

    if (
        no_one_has_bet(allowed, table)
        and not texture.get("wet", False)
        and active <= 4
        and (rank == 1 or has_preflop_advantage(table, my_seat))
    ):
        amount = probe_bet_amount(table, allowed, active)
        return "bet", amount, f"postflop pressure dry-board probe rank {rank}"

    return None


def anti_bully_defense(table, my_seat, blueprint):
    action, _amount, _message = blueprint
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if action != "fold" or "call" not in available:
        return None
    if not covered_by_larger_stack(table, my_seat):
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if not board_cards:
        return None

    pot = int(table.get("potChips") or 0)
    call_amount_value = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    stack = int(my_seat.get("stackChips") or 0)
    required = pot_odds(call_amount_value, pot)
    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    medium = rank == 1 or top_pair
    draw = has_good_draw(hole_cards, board_cards)
    cheap_stack_price = stack <= 0 or call_amount_value / stack <= 0.18

    if medium and required <= 0.26 and cheap_stack_price:
        return "call", call_amount_value, f"anti-bully cheap medium defense rank {rank}"
    if draw and required <= 0.20 and cheap_stack_price:
        return "call", call_amount_value, "anti-bully cheap draw defense"
    return None


def river_trips_price_cap(table, my_seat, blueprint):
    action, _amount, _message = blueprint
    if action != "call" or table.get("street") != "River":
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "call" not in available or "fold" not in available:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if made_hand_rank(hole_cards, board_cards) != 3:
        return None

    call_amount_value = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    pot = int(table.get("potChips") or 0)
    required = pot_odds(call_amount_value, pot)
    if required <= 0.40:
        return None

    return "fold", None, "folding overpriced river trips"


def postflop_raise_back_defense(table, my_seat, blueprint):
    action, _amount, _message = blueprint
    if action == "raise" or table.get("street", "Preflop") == "Preflop":
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    call_amount_value = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    if call_amount_value <= 0 or "call" not in available:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if not board_cards:
        return None

    street = table.get("street", "Flop")
    pot = int(table.get("potChips") or 0)
    required = pot_odds(call_amount_value, pot)
    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards)
    active = lookup_active_players(table)
    medium = rank == 1 or top_pair
    draw = has_good_draw(hole_cards, board_cards)
    can_raise = "raise" in available

    if rank >= 3:
        if can_raise:
            amount = postflop_raise_amount(table, allowed, strong=True)
            return "raise", amount, f"postflop value raise rank {rank}"
        if required <= 0.45:
            return "call", call_amount_value, f"postflop strong call rank {rank}"

    if board_dominated_two_pair(hole_cards, board_cards, rank):
        if required <= 0.16:
            return "call", call_amount_value, "board-dominated two-pair bluff catch"
        if "fold" in available:
            return "fold", None, "folding board-dominated two pair"
        return None

    if rank == 2:
        if paired_board_rank_two(hole_cards, board_cards, rank):
            if required <= 0.28:
                return "call", call_amount_value, "paired-board rank-2 pot control"
            if "fold" in available:
                return "fold", None, "folding fragile paired-board rank 2"
            return None
        if can_raise and street in {"Flop", "Turn"} and required <= 0.34:
            amount = postflop_raise_amount(table, allowed)
            return "raise", amount, "postflop value/protection raise rank 2"
        if required <= 0.40:
            return (
                "call",
                call_amount_value,
                "postflop strong pair/trips defense rank 2",
            )

    if medium:
        if (
            can_raise
            and active <= 3
            and not texture.get("paired", False)
            and not texture.get("wet", False)
            and required <= 0.18
            and stable_mix_percent(hole_cards, board_cards, street, pot) < 30
        ):
            amount = postflop_raise_amount(table, allowed)
            return "raise", amount, "postflop dry-board medium raise"
        if required <= 0.26:
            return "call", call_amount_value, f"postflop medium defense rank {rank}"

    if draw and street in {"Flop", "Turn"}:
        if (
            can_raise
            and active <= 3
            and not texture.get("paired", False)
            and required <= 0.18
            and stable_mix_percent(hole_cards, board_cards, street, pot) < 35
        ):
            amount = postflop_raise_amount(table, allowed, semi_bluff=True)
            return "raise", amount, "postflop semi-bluff raise"
        if required <= 0.22:
            return "call", call_amount_value, "postflop draw defense"

    return None


def high_card_bottom_pair_check(table, my_seat, blueprint):
    action, _amount, _message = blueprint
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if action != "bet" or "check" not in available:
        return None
    if table.get("street", "Preflop") == "Preflop":
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None

    texture = board_texture(board_cards) if board_cards else {"high": False}
    if not texture.get("high", False):
        return None

    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    if rank != 1 or top_pair:
        return None

    return "check", None, "high-card board: checking bottom pair"


def tag_opponent_paired_board_check(table, my_seat, blueprint):
    action, _amount, _message = blueprint
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if action != "bet" or "check" not in available:
        return None
    if table.get("street", "Preflop") == "Preflop":
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None

    rank = made_hand_rank(hole_cards, board_cards)
    if rank != 1:
        return None

    texture = board_texture(board_cards) if board_cards else {"paired": False}
    if not texture.get("paired", False):
        return None

    if not _has_tag_opponent(table, my_seat):
        return None

    return "check", None, "TAG opponent: checking medium hand on paired board"


def large_bet_medium_hand_fold(table, my_seat, blueprint):
    action, _amount, _message = blueprint
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if action != "call" or "fold" not in available:
        return None
    if table.get("street", "Preflop") == "Preflop":
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    rank = made_hand_rank(hole_cards, board_cards) if board_cards else 0
    top_pair = has_top_pair_or_better(hole_cards, board_cards) if board_cards else False
    if rank not in {1, 2} and not top_pair:
        return None

    price = call_amount(allowed)
    stack = int(my_seat.get("stackChips") or 0)
    stack_price = price / max(stack, 1)
    if stack_price <= 0.30:
        return None

    return "fold", None, "large bet: folding medium hand at high stack price"


def tag_opponent_medium_hand_fold(table, my_seat, blueprint):
    action, _amount, _message = blueprint
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if action != "call" or "fold" not in available:
        return None
    if table.get("street", "Preflop") == "Preflop":
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None

    rank = made_hand_rank(hole_cards, board_cards)
    if rank not in {1, 2}:
        return None

    if not _has_tag_opponent(table, my_seat):
        return None

    price = call_amount(allowed)
    stack = int(my_seat.get("stackChips") or 0)
    stack_price = price / max(stack, 1)
    if stack_price <= 0.50:
        return None

    return "fold", None, "TAG opponent: folding medium hand at high stack price"


def patch1_choose_action(table, my_seat):
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

    blueprint = survival_lookahead_choose_action(table, my_seat)
    raise_back = postflop_raise_back_defense(table, my_seat, blueprint)
    if raise_back is not None:
        return raise_back

    paired_check = tag_opponent_paired_board_check(table, my_seat, blueprint)
    if paired_check is not None:
        return paired_check

    defense = anti_bully_defense(table, my_seat, blueprint)
    if defense is not None:
        return defense

    river_trips = river_trips_price_cap(table, my_seat, blueprint)
    if river_trips is not None:
        return river_trips

    postflop = balanced_postflop_adjustment(table, my_seat, blueprint)
    if postflop is not None:
        return postflop

    tag_fold = tag_opponent_medium_hand_fold(table, my_seat, blueprint)
    if tag_fold is not None:
        return tag_fold

    large_bet_fold = large_bet_medium_hand_fold(table, my_seat, blueprint)
    if large_bet_fold is not None:
        return large_bet_fold

    action, amount, message = blueprint
    return action, amount, f"balanced postflop-pressure blueprint: {message}"


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
    position = position_bucket(table, my_seat)
    threshold = 40 if position in {"short", "late"} else 45
    if action in {"call", "check", "fold"} and score >= threshold:
        amount = balanced_raise_amount(table, allowed, score)
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
    if not no_one_has_bet(allowed, table):
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
    if rank >= 2 and fragile_rank_two(hole_cards, board_cards, rank):
        return None
    if paired and rank == 1 and not top_pair:
        return None
    # High-card boards dominate bottom pair with weak kickers. Don't pressure-bet
    # one-pair hands that don't use a top board card.
    if texture.get("high", False) and rank == 1 and not top_pair:
        return None
    if rank >= 1 or top_pair or score >= (42 if paired else 38):
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
    draw = has_good_draw(hole_cards, board_cards)

    if rank >= 2 and not fragile_rank_two(hole_cards, board_cards, rank):
        amount = pressure_raise_amount(table, allowed, 0.62)
        return "raise", amount, f"counter value/protection raise rank {rank}"

    if (
        draw
        and not texture.get("paired", False)
        and required <= 0.20
        and stable_mix_percent(
            hole_cards, board_cards, table.get("street", "Flop"), pot
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


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1d: Inlined Phase 3 logic
# (counter_adaptive, simple, adaptive, profiled_counter_adaptive,
#  anti_threshold, survival_sixmax, survival_lookup, survival_lookahead)
# ════════════════════════════════════════════════════════════════════════════════


# ── counter_adaptive helpers (new) ──────────────────────────────────────────────


def effective_spr(table, my_seat, call_amount=0):
    if my_seat is None:
        return None
    hero_stack = int(my_seat.get("stackChips") or 0)
    pot = effective_pot(table)
    if pot <= 0:
        return None
    hero_stack_after_call = max(0, hero_stack - int(call_amount or 0))
    pot_after_call = pot + int(call_amount or 0)
    return hero_stack_after_call / max(1, pot_after_call)


def spr_band(spr):
    if spr is None:
        return "unknown"
    if spr < 2:
        return "low"
    if spr < 5:
        return "medium"
    return "high"


def spr_bet_fraction(spr, strong=False):
    """Bet fraction based on SPR.

    Low SPR: smaller bets (pot-committed, less room to maneuver)
    Medium SPR: balanced sizing
    High SPR: larger bets (build the pot, charge draws)
    """
    band = spr_band(spr)
    if band == "low":
        return 0.30 if strong else 0.25
    if band == "medium":
        return 0.50 if strong else 0.40
    return 0.65 if strong else 0.55


def spr_raise_pressure(spr, strong=False):
    """Raise pressure based on SPR.

    Low SPR: larger pressure (pot-committed, build the pot)
    Medium SPR: balanced sizing
    High SPR: smaller pressure (more room to maneuver)
    """
    band = spr_band(spr)
    if band == "low":
        return 0.85 if strong else 0.65
    if band == "medium":
        return 0.70 if strong else 0.55
    return 0.55 if strong else 0.45


def small_pressure_bet(pot, allowed, spr=None):
    minimum = allowed.get("minBet", BIG_BLIND)
    if pot <= 0:
        return capped(max(minimum, BIG_BLIND), allowed)
    fraction = spr_bet_fraction(spr, strong=False)
    return capped(max(minimum, int(pot * fraction)), allowed)


def value_bet(pot, allowed, strong=False, spr=None):
    minimum = allowed.get("minBet", BIG_BLIND)
    if pot <= 0:
        return capped(max(minimum, BIG_BLIND), allowed)
    fraction = spr_bet_fraction(spr, strong=strong)
    return capped(max(minimum, int(pot * fraction)), allowed)


def raise_for_value(table, allowed, strong=False, my_seat=None):
    minimum = allowed.get("minRaiseTo")
    if minimum is None:
        return None
    pot = table.get("potChips", 0)
    current_bet = table.get("currentBet", 0)
    call_amount_value = call_amount(allowed)
    spr = effective_spr(table, my_seat, call_amount=call_amount_value)
    pressure = int(max(pot, BIG_BLIND) * spr_raise_pressure(spr, strong=strong))
    return capped(max(minimum, current_bet + pressure), allowed)


def counter_adaptive_choose_action(table, my_seat):
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if not available:
        return None, None, "No legal actions available"
    if not my_seat:
        if "fold" in available:
            return "fold", None, "Fallback: seat not found"
        return None, None, "No matching seat found"

    street = table.get("street", "Preflop")
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = table.get("potChips", 0)
    call_amount = allowed.get("callAmount", 0)
    required = pot_odds(call_amount, pot)

    if street == "Preflop":
        score = preflop_score(hole_cards)
        if "raise" in available and score >= 74:
            amount = raise_for_value(
                table, allowed, strong=score >= 94, my_seat=my_seat
            )
            return "raise", amount, f"Countering adaptive with score {score}"
        if "call" in available:
            if score >= 42 or required <= 0.08:
                return "call", call_amount, f"Calling controlled preflop score {score}"
            if "check" in available:
                return "check", None, f"Checking weak preflop score {score}"
            return "fold", None, f"Folding weak preflop score {score}"
        if "bet" in available and score >= 50:
            spr = effective_spr(table, my_seat)
            amount = small_pressure_bet(pot, allowed, spr=spr)
            return "bet", amount, f"Opening into adaptive score {score}"
        if "check" in available:
            return "check", None, f"Checking preflop score {score}"

    made_rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards)
    strong = made_rank >= 2
    medium = made_rank == 1 or top_pair
    triple_barrel = opponent_barrels_streets(table, my_seat)
    current_barrel = opponent_barrels_current_street(table, my_seat)

    if "raise" in available and strong:
        amount = raise_for_value(table, allowed, strong=made_rank >= 3, my_seat=my_seat)
        return "raise", amount, f"Value raising adaptive rank {made_rank}"

    if "call" in available:
        call_threshold = 0.18 if medium else 0.08
        if triple_barrel and medium:
            call_threshold = 0.10  # Be cautious vs triple barrels with medium hands
        if current_barrel and medium and street in {"Turn", "River"}:
            call_threshold = 0.12  # Be cautious vs turn/river barrels
        if strong or (medium and required <= call_threshold) or required <= 0.08:
            message = "Calling adaptive value range"
            if triple_barrel:
                message = "Calling triple-barrel price"
            return "call", call_amount, f"{message} rank {made_rank}"
        if "check" in available:
            return "check", None, "Checking marginal hand"
        return "fold", None, f"Overfolding to adaptive aggression rank {made_rank}"

    if "bet" in available:
        spr = effective_spr(table, my_seat)
        if strong:
            amount = value_bet(pot, allowed, strong=made_rank >= 3, spr=spr)
            return "bet", amount, f"Value betting adaptive rank {made_rank}"
        if medium:
            amount = value_bet(pot, allowed, spr=spr)
            return "bet", amount, "Thin value versus adaptive"
        if not texture["wet"]:
            amount = small_pressure_bet(pot, allowed, spr=spr)
            return "bet", amount, "Small pressure against adaptive checks"

    if "check" in available:
        return "check", None, "Checking no-edge spot"
    if "fold" in available:
        return "fold", None, "No profitable counter action"
    return None, None, "No supported legal action available"


# ── simple strategy (renamed choose_action to simple_choose_action) ────────────


def simple_choose_action(table, my_seat):
    """Choose a legal action using a simple hole-card strength heuristic."""
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if not available:
        return None, None, "No legal actions available"

    street = table.get("street", "Preflop")
    if not my_seat:
        if "fold" in available:
            return "fold", None, "Fallback: seat not found"
        return None, None, "No matching seat found"

    cards = my_seat.get("holeCards", [])
    stack = my_seat.get("stackChips", 0)
    pot = table.get("potChips", 0)

    ranks = [card[0] for card in cards] if cards else []
    suits = [card[1] for card in cards] if cards else []

    rank_values = {"A": 14, "K": 13, "Q": 12, "J": 11, "T": 10}
    for value in range(2, 10):
        rank_values[str(value)] = value

    vals = sorted([rank_values.get(rank, 0) for rank in ranks], reverse=True)
    is_pair = len(vals) == 2 and vals[0] == vals[1]
    is_suited = len(suits) == 2 and suits[0] == suits[1]
    high_card = vals[0] if vals else 0
    low_card = vals[1] if len(vals) > 1 else 0

    if is_pair:
        strength = 50 + high_card * 2
        if high_card >= 10:
            strength += 20
    else:
        strength = high_card * 3 + low_card
        if is_suited:
            strength += 10
        if high_card >= 12 and low_card >= 10:
            strength += 15
        if abs(high_card - low_card) == 1:
            strength += 5

    call_amount = allowed.get("callAmount", 0)
    min_raise = allowed.get("minRaiseTo")
    pot_odds = call_amount / (pot + call_amount) if (pot + call_amount) > 0 else 1.0

    action = None
    amount = None
    message = ""

    if street == "Preflop":
        if strength >= 60:
            if "raise" in available and min_raise:
                action = "raise"
                amount = min(
                    min_raise + (pot // 2), allowed.get("maxCommit", min_raise)
                )
                message = "Strong hand, raising to build the pot"
            elif "bet" in available:
                action = "bet"
                amount = max(pot // 2, allowed.get("minBet", 0))
                message = "Strong hand, betting for value"
            elif "call" in available:
                action = "call"
                amount = call_amount
                message = "Strong hand, calling to see flop"
            else:
                action = "check"
                message = "Strong hand, checking"
        elif strength >= 35:
            if "check" in available:
                action = "check"
                message = "Medium hand, seeing a cheap flop"
            elif "call" in available and pot_odds < 0.3:
                action = "call"
                amount = call_amount
                message = "Decent hand, good pot odds to call"
            else:
                action = "fold"
                message = "Marginal hand, folding to aggression"
        else:
            if "check" in available:
                action = "check"
                message = "Weak hand, checking behind"
            elif (
                "call" in available and pot_odds < 0.15 and call_amount <= stack * 0.05
            ):
                action = "call"
                amount = call_amount
                message = "Speculative hand, cheap call"
            else:
                action = "fold"
                message = "Weak hand, no reason to continue"
    else:
        if strength >= 50:
            if "bet" in available:
                action = "bet"
                amount = max(pot // 2, allowed.get("minBet", 0))
                message = f"Strong made hand, betting for value on {street}"
            elif "raise" in available and min_raise:
                action = "raise"
                amount = min_raise
                message = f"Strong hand, raising on {street}"
            elif "call" in available:
                action = "call"
                amount = call_amount
                message = "Strong hand, calling down"
            else:
                action = "check"
                message = "Strong hand, checking for pot control"
        elif strength >= 25:
            if "check" in available:
                action = "check"
                message = f"Marginal hand, checking on {street}"
            elif "call" in available and pot_odds < 0.25:
                action = "call"
                amount = call_amount
                message = "Draw or medium hand, calling"
            else:
                action = "fold"
                message = "Can't continue on this board, folding"
        else:
            if "check" in available:
                action = "check"
                message = f"Weak hand, giving up on {street}"
            else:
                action = "fold"
                message = "No hand, folding"

    if action and action not in available:
        if "check" in available:
            action = "check"
            message = "Fallback: checking"
        elif "fold" in available:
            action = "fold"
            message = "Fallback: folding"
        else:
            return None, None, "No fallback action available"

    if action in ("bet", "raise") and amount is None:
        if action == "bet" and "bet" in available:
            amount = allowed.get("minBet", 0)
        elif action == "raise" and "raise" in available:
            amount = allowed.get("minRaiseTo")
        else:
            if "fold" in available:
                return "fold", None, "Fallback: folding"
            return None, None, "No amount available"

    if action == "call" and amount is None:
        amount = call_amount

    return action, amount, message


# ── adaptive strategy (renamed choose_action to adaptive_choose_action) ───────


# ── adaptive strategy (renamed choose_action to adaptive_choose_action) ───────


def adaptive_has_overcard_pressure(hole_cards, board_cards):
    if not board_cards:
        return False
    board_high = max(card_values(board_cards))
    return max(card_values(hole_cards), default=0) > board_high


def adaptive_pressure_bet_amount(pot, allowed, texture):
    min_bet = allowed.get("minBet", BIG_BLIND)
    if pot <= 0:
        return capped(max(min_bet, BIG_BLIND), allowed)
    fraction = 0.65 if texture["wet"] else 0.45
    return capped(max(min_bet, int(pot * fraction)), allowed)


def adaptive_value_bet_amount(pot, allowed, strong=False):
    min_bet = allowed.get("minBet", BIG_BLIND)
    if pot <= 0:
        return capped(max(min_bet, BIG_BLIND), allowed)
    return capped(max(min_bet, int(pot * (0.62 if strong else 0.38))), allowed)


def adaptive_raise_amount(table, allowed, strong=False):
    minimum = allowed.get("minRaiseTo")
    if minimum is None:
        return None
    pot = table.get("potChips", 0)
    current_bet = table.get("currentBet", 0)
    pressure = int(max(pot, BIG_BLIND) * (0.62 if strong else 0.38))
    return capped(max(minimum, current_bet + pressure), allowed)


def adaptive_choose_action(table, my_seat):
    """Exploitative adaptive strategy tuned against the simple baseline."""
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if not available:
        return None, None, "No legal actions available"
    if not my_seat:
        if "fold" in available:
            return "fold", None, "Fallback: seat not found"
        return None, None, "No matching seat found"

    street = table.get("street", "Preflop")
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = table.get("potChips", 0)
    call_amount = allowed.get("callAmount", 0)
    required = pot_odds(call_amount, pot)

    if street == "Preflop":
        score = preflop_score(hole_cards)
        if "raise" in available and score >= 78:
            amount = adaptive_raise_amount(table, allowed, strong=score >= 95)
            return "raise", amount, f"Premium preflop score {score}, raising"
        if "call" in available:
            if score >= 46 or required <= 0.12:
                return "call", call_amount, f"Playable preflop score {score}, calling"
            if "check" in available:
                return "check", None, f"Weak preflop score {score}, checking"
            return "fold", None, f"Weak preflop score {score}, folding"
        if "bet" in available and score >= 62:
            amount = adaptive_value_bet_amount(pot, allowed, strong=score >= 88)
            return "bet", amount, f"Strong preflop score {score}, betting"
        if "check" in available:
            return "check", None, f"Preflop score {score}, checking"

    made_rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = (
        board_texture(board_cards)
        if board_cards
        else {"wet": False, "paired": False, "high": False}
    )
    strong = made_rank >= 2
    medium = made_rank == 1 or top_pair

    if "raise" in available and strong:
        amount = adaptive_raise_amount(table, allowed, strong=made_rank >= 3)
        return "raise", amount, f"Made hand rank {made_rank}, raising simple"

    if "call" in available:
        if strong or (medium and required <= 0.24) or required <= 0.10:
            return "call", call_amount, f"Defending with rank {made_rank}"
        if "check" in available:
            return "check", None, "Not advantageous, checking"
        return "fold", None, f"Rank {made_rank} below price, folding"

    if "bet" in available:
        # Paired boards punish thin value with medium-strength hands. Against
        # any opponent, middle pair with a weak kicker on a paired board is
        # vulnerable to sets and two pair. Check back rather than thin-value
        # bet into a board that dominates us.
        if (
            texture.get("paired", False)
            and made_rank == 1
            and not top_pair
            or texture.get("high", False)
            and made_rank == 1
            and not top_pair
        ):
            pass
        elif strong:
            amount = adaptive_value_bet_amount(pot, allowed, strong=made_rank >= 3)
            return "bet", amount, f"Value betting rank {made_rank}"
        elif medium:
            amount = adaptive_value_bet_amount(pot, allowed)
            return "bet", amount, "Thin value against simple"
        if (
            texture.get("high", False)
            and not texture.get("wet", False)
            and adaptive_has_overcard_pressure(hole_cards, board_cards)
        ):
            amount = adaptive_pressure_bet_amount(pot, allowed, texture)
            return "bet", amount, "Pressure betting dry high-card board"

    if "check" in available:
        return "check", None, "Not advantageous, checking"
    if "fold" in available:
        return "fold", None, "No profitable action, folding"
    return None, None, "No supported legal action available"


# ── profiled_counter_adaptive (renamed choose_action) ─────────────────────────


def profiled_table_profiles(table, my_seat):
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


def profiled_choose_action(table, my_seat):
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if not available:
        return None, None, "No legal actions available"
    if not my_seat:
        if "fold" in available:
            return "fold", None, "Fallback: seat not found"
        return None, None, "No matching seat found"

    opponents = active_opponents(table, my_seat)
    profiles = profiled_table_profiles(table, my_seat)
    tendencies = _table_tendencies(profiles)
    street = table.get("street", "Preflop")
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = table.get("potChips", 0)
    call_amount = allowed.get("callAmount", 0)
    required = pot_odds(call_amount, pot)

    if opponents <= 1 and not profiles:
        return counter_adaptive_choose_action(table, my_seat)

    if street == "Preflop":
        decision = medium_pocket_pair_vs_tight_three_bet(
            table, my_seat, ("raise", None, "base")
        )
        if decision is not None:
            return decision

        score = preflop_score(hole_cards)
        threshold = 48 + max(0, opponents - 2) * 6
        if tendencies["has_loose_passive"]:
            threshold -= 6
        if tendencies["has_tight_aggressive"]:
            threshold += 6
        if tendencies["has_bluffer"]:
            threshold -= 3
        if tendencies["has_station"]:
            threshold -= 3
        if tendencies["has_high_wtsd"]:
            threshold -= 2
        if tendencies["has_low_wtsd"]:
            threshold -= 2
        if tendencies["all_patient"]:
            threshold -= 4
        if tendencies["has_aggressive"]:
            threshold += 4

        if "raise" in available and score >= threshold + 30:
            amount = raise_for_value(
                table, allowed, strong=score >= 95, my_seat=my_seat
            )
            return "raise", amount, f"Profiled premium score {score}, raising"
        if "call" in available:
            price_cap = 0.08 if opponents <= 2 else 0.05
            if tendencies["has_bluffer"]:
                price_cap += 0.02
            if tendencies["has_station"]:
                price_cap += 0.01
            if score >= threshold or required <= price_cap:
                return "call", call_amount, f"Profiled preflop score {score}, calling"
            if "check" in available:
                return "check", None, f"Profiled weak score {score}, checking"
            return "fold", None, f"Profiled weak score {score}, folding"
        if "bet" in available and score >= threshold:
            spr = effective_spr(table, my_seat)
            amount = small_pressure_bet(pot, allowed, spr=spr)
            return "bet", amount, f"Profiled open score {score}"
        if "check" in available:
            return "check", None, f"Profiled score {score}, checking"

    made_rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards)
    strong_threshold = 2 if opponents <= 2 else 3
    strong = made_rank >= strong_threshold
    medium = made_rank == 1 or top_pair
    triple_barrel = opponent_barrels_streets(table, my_seat)
    current_barrel = opponent_barrels_current_street(table, my_seat)

    if "raise" in available and strong:
        if triple_barrel and tendencies["has_tight"]:
            return "fold", None, f"Tight triple-barrel caution rank {made_rank}"
        amount = raise_for_value(table, allowed, strong=made_rank >= 4, my_seat=my_seat)
        return "raise", amount, f"Profiled value raise rank {made_rank}"

    if "call" in available:
        if strong:
            return "call", call_amount, f"Profiled call rank {made_rank}"
        if triple_barrel and medium:
            if has_top_pair_good_kicker(hole_cards, board_cards):
                if tendencies["has_tight"]:
                    return "fold", None, f"Tight triple-barrel caution rank {made_rank}"
                return (
                    "call",
                    call_amount,
                    f"Top pair good kicker vs triple barrel rank {made_rank}",
                )
            if tendencies["has_loose"]:
                return (
                    "call",
                    call_amount,
                    f"Loose triple-barrel bluff catch rank {made_rank}",
                )
            return "fold", None, f"Triple-barrel caution rank {made_rank}"
        if current_barrel and medium and street in {"Turn", "River"}:
            return "fold", None, f"Turn/river barrel caution rank {made_rank}"
        if tendencies["has_bluffer"] and medium and required <= 0.30:
            return "call", call_amount, "Bluff-catching profiled opponent"
        if tendencies["has_station"] and medium and required <= 0.20:
            return "call", call_amount, "Station bluff-catch medium hand"
        if tendencies["has_high_wtsd"] and medium and required <= 0.18:
            return "call", call_amount, "High WTSD medium hand defense"
        if medium and opponents <= 2 and required <= 0.14:
            return "call", call_amount, "Heads-up medium hand defense"
        if (
            medium
            and has_overpair_to_board(hole_cards, board_cards)
            and required <= 0.40
        ):
            return "call", call_amount, "Overpair defense vs barrel"
        if "check" in available:
            return "check", None, "Profiled marginal hand, checking"
        return "fold", None, f"Profiled fold rank {made_rank}"

    if "bet" in available:
        spr = effective_spr(table, my_seat)
        if strong:
            amount = value_bet(
                pot,
                allowed,
                strong=made_rank >= 4
                or tendencies["has_station"]
                or tendencies["has_high_wtsd"],
                spr=spr,
            )
            return "bet", amount, f"Profiled value bet rank {made_rank}"
        if medium and opponents <= 2:
            amount = value_bet(pot, allowed, spr=spr)
            return "bet", amount, "Profiled thin value heads-up"
        if (
            opponents <= 2
            and (tendencies["all_patient"] or tendencies["has_low_wtsd"])
            and texture["high"]
            and not texture["wet"]
        ):
            amount = small_pressure_bet(pot, allowed, spr=spr)
            return "bet", amount, "Profiled pressure versus patient table"

    if "check" in available:
        return "check", None, "Profiled no-edge spot, checking"
    if "fold" in available:
        return "fold", None, "Profiled no profitable action"
    return None, None, "No supported legal action available"


def _table_tendencies(profiles):
    labels = [profile.label() for profile in profiles]
    station_labels = {"calling_station", "loose-passive", "loose-measured"}
    aggressive_labels = {
        "loose_aggressive",
        "tight_aggressive",
        "bluffer",
        "loose-aggressive",
        "balanced-aggressive",
    }
    patient_labels = {"patient_methodical", "unknown", "tight-passive"}

    def has_bluffer(profile):
        return profile.is_bluffer()

    def has_station(profile):
        return profile.is_station()

    def is_patient(profile):
        return profile.is_patient()

    def is_aggressive(profile):
        label = profile.label()
        if label in aggressive_labels:
            return True
        if profile.api_af is not None and profile.api_af > 2.0:
            return True
        if profile.api_pfr is not None and profile.api_pfr >= 0.25:
            return True
        return False

    def is_loose_passive(profile):
        return profile.is_loose_passive()

    def is_tight_aggressive(profile):
        return profile.is_tight_aggressive()

    def is_loose(profile):
        label = profile.label()
        return label in {
            "loose_aggressive",
            "loose-passive",
            "loose-measured",
            "calling_station",
            "balanced",
        }

    def is_tight(profile):
        label = profile.label()
        return label in {"tight_aggressive", "tight-passive", "patient_methodical"}

    return {
        "has_bluffer": any(has_bluffer(profile) for profile in profiles),
        "has_station": any(has_station(profile) for profile in profiles),
        "all_patient": bool(profiles)
        and all(is_patient(profile) for profile in profiles),
        "has_aggressive": any(is_aggressive(profile) for profile in profiles),
        "has_loose_passive": any(is_loose_passive(profile) for profile in profiles),
        "has_tight_aggressive": any(
            is_tight_aggressive(profile) for profile in profiles
        ),
        "has_loose": any(is_loose(profile) for profile in profiles),
        "has_tight": any(is_tight(profile) for profile in profiles),
        "has_high_wtsd": any(profile.has_high_wtsd() for profile in profiles),
        "has_low_wtsd": any(profile.has_low_wtsd() for profile in profiles),
    }


# ── anti_threshold (renamed choose_action) ────────────────────────────────────


def anti_threshold_should_cbet(texture, opponents):
    if not texture:
        return True
    return not texture.get("wet", False) or opponents <= 1


def anti_threshold_choose_action(table, my_seat):
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if not available:
        return None, None, "No legal actions available"
    if not my_seat:
        if "fold" in available:
            return "fold", None, "Fallback: seat not found"
        return None, None, "No matching seat found"

    street = table.get("street", "Preflop")
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = table.get("potChips", 0)
    call_amount = allowed.get("callAmount", 0)
    required = pot_odds(call_amount, pot)
    opponents = active_opponents(table, my_seat)

    seat_offset = None
    for i, s in enumerate(table.get("seats", [])):
        if s.get("agentId") == (my_seat or {}).get("agentId"):
            seat_offset = i
            break

    in_position = seat_offset is not None and opponents <= 2

    if street == "Preflop":
        score = preflop_score(hole_cards)
        play_threshold = 38 if in_position else 44
        raise_threshold = 72 if in_position else 78
        premium_threshold = 88 if in_position else 92

        if "raise" in available and score >= raise_threshold:
            amount = raise_for_value(
                table, allowed, strong=score >= premium_threshold, my_seat=my_seat
            )
            return "raise", amount, f"Premium score {score}, raising"
        if "call" in available:
            if score >= play_threshold or required <= 0.12:
                return "call", call_amount, f"Playable score {score}, calling"
            if "check" in available:
                return "check", None, f"Checking score {score}"
            return "fold", None, f"Folding weak score {score}"
        if "bet" in available and score >= 56:
            spr = effective_spr(table, my_seat)
            amount = value_bet(pot, allowed, spr=spr)
            return "bet", amount, f"Open betting score {score}"
        if "check" in available:
            return "check", None, f"Checking score {score}"

    made_rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards) if board_cards else {}
    strong_threshold = 2 if opponents <= 2 else 3
    strong = made_rank >= strong_threshold
    medium = made_rank == 1 or top_pair
    no_made = made_rank == 0

    if "raise" in available and strong:
        amount = raise_for_value(table, allowed, strong=made_rank >= 3, my_seat=my_seat)
        return "raise", amount, f"value raise rank {made_rank}"

    if "bet" in available and "check" in available and strong:
        spr = effective_spr(table, my_seat)
        amount = value_bet(pot, allowed, strong=made_rank >= 3, spr=spr)
        return "bet", amount, f"Value bet rank {made_rank}"

    if "call" in available:
        if strong:
            return "call", call_amount, f"Calling with value rank {made_rank}"
        if opponents <= 1 and required <= 0.25 and medium:
            return "call", call_amount, "Defending medium hand heads-up"
        if required <= 0.10:
            return "call", call_amount, "Good price for hand"
        if "check" in available:
            return "check", None, "Checking marginal hand"
        return "fold", None, f"Declining bad price, folding rank {made_rank}"

    if "bet" in available:
        spr = effective_spr(table, my_seat)
        if strong:
            amount = value_bet(pot, allowed, strong=made_rank >= 3, spr=spr)
            return "bet", amount, f"Value betting rank {made_rank}"
        if medium and opponents <= 1 and anti_threshold_should_cbet(texture, opponents):
            amount = value_bet(pot, allowed, spr=spr)
            return "bet", amount, "Thin value / c-bet"
        if (
            no_made
            and opponents <= 1
            and anti_threshold_should_cbet(texture, opponents)
        ):
            amount = small_pressure_bet(pot, allowed, spr=spr)
            return "bet", amount, "C-bet semi-bluff"
        if (
            medium
            and opponents <= 2
            and not (texture.get("paired", False) and not top_pair)
        ):
            amount = value_bet(pot, allowed, spr=spr)
            return "bet", amount, "Multiway thin bet"

    if "check" in available:
        return "check", None, "No edge, checking"
    if "fold" in available:
        return "fold", None, "No profitable action"
    return None, None, "No supported legal action"


# ── survival_sixmax (renamed choose_action + unique helpers) ──────────────────


# Note: many helpers (position_label, hand_class, active_seat_numbers,
# call_amount, min_raise_to, max_commit, min_bet, raise_to_amount, bet_amount,
# stack_total, pressure_seats, aggressive_profile_ids, bully_context) already
# exist locally in flattened_v5 with identical bodies; we reuse them.

_SIXMAX_OPEN_RANGES = {
    "UTG": {"AA", "KK", "QQ", "JJ", "TT", "99", "AKs", "AKo", "AQs", "AJs", "KQs"},
    "MP": {
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
        "AQo",
        "AJs",
        "ATs",
        "KQs",
        "KJs",
        "QJs",
    },
    "CO": {
        "AA",
        "KK",
        "QQ",
        "JJ",
        "TT",
        "99",
        "88",
        "77",
        "66",
        "AKs",
        "AKo",
        "AQs",
        "AQo",
        "AJs",
        "AJo",
        "ATs",
        "A9s",
        "KQs",
        "KQo",
        "KJs",
        "KTs",
        "QJs",
        "QTs",
        "JTs",
        "T9s",
    },
    "BTN": {
        "AA",
        "KK",
        "QQ",
        "JJ",
        "TT",
        "99",
        "88",
        "77",
        "66",
        "55",
        "44",
        "33",
        "22",
        "AKs",
        "AKo",
        "AQs",
        "AQo",
        "AJs",
        "AJo",
        "ATs",
        "ATo",
        "A9s",
        "A8s",
        "A7s",
        "A6s",
        "A5s",
        "A4s",
        "A3s",
        "A2s",
        "KQs",
        "KQo",
        "KJs",
        "KJo",
        "KTs",
        "K9s",
        "QJs",
        "QTs",
        "Q9s",
        "JTs",
        "J9s",
        "T9s",
        "98s",
        "87s",
        "76s",
    },
    "SB": {
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
        "AQo",
        "AJs",
        "ATs",
        "KQs",
        "KJs",
        "QJs",
    },
    "BB": {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs", "AQo", "AJs"},
}

_SIXMAX_DEFEND_RANGES = {
    "UTG": {"AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs"},
    "MP": {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs", "AQo", "AJs", "KQs"},
    "CO": _SIXMAX_OPEN_RANGES["MP"] | {"77", "ATs", "KTs", "QTs", "JTs"},
    "BTN": _SIXMAX_OPEN_RANGES["CO"] | {"55", "44", "33", "22", "A5s", "A4s", "K9s"},
    "SB": {"AA", "KK", "QQ", "JJ", "TT", "99", "AKs", "AKo", "AQs", "AJs", "KQs"},
    "BB": _SIXMAX_OPEN_RANGES["BTN"]
    | {"K8s", "Q8s", "J8s", "T8s", "97s", "86s", "75s"},
}

_SIXMAX_PREMIUMS = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
_SIXMAX_AGGRESSIVE_LABELS = {"bluffer", "loose_aggressive"}

# v6: profile-gated SB/BB blind defend widening. The first deterministic
# widening attempt (see PLAN_PREFLOP_PATCH_EV_LEAK.md) regressed vs tight
# heuristic baselines because we don't realise raw equity OOP. These sets
# are only consulted when _raiser_invites_wide_defense() says the opener
# is profiled as wide; otherwise preflop_positional_defense keeps the
# conservative behaviour and the sets are inert.
_PREFLOP_SB_FLAT_CALL = {
    "22",
    "33",
    "44",
    "55",
    "66",
    "77",  # set mining (>=30 BB effective)
    "65s",
    "76s",
    "87s",
    "98s",
    "T9s",  # low suited connectors
    "A2s",
    "A3s",
    "A4s",
    "A5s",  # suited wheel aces
}
_PREFLOP_BB_MIX_DEFEND = frozenset({"65s"})
# Minimum hands before the raiser's profile is trusted enough to widen
# our blind defend. 8 hands is enough to clear the 5-hand "unknown"
# label gate in OpponentProfile.label() with a small buffer.
_WIDE_DEFENSE_MIN_HANDS = 8
# fold_to_bet >= this signals a wide range (the opener folds a lot postflop
# because they have nothing), even when their label is balanced.
_WIDE_DEFENSE_FOLD_TO_BET = 0.55


def _is_late_position_sixmax(position):
    return position in {"CO", "BTN"}


def _sixmax_guarded_baseline(table, my_seat, max_call_fraction=0.10):
    action, amount, message = simple_choose_action(table, my_seat)
    available = table.get("allowedActions", {}).get("availableActions", [])
    if action not in available:
        return None
    if action == "check":
        return action, amount, f"Baseline free option: {message}"
    if action == "call":
        stack = int(my_seat.get("stackChips") or 0)
        price = call_amount(table.get("allowedActions", {}))
        if price <= max(BIG_BLIND, int(stack * max_call_fraction)):
            return action, amount, f"Baseline cheap continue: {message}"
    return None


def _is_tag_profile(profile):
    """Detect tight-aggressive profiles from VPIP/PFR ratios."""
    hands_seen = max(int(profile.hands_seen or 0), 1)
    vpip = profile.vpip / hands_seen
    pfr = profile.pfr / hands_seen
    label = profile.label().lower()
    return label in {"tight", "patient_methodical"} or (
        vpip <= 0.20 and pfr >= 0.10 and pfr / max(vpip, 0.01) >= 0.60
    )


def _has_tag_opponent(table, my_seat):
    profiles = profiled_table_profiles(table, my_seat)
    return any(_is_tag_profile(profile) for profile in profiles)


def _sixmax_anti_bully_action(table, my_seat):
    ctx = bully_context(table, my_seat)
    if ctx is None:
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    street = table.get("street", "Preflop")
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = int(table.get("potChips") or 0)
    price = call_amount(allowed)
    required = pot_odds(price, pot)
    score = preflop_score(hole_cards)
    made_rank = made_hand_rank(hole_cards, board_cards) if board_cards else 0
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    medium = made_rank in {1, 2} or top_pair
    weak = made_rank == 0
    strong = made_rank >= 3
    hand = hand_class(hole_cards)
    stack = int(my_seat.get("stackChips") or 0)
    stack_price = price / max(stack, 1)
    has_tag = _has_tag_opponent(table, my_seat)

    if has_tag and weak and "fold" in available:
        return "fold", None, "TAG opponent: folding weak hand"

    if has_tag and medium and stack_price > 0.50 and "fold" in available:
        return "fold", None, "TAG opponent: folding medium hand at high stack price"

    if street == "Preflop":
        if "raise" in available and (hand in _SIXMAX_PREMIUMS or score >= 88):
            target = max(BIG_BLIND * 5, int(pot * 1.25))
            return (
                "raise",
                raise_to_amount(table, allowed, target),
                f"anti-bully value backraise {hand}",
            )
        if (
            "call" in available
            and (score >= 58 or hand in _SIXMAX_DEFEND_RANGES["BB"])
            and (required <= 0.24 or ctx["known_aggressive"])
        ):
            return "call", price, f"anti-bully preflop defend {hand}"
        return None

    if "raise" in available and strong:
        current_bet = int(table.get("currentBet") or 0)
        target = current_bet + max(BIG_BLIND * 3, int(max(pot, BIG_BLIND) * 0.65))
        return (
            "raise",
            raise_to_amount(table, allowed, target),
            f"anti-bully value raise rank {made_rank}",
        )
    if "call" in available:
        if strong and required <= 0.45:
            return "call", price, f"anti-bully continue rank {made_rank}"
        if medium and required <= (0.32 if ctx["known_aggressive"] else 0.24):
            return "call", price, "anti-bully bluff catch"
        if required <= 0.10:
            return "call", price, "anti-bully tiny price"
    return None


def survival_sixmax_choose_action(table, my_seat):
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if not available:
        return None, None, "No legal actions available"
    if not my_seat:
        if "fold" in available:
            return "fold", None, "Fallback: seat not found"
        return None, None, "No matching seat found"

    street = table.get("street", "Preflop")
    bully_action = _sixmax_anti_bully_action(table, my_seat)
    if bully_action is not None:
        return bully_action

    if street == "Preflop":
        action, amount, message = profiled_choose_action(table, my_seat)
        return action, amount, f"survival preflop: {message}"

    if "bet" in available or "raise" in available:
        action, amount, message = adaptive_choose_action(table, my_seat)
        return action, amount, f"survival value pressure: {message}"

    action, amount, message = profiled_choose_action(table, my_seat)
    return action, amount, f"survival defense: {message}"

    # The block below is dead code in the original (unreachable after return);
    # preserved verbatim for behavioral parity if execution ever falls through.
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = int(table.get("potChips") or 0)
    current_bet = int(table.get("currentBet") or 0)
    stack = int(my_seat.get("stackChips") or 0)
    stack_bb = stack / BIG_BLIND if BIG_BLIND else 0
    required = pot_odds(call_amount(allowed), pot)
    opponents = active_opponents(table, my_seat)
    position = position_label(table, my_seat)
    hand = hand_class(hole_cards)

    baseline_action, baseline_amount, baseline_message = simple_choose_action(
        table, my_seat
    )
    if baseline_action in available:
        if (
            street == "Preflop"
            and baseline_action == "raise"
            and hand in _SIXMAX_PREMIUMS
        ):
            return baseline_action, baseline_amount, f"Premium {hand} from {position}"
        if street == "Preflop" and baseline_action == "fold":
            return (
                baseline_action,
                baseline_amount,
                (f"Preserving stack with {hand} from {position}"),
            )
        if street != "Preflop" and baseline_action == "fold":
            return (
                baseline_action,
                baseline_amount,
                (f"Avoiding chip leak rank {made_hand_rank(hole_cards, board_cards)}"),
            )
        return (
            baseline_action,
            baseline_amount,
            f"6-max chip baseline {position}: {baseline_message}",
        )

    if street == "Preflop":
        unopened = current_bet <= BIG_BLIND
        open_range = _SIXMAX_OPEN_RANGES.get(position, _SIXMAX_OPEN_RANGES["MP"])
        defend_range = _SIXMAX_DEFEND_RANGES.get(position, _SIXMAX_DEFEND_RANGES["MP"])

        if "raise" in available and hand in _SIXMAX_PREMIUMS:
            target = BIG_BLIND * (3.2 if opponents >= 4 else 2.6)
            all_in = stack_bb <= 10
            return (
                "raise",
                raise_to_amount(table, allowed, target, all_in=all_in),
                f"Premium {hand} from {position}, building chip EV",
            )

        if unopened and "raise" in available and hand in open_range:
            target = BIG_BLIND * (2.5 if _is_late_position_sixmax(position) else 3.0)
            return (
                "raise",
                raise_to_amount(table, allowed, target),
                f"Opening {hand} from {position}",
            )

        if "call" in available:
            cheap = required <= (0.08 if opponents >= 3 else 0.12)
            stack_safe = call_amount(allowed) <= max(BIG_BLIND, int(stack * 0.08))
            cheap_unopened = unopened and call_amount(allowed) <= BIG_BLIND
            score = preflop_score(hole_cards)
            if cheap_unopened and stack_safe and score >= 35:
                return "call", call_amount(allowed), f"Cheap unopened continue {hand}"
            if hand in defend_range and (cheap or stack_safe):
                return "call", call_amount(allowed), f"Controlled defend {hand}"
            if "check" in available:
                return "check", None, f"Free option with {hand}"
            baseline = _sixmax_guarded_baseline(table, my_seat, max_call_fraction=0.08)
            if baseline is not None:
                return baseline
            return "fold", None, f"Preserving stack with {hand} from {position}"

        if "check" in available:
            return "check", None, f"Checking {hand} from {position}"

    made_rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards) if board_cards else {"wet": False}
    strong_threshold = 2
    strong = made_rank >= strong_threshold
    monster = made_rank >= 4
    medium = made_rank == 1 or top_pair
    dry = not texture.get("wet", False)

    if "raise" in available and (strong or monster):
        fraction = 0.85 if monster else 0.55
        target = current_bet + max(BIG_BLIND, int(max(pot, BIG_BLIND) * fraction))
        return (
            "raise",
            raise_to_amount(table, allowed, target, all_in=monster and stack_bb <= 18),
            f"Value raise rank {made_rank}",
        )

    if "call" in available:
        price = call_amount(allowed)
        if strong and required <= 0.42:
            return "call", price, f"Continue strong rank {made_rank}"
        if medium and opponents <= 2 and required <= 0.24 and price <= stack * 0.18:
            return "call", price, "Heads-up medium hand at fair price"
        if medium and opponents <= 4 and required <= 0.18 and price <= stack * 0.14:
            return "call", price, "Small multiway price for medium hand"
        if top_pair and required <= 0.16 and price <= stack * 0.12:
            return "call", price, "Top-pair price is acceptable"
        if required <= 0.06 and price <= stack * 0.06:
            return "call", price, "Tiny price, preserving optionality"
        if "check" in available:
            return "check", None, "Free pot control"
        baseline = _sixmax_guarded_baseline(table, my_seat, max_call_fraction=0.12)
        if baseline is not None:
            return baseline
        return "fold", None, f"Avoiding chip leak rank {made_rank}"

    if "bet" in available:
        if strong:
            return (
                "bet",
                bet_amount(table, allowed, 0.72 if monster else 0.55),
                f"Value bet rank {made_rank}",
            )
        if medium and opponents <= 2:
            return "bet", bet_amount(table, allowed, 0.38), "Thin value heads-up"
        if medium and dry:
            return "bet", bet_amount(table, allowed, 0.34), "Small dry-board value"
        if opponents <= 1 and dry and _is_late_position_sixmax(position):
            return "bet", bet_amount(table, allowed, 0.25), "Low-risk late c-bet"

    if "check" in available:
        return "check", None, "Pot control, survival-first"
    if "fold" in available:
        return "fold", None, "No chip-positive path"
    return None, None, "No supported legal action"


# ── survival_lookup helpers + choose_action (renamed) ────────────────────────


_LOOKUP_AGGRESSIVE_LABELS = {"bluffer", "loose_aggressive", "tight_aggressive"}
_LOOKUP_LOOSE_LABELS = {"bluffer", "loose_aggressive", "calling_station"}
_LOOKUP_TIGHT_LABELS = {"patient_methodical", "tight_aggressive"}

_LOOKUP_STYLE_POLICY = {
    "unknown": {
        "preflop": "profiled",
        "pressure": "anti_threshold",
        "defense": "profiled",
        "open_adjust": 0,
    },
    "short_handed": {
        "preflop": "profiled",
        "pressure": "anti_threshold",
        "defense": "profiled",
        "open_adjust": -4,
    },
    "loose_aggressive": {
        "preflop": "profiled",
        "pressure": "anti_threshold",
        "defense": "profiled",
        "open_adjust": 4,
    },
    "loose_passive": {
        "preflop": "profiled",
        "pressure": "anti_threshold",
        "defense": "profiled",
        "open_adjust": -2,
    },
    "tight": {
        "preflop": "profiled",
        "pressure": "anti_threshold",
        "defense": "profiled",
        "open_adjust": -6,
    },
}

_LOOKUP_PREFLOP = {
    "tight": {"steal_scores": {"BTN": 42, "CO": 48, "SB": 52}, "value_3bet_score": 82},
    "loose_aggressive": {
        "steal_scores": {"BTN": 54, "CO": 60, "SB": 64},
        "value_3bet_score": 76,
    },
    "loose_passive": {
        "steal_scores": {"BTN": 48, "CO": 54, "SB": 58},
        "value_3bet_score": 80,
    },
    "unknown": {
        "steal_scores": {"BTN": 50, "CO": 56, "SB": 60},
        "value_3bet_score": 80,
    },
    "short_handed": {
        "steal_scores": {"BTN": 38, "CO": 44, "SB": 48},
        "value_3bet_score": 74,
    },
}

_LOOKUP_POSTFLOP = {
    ("loose_aggressive", "medium", "call"): {
        "max_pot_odds": 0.30,
        "policy": "bluff_catch",
    },
    ("loose_passive", "strong", "bet"): {"size": "large_value", "policy": "thin_value"},
    ("tight", "air", "bet"): {"size": "small_pressure", "policy": "steal_dry_boards"},
    ("unknown", "strong", "bet"): {"size": "value", "policy": "default_value"},
}


def lookup_active_players(table):
    seats = table.get("seats", [])
    return sum(
        1
        for seat in seats
        if not seat.get("folded", False) and not seat.get("hasFolded", False)
    )


def lookup_table_style(table, my_seat):
    player_count = lookup_active_players(table)
    if player_count <= 3:
        return "short_handed"

    labels = [profile.label() for profile in profiled_table_profiles(table, my_seat)]
    if not labels:
        return "unknown"

    aggressive = sum(label in _LOOKUP_AGGRESSIVE_LABELS for label in labels)
    loose = sum(label in _LOOKUP_LOOSE_LABELS for label in labels)
    tight = sum(label in _LOOKUP_TIGHT_LABELS for label in labels)
    stations = labels.count("calling_station")

    if aggressive >= max(1, len(labels) // 2):
        return "loose_aggressive"
    if stations >= max(1, len(labels) // 2) or loose >= max(2, len(labels) // 2):
        return "loose_passive"
    if tight >= max(2, len(labels) // 2):
        return "tight"
    return "unknown"


def lookup_choose_policy(street, available, style):
    policy = _LOOKUP_STYLE_POLICY.get(style, _LOOKUP_STYLE_POLICY["unknown"])
    if street == "Preflop":
        return policy["preflop"]
    if "call" in available:
        return policy["defense"]
    if "bet" in available or "raise" in available:
        return policy["pressure"]
    return policy["defense"]


def lookup_hand_bucket(hole_cards, board_cards):
    if not board_cards:
        score = preflop_score(hole_cards)
        if score >= 80:
            return "strong"
        if score >= 50:
            return "medium"
        return "air"

    rank = made_hand_rank(hole_cards, board_cards)
    if rank >= 2:
        return "strong"
    if rank == 1 or has_top_pair_or_better(hole_cards, board_cards):
        return "medium"
    return "air"


def lookup_dispatch(policy_name, table, my_seat):
    if policy_name == "anti_threshold":
        return anti_threshold_choose_action(table, my_seat)
    if policy_name == "profiled":
        return profiled_choose_action(table, my_seat)
    return survival_sixmax_choose_action(table, my_seat)


def lookup_style_adjustment(table, my_seat, style, hint):
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    street = table.get("street", "Preflop")
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = int(table.get("potChips") or 0)
    call_amount = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    required = pot_odds(call_amount, pot)

    if street == "Preflop" and "raise" in available:
        score = preflop_score(hole_cards)
        threshold = int(hint.get("value_3bet_score", 80))
        if score >= threshold and style == "loose_aggressive":
            amount = allowed.get("minRaiseTo") or (allowed.get("raiseRange") or {}).get(
                "min"
            )
            return "raise", amount, f"lookup value pressure score {score}"

    if street != "Preflop":
        bucket = lookup_hand_bucket(hole_cards, board_cards)
        rank = made_hand_rank(hole_cards, board_cards)
        texture = board_texture(board_cards) if board_cards else {"wet": False}
        if "call" in available and style == "short_handed":
            if rank >= 2 and required <= 0.32:
                return "call", call_amount, "lookup short-handed pot-control"
            if bucket == "medium" and required <= 0.26:
                return "call", call_amount, "lookup short-handed medium defense"
        if (
            "call" in available
            and style == "loose_aggressive"
            and bucket == "medium"
            and required <= float(hint.get("max_pot_odds", 0.0))
        ):
            return "call", call_amount, "lookup bluff catch loose table"
        if (
            "bet" in available
            and style == "tight"
            and bucket == "air"
            and not texture.get("wet", False)
        ):
            amount = max(
                allowed.get("minBet", BIG_BLIND),
                int(max(pot, BIG_BLIND) * 0.3),
            )
            return "bet", amount, "lookup dry-board steal"
    return None


def lookup_hint(table, my_seat, style):
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    street = table.get("street", "Preflop")
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])

    if street == "Preflop":
        return _LOOKUP_PREFLOP.get(style, _LOOKUP_PREFLOP["unknown"])

    bucket = lookup_hand_bucket(hole_cards, board_cards)
    action_type = "bet" if ("bet" in available or "raise" in available) else "call"
    return _LOOKUP_POSTFLOP.get(
        (style, bucket, action_type),
        _LOOKUP_POSTFLOP.get(("unknown", bucket, action_type), {}),
    )


def survival_lookup_choose_action(table, my_seat):
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if not available:
        return None, None, "No legal actions available"
    if not my_seat:
        if "fold" in available:
            return "fold", None, "Fallback: seat not found"
        return None, None, "No matching seat found"

    style = lookup_table_style(table, my_seat)
    hint = lookup_hint(table, my_seat, style)
    override = lookup_style_adjustment(table, my_seat, style, hint)
    if override is not None:
        return override

    policy_name = lookup_choose_policy(table.get("street", "Preflop"), available, style)
    action, amount, message = lookup_dispatch(policy_name, table, my_seat)
    return action, amount, f"lookup {style}/{policy_name}: {message}"


# ── survival_lookahead (renamed choose_action + helpers) ─────────────────────


_LOOKAHEAD_STYLES = {"unknown", "short_handed", "loose_aggressive", "tight"}


def _lookahead_clamp(value, low, high):
    return max(low, min(high, value))


def _lookahead_amount_delta(action, amount, my_seat):
    if action not in {"bet", "raise"} or amount is None:
        return 0
    current = int(my_seat.get("currentBetChips") or 0)
    return max(0, int(amount) - current)


def _lookahead_legal_amount(action, amount, allowed):
    if amount is None:
        return None
    maximum = int(allowed.get("maxCommit") or amount)
    if action == "bet":
        minimum = int(allowed.get("minBet") or BIG_BLIND)
    elif action == "raise":
        minimum = int(allowed.get("minRaiseTo") or amount)
    else:
        return int(amount)
    return _lookahead_clamp(int(amount), minimum, maximum)


def _lookahead_candidate_actions(table, my_seat, blueprint):
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    pot = int(table.get("potChips") or 0)
    current_bet = int(table.get("currentBet") or 0)
    candidates = []

    def add(action, amount=None):
        if action not in available:
            return
        normalized = (action, _lookahead_legal_amount(action, amount, allowed))
        if normalized not in candidates:
            candidates.append(normalized)

    add(*blueprint)
    add("fold")
    add("check")
    if "call" in available:
        add("call", int(allowed.get("callAmount") or allowed.get("callChips") or 0))
    if "bet" in available:
        minimum = int(allowed.get("minBet") or BIG_BLIND)
        add("bet", minimum)
        add("bet", max(minimum, int(max(pot, BIG_BLIND) * 0.35)))
        add("bet", max(minimum, int(max(pot, BIG_BLIND) * 0.60)))
    if "raise" in available:
        minimum = int(allowed.get("minRaiseTo") or current_bet + BIG_BLIND)
        add("raise", minimum)
        add("raise", max(minimum, current_bet + int(max(pot, BIG_BLIND) * 0.45)))
        add("raise", max(minimum, current_bet + int(max(pot, BIG_BLIND) * 0.75)))

    return candidates


def _lookahead_hand_equity_proxy(table, my_seat):
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if not board_cards:
        return _lookahead_clamp(preflop_score(hole_cards) / 135, 0.05, 0.90)

    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards)
    if rank >= 5:
        equity = 0.93
    elif rank == 4:
        equity = 0.86
    elif rank == 3:
        equity = 0.78
    elif rank == 2:
        equity = 0.64
    elif rank == 1 or top_pair:
        equity = 0.48
    else:
        equity = 0.20

    if texture.get("wet", False) and rank < 3:
        equity -= 0.06
    if lookup_active_players(table) > 3:
        equity -= 0.05
    return _lookahead_clamp(equity, 0.04, 0.96)


def _lookahead_style_fold_equity(style, active_count, action):
    base = {
        "tight": 0.44,
        "unknown": 0.24,
        "short_handed": 0.22,
        "loose_aggressive": 0.18,
        "loose_passive": 0.10,
    }.get(style, 0.20)
    if action == "raise":
        base += 0.04
    base -= max(0, active_count - 2) * 0.05
    return _lookahead_clamp(base, 0.04, 0.52)


def _lookahead_bust_risk_penalty(commit, stack, stack_after):
    if stack <= 0 or commit <= 0:
        return 0
    commit_ratio = commit / stack
    penalty = commit_ratio * commit * 0.35
    if stack_after < BIG_BLIND * 8:
        penalty += 28
    if commit_ratio >= 0.50:
        penalty += 35
    return penalty


def _lookahead_score_candidate(table, my_seat, style, candidate, blueprint):
    action, amount = candidate
    allowed = table.get("allowedActions", {})
    pot = int(table.get("potChips") or 0)
    stack = int(my_seat.get("stackChips") or 0)
    call_amount = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    required = pot_odds(call_amount, pot)
    active = lookup_active_players(table)
    equity = _lookahead_hand_equity_proxy(table, my_seat)
    rank = made_hand_rank(my_seat.get("holeCards", []), table.get("boardCards", []))
    bucket = lookup_hand_bucket(
        my_seat.get("holeCards", []), table.get("boardCards", [])
    )

    if action == "fold":
        return -call_amount * 0.35 - equity * max(pot, BIG_BLIND) * 0.10
    if action == "check":
        return equity * pot * 0.18 + 4

    if action == "call":
        commit = call_amount
        stack_after = max(0, stack - commit)
        score = equity * (pot + commit) - (1 - equity) * commit
        if bucket == "medium" and required <= 0.26:
            score += 18
        if rank >= 2 and required <= 0.32:
            score += 26
        if covered_by_larger_stack(table, my_seat) and bucket != "air":
            score += 10
        score -= _lookahead_bust_risk_penalty(commit, stack, stack_after)
    else:
        commit = _lookahead_amount_delta(action, amount, my_seat)
        stack_after = max(0, stack - commit)
        fold_equity = _lookahead_style_fold_equity(style, active, action)
        called_value = equity * (pot + commit) - (1 - equity) * commit
        score = fold_equity * pot + (1 - fold_equity) * called_value
        if rank >= 3:
            score += 18
        elif rank == 2 and call_amount > 0:
            score -= 42
        elif bucket == "air" and style == "tight":
            score += 12
        elif bucket == "air":
            score -= 20
        score -= _lookahead_bust_risk_penalty(commit, stack, stack_after)

    if candidate == blueprint:
        score += 6
    return score


def _lookahead_should_lookahead(table, my_seat, style, blueprint):
    street = table.get("street", "Preflop")
    if street != "Flop":
        return False
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "call" not in available:
        return False
    call_amount = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    required = pot_odds(call_amount, int(table.get("potChips") or 0))
    rank = made_hand_rank(my_seat.get("holeCards", []), table.get("boardCards", []))
    bucket = lookup_hand_bucket(
        my_seat.get("holeCards", []), table.get("boardCards", [])
    )
    has_profiles = bool(table.get("opponentProfiles"))
    if not has_profiles or style != "loose_aggressive":
        return False
    if blueprint[0] == "fold":
        return rank >= 2 or (bucket == "medium" and required <= 0.24)
    if blueprint[0] == "raise":
        return rank == 2 and required <= 0.32
    return False


def _lookahead_action(table, my_seat, blueprint_action, blueprint_amount):
    blueprint = (blueprint_action, blueprint_amount)
    style = lookup_table_style(table, my_seat)
    if not _lookahead_should_lookahead(table, my_seat, style, blueprint):
        return None

    allowed = table.get("allowedActions", {})
    call_amount = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    candidates = [blueprint, ("fold", None), ("call", call_amount)]
    if not candidates:
        return None

    scored = [
        (
            _lookahead_score_candidate(table, my_seat, style, candidate, blueprint),
            candidate,
        )
        for candidate in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    blueprint_score = _lookahead_score_candidate(
        table, my_seat, style, blueprint, blueprint
    )
    if best == blueprint or best_score < blueprint_score + 5:
        return None
    action, amount = best
    return (
        action,
        amount,
        (
            f"lookahead {style}: {action}"
            f" score {best_score:.1f} over blueprint {blueprint_score:.1f}"
        ),
    )


def survival_lookahead_choose_action(table, my_seat):
    action, amount, message = survival_lookup_choose_action(table, my_seat)
    if action is None or not my_seat:
        return action, amount, message

    override = _lookahead_action(table, my_seat, action, amount)
    if override is not None:
        return override
    return action, amount, f"lookahead blueprint: {message}"


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


def preflop_spr(table, my_seat):
    """Estimate preflop SPR after calling the current raise."""
    stack = int(my_seat.get("stackChips") or 0)
    call = int(table.get("allowedActions", {}).get("callAmount") or 0)
    pot = int(table.get("potChips") or 0)
    if call <= 0 or stack <= 0:
        return 0.0
    remaining = max(0, stack - call)
    future_pot = pot + (call * 2)
    return remaining / future_pot if future_pot > 0 else 0.0


def has_tight_active_opponent(table, my_seat):
    """Return True if any active opponent is tight (VPIP <= 18%)."""
    profiles = profiled_table_profiles(table, my_seat)
    for profile in profiles:
        vpip = profile.vpip_frequency
        hands = int(profile.hands_seen or 0)
        if hands >= 10 and vpip <= SET_MINING_TIGHT_OPPONENT_VPIP:
            return True
    return False


def top_pair_good_kicker_vs_loose_bad_price(
    table, my_seat, base
) -> ActionDecision | None:
    """Continue with top pair good kicker against loose opponents at bad prices.

    Loose opponents have wider betting ranges with more bluffs and thinner value.
    Top pair with a strong kicker (K or better) should continue even when the
    immediate pot odds are slightly unfavorable.
    """
    if table.get("street", "Flop") == "Preflop":
        return None

    action, _amount, _message = base
    if action != "fold":
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "call" not in available or "fold" not in available:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if not has_top_pair_good_kicker(hole_cards, board_cards):
        return None

    profiles = profiled_table_profiles(table, my_seat)
    if not profiles or not any(
        profile.label()
        in {
            "loose_aggressive",
            "loose-passive",
            "loose-measured",
            "calling_station",
            "balanced",
        }
        for profile in profiles
    ):
        return None

    return (
        "call",
        int(allowed.get("callAmount") or 0),
        "loose opponent top-pair good kicker bluff catch",
    )


def medium_pocket_pair_vs_tight_three_bet(
    table, my_seat, base
) -> ActionDecision | None:
    """Fold medium pocket pairs (77-TT) to tight 3-bets at bad prices.

    Medium pocket pairs need good implied odds to set-mine. Against a tight
    opponent's 3-bet at a large price (>15% of stack), the range is too
    condensed toward overpairs to continue profitably.
    """
    if table.get("street", "Preflop") != "Preflop":
        return None

    action, _amount, _message = base
    if action not in {"raise", "call"}:
        return None

    hole_cards = my_seat.get("holeCards", [])
    rank = hole_pair_rank(hole_cards)
    if rank is None or rank < RANK_VALUES["7"] or rank > RANK_VALUES["T"]:
        return None

    allowed = table.get("allowedActions", {})
    call = int(allowed.get("callAmount") or 0)
    stack = int(my_seat.get("stackChips") or 0)
    if call <= 0 or stack <= 0:
        return None

    price_to_stack = call / stack
    if price_to_stack <= 0.15:
        return None

    if not has_tight_active_opponent(table, my_seat):
        return None

    return (
        "fold",
        None,
        f"tight 3-bet: fold {rank}{hole_cards[0][1]}{hole_cards[1][1]} at {price_to_stack:.1%} stack",
    )


def pocket_pair_set_mining_guard(table, my_seat, base) -> ActionDecision | None:
    """Force a call with small pocket pairs at good set-mining prices.

    Small pocket pairs (22-66) need cheap entry and deep implied odds to be +EV.
    This guard overrides the base fold when:
    - It's preflop
    - Hero has a small pocket pair (22-66)
    - The base action is fold
    - The price is <= 11.8% of stack (8:1 implied odds)
    - No active opponent is tight (VPIP <= 18%)
    - SPR >= 8.0 (deep enough to realize implied odds)

    Targeted patch: only fires on the specific scenario where the base
    strategy over-folds small pocket pairs at cheap prices. Does not
    override raises.
    """
    if table.get("street", "Preflop") != "Preflop":
        return None

    action, _amount, _message = base
    if action != "fold":
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "call" not in available:
        return None

    hole_cards = my_seat.get("holeCards", [])
    rank = hole_pair_rank(hole_cards)
    if rank is None or rank > RANK_VALUES["6"]:
        return None

    call = int(allowed.get("callAmount") or 0)
    stack = int(my_seat.get("stackChips") or 0)
    if call <= 0 or stack <= 0:
        return None

    price_to_stack = call / stack
    if price_to_stack > SET_MINING_MAX_PRICE:
        return None

    if has_tight_active_opponent(table, my_seat):
        return None

    spr = preflop_spr(table, my_seat)
    if spr < SET_MINING_MIN_SPR:
        return None

    return (
        "call",
        call,
        f"set-mining guard: {rank}{hole_cards[0][1]}{hole_cards[1][1]} at {price_to_stack:.1%} stack, SPR {spr:.1f}",
    )


def small_pair_multiway_fold_guard(table, my_seat, base) -> ActionDecision | None:
    """Fold small pocket pairs (22-66) when FACING a raise in 3+ player pots.

    Evidence (selfplay_s2vbase_pair_board.sqlite, 50k hands):
    - 22-77 lose -1000 chips avg with 0 wins in 3+ player pots
    - SPR drops below 8.0 in multi-way, making set-mining unprofitable
    - More players = more likely someone has a higher set/flush

    Conditions (all must be true):
    - Preflop
    - Hero is FACING a raise (not opening) — check raise_seat != hero_seat
    - Hero has small pocket pair (22-66)
    - active_players >= 3 (multi-way pot)
    - Price > 5% of stack (multi-way requires very cheap entry)

    Why this scope: Opening 55 from MP is +EV when it takes down the
    blinds. The leak is CALLING a raise in multi-way, not opening.
    """
    if table.get("street", "Preflop") != "Preflop":
        return None

    # Only fire when hero is FACING a raise (not opening).
    raise_seat = table.get("raiseSeatNumber")
    hero_seat_num = my_seat.get("seatNumber")
    if raise_seat is None or raise_seat == hero_seat_num:
        return None

    action, _amount, _message = base
    if action not in ("fold", "raise", "call"):
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "fold" not in available:
        return None

    active = lookup_active_players(table)
    if active < SMALL_PAIR_MULTIWAY_MIN_PLAYERS:
        return None

    hole_cards = my_seat.get("holeCards", [])
    rank = hole_pair_rank(hole_cards)
    if rank is None or rank > RANK_VALUES["7"]:
        return None

    call = int(allowed.get("callAmount") or 0)
    stack = int(my_seat.get("stackChips") or 0)
    if call <= 0 or stack <= 0:
        return None

    price_to_stack = call / stack
    if price_to_stack <= SMALL_PAIR_MULTIWAY_MAX_PRICE:
        return None

    return (
        "fold",
        None,
        f"small-pair multiway guard: {rank}{hole_cards[0][1]}{hole_cards[1][1]} "
        f"folded at {price_to_stack:.1%} stack in {active}-way pot",
    )


def medium_pair_paired_board_fold_guard(table, my_seat, base) -> ActionDecision | None:
    """Fold 77 on paired boards — the hand is crushed against any reasonable range.

    Evidence: selfplay_s2vbase_pair_board.sqlite shows 77 on paired boards
    loses massively. Against a 3-bettor's range on a paired board, 77 has
    ~5-10% equity. The bot was calling/raising instead of folding.

    Conditions (all must be true):
    - Postflop
    - Hero has exactly 77 (rank == 7)
    - Board is paired
    - There is a bet to face (price > 0)

    Targeted: only fires for exactly 77 on paired boards. Does not affect
    88, 99, TT (which have more showdown value) or 22-66 (handled by
    other guards).
    """
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    if action not in ("call", "raise"):
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "fold" not in available:
        return None

    hole_cards = my_seat.get("holeCards", [])
    rank = hole_pair_rank(hole_cards)
    if rank != RANK_VALUES["7"]:
        return None

    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None
    if not board_has_pair(board_cards):
        return None

    price = call_amount(allowed)
    if price <= 0:
        return None

    return (
        "fold",
        None,
        "77 on paired board: crushed against any reasonable range",
    )


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

    cap = 0.35
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
    if private_made_hand_rank(hole_cards, board_cards) != 1:
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


def medium_hand_multiway_fold_guard(table, my_seat, base) -> ActionDecision | None:
    """Fold medium-strength hands postflop in 3+ player pots at bad prices.

    Evidence (selfplay_s2vbase_pair_board.sqlite, 50k hands):
    - "medium" hand bucket calling postflop loses -22 chips avg in 3-way pots
    - "medium" hand bucket calling postflop loses -41 chips avg in 4-way pots
    - Top pair / one pair value drops significantly in multi-way

    Conditions (all must be true):
    - Postflop
    - Hero has a medium hand (one pair, not two pair+)
    - active_players >= 3 (multi-way pot)
    - Required pot odds > 20% (tighter than 35% single-way)

    Why this scope: In heads-up, top pair with good kicker is a clear
    call at 2:1. In multi-way, the same hand has only ~30% equity vs
    two ranges, and reverse implied odds from draws make it worse.
    """
    if table.get("street", "Preflop") != "Preflop" and table.get("street") is None:
        return None
    street = table.get("street", "Preflop")
    if street == "Preflop":
        return None

    action, _amount, _message = base
    if action not in ("call", "raise"):
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "fold" not in available:
        return None

    active = lookup_active_players(table)
    if active < MEDIUM_HAND_MULTIWAY_MIN_PLAYERS:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None

    # Only trigger on medium hands (one pair). Two pair+ is value territory.
    rank = private_made_hand_rank(hole_cards, board_cards)
    if rank != 1:
        return None

    # Overpairs (AA/KJ/QQ) retain high equity in multi-way pots even at
    # 30-40% pot odds. Don't fold them to a single barrel — call instead.
    if has_overpair_to_board(hole_cards, board_cards):
        return None

    price = call_amount(allowed)
    if price <= 0:
        return None

    pot = int(table.get("potChips") or 0)
    required = pot_odds(price, pot)
    if required <= MEDIUM_HAND_MULTIWAY_MAX_PRICE:
        return None

    return (
        "fold",
        None,
        f"medium-hand multiway guard: {rank}-pair folded at {required:.0%} price in {active}-way pot",
    )


def is_board_made_or_kicker_vulnerable(hole_cards, board_cards) -> bool:
    """Detect hands where hero has no real private edge.

    True when:
    - Pure board-made: the best 5-card hand equals the board's hand
      (every seat shares that five-card strength), or
    - Board trips + kicker only: the board has trips, hero doesn't hold
      the trip rank, and the only private improvement is the kicker
      (e.g. Qh Kd on 33385 — beats players playing the board, but loses
      to anyone with a K and to anyone with a pocket pair / full house).

    In both cases hero shares the hand's strength with everyone at the
    table; value-raising or stacking off is -EV.
    """
    if len(board_cards) != 5:
        return False
    if evaluate_hand(list(hole_cards) + list(board_cards)) == evaluate_hand(
        board_cards
    ):
        return True
    return board_trips_with_kicker_only(hole_cards, board_cards)


def sliver_shove_guard(table, my_seat, base) -> ActionDecision | None:
    """Override rank-based folds when the call is priced in as a sliver.

    When ``call / (pot + call) <= SLIVER_SHOVE_POT_ODDS_FLOOR`` (currently
    0.10), any two cards have >10% equity vs any plausible range — so the
    ``Profiled fold rank 0`` path leaks EV. This guard fires only when the
    base action is ``"fold"``: it never converts a call/raise into anything
    else, so it can only make existing decisions more aggressive on the
    cheap-call axis.

    Currently restricted to the river (no future streets → equity realization
    ≈ 100%). To extend to turn/flop, see the comment on
    ``SLIVER_SHOVE_POT_ODDS_FLOOR``.
    """
    if table.get("street", "Preflop") != "River":
        return None

    action, _amount, _message = base
    if action != "fold":
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "call" not in available:
        return None

    price = call_amount(allowed)
    if price <= 0:
        return None

    pot = int(table.get("potChips") or 0)
    required = pot_odds(price, max(pot, 1))
    if required <= SLIVER_SHOVE_POT_ODDS_FLOOR:
        return (
            "call",
            price,
            f"sliver-shove floor: {required:.1%} pot odds",
        )
    return None


def board_made_hand_guard(table, my_seat, base) -> ActionDecision | None:
    """Avoid value-raising or stacking off when hero has no real private edge.

    Fires on both pure board-made hands (trips on a paired board, board
    flush, board straight, board full house) and board-trips-with-kicker-only
    hands (Qh Kd on 33385). Together these cover the Fielding spec in
    ``test_from_fielding.py`` — the "playing the board" leak where every
    seat shares the same five-card strength.

    Behavior matches the legacy ``board_made_air_guard`` and
    ``vulnerable_board_trips_guard`` exactly:
    - raise → fold (no private value to raise for)
    - bet → check (check back the shared strength)
    - call at ≥33% pot odds → fold (don't stack off air)
    """
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    if action == "fold":
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if not is_board_made_or_kicker_vulnerable(hole_cards, board_cards):
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    price = call_amount(allowed)
    pot = int(table.get("potChips") or 0)

    if action == "raise" and "fold" in available:
        return ("fold", None, "board-made hand: no private value to raise")

    if action == "bet" and "check" in available:
        return ("check", None, "board-made hand: check back shared strength")

    if action == "call" and price > 0:
        required = pot_odds(price, max(pot, 1))
        if required >= 0.33 and "fold" in available:
            return (
                "fold",
                None,
                f"board-made hand: folded large bet at {required:.0%} price",
            )

    return None


def flush_ranks(hole_cards, board_cards):
    """Return sorted flush ranks for hero if hero has a flush, else None."""
    cards = list(hole_cards) + list(board_cards)
    by_suit = {}
    for card in cards:
        if len(card) < 2:
            continue
        by_suit.setdefault(card[1].lower(), []).append(card)

    best_suit = max(by_suit, key=lambda suit: len(by_suit[suit]), default=None)
    if best_suit is None or len(by_suit[best_suit]) < 5:
        return None
    return sorted(
        (card_values([card])[0] for card in by_suit[best_suit]),
        reverse=True,
    )


def vulnerable_non_nut_flush_on_paired_board(hole_cards, board_cards) -> bool:
    """Detect a vulnerable non-nut flush on a paired board, where full houses are possible.

    K-high flush and better are strong enough to play aggressively. The guard
    only applies to Q-high flush or worse, where the hand is more vulnerable to
    A-high flush and full houses on paired boards.
    """
    texture = board_texture(board_cards)
    if not texture.get("paired", False):
        return False

    ranks = flush_ranks(hole_cards, board_cards)
    if ranks is None:
        return False

    # K-high flush or better plays normally. Q-high or worse is vulnerable.
    highest = max(ranks[:5])
    return highest < RANK_VALUES["K"]


def vulnerable_flush_guard(table, my_seat, base) -> ActionDecision | None:
    """Avoid overplaying non-nut flushes on paired boards."""
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    if action == "fold":
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if not vulnerable_non_nut_flush_on_paired_board(hole_cards, board_cards):
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    price = call_amount(allowed)
    pot = int(table.get("potChips") or 0)

    if action == "raise" and "fold" in available:
        if int(table.get("facing_bet") or 0) == 1:
            return (
                "call",
                price,
                "non-nut flush on paired board: bluff catch",
            )
        return (
            "fold",
            None,
            "non-nut flush on paired board: folded value-raise",
        )

    if action == "bet" and "check" in available:
        return ("check", None, "non-nut flush on paired board: check back")

    if action == "call" and price > 0:
        required = pot_odds(price, max(pot, 1))
        if required >= 0.33 and "fold" in available:
            return (
                "fold",
                None,
                (
                    "non-nut flush on paired board: folded large bet "
                    f"at {required:.0%} price"
                ),
            )

    return None


def board_trips_with_kicker_only(hole_cards, board_cards) -> bool:
    """Detect trips made from board trips plus a hole-card kicker."""
    if len(board_cards) != 5:
        return False
    trip_ranks = [
        rank for rank, count in rank_counts(board_cards).items() if count == 3
    ]
    if len(trip_ranks) != 1:
        return False

    trip_rank = trip_ranks[0]
    if trip_rank in card_values(hole_cards):
        return False

    full_rank = evaluate_hand(list(hole_cards) + list(board_cards))
    return full_rank[0] == 3


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
        if board_dominated_two_pair(hole_cards, board_cards, 2):
            if "fold" in available:
                return (
                    "fold",
                    None,
                    "folded board-dominated two pair on paired board",
                )
            return None
        if fragile_rank_two and (
            required > PAIRED_BOARD_MIN_FOLD_PRICE or price > max(stack, 1)
        ):
            if "fold" in available:
                return (
                    "fold",
                    None,
                    f"folded fragile paired-board hand at {required:.0%} price",
                )
            return None
        if (
            fragile_rank_two
            and texture.get("high", False)
            and texture.get("paired", False)
            and required > 0.25
        ):
            if "fold" in available:
                return (
                    "fold",
                    None,
                    f"folded vulnerable two pair on A-high paired board at {required:.0%}",
                )
            return None
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


def board_assisted_two_pair_guard(table, my_seat, blueprint) -> ActionDecision | None:
    """Prevent over-valuing two pair when one pair is fully on the board.

    On a paired board, a hand classified as ``made_hand_rank == 2`` (two pair)
    can be either:
      - **Real two pair**: hero holds one card from each pair (e.g. 7d Th on
        7c 4s Tc 4d). Both pairs use a private card. Genuinely strong.
      - **Board-assisted two pair**: one pair is fully on the board and hero
        only contributes the other pair (e.g. Td 6d on 7c 4s Tc 4d). The fours
        come entirely from the board. Fragile — any opponent 4 is a full house,
        JJ+ beats us, and AT/KT/QT/JT dominate.

    This guard detects the board-assisted case and converts raises to
    check/call, and calls to fold against tight opponents.
    """
    action, _amount, _message = blueprint
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if table.get("street", "Preflop") == "Preflop":
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None
    if not board_has_pair(board_cards):
        return None

    # Evaluate the best 5-card hand to get the two pair ranks.
    all_cards = list(hole_cards) + list(board_cards)
    hand_rank = evaluate_hand(all_cards)
    if hand_rank[0] != 2:
        return None

    pair_high_rank = hand_rank[1]  # numeric rank of the higher pair
    pair_low_rank = hand_rank[2]  # numeric rank of the lower pair

    # Count how many of the two pair ranks appear in hole cards.
    hole_ranks = {card[0] for card in hole_cards}
    rank_to_value = {rank: idx for idx, rank in enumerate("23456789TJQKA", start=2)}
    hole_values = {rank_to_value.get(r, 0) for r in hole_ranks}

    high_in_hole = pair_high_rank in hole_values
    low_in_hole = pair_low_rank in hole_values

    # If both pair ranks are in the hole, it's a real two pair — don't guard.
    if high_in_hole and low_in_hole:
        return None

    # One or both pair ranks are board-only → board-assisted two pair.
    # Against a tight opponent, this hand is a bluff-catcher at best.
    profiles = table.get("opponentProfiles", {})
    is_tight = False
    for profile in profiles.values():
        if isinstance(profile, dict):
            vpip = profile.get("vpip", 0)
            hands = profile.get("hands_seen", 0)
            if hands > 0 and (vpip / hands) < 0.25:
                is_tight = True
                break

    # Convert raise/bet → check (if available) or call. Both branches
    # require is_tight — against wider-range opponents the value-bet
    # extracts chips from one-pair hands that would have folded to the
    # bet, and a check-back lets them realize equity instead.
    if action in ("raise", "bet") and is_tight:
        if "check" in available and no_one_has_bet(table, allowed):
            return "check", None, "board-assisted two pair: check back vs tight"
        if "call" in available:
            price = call_amount(allowed)
            pot = int(table.get("potChips") or 0)
            required = pot_odds(price, pot)
            if required > 0.25 and "fold" in available:
                return "fold", None, "folded board-assisted two pair vs tight"
            return "call", price, "board-assisted two pair: call vs tight"

    # Facing a bet: fold at bad prices vs tight, otherwise call.
    if action == "call" and is_tight:
        price = call_amount(allowed)
        pot = int(table.get("potChips") or 0)
        required = pot_odds(price, pot)
        if required > 0.25 and "fold" in available:
            return "fold", None, "folded board-assisted two pair vs tight bet"

    return None


def river_two_pair_raise_guard(table, my_seat, blueprint) -> ActionDecision | None:
    """Prevent value-raising with two pair on the river vs tight opponents.

    Two pair on the river looks strong but is often beaten by sets, straights,
    flushes, or better two pair. Against a tight opponent's river betting range,
    the hand is a bluff-catcher at best — it should check/call, not raise/bet.

    Fires only when:
    - River street
    - made_hand_rank == 2 (two pair)
    - Base action is raise or bet
    - Opponent profile shows VPIP% < 25% (tight)
    """
    action, _amount, _message = blueprint
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if table.get("street") != "River":
        return None
    if action not in ("raise", "bet"):
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 5:
        return None

    all_cards = list(hole_cards) + list(board_cards)
    hand_rank = evaluate_hand(all_cards)
    if hand_rank[0] != 2:
        return None

    # Check if opponent is tight (VPIP% < 25%)
    profiles = table.get("opponentProfiles", {})
    is_tight = False
    for profile in profiles.values():
        if isinstance(profile, dict):
            vpip = profile.get("vpip", 0)
            hands = profile.get("hands_seen", 0)
            if hands > 0 and (vpip / hands) < 0.25:
                is_tight = True
                break

    if not is_tight:
        return None

    # Convert raise/bet → check (if no bet to face) or call
    if action == "bet" and "check" in available:
        return "check", None, "river two pair: check back vs tight"
    if action == "raise" and "check" in available and no_one_has_bet(table, allowed):
        return "check", None, "river two pair: check back vs tight"
    if "call" in available:
        price = call_amount(allowed)
        return "call", price, "river two pair: call vs tight"

    return None


def preflop_min_raise_war_cap(table, my_seat, blueprint) -> ActionDecision | None:
    """Cap preflop raises after 3+ raise-backs to prevent min-raise wars.

    In a min-raise war, the bot and opponent keep min-raising each other
    preflop. After 3+ raise-backs, the SPR is very low (< 2.0) and further
    raises just bloated the pot without changing the outcome. The bot should
    call or check instead.

    Fires when:
    - Preflop street
    - Base action is raise or bet
    - Hero has already raised 3+ times this hand (action history)
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

    # Cap hit: convert raise to call or check
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if action == "raise" and "call" in available:
        price = call_amount(allowed)
        return "call", price, "preflop war cap: call after 3 raise-backs"
    if action == "bet" and "check" in available:
        return "check", None, "preflop war cap: check after 3 raise-backs"

    return None


def river_one_pair_over_call(table, my_seat, blueprint) -> ActionDecision | None:
    """Fold one pair on the river when facing a bet > 30% of pot.

    The bot calls river bets with one pair (medium bucket, rank 1) 24,374
    times, losing -380k chips. One pair on the river is a bluff-catcher and
    should fold to medium/large bets where it's likely behind.

    Fires when:
    - River street
    - Base action is call
    - Made hand rank is 1 (one pair)
    - Facing a bet > 30% of pot
    """
    action, _amount, _message = blueprint
    if table.get("street") != "River":
        return None
    if action != "call":
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 5:
        return None

    rank = made_hand_rank(hole_cards, board_cards)
    if rank != 1:
        return None

    allowed = table.get("allowedActions", {})
    call_amt = int(allowed.get("callAmount") or 0)
    if call_amt <= 0:
        return None

    pot = int(table.get("potChips") or 0)
    if pot <= 0:
        return None

    # Fold if bet is > 50% of pot
    if call_amt / pot > 0.50:
        return "fold", None, "river one pair: fold vs >50% pot bet"

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
    tracker_summary = tracker_pressure_summary(table, my_seat)
    return {
        "opponents": opponents,
        "profile_count": len(profiles),
        "profile_confidence": profile_confidence,
        "fold_to_bet": fold_to_bet,
        "call_frequency": call_frequency,
        "hero_strength": hero_strength,
        "opponent_strength": opponent_strength,
        "range_disadvantage": range_disadvantage,
        "tracker_strength": tracker_summary["tracker_strength"],
        "tracker_range_advantage": tracker_summary["tracker_range_advantage"],
        "tracker_bluff_frequency": tracker_summary["tracker_bluff_frequency"],
        "tracker_value_frequency": tracker_summary["tracker_value_frequency"],
        "tracker_capped_probability": tracker_summary["tracker_capped_probability"],
        "tracker_confidence": tracker_summary["tracker_confidence"],
        "tracker_samples": tracker_summary["tracker_samples"],
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
    if summary.get("tracker_confidence", 0.0) >= 0.2:
        probability += summary.get("tracker_capped_probability", 0.0) * 0.10
        probability += summary.get("tracker_bluff_frequency", 0.0) * 0.08
        probability -= summary.get("tracker_value_frequency", 0.0) * 0.05
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
    draw = has_good_draw(
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
                "v004 mixed pot control weak non-top pair wet board",
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

    rank = private_made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards)
    draw = has_good_draw(
        hole_cards,
        board_cards,
    )
    if rank == 0 and not top_pair and not draw:
        return None

    _, hole_used = private_made_hand(hole_cards, board_cards)
    if hole_used == 0 and rank == 0:
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

    rank = private_made_hand_rank(hole_cards, board_cards) if board_cards else 0
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

    amount = bet_amount_frac(table, allowed, 0.26)
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
    draw = has_good_draw(hole_cards, board_cards)
    texture = board_texture(board_cards) if board_cards else {"wet": False}

    if rank >= 2 and not fragile_rank_two(
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


def postflop_draw_continue(table, my_seat, base):
    """Continue with strong draws (FD/OESD) at non-cheap prices.

    Closes the OOP draw-call leak: `cheap_postflop_continue` only
    handles `required <= 0.12`, so any 25%+ c-bet forces the bot
    to fold the draw outright. Strong draws have 30-40% raw
    equity on the flop and 17-25% on the turn, so calls at
    non-cheap prices are +EV (and draws realise their equity
    without needing postflop play — either you hit or
    check-fold).

    Strictly additive: only converts a fold to a call when the
    draw is strong and the price is reasonable. No new raises,
    no new folds, no effect on made-hand decisions.
    """
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    if action != "fold":
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "call" not in available:
        return None
    price = call_amount(allowed)
    if price <= 0:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None

    has_fd = has_flush_draw(hole_cards, board_cards)
    has_oesd = has_open_ended_straight_draw(hole_cards, board_cards)
    if not has_fd and not has_oesd:
        return None

    street = table.get("street", "Flop")
    if street not in ("Flop", "Turn"):
        return None
    opponents = active_opponents(table, my_seat)

    # Street- and opponent-count-dependent price cap. Loose:
    # flop HU 0.30 (raw equity 30-40%, +EV with implied odds),
    # flop multiway 0.22 (variance reduces draw equity),
    # turn HU 0.20 (one card to come, raw equity 17-25%),
    # turn multiway 0.15.
    if street == "Flop":
        cap = 0.30 if opponents <= 2 else 0.22
    else:  # Turn
        cap = 0.20 if opponents <= 2 else 0.15

    pot = effective_pot(table)
    required = pot_odds(price, pot)
    if required > cap:
        return None

    # Stack guard: don't over-commit with a single-street draw.
    stack = int(my_seat.get("stackChips") or 0)
    if stack > 0 and price > stack * 0.20:
        return None

    draw_label = "+".join(
        label for label, present in (("FD", has_fd), ("OESD", has_oesd)) if present
    )
    return (
        "call",
        price,
        f"v006 draw continue {draw_label} street {street} opp {opponents} required {required:.0%} cap {cap:.0%}",
    )


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
    for low in range(2, 11):
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


def active_seat_numbers(table):
    return [
        int(seat.get("seatNumber"))
        for seat in table.get("seats", [])
        if not seat.get("folded", False)
        and not seat.get("hasFolded", False)
        and seat.get("seatNumber") is not None
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
        if seat.get("agentId") != my_id
        and not seat.get("folded", False)
        and not seat.get("hasFolded", False)
    )


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


def no_one_has_bet(table, allowed):
    return call_amount(allowed) == 0 and int(table.get("currentBet") or 0) == 0


def no_large_preflop_raise(table, allowed):
    blind = blind_size(allowed, table)
    return int(table.get("currentBet") or 0) <= blind and call_amount(allowed) <= blind


def unopened_preflop(table, allowed):
    return effective_pot(table) <= int(1.5 * blind_size(allowed, table))


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
        if seat.get("folded", False) or seat.get("hasFolded", False):
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


def profile_aggression_frequency_merged(profile):
    """Aggression frequency, falling back to the API-derived value
    when the local sample is sparse.

    The local-vs-API merge (``apply_external_stats_merge``) sets
    ``profile.api_aggr_freq`` whenever it overrode local counters
    with API data. In that case, the local
    ``aggression_frequency`` (computed from observed
    calls/bets/raises/folds) is unreliable because the sample is
    too small to be meaningful — often 0.0 with no observed
    actions. The API's aggression frequency is a better read.

    Returns:
      0.0 if profile is None
      api_aggr_freq if it was set and the merge overrode this profile
      local aggression_frequency otherwise
    """
    if profile is None:
        return 0.0
    api_freq = getattr(profile, "api_aggr_freq", None)
    api_used = getattr(profile, "api_source_used", False)
    if api_freq is not None and api_used:
        return float(api_freq)
    local = profile_value(profile, "aggression_frequency")
    return float(local) if local is not None else 0.0


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
            if not seat.get("folded", False) and not seat.get("hasFolded", False)
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
                "aggression": float(profile_aggression_frequency_merged(profile)),
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


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6a: OPPONENT-EXPLOIT CONTEXT HELPERS  (v5 new)
# ════════════════════════════════════════════════════════════════════════════════


LOOSE_LABELS = frozenset({"calling_station", "loose_aggressive"})
AGGRO_LABELS = frozenset({"bluffer", "tight_aggressive", "loose_aggressive"})


def _label_from_profile(profile):
    """Return the profile label() string, falling back to empty string."""
    if profile is None:
        return ""
    if hasattr(profile, "label"):
        try:
            return profile.label()
        except Exception:
            pass
    label = profile_value(profile, "label")
    return str(label) if label else ""


def opponent_exploit_context(table, my_seat):
    """Build a per-seat-aware exploit summary from opponent profiles.

    Fields returned:
      table_type:      "passive" / "loose" / "station" / "aggressive" / "mixed"
      passive_count:   count of 'patient_methodical' labelled active opponents
      loose_count:     count of 'calling_station' or 'loose_aggressive'
      aggro_count:     count of 'bluffer' / 'tight_aggressive' / 'loose_aggressive'
      has_big_stack_loose: bool — at least one live opponent has stack > 1.4x hero
                             AND their label is loose/station
      profile_confidence: fraction of opponents with >=15 hands seen (0-1).
                          Capped at 0.5 when any_api_source is True — the
                          API merge uses the opponent's full-competition
                          sample size for "hands seen", but we have not
                          personally observed them that much at our table.
                          Capping prevents over-exploitation of opponents
                          whose data is API-derived.
      any_api_source:  bool — True if at least one active opponent's
                       profile had its counters overridden by the API
                       stats merge (i.e. local hands_seen was below
                       LOCAL_MIN_HANDS=20 and the API had fresh data).
                       Strategy branches that gate on profile_confidence
                       >= 0.5 are deliberately not triggered when this
                       is True.
      avg_fold_to_bet:     Bayesian average fold-to-bet across active opponents
      avg_call_freq:       Bayesian average call frequency across active opponents
    """
    my_id = (my_seat or {}).get("agentId")
    hero_stack = int((my_seat or {}).get("stackChips") or 0)
    opponent_profiles_raw = table.get("opponentProfiles") or {}

    active_seats = [
        seat
        for seat in table.get("seats", [])
        if seat.get("agentId") != my_id
        and not seat.get("folded", False)
        and not seat.get("hasFolded", False)
    ]
    total_active = max(len(active_seats), 1)

    passive_count = 0
    loose_count = 0
    aggro_count = 0
    has_big_stack_loose = False
    confident_count = 0

    fold_to_bet_values = []
    call_freq_values = []
    any_api_source = False

    for seat in active_seats:
        agent_id = seat.get("agentId")
        profile = opponent_profiles_raw.get(agent_id)
        if profile is None:
            continue

        hands_seen = int(profile_value(profile, "hands_seen") or 0)
        if hands_seen >= 15:
            confident_count += 1

        # Track when the local-vs-API merge overrode this profile's
        # counters with API data. The merge sets ``api_source_used``
        # only when the local sample was below LOCAL_MIN_HANDS and
        # the API had fresh, large-sample data. When True, the
        # ``hands_seen`` we're using is the API's view, not ours.
        if getattr(profile, "api_source_used", False):
            any_api_source = True

        label = _label_from_profile(profile)

        if label == "patient_methodical":
            passive_count += 1
        elif label in LOOSE_LABELS:
            loose_count += 1
        if label in AGGRO_LABELS:
            aggro_count += 1

        # Bayesian averages
        ftb = bayesian_fold_to_bet_frequency(profile)
        cf = bayesian_call_frequency(profile)
        fold_to_bet_values.append(ftb)
        call_freq_values.append(cf)

        # Big-stack loose detection
        if label in LOOSE_LABELS:
            opp_stack = int(seat.get("stackChips") or 0)
            if hero_stack > 0 and opp_stack > hero_stack * 1.4:
                has_big_stack_loose = True

    # Table-type classification
    if len(fold_to_bet_values) < 2:
        table_type = "mixed"
    else:
        avg_fold = _average(fold_to_bet_values, FOLD_TO_BET_PRIOR)
        if passive_count >= total_active * 0.6 and avg_fold >= 0.55:
            table_type = "passive"
        elif loose_count >= total_active * 0.5:
            table_type = "station" if avg_fold < 0.40 else "loose"
        elif aggro_count >= 2:
            table_type = "aggressive"
        else:
            table_type = "mixed"

    # When the API was used to override any profile, cap
    # profile_confidence at 0.5. The strategy's exploit branches
    # gated on >= 0.4 still trigger (we have *some* signal), but
    # those gated on >= 0.5 do not — preventing over-exploitation
    # of opponents we have not personally observed enough to trust.
    raw_confidence = confident_count / total_active
    profile_confidence = min(0.5, raw_confidence) if any_api_source else raw_confidence

    return {
        "table_type": table_type,
        "passive_count": passive_count,
        "loose_count": loose_count,
        "aggro_count": aggro_count,
        "has_big_stack_loose": has_big_stack_loose,
        "profile_confidence": profile_confidence,
        "any_api_source": any_api_source,
        "avg_fold_to_bet": _average(fold_to_bet_values, FOLD_TO_BET_PRIOR),
        "avg_call_freq": _average(call_freq_values, CALL_FREQUENCY_PRIOR),
    }


def _get_raiser_profile(table, my_seat, allowed):
    """Return the OpponentProfile for the seat that made the current raise.

    Scans live opponents and returns the profile of the one whose
    currentBetChips matches the currentBet (the aggressor).
    Returns None if no matching aggressor is found.
    """
    current_bet = int(table.get("currentBet") or 0)
    if current_bet <= 0:
        return None
    blind = blind_size(allowed, table)
    # The aggressor has currentBetChips == current_bet and current_bet > blind
    if current_bet <= blind:
        return None
    my_id = (my_seat or {}).get("agentId")
    profiles = table.get("opponentProfiles") or {}
    for seat in table.get("seats", []):
        if seat.get("agentId") == my_id:
            continue
        if seat.get("folded", False) or seat.get("hasFolded", False):
            continue
        if int(seat.get("currentBetChips") or 0) == current_bet:
            return profiles.get(seat.get("agentId"))
    return None


def _opener_is_calling_station(table, my_seat, allowed):
    """True if the preflop opener (the raiser we are facing) is a calling
    station or loose-aggressive player."""
    profile = _get_raiser_profile(table, my_seat, allowed)
    if profile is None:
        return False
    label = _label_from_profile(profile)
    return label in LOOSE_LABELS


def _raiser_invites_wide_defense(raiser_profile):
    """True when the raiser's profile indicates a wide opening range.

    Used to gate the SB flat-call set and BB mix-defend in
    preflop_positional_defense. The widening only fires when the
    raiser is profiled as wide; against unprofiled or tight
    opponents the bot stays in the prior conservative behaviour.

    A profile "invites wide defense" when ANY of:
      * label is in LOOSE_LABELS (loose_aggressive or calling_station)
      * observed fold_to_bet frequency is high (signals a wide range
        regardless of label — the opener folds postflop because they
        have nothing)

    Requires at least _WIDE_DEFENSE_MIN_HANDS of data. Returns False
    if the profile is missing or below the confidence threshold.
    """
    if raiser_profile is None:
        return False
    hands_seen = int(profile_value(raiser_profile, "hands_seen") or 0)
    if hands_seen < _WIDE_DEFENSE_MIN_HANDS:
        return False
    label = _label_from_profile(raiser_profile)
    if label in LOOSE_LABELS:
        return True
    if bayesian_fold_to_bet_frequency(raiser_profile) >= _WIDE_DEFENSE_FOLD_TO_BET:
        return True
    return False


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
        and not seat.get("folded", False)
        and not seat.get("hasFolded", False)
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
    if "raise" not in available or not no_large_preflop_raise(table, allowed):
        return None

    hole_cards = my_seat.get("holeCards", [])
    hand = hand_class(hole_cards)
    score = preflop_score(hole_cards)
    opponents = active_opponents(table, my_seat)
    if opponents < 2:
        return None

    if hand in PREMIUM_OPEN_CLASSES or score >= 75:
        limpers = count_limpers(table, allowed)
        base_multiplier = 4.0 if score >= 96 else 3.0
        amount = raise_to_amount(
            table, allowed, BIG_BLIND * (base_multiplier + limpers)
        )
        return "raise", amount, f"v003 six-max premium open pressure {hand}/{score}"
    return None


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 8: EXACT POSTFLOP LOGIC FROM auto_research_v007
# (Note: SIMPLE_PRESSURE_*, PAIRED_BOARD_*, VALUE_HEAVY_* constants are
#  already defined in Section 1 and are not re-declared here.)


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


def tracker_pressure_summary(table, my_seat) -> dict:
    """Return compact Bayesian range features for live opponents."""
    allowed = table.get("allowedActions", {})
    known_cards = [
        *list(my_seat.get("holeCards", [])),
        *list(table.get("boardCards", [])),
    ]
    summaries = []
    for seat in live_opponent_seats(table, my_seat):
        agent_id = seat.get("agentId") or seat.get("seatNumber")
        if agent_id is None:
            continue
        state = _RANGE_TRACKER.update(
            str(agent_id),
            position=position_label(table, seat),
            situation=_range_situation_for_seat(table, seat, allowed),
            action=_range_action_for_seat(table, seat, allowed),
            known_cards=known_cards,
            amount=seat.get("currentBetChips"),
            pot=table.get("potChips"),
        )
        summaries.append(state_to_tracker_summary(state))
    return average_summary(summaries)


def state_to_tracker_summary(state) -> dict:
    return {
        "agent_id": state.agent_id,
        "posterior_strength": _weighted_strength_from_range(state.posterior_range),
        "prior_strength": _weighted_strength_from_range(state.prior_range),
        "range_advantage": _weighted_strength_from_range(state.posterior_range)
        - _weighted_strength_from_range(state.prior_range),
        "bluff_frequency": state.confidence * 0.15 if state.action_history else 0.0,
        "value_frequency": state.confidence * 0.20 if state.action_history else 0.0,
        "capped_probability": 0.0,
        "top_classes": state.posterior_range.top_classes(8),
        "confidence": state.confidence,
        "samples": len(state.action_history),
        "showdowns": len(state.showdown_hands),
    }


def _weighted_strength_from_range(hand_range) -> float:
    total = hand_range.total_weight()
    if total <= 0:
        return 0.0
    weighted = sum(
        weight * class_strength(combo_class(combo))
        for combo, weight in hand_range.weights.items()
        if weight > 0
    )
    return max(0.0, min(1.0, weighted / total))


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
    if len(board_cards) < 3 or private_made_hand_rank(hole_cards, board_cards) != 1:
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
            "v007 mixed range/CFR paired-board fold: "
            f"dist {decision.summary()} roll {decision.roll:.2f}, "
            f"required {required:.0%}, range edge {range_edge:+.2f}"
        ),
    )


def preflop_open_raise(table, my_seat, base) -> ActionDecision | None:
    """Proactively open-raise preflop with playable hands based on position.

    GTO 6-max open-raise frequencies (v4 tuned):
      BTN ~44% → score ≥52, CO ~30% → ≥57, HJ/MP ~20–24% → ≥62,
      UTG ~16% → ≥68, SB ~38% → ≥57. Default (unknown pos) uses MP bar.
    Sizing: 2.5x BB on BTN (IP, incentivise calls), 3x BB elsewhere.

    v5 opponent-adaptive adjustments:
      - Passive field: widen opening range by -3 score points (they fold much)
      - Loose/station field: tighten slightly (+2) but size up to 3.5x
    """
    if table.get("street", "Preflop") != "Preflop":
        return None

    action, _amount, _message = base
    if action in {"raise", "bet"}:
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "raise" not in available:
        return None

    if not unopened_preflop(table, allowed):
        return None

    hole_cards = my_seat.get("holeCards", [])
    score = preflop_score(hole_cards)
    pos = position_label(table, my_seat)

    # GTO position-aware minimum score thresholds (v4 fix: was 99 default)
    min_score = 65  # MP-equivalent fallback for any unrecognised seat label
    size_bb = 3.0
    if pos == "BTN":
        min_score = 52  # ~44% open frequency
        size_bb = 2.5  # BTN raises smaller to invite calls
    elif pos in {"CO", "SB"}:
        min_score = 57  # ~30–38% open frequency
    elif pos in {"HJ", "MP"}:
        min_score = 62  # ~20–24% open frequency
    elif pos == "UTG":
        min_score = 68  # ~16% open frequency

    # v5: opponent-adaptive range/sizing adjustments
    ctx = opponent_exploit_context(table, my_seat)
    if ctx["profile_confidence"] >= 0.4:
        if ctx["table_type"] == "passive":
            # Passive field: widen opening range by -3 score points, keep sizing
            min_score = max(48, min_score - 3)
        elif ctx["table_type"] in {"loose", "station"}:
            # Loose/station field: tighten slightly but size up to 3.5x
            min_score = min_score + 2
            size_bb = max(size_bb, 3.5)  # never below GTO base for this position

    if score >= min_score:
        amount = raise_to_amount(table, allowed, BIG_BLIND * size_bb)
        return (
            "raise",
            amount,
            f"preflop open raise: score {score} pos {pos} size {size_bb}x table {ctx['table_type']}",
        )
    return None


def count_limpers(table, allowed):
    if table.get("street", "Preflop") != "Preflop":
        return 0
    pot = effective_pot(table)
    return max(0, (pot - 3) // BIG_BLIND)


def preflop_isolation_raise(table, my_seat, base) -> ActionDecision | None:
    """Isolate limpers preflop with strong hands by raising."""
    if table.get("street", "Preflop") != "Preflop":
        return None

    action, _amount, _message = base
    if action in {"raise", "bet"}:
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "raise" not in available:
        return None

    # We only isolate if the pot has limpers (opened, but no large raise)
    if unopened_preflop(table, allowed) or not no_large_preflop_raise(table, allowed):
        return None

    hole_cards = my_seat.get("holeCards", [])
    score = preflop_score(hole_cards)
    pos = position_label(table, my_seat)

    # Graded isolation thresholds: slightly tighter than open raising
    min_score = 99
    if pos in {"BTN", "CO"}:
        min_score = 60
    elif pos in {"HJ", "MP"}:
        min_score = 65
    elif pos in {"UTG", "SB"}:
        min_score = 70

    if score >= min_score:
        limpers = count_limpers(table, allowed)
        # Raise sizing: 3x BB + 1x BB per limper. If we are in the SB, we add another 1x BB.
        multiplier = 3.0 + limpers
        if pos == "SB":
            multiplier += 1.0
        amount = raise_to_amount(table, allowed, BIG_BLIND * multiplier)
        return (
            "raise",
            amount,
            f"v003 preflop isolation raise: score {score} pos {pos} limpers {limpers}",
        )
    return None


def preflop_positional_defense(table, my_seat, base) -> ActionDecision | None:
    """Defend/call wider based on position, but ONLY when facing a raise.

    GTO discipline (v4):
    - UTG/HJ/MP should 3-bet-or-fold vs a raise; flat-calling those seats is
      a postflop EV leak. preflop_three_bet runs first and catches the 3-bet
      hands; the rest fold here.
    - BTN and BB are the primary flat-call seats (IP or closing action).
    - CO may flat occasionally (semi-IP) vs EP openers.
    - BB stack guard raised to 20% (was 15%); BB has chips already invested.

    v6 profile-aware widening (gated):
    - SB flat-call set: set-mining pairs (>=30 BB effective), low suited
      connectors, and suited wheel aces. Only fires when the raiser is
      profiled as wide (see _raiser_invites_wide_defense). Against unprofiled
      or tight opponents SB stays in the prior 3-bet-or-fold discipline.
    - BB hand-class fallback to _PREFLOP_BB_MIX_DEFEND (65s only): the
      score-only threshold (40) misses 65s (36), which is the canonical GTO
      mix-defend hand. Gated on the same profile signal so we don't
      over-defend vs tight heuristic baselines.
    """
    if table.get("street", "Preflop") != "Preflop":
        return None

    action, _amount, _message = base
    if action != "fold":
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    price = call_amount(allowed)

    if "call" not in available or price <= 0:
        return None

    # ONLY defend if facing a raise (pot is already opened). Prevents unopened limping.
    if unopened_preflop(table, allowed):
        return None

    hole_cards = my_seat.get("holeCards", [])
    score = preflop_score(hole_cards)
    hand = hand_class(hole_cards)
    pos = position_label(table, my_seat)

    # Compute the raiser's profile + the wide-defense gate once. Used by both
    # the SB flat-call path and the BB hand-class fallback.
    raiser_profile = _get_raiser_profile(table, my_seat, allowed)
    wide_defense_ok = _raiser_invites_wide_defense(raiser_profile)

    # v6: SB small flat-call set, GATED on wide_defense_ok. SB closes the
    # action with a discount; set-mining pairs and low suited connectors are
    # +EV at cheap prices ONLY when the raiser has a wide range. Against
    # unprofiled or tight opponents we keep the prior 3-bet-or-fold
    # discipline (return None below) so we don't over-defend OOP.
    if pos == "SB":
        if not wide_defense_ok:
            return None
        if hand not in _PREFLOP_SB_FLAT_CALL:
            return None
        pot = effective_pot(table)
        required = pot_odds(price, pot)
        if required > 0.40:  # SB vs 2.5-3x opens costs 30-36%, blocks 4x+ raises
            return None
        stack = int(my_seat.get("stackChips") or 0)
        if stack > 0 and price > stack * 0.15:
            return None
        # Set mining needs deep effective stacks for implied odds.
        if hand in {"22", "33", "44", "55", "66", "77"} and stack < 30 * BIG_BLIND:
            return None
        return (
            "call",
            price,
            f"v006 SB flat-call: hand {hand} score {score} required {required:.0%}",
        )

    # GTO: sandwiched OOP positions (UTG, HJ, MP) 3-bet-or-fold, never flat-call.
    if pos in {"UTG", "HJ", "MP"}:
        return None

    # Position-aware thresholds: (min_score, max_price)
    # BTN and BB are the primary flat-call seats; CO may flat vs EP/MP only.
    thresholds = {
        "BTN": (48, 0.42),
        "CO": (55, 0.40),
        "BB": (40, 0.45),
    }

    min_score, max_price = thresholds.get(pos, (65, 0.20))  # default to strict OOP

    # v5: Raiser-profile adjustment — lookup the opener's profile and adjust
    if raiser_profile is not None:
        raiser_label = _label_from_profile(raiser_profile)
        if raiser_label == "patient_methodical":
            # Tight opener: tighten our call range by +5 score points
            min_score += 5
        elif raiser_label in LOOSE_LABELS:
            # Loose opener: widen our call range by -5 score points
            min_score = max(35, min_score - 5)
        elif raiser_label == "bluffer":
            # Bluffer opener: widen significantly — their range is wide and weak
            min_score = max(30, min_score - 8)

    pot = effective_pot(table)
    required = pot_odds(price, pot)
    stack = int(my_seat.get("stackChips") or 0)

    # v6: BB hand-class fallback, GATED on wide_defense_ok. The score-based
    # threshold (40) misses 65s (36) that the GTO BB defend range covers
    # at 20-50% frequency vs a wide opener. Gated so we don't over-defend
    # vs unprofiled or tight opponents.
    if (
        pos == "BB"
        and score < min_score
        and hand in _PREFLOP_BB_MIX_DEFEND
        and wide_defense_ok
    ):
        if required > max_price:
            return None
        if stack > 0 and price > stack * 0.20:
            # Same short-stack rescue as the score-driven BB path.
            if not (price <= BIG_BLIND * 8 and score >= 45):
                return None
        return (
            "call",
            price,
            f"v006 BB hand-class defend: hand {hand} score {score} required {required:.0%}",
        )

    if score < min_score:
        return None

    if required > max_price:
        return None

    # Stack-depth guard: position-aware.
    # BB has chips invested and closes action, so allow wider calls.
    stack_guard = 0.20 if pos == "BB" else 0.15
    if stack > 0 and price > stack * stack_guard:
        # BB exception: cheap raises (≤8 BB) with any playable hand always defend
        if pos == "BB" and price <= BIG_BLIND * 8 and score >= 45:
            pass  # allow the call regardless of stack guard
        else:
            return None

    return (
        "call",
        price,
        f"v006 preflop positional defense: score {score} pos {pos} required {required:.0%} cap {max_price:.0%}",
    )


def preflop_blind_defense(table, my_seat, base) -> ActionDecision | None:
    """Compatibility name for v5's preflop defense rescue.

    v5 routes this through ``preflop_positional_defense`` so the same tests can
    exercise the current strategy without depending on the older v2 function
    name.
    """
    return preflop_positional_defense(table, my_seat, base)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 9: GTO PREFLOP 3-BET AND SQUEEZE LOGIC (v4 new)
# ════════════════════════════════════════════════════════════════════════════════

# Hands that always 3-bet for value (all positions)
_3BET_VALUE_CLASSES = {"AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs"}

# Suited bluff 3-bet hands (blocking villain value, good playability)
# Used from BTN/SB vs EP/MP openers
_3BET_BLUFF_SUITED = {"A5s", "A4s", "A3s", "KQs", "QJs"}


def _count_preflop_raises(table, allowed):
    """Estimate how many raises are already in the pot this street.

    Uses the current bet size relative to BB to distinguish:
      0  = no raise yet (pot ≤ 1 BB)
      1  = one open raise (pot between 1 BB and ~5 BB per seat)
      2+ = there has already been a 3-bet or more
    """
    blind = blind_size(allowed, table)
    current_bet = int(table.get("currentBet") or 0)
    if current_bet <= blind:
        return 0
    # A standard open-raise is 2.5–4x BB; a 3-bet is typically ≥9x BB
    if current_bet >= blind * 8:
        return 2  # 3-bet or 4-bet already in
    return 1


def _count_preflop_callers(table, allowed):
    """Count seats that have called the current bet (excluding the raiser)."""
    blind = blind_size(allowed, table)
    current_bet = int(table.get("currentBet") or 0)
    if current_bet <= blind:
        return 0
    callers = 0
    for seat in table.get("seats", []):
        seat_bet = int(seat.get("currentBetChips") or 0)
        if (
            seat_bet == current_bet
            and not seat.get("folded")
            and not seat.get("hasFolded")
        ):
            # This seat matched the current bet (i.e., called, not raised)
            callers += 1
    # Subtract 1 for the raiser themselves (who set the current bet)
    return max(0, callers - 1)


def preflop_three_bet(table, my_seat, base) -> ActionDecision | None:
    """3-bet preflop with a balanced value + bluff range vs a single open-raise.

    GTO 3-bet frequencies (v4):
      Value (all positions): AA, KK, QQ, JJ, AKs, AKo, AQs → always 3-bet.
      Bluff (BTN/SB vs EP/MP only): A5s, A4s, A3s, KQs, QJs → polar 3-bets.
      BB merged range vs BTN: score ≥ 78 (TT+, AJs+, KQs in addition to value).

    Sizing:
      In-position (BTN vs CO/earlier): 3× the open-raise size.
      Out-of-position (all other seats): 3.5–4× the open-raise size.
      Minimum sizing: 9 BB (never a min-3-bet).

    v5 opponent-adaptive:
      - Bluff 3-bets suppressed when opener is a calling station (they call).
      - Bluff 3-bets expanded vs passive tables (they fold too much).
      - Sizing scaled up vs stations (build pot).
    """
    if table.get("street", "Preflop") != "Preflop":
        return None

    action, _amount, _message = base
    # Don't override an already-aggressive action
    if action in {"raise", "bet"}:
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "raise" not in available:
        return None

    # Must be facing exactly one open-raise (no 3-bet already in)
    if _count_preflop_raises(table, allowed) != 1:
        return None

    hole_cards = my_seat.get("holeCards", [])
    hc = hand_class(hole_cards)
    score = preflop_score(hole_cards)
    pos = position_label(table, my_seat)

    should_three_bet = False
    is_value = False

    # --- Value 3-bets (every position) ---
    if hc in _3BET_VALUE_CLASSES or score >= 90:
        should_three_bet = True
        is_value = True

    # --- BB merged 3-bet vs BTN/CO opens (TT+, AJs+, KQs) ---
    if not should_three_bet and pos == "BB" and score >= 78:
        should_three_bet = True
        is_value = True

    # --- SB value + polar 3-bets vs BTN open ---
    if not should_three_bet and pos == "SB" and score >= 80:
        should_three_bet = True
        is_value = True

    # --- BTN polar 3-bets vs CO/earlier ---
    if not should_three_bet and pos == "BTN" and score >= 82:
        should_three_bet = True
        is_value = True

    # --- v5: profile-aware bluff 3-bet gating ---
    if not should_three_bet and pos in {"BTN", "SB"} and hc in _3BET_BLUFF_SUITED:
        ctx = opponent_exploit_context(table, my_seat)
        opener_is_station = _opener_is_calling_station(table, my_seat, allowed)
        if not opener_is_station and ctx["table_type"] in {"passive", "mixed"}:
            should_three_bet = True
        elif ctx["table_type"] == "passive" and ctx["avg_fold_to_bet"] >= 0.55:
            # Very passive table — bluff 3-bet even when non-BTN/SB positions
            should_three_bet = True

    if not should_three_bet:
        return None

    # --- v5: sizing adjusted for table type ---
    current_bet = int(table.get("currentBet") or 0)
    blind = blind_size(allowed, table)
    ip_positions = {"BTN"}
    multiplier = 3.0 if pos in ip_positions else 3.5
    ctx = opponent_exploit_context(table, my_seat)
    if is_value and ctx["table_type"] in {"station", "loose"}:
        # Size up for value vs stations — they will call larger anyway
        multiplier = 3.5 if pos in ip_positions else 4.0
    target = max(
        BIG_BLIND * 10 if is_value else BIG_BLIND * 9, int(current_bet * multiplier)
    )
    amount = raise_to_amount(table, allowed, target)
    if amount is None:
        return None

    label = "value" if is_value else "bluff"
    return (
        "raise",
        amount,
        f"preflop 3-bet {label}: hand {hc} score {score} pos {pos} "
        f"size {target} table {ctx['table_type']}",
    )


def preflop_squeeze(table, my_seat, base) -> ActionDecision | None:
    """Squeeze preflop when facing a raise + at least one caller.

    Callers in front have capped, speculative ranges that cannot continue
    against a big re-raise, giving this play strong fold equity even with
    semi-strong hands. GTO squeeze range is tighter than a 3-bet range
    because we need more raw equity to justify the larger sizing.

    Trigger: one raise + ≥1 caller already in the pot.
    Range:
      Always squeeze: AA, KK, QQ, JJ, AKs, AKo, AQs, AQo
      BTN/CO squeeze extension: score ≥ 80 (TT, AJs, KQs)
      Bluff squeeze (BTN/SB only, 30% frequency): A5s, A4s, KQs
    Sizing: 4× original raise + 2 BB per caller.
    """
    if table.get("street", "Preflop") != "Preflop":
        return None

    action, _amount, _message = base
    if action in {"raise", "bet"}:
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "raise" not in available:
        return None

    # Exactly one raise already in (not a 3-bet war)
    if _count_preflop_raises(table, allowed) != 1:
        return None

    # Need at least one caller already in to qualify as a squeeze
    callers = _count_preflop_callers(table, allowed)
    if callers < 1:
        return None

    hole_cards = my_seat.get("holeCards", [])
    hc = hand_class(hole_cards)
    score = preflop_score(hole_cards)
    pos = position_label(table, my_seat)

    should_squeeze = False

    # Always squeeze with premiums
    if hc in {"AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs", "AQo"} or score >= 88:
        should_squeeze = True

    # BTN/CO extension (TT, AJs, KQs)
    if not should_squeeze and pos in {"BTN", "CO"} and score >= 80:
        should_squeeze = True

    # Probabilistic bluff squeeze from BTN/SB — only if callers are NOT stations
    if not should_squeeze and pos in {"BTN", "SB"} and hc in {"A5s", "A4s", "KQs"}:
        ctx = opponent_exploit_context(table, my_seat)
        if (
            ctx["table_type"] not in {"station", "loose"}
            and ctx["avg_fold_to_bet"] >= 0.45
        ):
            chosen = choose_weighted(
                (("squeeze", 0.30), ("pass", 0.70)),
                "preflop-bluff-squeeze",
                table,
                my_seat,
                strategy="flattened_v005",  # TODO: Check if this is to log message to db
                extra=(hc, pos, callers),
            )
            if chosen == "squeeze":
                should_squeeze = True

    if not should_squeeze:
        return None

    # Sizing: 4× current raise + 2 BB per caller
    current_bet = int(table.get("currentBet") or 0)
    target = max(BIG_BLIND * 10, int(current_bet * 4) + callers * BIG_BLIND * 2)
    amount = raise_to_amount(table, allowed, target)
    if amount is None:
        return None

    return (
        "raise",
        amount,
        f"preflop squeeze: hand {hc} score {score} pos {pos} callers {callers} size {target}",
    )


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 10: BIG-STACK-LOOSE OPPONENT POSTFLOP EXPLOIT  (v5 new)
# ════════════════════════════════════════════════════════════════════════════════


def big_stack_loose_postflop_adjust(table, my_seat, base) -> ActionDecision | None:
    """When the active opponent is a large-stack loose/station player:
    - Convert river bluff bets to checks
    - Expand thin value-bet range on flop/turn (any pair = bet)
    - Do NOT c-bet with complete air
    """
    if table.get("street", "Preflop") == "Preflop":
        return None

    ctx = opponent_exploit_context(table, my_seat)
    if not ctx["has_big_stack_loose"] or ctx["profile_confidence"] < 0.35:
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    street = table.get("street", "Flop")
    rank = (
        private_made_hand_rank(hole_cards, board_cards) if len(board_cards) >= 3 else 0
    )

    # River: suppress bluff bets (air/draw) vs big-stack loose caller
    if street == "River" and action == "bet" and rank == 0:
        if "check" in available:
            return "check", None, "big-stack-loose suppress river bluff"

    # Flop/Turn: thin value bet with any pair vs stations
    if street in {"Flop", "Turn"} and action == "check" and rank >= 1:
        if "bet" in available and no_one_has_bet(table, allowed):
            pot = int(table.get("potChips") or 0)
            bet_size = max(min_bet(allowed), int(max(pot, BIG_BLIND) * 0.40))
            return (
                "bet",
                capped(bet_size, allowed),
                f"thin value vs big-stack-loose: rank {rank}",
            )

    return None


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

    # TAG opponents with 0 all-ins are value-heavy when they jam. Don't rescue
    # medium-strength hands (one pair/two pair) from folding at high stack prices.
    if hand_rank in {1, 2} and _has_tag_opponent(table, my_seat):
        stack = int(my_seat.get("stackChips") or 0)
        stack_price = price / max(stack, 1)
        if stack_price > 0.50:
            return None

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

    # Calculate SPR after the contemplated call: remaining effective stack over
    # the pot after the call. This is the commitment metric that matters for a
    # fold/call decision, not the raw current pot or the live bet total.
    hero_stack_after_call = max(0, int(my_seat.get("stackChips") or 0) - price)
    opponent_stacks_after_call = [
        max(0, int(seat.get("stackChips") or 0))
        for seat in live_opponent_seats(table, my_seat)
    ]
    effective_stack_after_call = min(
        [hero_stack_after_call, *opponent_stacks_after_call]
        if opponent_stacks_after_call
        else [hero_stack_after_call]
    )
    pot_after_call = effective_pot(table) + price
    spr = effective_stack_after_call / max(1, pot_after_call)

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


def is_hero_in_position(table, my_seat):
    # postflop action order: SB, BB, UTG, MP, CO, BTN
    order = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
    hero_pos = position_label(table, my_seat)
    if hero_pos not in order:
        return False
    hero_idx = order.index(hero_pos)
    opponents = live_opponent_seats(table, my_seat)
    for opp in opponents:
        opp_pos = position_label(table, opp)
        if opp_pos in order:
            opp_idx = order.index(opp_pos)
            if opp_idx > hero_idx:
                return False
    return True


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

    # Restrict to a genuine heads-up pot. active_opponents-based routing
    # ensures this fires when the hand has narrowed down to 1 opponent,
    # rather than failing on 6-max tables where seated_players is always 6.
    if active_opponents(table, my_seat) > 1:
        return None

    # 1. Structural: Must be In Position
    if not is_hero_in_position(table, my_seat):
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

    # Restrict to a genuine heads-up pot (see ip_dry_board_cbet_exploit).
    if active_opponents(table, my_seat) > 1:
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


def semi_bluff_exploit(table, my_seat, base) -> ActionDecision | None:
    """Aggressively bet/raise with strong draws in position instead of passively calling.

    Intercepts check or call decisions from the base strategy and converts them
    to a bet when holding strong draws (flush draw or open-ended straight draw)
    in position. This builds the pot with equity and generates fold equity
    instead of passively calling.
    """
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])

    # Only intercept passive actions (check or fold-with-call-available)
    if action not in {"check", "fold"}:
        return None
    if "bet" not in available:
        return None
    if not no_one_has_bet(table, allowed):
        return None

    # Must be in position
    if not is_hero_in_position(table, my_seat):
        return None
    pos = position_label(table, my_seat)

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None

    # Must have a strong draw
    has_fd = has_flush_draw(hole_cards, board_cards)
    has_oesd = has_open_ended_straight_draw(hole_cards, board_cards)
    if not has_fd and not has_oesd:
        return None

    # Don't semi-bluff into paired boards (reduced fold equity)
    texture = board_texture(board_cards)
    if texture.get("paired", False):
        return None

    # Sizing: 50-60% pot for maximum pressure with good risk/reward
    pot = int(table.get("potChips") or 0)
    bet_size = max(min_bet(allowed), int(max(pot, BIG_BLIND) * 0.55))

    draw_type = []
    if has_fd:
        draw_type.append("FD")
    if has_oesd:
        draw_type.append("OESD")
    draw_label = "+".join(draw_type)

    return (
        "bet",
        capped(bet_size, allowed),
        f"v003 semi-bluff {draw_label} IP pos {pos}",
    )


def sixmax_adjustment(table, my_seat, base) -> ActionDecision | None:
    # 0. Universal mathematical truth: Pot commitment overrides everything (all table sizes)
    decision = spr_commitment_lock(table, my_seat, base)
    if decision is not None:
        return decision

    # 0.4. A priced-in river sliver is +EV vs any plausible range.
    #      Restricted to River until data confirms turn/flop extension.
    decision = sliver_shove_guard(table, my_seat, base)
    if decision is not None:
        return decision

    # 0.5. Board-made / kicker-only hands share strength with the table.
    #      Covers pure board-made (trips/flush/straight/full house from
    #      the board) AND board trips + kicker only (Qh Kd on 33385).
    decision = board_made_hand_guard(table, my_seat, base)
    if decision is not None:
        return decision

    # 0.6. Non-nut flushes on paired boards have severe reverse implied odds.
    decision = vulnerable_flush_guard(table, my_seat, base)
    if decision is not None:
        return decision

    # 0.7a. Targeted multi-way patch: fold small pairs in 3+ way pots.
    # Must run before preflop open-raise / isolation / 3-bet so it can
    # override the wide MP open range in 6-max games.
    decision = small_pair_multiway_fold_guard(table, my_seat, base)
    if decision is not None:
        return decision

    # 1. GTO preflop 3-bet vs a single open-raise (v5: profile-gated bluffs + sizing)
    decision = preflop_three_bet(table, my_seat, base)
    if decision is not None:
        return decision

    # 2. GTO preflop squeeze vs raise + callers (v5: profile-gated bluff arm)
    decision = preflop_squeeze(table, my_seat, base)
    if decision is not None:
        return decision

    # 3. Proactive preflop open raising (v5: profile-aware range/sizing)
    decision = preflop_open_raise(table, my_seat, base)
    if decision is not None:
        return decision

    # 3.5. Preflop isolation raise over limpers
    decision = preflop_isolation_raise(table, my_seat, base)
    if decision is not None:
        return decision

    # 3.6. Targeted multi-way patch: fold small pairs in 3+ way pots.
    decision = small_pair_multiway_fold_guard(table, my_seat, base)
    if decision is not None:
        return decision

    # 3.7. Targeted set-mining patch: call with small pairs (22-66) at cheap prices.
    decision = top_pair_good_kicker_vs_loose_bad_price(table, my_seat, base)
    if decision is not None:
        return decision

    decision = medium_pocket_pair_vs_tight_three_bet(table, my_seat, base)
    if decision is not None:
        return decision

    decision = pocket_pair_set_mining_guard(table, my_seat, base)
    if decision is not None:
        return decision

    # 4. Preflop positional defense (v5: raiser-profile-aware score threshold)
    decision = preflop_positional_defense(table, my_seat, base)
    if decision is not None:
        return decision

    # 5. Postflop exploit: suppress bluffs / expand thin value vs big-stack-loose opponents
    decision = big_stack_loose_postflop_adjust(table, my_seat, base)
    if decision is not None:
        return decision

    # 6. Tightly-gated postflop exploits. Each self-guards on table size and
    # profile confidence, so they only fire where they are +EV.
    for exploit in (
        ip_dry_board_cbet_exploit,
        high_wtsd_thin_value_bet,
        semi_bluff_exploit,
    ):
        if (decision := exploit(table, my_seat, base)) is not None:
            return decision

    # 7. Universal fallbacks: reproduce flattened's exact cascade order so that
    # v4 never gives back EV in spots the specialised branches did not handle.
    for adjustment in (
        preflop_premium_pressure,  # from v003
        medium_pair_paired_board_fold_guard,  # 77 on paired boards (must run first)
        paired_board_pot_control,  # from v005
        board_assisted_two_pair_guard,  # over-value leak: board-assisted two pair on paired board
        river_two_pair_raise_guard,  # river two-pair value-raise leak vs tight opponents
        preflop_min_raise_war_cap,  # cap preflop min-raise wars after 3 raise-backs
        river_one_pair_over_call,  # fold one pair on river vs >30% pot bet
        paired_board_range_fold,  # from v007
        medium_hand_multiway_fold_guard,  # Patch B: tighten multi-way medium hands
        weak_pair_wet_board_pot_control,  # from v004 (was dropped in v2)
        strong_top_pair_defense,  # from v005
        high_card_bottom_pair_check,  # thin-value leak: bottom pair on A/K-high boards
        mixed_threshold_pressure_response,  # from v004 (was dropped in v2)
        simple_profile_river_bluff_catch,  # from v007
        range_mixed_dry_probe,  # from v003
        cheap_postflop_continue,  # from v003
        postflop_draw_continue,  # from v006 — non-cheap draw calls
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

    base = survival_sixmax_choose_action(table, my_seat)
    if seated_players(table) < 4:
        base = patch1_choose_action(table, my_seat)
        counter_adj = heads_up_counter_action(table, my_seat, base)
        if counter_adj is not None:
            base = counter_adj
        research = research_short_handed_action(table, my_seat, base)
        if research is not None:
            base = research
    adjusted = sixmax_adjustment(table, my_seat, base)
    if adjusted is not None:
        return adjusted

    action, amount, message = base
    return action, amount, f"{message}"
