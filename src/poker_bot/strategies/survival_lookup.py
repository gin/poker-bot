"""Lookup-first 6-max survival strategy.

This strategy keeps runtime decisions fast and network-free. The lookup tables
act as local GTO-style priors by table size and opponent composition, then the
policy falls back to tested heuristic modules for concrete action selection.
"""

from poker_bot.strategies import (
    anti_threshold,
    survival_sixmax,
)
from poker_bot.strategies import (
    profiled_counter_adaptive as profiled,
)
from poker_bot.strategies.adaptive import (
    BIG_BLIND,
    board_texture,
    has_top_pair_or_better,
    made_hand_rank,
    pot_odds,
    preflop_score,
)

ActionDecision = tuple[str | None, int | None, str]

AGGRESSIVE_LABELS = {"bluffer", "loose_aggressive", "tight_aggressive"}
LOOSE_LABELS = {"bluffer", "loose_aggressive", "calling_station"}
TIGHT_LABELS = {"patient_methodical", "tight_aggressive"}

STYLE_POLICY = {
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

PREFLOP_LOOKUP = {
    "tight": {
        "steal_scores": {"BTN": 42, "CO": 48, "SB": 52},
        "value_3bet_score": 82,
    },
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

POSTFLOP_LOOKUP = {
    ("loose_aggressive", "medium", "call"): {
        "max_pot_odds": 0.30,
        "policy": "bluff_catch",
    },
    ("loose_passive", "strong", "bet"): {
        "size": "large_value",
        "policy": "thin_value",
    },
    ("tight", "air", "bet"): {
        "size": "small_pressure",
        "policy": "steal_dry_boards",
    },
    ("unknown", "strong", "bet"): {
        "size": "value",
        "policy": "default_value",
    },
}

# CFR-style priors: static action-frequency hints for abstract buckets. This is
# not live CFR; it is the seam where offline regret-minimization output can be
# stored later without changing the strategy shape.
CFR_PRIORS = {
    ("preflop", "short_handed"): {"profiled": 0.70, "anti_threshold": 0.30},
    ("preflop", "loose_aggressive"): {"profiled": 0.80, "anti_threshold": 0.20},
    ("postflop_pressure", "unknown"): {"anti_threshold": 0.85, "profiled": 0.15},
    ("postflop_defense", "loose_aggressive"): {
        "profiled": 0.75,
        "anti_threshold": 0.25,
    },
}


def active_players(table):
    seats = table.get("seats", [])
    return sum(
        1
        for seat in seats
        if not seat.get("folded", False) and not seat.get("hasFolded", False)
    )


def table_style(table, my_seat):
    player_count = active_players(table)
    if player_count <= 3:
        return "short_handed"

    labels = [profile.label() for profile in profiled.table_profiles(table, my_seat)]
    if not labels:
        return "unknown"

    aggressive = sum(label in AGGRESSIVE_LABELS for label in labels)
    loose = sum(label in LOOSE_LABELS for label in labels)
    tight = sum(label in TIGHT_LABELS for label in labels)
    stations = labels.count("calling_station")

    if aggressive >= max(1, len(labels) // 2):
        return "loose_aggressive"
    if stations >= max(1, len(labels) // 2) or loose >= max(2, len(labels) // 2):
        return "loose_passive"
    if tight >= max(2, len(labels) // 2):
        return "tight"
    return "unknown"


def choose_policy(street, available, style):
    policy = STYLE_POLICY.get(style, STYLE_POLICY["unknown"])
    if street == "Preflop":
        return policy["preflop"]
    if "call" in available:
        return policy["defense"]
    if "bet" in available or "raise" in available:
        return policy["pressure"]
    return policy["defense"]


def hand_bucket(hole_cards, board_cards):
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


def lookup_hint(table, my_seat, style):
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    street = table.get("street", "Preflop")
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])

    if street == "Preflop":
        return PREFLOP_LOOKUP.get(style, PREFLOP_LOOKUP["unknown"])

    bucket = hand_bucket(hole_cards, board_cards)
    action_type = "bet" if ("bet" in available or "raise" in available) else "call"
    return POSTFLOP_LOOKUP.get(
        (style, bucket, action_type),
        POSTFLOP_LOOKUP.get(("unknown", bucket, action_type), {}),
    )


def dispatch(policy_name, table, my_seat):
    if policy_name == "anti_threshold":
        return anti_threshold.choose_action(table, my_seat)
    if policy_name == "profiled":
        return profiled.choose_action(table, my_seat)
    return survival_sixmax.choose_action(table, my_seat)


def style_adjustment(table, my_seat, style, hint):
    """Small lookup-driven overrides before module fallback."""
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
        bucket = hand_bucket(hole_cards, board_cards)
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


def choose_action(table, my_seat) -> ActionDecision:
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if not available:
        return None, None, "No legal actions available"
    if not my_seat:
        if "fold" in available:
            return "fold", None, "Fallback: seat not found"
        return None, None, "No matching seat found"

    style = table_style(table, my_seat)
    hint = lookup_hint(table, my_seat, style)
    override = style_adjustment(table, my_seat, style, hint)
    if override is not None:
        return override

    policy_name = choose_policy(table.get("street", "Preflop"), available, style)
    action, amount, message = dispatch(policy_name, table, my_seat)
    return action, amount, f"lookup {style}/{policy_name}: {message}"
