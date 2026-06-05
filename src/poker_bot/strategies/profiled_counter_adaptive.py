"""Counter-adaptive strategy with opponent profiles and multiway awareness."""

from poker_bot.opponents import OpponentProfile, profile_from_mapping
from poker_bot.strategies import counter_adaptive
from poker_bot.strategies.adaptive import (
    board_texture,
    has_top_pair_or_better,
    made_hand_rank,
    pot_odds,
    preflop_score,
)

type ActionDecision = tuple[str | None, int | None, str]


def active_opponents(table, my_seat):
    seats = table.get("seats", [])
    my_id = (my_seat or {}).get("agentId")
    opponents = [
        seat
        for seat in seats
        if seat.get("agentId") != my_id
        and not seat.get("folded", False)
        and not seat.get("hasFolded", False)
    ]
    return max(1, len(opponents))


def table_profiles(table, my_seat):
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


def choose_action(table, my_seat) -> ActionDecision:
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if not available:
        return None, None, "No legal actions available"
    if not my_seat:
        if "fold" in available:
            return "fold", None, "Fallback: seat not found"
        return None, None, "No matching seat found"

    opponents = active_opponents(table, my_seat)
    profiles = table_profiles(table, my_seat)
    tendencies = table_tendencies(profiles)
    street = table.get("street", "Preflop")
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = table.get("potChips", 0)
    call_amount = allowed.get("callAmount", 0)
    required = pot_odds(call_amount, pot)

    if opponents <= 1 and not profiles:
        return counter_adaptive.choose_action(table, my_seat)

    if street == "Preflop":
        score = preflop_score(hole_cards)
        threshold = 48 + max(0, opponents - 2) * 6
        if tendencies["all_patient"]:
            threshold -= 4
        if tendencies["has_aggressive"]:
            threshold += 4

        if "raise" in available and score >= threshold + 30:
            amount = counter_adaptive.raise_for_value(
                table, allowed, strong=score >= 95
            )
            return "raise", amount, f"Profiled premium score {score}, raising"
        if "call" in available:
            price_cap = 0.08 if opponents <= 2 else 0.05
            if score >= threshold or required <= price_cap:
                return "call", call_amount, f"Profiled preflop score {score}, calling"
            if "check" in available:
                return "check", None, f"Profiled weak score {score}, checking"
            return "fold", None, f"Profiled weak score {score}, folding"
        if "bet" in available and score >= threshold:
            amount = counter_adaptive.small_pressure_bet(pot, allowed)
            return "bet", amount, f"Profiled open score {score}"
        if "check" in available:
            return "check", None, f"Profiled score {score}, checking"

    made_rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards)
    strong_threshold = 2 if opponents <= 2 else 3
    strong = made_rank >= strong_threshold
    medium = made_rank == 1 or top_pair

    if "raise" in available and strong and not tendencies["has_aggressive"]:
        amount = counter_adaptive.raise_for_value(table, allowed, strong=made_rank >= 4)
        return "raise", amount, f"Profiled value raise rank {made_rank}"

    if "call" in available:
        if strong:
            return "call", call_amount, f"Profiled call rank {made_rank}"
        if tendencies["has_bluffer"] and medium and required <= 0.28:
            return "call", call_amount, "Bluff-catching profiled opponent"
        if medium and opponents <= 2 and required <= 0.14:
            return "call", call_amount, "Heads-up medium hand defense"
        if "check" in available:
            return "check", None, "Profiled marginal hand, checking"
        return "fold", None, f"Profiled fold rank {made_rank}"

    if "bet" in available:
        if strong:
            amount = counter_adaptive.value_bet(
                pot, allowed, strong=made_rank >= 4 or tendencies["has_station"]
            )
            return "bet", amount, f"Profiled value bet rank {made_rank}"
        if medium and opponents <= 2:
            amount = counter_adaptive.value_bet(pot, allowed)
            return "bet", amount, "Profiled thin value heads-up"
        if (
            opponents <= 2
            and tendencies["all_patient"]
            and texture["high"]
            and not texture["wet"]
        ):
            amount = counter_adaptive.small_pressure_bet(pot, allowed)
            return "bet", amount, "Profiled pressure versus patient table"

    if "check" in available:
        return "check", None, "Profiled no-edge spot, checking"
    if "fold" in available:
        return "fold", None, "Profiled no profitable action"
    return None, None, "No supported legal action available"
