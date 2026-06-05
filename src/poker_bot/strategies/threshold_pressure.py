"""Threshold-pressure strategy tuned against profiled counter-adaptive."""

from poker_bot.strategies.adaptive import (
    BIG_BLIND,
    board_texture,
    capped,
    has_top_pair_or_better,
    made_hand_rank,
    pot_odds,
    preflop_score,
)
from poker_bot.strategies.profiled_counter_adaptive import active_opponents

ActionDecision = tuple[str | None, int | None, str]


def pressure_amount_for_threshold(pot, allowed, target_required=0.16):
    minimum = allowed.get("minBet", BIG_BLIND)
    if pot <= 0:
        return capped(max(minimum, BIG_BLIND), allowed)
    raw = int((target_required * pot) / max(0.01, 1 - target_required))
    amount = max(minimum, raw + 1, int(pot * 0.25))
    return capped(amount, allowed)


def value_bet_amount(pot, allowed, strong=False):
    minimum = allowed.get("minBet", BIG_BLIND)
    if pot <= 0:
        return capped(max(minimum, BIG_BLIND), allowed)
    return capped(max(minimum, int(pot * (0.55 if strong else 0.35))), allowed)


def threshold_raise_amount(table, allowed, strong=False):
    minimum = allowed.get("minRaiseTo")
    if minimum is None:
        return None
    pot = table.get("potChips", 0)
    current_bet = table.get("currentBet", 0)
    target = current_bet + int(max(pot, BIG_BLIND) * (0.75 if strong else 0.45))
    return capped(max(minimum, target), allowed)


def should_pressure(texture, opponents, medium):
    if opponents > 2:
        return False
    if medium:
        return True
    return not texture["wet"]


def choose_action(table, my_seat) -> ActionDecision:
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

    if street == "Preflop":
        score = preflop_score(hole_cards)
        if "raise" in available and score >= 74:
            amount = threshold_raise_amount(table, allowed, strong=score >= 92)
            return "raise", amount, f"Threshold preflop pressure score {score}"
        if "call" in available:
            if score >= 42 or required <= 0.08:
                return "call", call_amount, f"Controlled call score {score}"
            if "check" in available:
                return "check", None, f"Checking weak score {score}"
            return "fold", None, f"Folding weak score {score}"
        if "bet" in available and score >= 50:
            amount = pressure_amount_for_threshold(pot, allowed, 0.14)
            return "bet", amount, f"Opening threshold pressure score {score}"
        if "check" in available:
            return "check", None, f"Checking score {score}"

    made_rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards)
    strong = made_rank >= (2 if opponents <= 2 else 3)
    medium = made_rank == 1 or top_pair

    if "raise" in available and strong:
        amount = threshold_raise_amount(table, allowed, strong=made_rank >= 3)
        return "raise", amount, f"Threshold value raise rank {made_rank}"

    if "call" in available:
        if strong or (medium and required <= 0.18) or required <= 0.08:
            return "call", call_amount, f"Calling pressure response rank {made_rank}"
        if "check" in available:
            return "check", None, "Checking marginal pressure response"
        return "fold", None, f"Folding to value-heavy line rank {made_rank}"

    if "bet" in available:
        if strong:
            amount = value_bet_amount(pot, allowed, strong=made_rank >= 3)
            return "bet", amount, f"Threshold value bet rank {made_rank}"
        if should_pressure(texture, opponents, medium):
            target = 0.18 if medium else 0.15
            amount = pressure_amount_for_threshold(pot, allowed, target)
            return "bet", amount, "Betting just over defend threshold"

    if "check" in available:
        return "check", None, "No threshold edge, checking"
    if "fold" in available:
        return "fold", None, "No profitable threshold action"
    return None, None, "No supported legal action available"
