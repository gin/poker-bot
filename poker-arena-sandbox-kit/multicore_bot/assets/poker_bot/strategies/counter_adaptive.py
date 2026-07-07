"""Counter-strategy tuned to exploit the adaptive baseline."""

from poker_bot.strategies.adaptive import (
    BIG_BLIND,
    board_texture,
    capped,
    has_top_pair_or_better,
    made_hand_rank,
    pot_odds,
    preflop_score,
)

ActionDecision = tuple[str | None, int | None, str]


def small_pressure_bet(pot, allowed):
    minimum = allowed.get("minBet", BIG_BLIND)
    if pot <= 0:
        return capped(max(minimum, BIG_BLIND), allowed)
    return capped(max(minimum, int(pot * 0.25)), allowed)


def value_bet(pot, allowed, strong=False):
    minimum = allowed.get("minBet", BIG_BLIND)
    if pot <= 0:
        return capped(max(minimum, BIG_BLIND), allowed)
    return capped(max(minimum, int(pot * (0.55 if strong else 0.35))), allowed)


def raise_for_value(table, allowed, strong=False):
    minimum = allowed.get("minRaiseTo")
    if minimum is None:
        return None
    pot = table.get("potChips", 0)
    current_bet = table.get("currentBet", 0)
    pressure = int(max(pot, BIG_BLIND) * (0.75 if strong else 0.45))
    return capped(max(minimum, current_bet + pressure), allowed)


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

    if street == "Preflop":
        score = preflop_score(hole_cards)
        if "raise" in available and score >= 74:
            amount = raise_for_value(table, allowed, strong=score >= 94)
            return "raise", amount, f"Countering adaptive with score {score}"
        if "call" in available:
            if score >= 42 or required <= 0.08:
                return "call", call_amount, f"Calling controlled preflop score {score}"
            if "check" in available:
                return "check", None, f"Checking weak preflop score {score}"
            return "fold", None, f"Folding weak preflop score {score}"
        if "bet" in available and score >= 50:
            amount = small_pressure_bet(pot, allowed)
            return "bet", amount, f"Opening into adaptive score {score}"
        if "check" in available:
            return "check", None, f"Checking preflop score {score}"

    made_rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards)
    strong = made_rank >= 2
    medium = made_rank == 1 or top_pair

    if "raise" in available and strong:
        amount = raise_for_value(table, allowed, strong=made_rank >= 3)
        return "raise", amount, f"Value raising adaptive rank {made_rank}"

    if "call" in available:
        if strong or (medium and required <= 0.18) or required <= 0.08:
            return "call", call_amount, f"Calling adaptive value range rank {made_rank}"
        if "check" in available:
            return "check", None, "Checking marginal hand"
        return "fold", None, f"Overfolding to adaptive aggression rank {made_rank}"

    if "bet" in available:
        if strong:
            amount = value_bet(pot, allowed, strong=made_rank >= 3)
            return "bet", amount, f"Value betting adaptive rank {made_rank}"
        if medium:
            amount = value_bet(pot, allowed)
            return "bet", amount, "Thin value versus adaptive"
        if not texture["wet"]:
            amount = small_pressure_bet(pot, allowed)
            return "bet", amount, "Small pressure against adaptive checks"

    if "check" in available:
        return "check", None, "Checking no-edge spot"
    if "fold" in available:
        return "fold", None, "No profitable counter action"
    return None, None, "No supported legal action available"
