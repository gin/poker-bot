"""Counter-strategy tuned to exploit threshold-pressure sizing."""

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

# type ActionDecision = tuple[str | None, int | None, str]
ActionDecision = tuple[str | None, int | None, str]


def pressure_amount(pot, allowed, target_required=0.20):
    minimum = allowed.get("minBet", BIG_BLIND)
    if pot <= 0:
        return capped(max(minimum, BIG_BLIND), allowed)
    raw = int((target_required * pot) / max(0.01, 1 - target_required))
    amount = max(minimum, raw + 1, int(pot * 0.30))
    return capped(amount, allowed)


def value_bet_amount(pot, allowed, strong=False):
    minimum = allowed.get("minBet", BIG_BLIND)
    if pot <= 0:
        return capped(max(minimum, BIG_BLIND), allowed)
    size = 0.50 if strong else 0.33
    return capped(max(minimum, int(pot * size)), allowed)


def raise_amount(table, allowed, strong=False):
    minimum = allowed.get("minRaiseTo")
    if minimum is None:
        return None
    pot = table.get("potChips", 0)
    current_bet = table.get("currentBet", 0)
    size = 0.66 if strong else 0.40
    target = current_bet + int(max(pot, BIG_BLIND) * size)
    return capped(max(minimum, target), allowed)


def should_cbet(texture, opponents):
    if not texture:
        return True
    return not texture.get("wet", False) or opponents <= 1


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

    # Seat position estimate — lower seat offset = earlier position
    seat_offset = None
    for i, s in enumerate(table.get("seats", [])):
        if s.get("agentId") == (my_seat or {}).get("agentId"):
            seat_offset = i
            break

    in_position = seat_offset is not None and opponents <= 2

    if street == "Preflop":
        score = preflop_score(hole_cards)

        # Play looser in position (button/CO), tighter out of position
        play_threshold = 38 if in_position else 44
        raise_threshold = 72 if in_position else 78
        premium_threshold = 88 if in_position else 92

        if "raise" in available and score >= raise_threshold:
            amount = raise_amount(table, allowed, strong=score >= premium_threshold)
            return "raise", amount, f"Premium score {score}, raising"
        if "call" in available:
            if score >= play_threshold or required <= 0.12:
                return "call", call_amount, f"Playable score {score}, calling"
            if "check" in available:
                return "check", None, f"Checking score {score}"
            return "fold", None, f"Folding weak score {score}"
        if "bet" in available and score >= 56:
            amount = value_bet_amount(pot, allowed)
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

    # Raise with strong hands
    if "raise" in available and strong:
        amount = raise_amount(table, allowed, strong=made_rank >= 3)
        return "raise", amount, f"value raise rank {made_rank}"

    # Check-raise / donk bet with strong hands when out of position
    if "bet" in available and "check" in available and strong:
        amount = value_bet_amount(pot, allowed, strong=made_rank >= 3)
        return "bet", amount, f"Value bet rank {made_rank}"

    # Call decisions
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

    # Bet decisions (when check isn't available — in position as preflop aggressor)
    if "bet" in available:
        if strong:
            amount = value_bet_amount(pot, allowed, strong=made_rank >= 3)
            return "bet", amount, f"Value betting rank {made_rank}"
        if medium and opponents <= 1 and should_cbet(texture, opponents):
            amount = value_bet_amount(pot, allowed)
            return "bet", amount, "Thin value / c-bet"
        if no_made and opponents <= 1 and should_cbet(texture, opponents):
            amount = pressure_amount(pot, allowed, 0.16)
            return "bet", amount, "C-bet semi-bluff"
        if medium and opponents <= 2:
            amount = value_bet_amount(pot, allowed)
            return "bet", amount, "Multiway thin bet"

    if "check" in available:
        return "check", None, "No edge, checking"
    if "fold" in available:
        return "fold", None, "No profitable action"
    return None, None, "No supported legal action"
