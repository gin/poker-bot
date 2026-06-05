"""Simple baseline Texas Hold'em strategy."""

from poker_bot.table import find_agent_seat

type ActionDecision = tuple[str | None, int | None, str]


def choose_action(table, agent_id) -> ActionDecision:
    """Choose a legal action using a simple hole-card strength heuristic."""
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if not available:
        return None, None, "No legal actions available"

    street = table.get("street", "Preflop")
    my_seat = find_agent_seat(table, agent_id)

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
