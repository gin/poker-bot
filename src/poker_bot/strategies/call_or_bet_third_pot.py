"""Strategy that always calls or bets one-third of the pot."""

ActionDecision = tuple[str | None, int | None, str]


def _third_pot_bet(table: dict, allowed: dict) -> int:
    """Return the nearest legal bet size at or above one-third pot."""
    pot = int(table.get("potChips") or 0)
    target = pot // 3

    bet_range = allowed.get("betRange") or {}
    minimum = int(bet_range.get("min", allowed.get("minBet") or 0))
    maximum = bet_range.get("max", allowed.get("maxCommit"))

    target = max(target, minimum)
    if maximum is not None:
        target = min(target, int(maximum))
    return target


def choose_action(table: dict, my_seat: dict | None) -> ActionDecision:
    """Call when facing a bet; otherwise bet one-third of the pot."""
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if not available:
        return None, None, "No legal actions available"

    if "call" in available:
        call_amount = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
        return "call", call_amount, "Calling every time"

    if "bet" in available:
        amount = _third_pot_bet(table, allowed)
        return "bet", amount, "Betting one-third of the pot"

    if "check" in available:
        return "check", None, "No call or bet available, checking"
    if "fold" in available:
        return "fold", None, "No call or bet available"

    return None, None, "No supported legal action available"
