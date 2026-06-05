"""All-in everytime strategy."""

ActionDecision = tuple[str | None, int | None, str]


def choose_action(table, my_seat) -> ActionDecision:
    """Choose the most aggressive legal action, committing the full stack."""
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if not available:
        return None, None, "No legal actions available"

    if not my_seat:
        if "fold" in available:
            return "fold", None, "Fallback: seat not found"
        return None, None, "No matching seat found"

    max_commit = allowed.get(
        "maxCommit",
        my_seat.get("currentBetChips", 0) + my_seat.get("stackChips", 0),
    )

    if "raise" in available:
        return "raise", max_commit, "All-in every time"
    if "bet" in available:
        return "bet", max_commit, "All-in every time"
    if "call" in available:
        return "call", allowed.get("callAmount", 0), "Calling all available chips"
    if "check" in available:
        return "check", None, "No bet available, checking"
    if "fold" in available:
        return "fold", None, "No aggressive action available"

    return None, None, "No supported legal action available"
