"""Arena adapter — wraps my pre-existing `my_bot.decide()` into act(table).

Per SUBMITTING.md §4(b) "I already have a poker bot":
  obs  = features(table)         # table dict  -> MY bot's input
  move = my_bot.decide(obs)      # my untouched logic
  return to_arena(move, allowed) # MY output   -> a legal action

`amount` returned to the server = TOTAL chips committed on this street after
acting (not the delta). I size off the pot and then clamp into the legal range.
"""
from my_bot import decide


def _hero_hole(table: dict) -> list:
    seats = table.get("seats") or []
    me = table.get("selfSeatNumber")
    seat = next((s for s in seats if s.get("seatNumber") == me), {})
    return list(seat.get("holeCards") or [])


def act(table: dict) -> dict:
    allowed = table.get("allowedActions") or {}
    avail = allowed.get("availableActions") or []
    call_chips = int(allowed.get("callChips") or 0)
    pot = int(table.get("potChips") or 0)
    board = list(table.get("boardCards") or [])
    hole = _hero_hole(table)

    # --- map table -> my bot's world, call my untouched logic ---
    action, frac = decide(hole, board, call_chips, pot)

    # --- map my output -> a LEGAL action (this is the whole contract) ---

    # Raise/bet: my bot says "raise frac of pot". The server only ever offers
    # ONE of {bet, raise} depending on state, with its own min/max. Pick whichever
    # is legal, size off pot, clamp.
    if action == "raise":
        target_key = "raise" if "raise" in avail else ("bet" if "bet" in avail else None)
        if target_key:
            rng = allowed.get("raiseRange" if target_key == "raise" else "betRange") or {}
            lo, hi = int(rng.get("min") or 0), int(rng.get("max") or 0)
            if hi > 0:
                # amount = TOTAL on this street: pot * frac, clamped to [lo, hi]
                amount = max(lo, min(int(pot * frac) or lo, hi))
                return {"action": target_key, "amount": amount}
        # couldn't raise/bet -> degrade gracefully
        action = "call" if call_chips > 0 else "check"

    if action == "call":
        if "call" in avail:
            return {"action": "call"}
        action = "check"  # nothing to call

    if action == "check":
        if "check" in avail:
            return {"action": "check"}
        # can't check (facing a bet) -> fall through to fold

    return {"action": "fold"} if "fold" in avail else {"action": "check"}
