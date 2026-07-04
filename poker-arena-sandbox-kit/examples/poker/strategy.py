"""Sample strategy — THIS is exactly what you upload to dev.fun Arena.

Export `act(table)` (or `choose_action(table)`). Read the `table` dict, return
one action as a dict (or a bare string). `amount` = TOTAL chips committed on this
street after acting (not the delta). For fold/check/call, omit `amount`.

SELF-CONTAINED: your submitted strategy.py runs with only stdlib + numpy + torch
on the server — it can NOT import `arena_sdk`. So the position helper below is
inlined here. (When iterating locally you can `from arena_sdk.poker.read import
is_button, to_call, pot_odds` — the same logic, for local iteration.)

This baseline is tight-aggressive and **position-aware**: it opens wider as the
button (in position) and plays tighter out of position. Replace the logic with
your own (rules -> solver -> self-play -> RL).
"""
RANKS = "23456789TJQKA"


def _rank(card: str) -> int:
    return RANKS.index(card[0].upper()) if card and card[0].upper() in RANKS else 0


def _is_button(table: dict) -> bool:
    """Heads-up, the button posts the small blind — find that seat in recentEvents.
    The table has NO button/position field, so you derive it. (Postflop the button
    acts last = in position.)"""
    sb = table.get("smallBlindChips")
    for ev in table.get("recentEvents") or []:
        s = ev.get("summary") or {}
        if ev.get("type") == "BlindPosted" and s.get("amount") == sb:
            return s.get("seatNumber") == table.get("selfSeatNumber")
    return False


def act(table: dict) -> dict:
    allowed = table.get("allowedActions") or {}
    avail = allowed.get("availableActions") or []
    call_chips = int(allowed.get("callChips") or 0)
    pot = int(table.get("potChips") or 0)
    board = list(table.get("boardCards") or [])

    # YOUR hole cards live under YOUR seat, not at table["holeCards"].
    seat = next((s for s in (table.get("seats") or [])
                 if s.get("seatNumber") == table.get("selfSeatNumber")), {})
    hole = list(seat.get("holeCards") or [])

    pair = len(hole) == 2 and hole[0][0] == hole[1][0]
    high = max((_rank(c) for c in hole), default=0)
    suited = len(hole) == 2 and hole[0][-1] == hole[1][-1]
    in_position = _is_button(table)

    def raise_to(frac: float) -> dict:
        rr = allowed.get("raiseRange") or {}
        lo, hi = int(rr.get("min") or 0), int(rr.get("max") or 0)
        if lo <= 0:
            return {"action": "call"} if "call" in avail else {"action": "check"}
        return {"action": "raise", "amount": max(lo, min(int(pot * frac) or lo, hi))}

    # ── Preflop: open wider in position, tighter out of position ──────────
    if not board:
        strong = pair or high >= 11 or (suited and high >= 9)
        playable = strong or (in_position and (high >= 9 or suited))  # button opens wider
        if "raise" in avail and strong:
            return raise_to(3.0)
        if call_chips == 0:
            return {"action": "check"} if "check" in avail else {"action": "fold"}
        if playable and call_chips <= pot:
            return {"action": "call"}
        return {"action": "fold"}

    # ── Postflop ─────────────────────────────────────────────────────────
    connects = pair or bool({c[0].upper() for c in hole} & {c[0].upper() for c in board})
    if call_chips == 0:
        # bet for value / semi-bluff when we connect; in position, bet a bit thinner
        if (connects or in_position) and "bet" in avail:
            br = allowed.get("betRange") or {}
            lo, hi = int(br.get("min") or 0), int(br.get("max") or 0)
            if lo > 0:
                frac = 0.5 if connects else 0.33
                return {"action": "bet", "amount": max(lo, min(int(pot * frac), hi))}
        return {"action": "check"} if "check" in avail else {"action": "fold"}
    if connects and call_chips <= max(int(pot * 0.6), 1):
        return {"action": "call"}
    return {"action": "fold"}
