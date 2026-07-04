"""Skeleton: pick a uniformly random legal action (sizing = half pot). Useful as
a noisy opponent and to prove the action/amount contract end-to-end."""
import random


def act(table: dict) -> dict:
    allowed = table.get("allowedActions") or {}
    avail = list(allowed.get("availableActions") or [])
    if not avail:
        return {"action": "fold"}
    pick = random.choice(avail)
    pot = int(table.get("potChips") or 0)
    if pick in ("bet", "raise"):
        rng = allowed.get("raiseRange" if pick == "raise" else "betRange") or {}
        lo, hi = int(rng.get("min") or 1), int(rng.get("max") or 1)
        return {"action": pick, "amount": max(lo, min(int(pot * 0.5) or lo, hi))}
    return {"action": pick}
