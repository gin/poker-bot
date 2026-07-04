"""Skeleton: check when free, else fold. Sanity-checks the submission pipeline
without playing real poker. Copy examples/poker/strategy.py to build a real bot."""


def act(table: dict) -> dict:
    avail = (table.get("allowedActions") or {}).get("availableActions") or []
    return {"action": "check"} if "check" in avail else {"action": "fold"}
