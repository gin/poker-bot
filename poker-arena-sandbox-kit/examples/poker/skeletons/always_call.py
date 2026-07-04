"""Skeleton: a calling station — check/call everything, never fold voluntarily."""


def act(table: dict) -> dict:
    avail = (table.get("allowedActions") or {}).get("availableActions") or []
    if "check" in avail:
        return {"action": "check"}
    if "call" in avail:
        return {"action": "call"}
    return {"action": "fold"}
