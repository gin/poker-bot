from typing import Any


class PokerGuard:
    @staticmethod
    def evaluate(
        table: dict[str, Any], hero_seat: dict[str, Any], proposed_action: str
    ) -> tuple[str, str]:
        pot = table.get("potChips") or 0
        allowed = table.get("allowedActions") or {}
        call_amt = allowed.get("callAmount", 0) or 0
        if proposed_action == "fold" and call_amt > 0:
            pot_odds = call_amt / (pot + call_amt + 1e-6)
            if pot_odds < 0.15:
                return "call", "override: sliver_shove_floor"
        if proposed_action == "raise" and call_amt > 0:
            raise_range = allowed.get("raiseRange") or {}
            raise_amt = raise_range.get("min", 0) or 0
            if raise_amt > pot * 3:
                return "call", "cap: excessive_bet_size"
        return proposed_action, "approved"
