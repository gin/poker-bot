from typing import Any


class PokerGuard:
    """
    The 'Safety Sail' layer. Implements hard constraints to prevent catastrophic
    neural network errors based on proven poker theory.
    """

    @staticmethod
    def evaluate(
        table: dict[str, Any], hero_seat: dict[str, Any], proposed_action: str
    ) -> tuple[str, str]:
        """
        Validates the proposed action against hard safety rules.

        Returns:
            (final_action, reason)
            reason: "approved" if no override, otherwise the reason for override.
        """
        pot = table.get("potChips") or 0
        allowed = table.get("allowedActions") or {}
        call_amt = allowed.get("callAmount", 0) or 0

        # --- RULE 1: Sliver Shove / Pot Odds Floor ---
        # If the cost to call is extremely low compared to the pot (e.g., < 15%),
        # it is almost always a mistake to fold, regardless of what the NN thinks.
        if proposed_action == "fold" and call_amt > 0:
            pot_odds = call_amt / (pot + call_amt + 1e-6)
            if pot_odds < 0.15:
                return "call", "override: sliver_shove_floor"

        # --- RULE 2: Nut Hand Protection ---
        # (Requires hand rank from encoder/evaluator)
        # If hero has the absolute nuts or near-nuts, we must not fold.
        # Note: Implementation depends on hand_rank being passed in.

        # --- RULE 3: Bet Sizing Cap ---
        # Prevent the NN from making absurdly large bets that the heuristics
        # would never make (e.g., 10x pot on river).
        if proposed_action == "raise" and call_amt > 0:
            raise_range = allowed.get("raiseRange") or {}
            raise_amt = raise_range.get("min", 0) or 0
            # If the min raise is already > 3x pot, we might cap it
            if raise_amt > pot * 3:
                return "call", "cap: excessive_bet_size"

        return proposed_action, "approved"
