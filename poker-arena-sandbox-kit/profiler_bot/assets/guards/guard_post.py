"""Post-decision guards for the sandbox."""

from __future__ import annotations

from poker_bot.guards.context import GuardContext
from poker_bot.guards.registry import GuardRail
from poker_bot.hand_utils import (
    card_values, evaluate_hand, board_has_pair, board_dominated_two_pair,
    paired_board_ranks, pot_odds, profile_value, profile_vpip_frequency,
    profile_call_frequency, profile_fold_to_bet_frequency,
    profile_aggression_frequency_merged, single_opponent_profile,
    opponent_is_bluffy, is_tight_opponent, call_amount, no_one_has_bet,
)

ActionDecision = tuple[str, int | None, str]

# ── Thresholds (from hu009-hu012, benchmark-validated) ─────────────────────
_TURN_TWO_PAIR_SUPPRESS_VPIP = 0.30
_TURN_TWO_PAIR_SUPPRESS_FOLD_TO_BET = 0.55

guard_rail = GuardRail()
guard_post = guard_rail

@guard_rail.register(
    "simple_test_guard",
    "post", 20, ["hu", "6max"],
    "Simple test guard for validation",
)
def simple_test_guard(ctx: GuardContext, proposed: ActionDecision) -> ActionDecision | None:
    # If proposed action is raise, let it through
    if proposed[0] == "raise":
        return None
    # Otherwise, return the proposed action
    return proposed
