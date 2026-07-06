"""Pre-decision guards for the sandbox."""

from __future__ import annotations

from poker_bot.guards.context import GuardContext
from poker_bot.guards.registry import GuardRail
from poker_bot.hand_utils import (
    pot_odds,
    effective_pot,
    call_amount,
    live_opponent_seats,
    is_board_made_or_kicker_vulnerable,
    royal_flush_possible,
    is_aks,
    card_values,
    rank_counts,
    evaluate_hand,
    board_texture,
    RANK_VALUES,
)

ActionDecision = tuple[str, int | None, str]

guard_rail = GuardRail()
guard_pre = guard_rail

# Pre-Guard 1: spr_commitment_lock
# ══════════════════════════════════════════════════════════════


@guard_rail.register(
    "spr_commitment_lock",
    "pre",
    0,
    ["hu", "6max"],
    "Pot-committed: call with strong hand (two pair+) when SPR is low",
)
def spr_commitment_lock(ctx: GuardContext) -> ActionDecision | None:
    return None
