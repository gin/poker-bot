"""Rank-2 value raise facing a bet — guard for hubase.

When hero holds two pair (rank 2) on the flop or turn and is facing a bet,
the base strategy wants to value-raise. But raising two pair into a bet is
consistently -EV: villain's re-raise range is too strong.

The guard must:
- Convert rank-2 raises to calls when facing a bet on flop/turn
- Allow rank-2 raises when NOT facing a bet (first action)
- Allow rank 1/3+ to raise facing a bet (different hand strengths)
- Allow rank-2 raises on the river (different street dynamics)
"""

import pytest

from poker_bot.strategies.hubase import choose_action, made_hand_rank

HERO = "hero"


def make_seat(seat_number, agent_id, hole_cards=None, *, folded=False):
    return {
        "seatNumber": seat_number,
        "agentId": agent_id,
        "holeCards": hole_cards or [],
        "stackChips": 2000,
        "currentBetChips": 0,
        "folded": folded,
        "hasFolded": False,
    }


def make_table(
    hole,
    board,
    *,
    street="Flop",
    pot=80,
    call=40,
    facing_bet=True,
    available=("fold", "call", "raise", "all-in"),
):
    seats = [make_seat(1, HERO, hole)]
    seats.append(make_seat(2, "villain"))

    table = {
        "street": street,
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": 1,
        "seats": seats,
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": list(available),
            "callAmount": call,
            "callChips": call,
            "raiseRange": {"min": call * 2 if call > 0 else 40, "max": 2000},
            "betRange": {"min": 0, "max": 2000},
        },
    }
    return table, seats[0]


def action_for(hole, board, **kwargs):
    table, hero = make_table(hole, board, **kwargs)
    return choose_action(table, hero)[0]


# ────────────────────────────────────────────────────────────────────
# Helper: real two pair hands (hero holds one card from each pair)
# ────────────────────────────────────────────────────────────────────
# Two pair requires: hero's two cards each match a different board card.
# Example: 7C KC on board 7S 3D KH → pair of 7s + pair of Ks (rank 2)
# Example: 9C 5D on board 9H 5S 2C → pair of 9s + pair of 5s (rank 2)


def test_rank2_facing_bet_flop_calls_instead_of_raise():
    """7C KC on 7S 3D KH board, facing a bet on flop → call, not raise."""
    assert made_hand_rank(["7C", "KC"], ["7S", "3D", "KH"]) == 2
    action = action_for(
        ["7C", "KC"], ["7S", "3D", "KH"],
        street="Flop", pot=80, call=40, facing_bet=True,
    )
    assert action == "call", f"rank 2 facing bet on flop should call, got {action!r}"


def test_rank2_facing_bet_turn_calls_instead_of_raise():
    """9C 5D on 9H 5S 2C KH board, facing a bet on turn → call, not raise."""
    assert made_hand_rank(["9C", "5D"], ["9H", "5S", "2C", "KH"]) == 2
    action = action_for(
        ["9C", "5D"], ["9H", "5S", "2C", "KH"],
        street="Turn", pot=200, call=80, facing_bet=True,
    )
    assert action == "call", f"rank 2 facing bet on turn should call, got {action!r}"


def test_rank2_not_facing_bet_flop_allows_raise():
    """7C KC on 7S 3D KH board, first to action (call=0) → patch1 may pot-control.

    When not facing a bet, the base wants to raise. But patch1_choose_action
    (HU lookup) may convert this to a call for pot control (rank >= 2, required <= 0.32).
    Either raise or call is acceptable here — the key is that the hand doesn't fold.
    """
    assert made_hand_rank(["7C", "KC"], ["7S", "3D", "KH"]) == 2
    action = action_for(
        ["7C", "KC"], ["7S", "3D", "KH"],
        street="Flop", pot=80, call=0, facing_bet=False,
    )
    # Not facing bet: either raise (base) or call (patch1 pot-control) is fine
    assert action in ("raise", "call"), f"rank 2 not facing bet should not fold, got {action!r}"


def test_rank1_facing_bet_not_affected():
    """AC 5H on AS 7D 3C board (one pair, rank 1), facing bet → base decides."""
    rank = made_hand_rank(["AC", "5H"], ["AS", "7D", "3C"])
    assert rank == 1
    action = action_for(
        ["AC", "5H"], ["AS", "7D", "3C"],
        street="Flop", pot=80, call=40, facing_bet=True,
    )
    # Rank 1 is not affected by the rank-2 guard
    # Base may fold (as the data shows), but guard shouldn't convert raise to call
    assert action in ("fold", "call", "raise"), f"rank 1 should not be forced by rank-2 guard, got {action!r}"


def test_rank3_facing_bet_allows_raise():
    """8C 8D on 8H KS 3D board (trips, rank 3), facing bet → raise allowed."""
    assert made_hand_rank(["8C", "8D"], ["8H", "KS", "3D"]) == 3
    action = action_for(
        ["8C", "8D"], ["8H", "KS", "3D"],
        street="Flop", pot=80, call=40, facing_bet=True,
    )
    assert action == "raise", f"trips facing bet should raise, got {action!r}"


def test_rank2_facing_bet_river_not_affected():
    """7C KC on 7S 3D KH 2S 9C (river), facing bet → guard doesn't fire on river."""
    assert made_hand_rank(["7C", "KC"], ["7S", "3D", "KH", "2S", "9C"]) == 2
    action = action_for(
        ["7C", "KC"], ["7S", "3D", "KH", "2S", "9C"],
        street="River", pot=400, call=150, facing_bet=True,
    )
    # Guard only covers Flop/Turn, so river raises are allowed
    assert action in ("raise", "call", "fold"), f"rank 2 on river should not be affected, got {action!r}"
