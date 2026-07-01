"""River one-pair over-call leak — behavioral spec for s3base.

The bot calls river bets with one pair (medium bucket, rank 1) 24,374 times,
losing -380k chips. One pair on the river is a bluff-catcher and should
fold to large bets (> 50% pot).

The fix: add a guard that folds one pair on the river when facing a bet
greater than 50% of pot. One pair can call smaller bets but should fold
to larger bets where it's likely behind a stronger hand.
"""

import pytest

from poker_bot.strategies.s3base import choose_action

HERO = "hero"


def make_seat(
    seat_number, agent_id, hole_cards=None, *, folded=False, stack=2000, current_bet=0
):
    return {
        "seatNumber": seat_number,
        "agentId": agent_id,
        "holeCards": hole_cards or [],
        "stackChips": stack,
        "currentBetChips": current_bet,
        "folded": folded,
        "hasFolded": False,
    }


def make_table(hole, board, *, call_amount=50, pot=150):
    """Build a river table where hero has one pair and faces a bet."""
    seats = [
        make_seat(1, HERO, hole, current_bet=0),
        make_seat(2, "villain", current_bet=call_amount),
    ]
    table = {
        "street": "River",
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": 1,
        "seats": seats,
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": ["fold", "call", "raise", "all-in"],
            "callAmount": call_amount,
            "callChips": call_amount,
            "raiseRange": {"min": call_amount * 2, "max": 2000},
            "betRange": {"min": 0, "max": 2000},
        },
    }
    return table, seats[0]


# One pair on the river: hero has one pair on various boards
ONE_PAIR_HANDS = [
    (["KS", "QH"], ["KH", "7D", "4C", "2H", "8C"]),  # Pair of kings
    (["JS", "TC"], ["JH", "5D", "3C", "7H", "2S"]),  # Pair of jacks
    (["9D", "8H"], ["9C", "5S", "3D", "2C", "TH"]),  # Pair of nines
]


def action_for(hole, board, **kwargs):
    table, hero = make_table(hole, board, **kwargs)
    return choose_action(table, hero)[0]


@pytest.mark.parametrize("hole,board", ONE_PAIR_HANDS)
def test_one_pair_folds_vs_large_river_bet(hole, board):
    """One pair on the river should fold vs a bet > 50% of pot."""
    # pot=150, call=100: 100/150 = 67% > 50% → should fold
    action = action_for(hole, board, call_amount=100, pot=150)
    assert action == "fold", (
        f"one pair {hole} on {board} should fold vs large river bet, got {action!r}"
    )


@pytest.mark.parametrize("hole,board", ONE_PAIR_HANDS)
def test_one_pair_calls_vs_medium_river_bet(hole, board):
    """One pair on the river can call medium bets (< 50% pot)."""
    # pot=150, call=50: 50/150 = 33% < 50% → should call
    action = action_for(hole, board, call_amount=50, pot=150)
    assert action == "call", (
        f"one pair {hole} on {board} should call vs medium river bet, got {action!r}"
    )


@pytest.mark.parametrize("hole,board", ONE_PAIR_HANDS)
def test_one_pair_calls_vs_small_river_bet(hole, board):
    """One pair on the river can call small bets (< 30% pot)."""
    # pot=150, call=20: 20/150 = 13% < 50% → should call
    action = action_for(hole, board, call_amount=20, pot=150)
    assert action == "call", (
        f"one pair {hole} on {board} should call vs small river bet, got {action!r}"
    )
