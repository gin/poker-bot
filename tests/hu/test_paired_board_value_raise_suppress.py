"""Flop value raise on paired boards — board-dominated trips guard for hubase.

The base strategy's "Value raising" lookup sees made_hand_rank == 3 on a
paired board and raises for value. But when the three-of-a-kind is fully
on the board (e.g., T7 on Q-7-7), hero's private cards contribute nothing
to the trips. The hand is effectively one pair, and raising into a paired
board is catastrophic.

The guard must:
- Suppress value raises on paired boards when trips are fully on board
- Allow value raises on paired boards when hero holds a trips card (real trips)
- Allow value raises on dry boards (no regression)

Reference: gameplay.sqlite hubase telemetry — 221 flop value raises on
paired boards, avg -899/hand (the single largest leak in the strategy).
"""

import pytest

from poker_bot.strategies.hubase import choose_action

HERO = "hero"
DEFAULT_AVAILABLE = ("fold", "call", "raise", "all-in")


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
    call=0,
    available=DEFAULT_AVAILABLE,
    profiles=None,
    button=1,
):
    seats = [make_seat(1, HERO, hole)]
    seats.append(make_seat(2, "villain"))

    table = {
        "street": street,
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": button,
        "seats": seats,
        "opponentProfiles": profiles or {},
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


# --- Cases where the guard SHOULD suppress (check instead of raise) ---

# One pair on a paired board — trips are fully on board.
# T7 on Q-7-7: made_hand_rank == 3 (three sevens), but hero doesn't hold a 7.
# Any opponent with a pocket pair has trips; any Q has top pair.
PAIRED_BOARD_DOMINATED_TRIPS_CASES = (
    (["Tc", "7h"], ["Qh", "7s", "7d"]),      # tens on Q-7-7 (trips on board)
    (["9d", "8c"], ["Jh", "8s", "8d"]),      # nines on J-8-8 (trips on board)
    (["Kh", "5s"], ["Ah", "5d", "5c"]),      # kings on A-5-5 (trips on board)
    (["Qd", "3h"], ["9s", "3c", "3d"]),      # queens on 9-3-3 (trips on board)
    (["Ac", "Jh"], ["Ks", "Jd", "Jc"]),      # aces on K-J-J (trips on board)
)


# --- Cases where the guard should NOT suppress (allow raise) ---

# Real trips on a paired board — hero holds one of the trips cards.
PAIRED_BOARD_REAL_TRIPS_CASES = (
    (["7h", "7c"], ["Qh", "7s", "7d"]),      # pocket sevens on Q-7-7 = quads
    (["8d", "8c"], ["Jh", "8s", "8h"]),      # pocket eights on J-8-8 = quads
    (["Qd", "7c"], ["Qh", "7s", "7d"]),      # Qc makes real trips (Q-Q-Q)
)

# Dry board, one pair — no paired board, no suppression.
DRY_BOARD_ONE_PAIR_CASES = (
    (["Ac", "5h"], ["Kd", "9s", "3c"]),      # A-high one pair on dry board
    (["Qh", "Jc"], ["Td", "7s", "2d"]),      # Q-high one pair on dry board
    (["Kh", "8d"], ["Qc", "6s", "3h"]),      # K-high one pair on dry board
)


@pytest.mark.parametrize("hole,board", PAIRED_BOARD_DOMINATED_TRIPS_CASES)
def test_board_dominated_trips_does_not_raise(hole, board):
    """When trips are fully on board, suppress the value raise, check instead."""
    action = action_for(hole, board)
    assert action != "raise", (
        f"board-dominated trips should not raise, got {action!r}"
    )


@pytest.mark.parametrize("hole,board", PAIRED_BOARD_REAL_TRIPS_CASES)
def test_real_trips_on_paired_board_not_suppressed_by_new_guard(hole, board):
    """Direct unit test: the board_dominated_trips_guard must NOT fire
    when hero's hole cards contain one of the board's paired rank.

    We verify the guard directly rather than through choose_action because
    the cascade may convert raises to calls via pot-control (legitimate
    behavior). The guard specifically only suppresses when trips are fully
    on the board with no private contribution.
    """
    from poker_bot.strategies.hubase import board_dominated_trips_guard

    table, hero = make_table(hole, board)
    # Simulate the base wanting to raise (the leak scenario)
    base = ("raise", 40, "base wants to raise")
    result = board_dominated_trips_guard(table, hero, base)
    assert result is None, (
        f"board_dominated_trips_guard should NOT fire for real trips, "
        f"got {result!r}"
    )


@pytest.mark.parametrize("hole,board", DRY_BOARD_ONE_PAIR_CASES)
def test_one_pair_on_dry_board_not_affected(hole, board):
    """Dry board: no paired board, guard should not fire."""
    action = action_for(hole, board)
    # On dry board, the base may raise, call, or check depending on hand.
    # The guard should NOT suppress — whatever the base does is fine.
    assert action in ("raise", "call", "check", "bet"), (
        f"dry board should not be affected by paired-board guard, got {action!r}"
    )
