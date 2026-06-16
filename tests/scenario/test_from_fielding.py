"""
Board-made-hand ("playing the board") behavioral spec for flattened_v5.

When the board itself makes the hand — trips on a paired board, a five-card
flush, a five-card straight, or a full house — every seat shares that five-card
strength. The danger is treating that shared board hand as a private monster and
value-raising or stacking off with air.

This file is intentionally conservative for a deterministic bot baseline:
board-made air should not value-raise, and it should fold to a large river bet.
Real private improvements, like a set, nut flush, or straight using a hole card,
should still continue or build value.
"""

import pytest

from poker_bot.strategies.flattened_v5 import choose_action

HERO = "hero"
DEFAULT_AVAILABLE = ("fold", "call", "raise")

BOARD_MADE_AIR_CASES = (
    (["2c", "3h"], ["8d", "8h", "8s", "Kd", "4c"]),  # board trips, no kicker
    (["Ks", "Qc"], ["Kh", "Qh", "9h", "4h", "2h"]),  # board flush, no heart
    (["Ks", "2c"], ["5d", "6h", "7s", "8c", "9d"]),  # board straight, no 4/T
    (["2c", "3h"], ["8d", "8h", "8s", "Kd", "Kc"]),  # board full house, air
)

VULNERABLE_MADE_HAND_CASES = (
    (
        ["Qh", "Kd"],
        ["3d", "8h", "3h", "3s", "5h"],
    ),  # trips with kicker only on paired board; reverse implied odds
)

NO_VALUE_RAISE_CASES = BOARD_MADE_AIR_CASES + VULNERABLE_MADE_HAND_CASES

REAL_HAND_CASES = (
    (["8c", "8s"], ["8d", "Kh", "2s", "Qc", "4d"]),  # real set
    (["Ah", "Jc"], ["Kh", "9h", "4h", "2h", "Ts"]),  # nut flush
    (["9d", "2c"], ["5d", "6h", "7s", "8c", "Kd"]),  # real straight
)


@pytest.fixture
def choose_flattened_v5():
    return choose_action


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
    street="River",
    pot=300,
    call=200,
    available=DEFAULT_AVAILABLE,
    villains=2,
    profiles=None,
    button=1,
):
    """Build a minimal flattened_v5 table fixture.

    The helper includes the fields flattened_v5 commonly expects, while keeping
    the scenario focused on action selection rather than arena plumbing.
    """
    seats = [make_seat(1, HERO, hole)]
    for index in range(villains):
        seats.append(make_seat(2 + index, f"v{index}"))

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
            "raiseRange": {"min": call * 2, "max": 2000},
            "betRange": {"min": 0, "max": 2000},
        },
    }
    return table, seats[0]


def choose_action_for(hole, board, choose_action_fn, **kwargs):
    table, hero = make_table(hole, board, **kwargs)
    return choose_action_fn(table, hero)


def action_for(hole, board, choose_action_fn, **kwargs):
    return choose_action_for(hole, board, choose_action_fn, **kwargs)[0]


def assert_not_value_raise(action):
    assert action != "raise"


def assert_folds_to_large_bet(action):
    assert action == "fold"


def assert_value_action(action):
    assert action in {"raise", "call", "bet"}


@pytest.mark.parametrize("hole,board", NO_VALUE_RAISE_CASES)
def test_no_private_edge_or_vulnerable_flush_does_not_value_raise(
    choose_flattened_v5,
    hole,
    board,
):
    action = action_for(hole, board, choose_flattened_v5)
    assert_not_value_raise(action)


@pytest.mark.parametrize("hole,board", NO_VALUE_RAISE_CASES)
def test_no_private_edge_or_vulnerable_flush_folds_to_large_bet(
    choose_flattened_v5,
    hole,
    board,
):
    action = action_for(hole, board, choose_flattened_v5, pot=38013, call=33150)
    assert_folds_to_large_bet(action)


@pytest.mark.parametrize("hole,board", REAL_HAND_CASES)
def test_real_private_hands_still_get_value(choose_flattened_v5, hole, board):
    action = action_for(hole, board, choose_flattened_v5)
    assert_value_action(action)


# Original stricter tests are kept as comments instead of deleted. They were
# useful as a first-pass contract, but the replacement tests above separate the
# robust rule ("no value raise with board-made air") from the more context-
# dependent question of whether a call is ever correct.
#
# def test_board_trips_with_air_does_not_raise():
#     # board shows trip eights; we hold two unrelated cards -> everyone has trips
#     assert act(["9c", "Tc"], ["8d", "8h", "8s", "Kd", "4c"]) != "raise"
#
#
# def test_board_trips_with_air_does_not_stack_off():
#     # facing a big bet on a trips board with air: fold, never call it off
#     assert act(["9c", "Tc"], ["8d", "8h", "8s", "Kd", "4c"]) not in ("raise", "call")
#
#
# def test_board_flush_we_hold_no_suit_does_not_raise():
#     # five hearts on the board, we hold none -> we play the board's flush
#     assert act(["Ks", "Qc"], ["Kh", "Qh", "9h", "4h", "2h"]) != "raise"
#
#
# def test_board_straight_unimproved_does_not_stack_off():
#     # 5-6-7-8-9 on the board; our cards make no higher straight
#     assert act(["Ks", "2c"], ["5d", "6h", "7s", "8c", "9d"]) not in (
#         "raise",
#         "call",
#     )
#
#
# def test_board_full_house_with_air_does_not_stack_off():
#     # board is eights-full-of-kings; neither hole card improves it
#     assert act(["2c", "3h"], ["8d", "8h", "8s", "Kd", "Kc"]) not in (
#         "raise",
#         "call",
#     )
#
#
# def test_real_set_still_gets_value():
#     # pocket eights + one board eight = our own set (not the board's trips)
#     assert act(["8c", "8s"], ["8d", "Kh", "2s", "Qc", "4d"]) in (
#         "raise",
#         "call",
#         "bet",
#     )
#
#
# def test_nut_flush_still_gets_value():
#     # we hold the ace of the flushed suit
#     assert act(["Ah", "Jc"], ["Kh", "9h", "4h", "2h", "Ts"]) in (
#         "raise",
#         "call",
#         "bet",
#     )
#
#
# def test_real_straight_using_our_card_still_gets_value():
#     # board is only four-to-a-straight; our nine completes a real straight
#     assert act(["9d", "2c"], ["5d", "6h", "7s", "8c", "Kd"]) in (
#         "raise",
#         "call",
#         "bet",
#     )
