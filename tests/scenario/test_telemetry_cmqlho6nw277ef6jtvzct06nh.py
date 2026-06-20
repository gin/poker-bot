"""Telemetry-driven medium-hand value-betting flaw for s2base.

Reference hand from telemetry-luigi-tournament.sqlite (strategy='s2v012'):
  - cmqlho6nw277ef6jtvzct06nh: 5h Qs on 4h Js Ad 5c Ts
  - Hero has Q5o, bottom pair on the Turn, bottom two on the River
  - The bot thin-value bet both streets and lost the hand

The flaw: the bot treats made_rank == 1 (one pair) as "medium" and value-bets
it for thin value. But bottom pair with a weak kicker on high-card boards is
dominated by most continuing ranges. These hands need pot control, not value
bets.
"""

import pytest

from poker_bot.strategies.s2base import choose_action


HERO = "hero"
DEFAULT_AVAILABLE = ("fold", "check", "bet", "all-in")


@pytest.fixture
def choose_s2base():
    return choose_action


def make_seat(seat_number, agent_id, hole_cards=None, *, folded=False):
    return {
        "seatNumber": seat_number,
        "agentId": agent_id,
        "holeCards": hole_cards or [],
        "stackChips": 12000,
        "currentBetChips": 0,
        "folded": folded,
    }


def make_table(
    hole,
    board,
    *,
    street="Turn",
    pot=180,
    call=0,
    available=DEFAULT_AVAILABLE,
    villains=1,
    profiles=None,
    button=1,
):
    """Build a minimal s2base table fixture for the Q5o telemetry scenario."""
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
            "minBet": 10,
            "minRaiseTo": max(20, call * 2),
        },
    }
    return table, seats[0]


def choose_action_for(hole, board, choose_action_fn, **kwargs):
    table, hero = make_table(hole, board, **kwargs)
    return choose_action_fn(table, hero)


def action_for(hole, board, choose_action_fn, **kwargs):
    return choose_action_for(hole, board, choose_action_fn, **kwargs)[0]


def assert_checks_medium_pair(action):
    assert action == "check"


def assert_does_not_thin_value_bet(action):
    assert action != "bet"


@pytest.mark.parametrize(
    "hole,board",
    (
        (["5h", "Qs"], ["4h", "Js", "Ad", "5c"]),  # bottom pair, Q kicker
    ),
)
def test_bottom_pair_high_card_board_checks_turn(
    choose_s2base,
    hole,
    board,
):
    """Q5o on Ad Js 4h 5c should check, not thin-value bet bottom pair.

    Reference: cmqlho6nw277ef6jtvzct06nh from telemetry.
    The bot bet 68 into 180 on the Turn with bottom pair and a Q kicker. This is
    dominated by most continuing ranges on an A-high, J-high board.
    """
    action = action_for(hole, board, choose_s2base, pot=180)
    assert_checks_medium_pair(action)


@pytest.mark.parametrize(
    "hole,board",
    (
        (["5h", "Qs"], ["4h", "Js", "Ad", "5c", "Ts"]),  # bottom two, weak kicker
    ),
)
def test_bottom_two_high_card_board_checks_river(
    choose_s2base,
    hole,
    board,
):
    """Q5o on Ad Js 4h 5c Ts should check, not thin-value bet bottom two.

    Reference: cmqlho6nw277ef6jtvzct06nh from telemetry.
    The bot bet 120 into 316 on the River with bottom two. The T on the river
    gives opponents AT, JT, 5T possibilities, and the A/J high cards dominate
    this hand. This is not a value bet.
    """
    action = action_for(hole, board, choose_s2base, pot=316)
    assert_checks_medium_pair(action)


@pytest.mark.parametrize(
    "hole,board",
    (
        (["5h", "Qs"], ["4h", "Js", "Ad", "5c"]),
        (["5h", "Qs"], ["4h", "Js", "Ad", "5c", "Ts"]),
    ),
)
def test_medium_pair_does_not_thin_value_bet_high_card_boards(
    choose_s2base,
    hole,
    board,
):
    """Q5o should not thin-value bet on A/J high-card boards.

    This is the core flaw from cmqlho6nw277ef6jtvzct06nh: the bot's
    adaptive_choose_action() treats made_rank == 1 as "medium" and value-bets it
    with the message "Thin value against simple". That logic is too loose for
    bottom pair / bottom two with weak kickers.
    """
    action = action_for(hole, board, choose_s2base, pot=180 if len(board) == 4 else 316)
    assert_does_not_thin_value_bet(action)
