"""River two-pair value-raise leak — behavioral spec for s3base.

On the river, the bot value-raises with two pair against tight opponents
and gets punished. Two pair on the river is often beaten by sets, straights,
flushes, or better two pair. Against a tight opponent's river betting range,
the hand is a bluff-catcher at best.

The guard should convert raise/bet → check/call when:
- River street
- made_hand_rank == 2 (two pair)
- Opponent is tight (VPIP% < 25%)

Reference: benchmark.sqlite analysis (10k hands × 5 seeds). River two-pair
raises lost -389,419 chips across all opponents, making it the single biggest
loss source. Examples:
  - KS 6C on 2H 5C 6D 3D 2D (sixes-and-deuces, board trips) → raise
  - 6S KH on 9C KC TS QS QH (kings-and-queens, board straight) → raise
  - AD 3D on QS QH 4S 5S AS (aces-and-fives) → raise
"""

import pytest

from poker_bot.strategies.s3base import choose_action

HERO = "hero"
DEFAULT_AVAILABLE = ("fold", "call", "raise", "all-in")
CHECK_BET_AVAILABLE = ("fold", "check", "bet", "all-in")

# Two pair on the river where value-raising is incorrect vs tight opponents
RIVER_TWO_PAIR_RAISE_CASES = (
    # sixes-and-deuces on a paired board (2H 5C 6D 3D 2D) — board has trip 2s, hero has 6s+2s
    (["KS", "6C"], ["2H", "5C", "6D", "3D", "2D"]),
    # kings-and-queens on a straight board (9C KC TS QS QH) — board has Q-J-T-9 straight, hero has Ks+Qs
    (["6S", "KH"], ["9C", "KC", "TS", "QS", "QH"]),
    # aces-and-fives on a paired board (QS QH 4S 5S AS) — board has trip Qs, hero has As+5s
    (["AD", "3D"], ["QS", "QH", "4S", "5S", "AS"]),
    # eights-and-sevens on a paired board (8H 3D 3S 7C 2H) — board has trip 3s, hero has 8s+7s
    (["8S", "7H"], ["8H", "3D", "3S", "7C", "2H"]),
)

# Same hands vs a loose opponent — value-raising is still correct
LOOSE_OPPONENT = {
    "villain": {"hands_seen": 50, "vpip": 30, "pfr": 20},
}

TIGHT_OPPONENT = {
    "villain": {"hands_seen": 50, "vpip": 12, "pfr": 8},
}


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
    call=100,
    available=DEFAULT_AVAILABLE,
    profiles=None,
    button=1,
):
    """Build a minimal heads-up river fixture."""
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
            "raiseRange": {"min": call * 2, "max": 2000},
            "betRange": {"min": 0, "max": 2000},
        },
    }
    return table, seats[0]


def action_for(hole, board, **kwargs):
    table, hero = make_table(hole, board, **kwargs)
    return choose_action(table, hero)[0]


def assert_does_not_raise(action):
    assert action != "raise", f"river two pair over-valued: chose {action!r}"


def assert_does_not_value_bet(action):
    assert action not in ("raise", "bet"), (
        f"river two pair should not value-bet: chose {action!r}"
    )


def assert_value_action(action):
    assert action in {"raise", "bet"}, (
        f"two pair vs loose should still value: chose {action!r}"
    )


@pytest.mark.parametrize("hole,board", RIVER_TWO_PAIR_RAISE_CASES)
def test_river_two_pair_does_not_raise_vs_tight(hole, board):
    """River two pair vs tight opponent → check, not raise.

    The bot's profiled value-raise rank 2 fires with two pair on the river,
    but against a tight opponent the river betting range is condensed toward
    sets, straights, and flushes. Two pair is a bluff-catcher at best.
    """
    action = action_for(
        hole,
        board,
        available=CHECK_BET_AVAILABLE,
        profiles=TIGHT_OPPONENT,
    )
    assert_does_not_value_bet(action)


@pytest.mark.parametrize("hole,board", RIVER_TWO_PAIR_RAISE_CASES)
def test_river_two_pair_calls_vs_tight_facing_bet(hole, board):
    """River two pair facing a bet from tight opponent → call, not raise.

    When facing a bet, two pair is a reasonable call (getting 2:1+), but
    re-raising is -EV against a tight range.
    """
    action = action_for(
        hole,
        board,
        pot=300,
        call=100,
        profiles=TIGHT_OPPONENT,
    )
    assert_does_not_raise(action)


@pytest.mark.parametrize("hole,board", RIVER_TWO_PAIR_RAISE_CASES)
def test_river_two_pair_still_raises_vs_loose(hole, board):
    """River two pair vs loose opponent → still raise for value.

    Against a loose opponent, the river betting range is wider with more
    one-pair hands and bluffs. Two pair can extract value.
    """
    action = action_for(
        hole,
        board,
        available=CHECK_BET_AVAILABLE,
        profiles=LOOSE_OPPONENT,
    )
    assert_value_action(action)
