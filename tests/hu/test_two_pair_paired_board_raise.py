"""Board-assisted two pair on a paired board — behavioral spec for s3base.

When the turn pairs a previously single-paired board, hero's "two pair" is
often built on a board card rather than from a private second pair. That
hand looks like ``made_hand_rank == 2`` to the classifier, but it is
fragile: any opponent 4 makes a full house, an overpair beats us, and
top-pair-better-kicker dominates. The bot must pot-control (check, or
fold to a bet) instead of value-raising into a tight range.

Reference: telemetry-luigi-playground.sqlite hand
``cmqmv6leork81f6jtmytkgs2r``. Hero held Td 6d on 7c 4s Tc 4d, raised
on the turn, 3-bet/4-bet the turn, and called off the rest of the stack
with tens-and-fours — losing 1015 chips to a tight opponent.
"""

import pytest

from poker_bot.strategies.s3base import choose_action

HERO = "hero"
DEFAULT_AVAILABLE = ("fold", "call", "raise", "all-in")

# Hero holds both cards of one pair privately → real two pair.
# JJ on Js 6s 6c is a set (always raises), so it is not the right
# control. The right control: hero holds one of the paired board cards
# privately so both pairs are private+board, which is "real" two pair
# and SHOULD still raise. E.g. 7d Th on 7c 4s Tc 4d gives sevens-and-
# tens where hero holds both 7 and T privately — that's a genuine
# made hand, not board-assisted.
REAL_TWO_PAIR_CASES = (
    (
        ["7d", "Th"],
        ["7c", "4s", "Tc", "4d"],
    ),  # sevens (private+board) + tens (private+board)
    (
        ["4d", "Tc"],
        ["7c", "4s", "Tc", "4d"],
    ),  # fours (private+board) + tens (private+board)
)

# Hero holds neither card of the paired board → board-assisted two pair.
# On 7c 4s Tc 4d, Td 6d makes tens-and-fours where the 4s come from the
# board. This is fragile: any 4 is a full house, JJ+ beats us, AT/KT/QT/JT
# dominate. Must NOT raise; should check or fold to a bet.
BOARD_ASSISTED_TWO_PAIR_CASES = (
    # Reference: cmqmv6leork81f6jtmytkgs2r — hero lost 1015 chips
    (["Td", "6d"], ["7c", "4s", "Tc", "4d"]),
    # Generic equivalent: another ten with no private 4
    (["Th", "8h"], ["7c", "4s", "Tc", "4d"]),
    # Kd 5d on 7c 4s Kc 4d → kings-and-fours where the 4s come from board
    (["Kd", "5d"], ["7c", "4s", "Kc", "4d"]),
)

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
    street="Turn",
    pot=80,
    call=40,
    available=DEFAULT_AVAILABLE,
    profiles=None,
    button=1,
):
    """Build a minimal heads-up postflop fixture.

    The helper mirrors test_from_fielding.py: 2 seats (hero + 1 villain),
    keyword-only postflop knobs, no arena plumbing.
    """
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
    assert action != "raise", f"board-assisted two pair over-valued: chose {action!r}"


def assert_folds_to_bet(action):
    assert action == "fold", (
        f"fragile board-assisted two pair should fold to a bet, got {action!r}"
    )


def assert_value_action(action):
    assert action in {"raise", "call", "bet"}, (
        f"real two pair should still get value, got {action!r}"
    )


@pytest.mark.parametrize("hole,board", BOARD_ASSISTED_TWO_PAIR_CASES)
def test_board_assisted_two_pair_does_not_raise_vs_tight(hole, board):
    """Td 6d on 7c 4s Tc 4d (Turn) vs a tight opponent → check, not raise.

    Reference: cmqmv6leork81f6jtmytkgs2r. The bot value-raised tens-and-
    fours on the turn against a tight opponent and bled 1015 chips. The
    4 on the board means villain's range is heavy with full houses
    (any 4) and overpairs; the hand is a bluff-catcher at best.
    """
    action = action_for(hole, board, profiles=TIGHT_OPPONENT)
    assert_does_not_raise(action)


@pytest.mark.parametrize("hole,board", BOARD_ASSISTED_TWO_PAIR_CASES)
def test_board_assisted_two_pair_folds_to_bet_vs_tight(hole, board):
    """Same hand, facing a half-pot bet from a tight opponent → fold.

    When villain bets, the bet range is condensed toward 4x (full house)
    and overpairs. Even at 2:1 odds, board-assisted two pair has
    insufficient equity against a tight range to continue.
    """
    action = action_for(hole, board, pot=80, call=40, profiles=TIGHT_OPPONENT)
    assert_folds_to_bet(action)


@pytest.mark.parametrize("hole,board", REAL_TWO_PAIR_CASES)
def test_real_two_pair_still_raises_for_value(hole, board):
    """7d Th / 4d Tc on 7c 4s Tc 4d (Turn) vs tight opponent → raise.

    Positive control: when hero holds both cards of the two pair
    privately (or one card from each pair), the two pair is genuinely
    strong and should still raise for value. The guard must not
    over-correct into folding real two pair.
    """
    action = action_for(hole, board, profiles=TIGHT_OPPONENT)
    assert_value_action(action)
