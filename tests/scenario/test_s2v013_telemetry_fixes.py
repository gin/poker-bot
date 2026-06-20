"""Behavioral spec for paired-board leaks observed in s2v013 telemetry.

Reference: ``./PLAN_FIX_S2V013_FLAWS.md``. Telemetry review of 166 hands /
227 decisions showed three coupled leaks on paired boards that account for
~3,000 chips of loss:

1. **3-barrel catastrophe (-913 across 3 hands)** — bot leads flop with
   J-high / bottom pair / no pair on a paired board, then barrels turn
   and river into a paired board where opponents easily have it.

2. **AA over-fold (-93)** — AdAs on 7-3-J folds to a single flop barrel
   of ~31% pot. The ``Rank 1 below price`` rule over-folds overpairs.

3. **River calls with dominated / board-played hands (-266 across 2 calls)**
   — Q-T on Q-J-A-J calls river 49 dominated by Ax; 6-J on double-paired
   4-T-T-2-4 calls river 102 while literally playing the board.

This file locks in the correct behavior. Tests should make ``s2base.py``
return non-bet / non-fold / non-call actions on these scenarios.

Test style mirrors ``./tests/scenario/test_from_fielding.py``: module
docstring → named case tuples → ``make_table`` helper → parametrize-driven
tests with explicit ``assert_*`` helpers.
"""

import pytest

from poker_bot.strategies.s2base import choose_action

HERO = "hero"
DEFAULT_AVAILABLE = ("fold", "call", "raise", "all-in")
CHECK_BET_AVAILABLE = ("check", "bet")


# ── Group A: 3-street barrel catastrophe on paired boards ──────────────
#
# Each of these hero hands led the flop with no real hand on a paired
# board, then barreled turn and river into a paired board where villain's
# range is heavy with sets / two pair. Net loss: 443 + 293 + 177 = -913.

NO_LEAD_BARREL_CASES = (
    # T♣J♥ on Q♣Q♦: hero has J-high, plays the board. -443
    (["Tc", "Jh"], ["5c", "Qc", "Qd"]),
    # J♥A♥ on 4♣4♠: hero has bottom pair 4s with A kicker, marginal. -293
    (["Jh", "Ah"], ["5h", "4c", "4s"]),
    # 6dJd on T♣T♥: hero has no pair (board pairs Ts; later 4s/4d also pair). -177
    (["6d", "Jd"], ["4s", "Tc", "Th"]),
)


# ── Group B: AA defends single flop barrel ──────────────────────────────

AA_DEFEND_CASE = (
    ["Ad", "As"],
    ["7c", "3d", "Jh"],
)


# ── Group C: river call discipline on paired boards (4-way pot) ─────────
#
# Both calls were at 27-28% pot odds in 4-handed pots. The correct play is
# to fold: top pair with T kicker on A-high paired board is dominated by
# any Ax; 6-J on double-paired board literally plays the board.

NO_RIVER_CALL_PLAYED_BOARD_CASES = (
    # Q♣T♣ on Q♦J♠6♦A♠J♦ river: dominated top pair (Q kicker is dead vs Ax). -89
    (["Qc", "Tc"], ["Qd", "Js", "6d", "As", "Jd"]),
    # 6dJd on 4♠T♣T♥2♣4♦ river: hero plays the board T-T-4-4-2. -177
    (["6d", "Jd"], ["4s", "Tc", "Th", "2c", "4d"]),
)


# ── Group D: positive controls — real hands still value on paired boards
#
# These lock in that the "don't barrel with no pair on paired board" guard
# does NOT over-correct into folding made hands. Sets / two pair on the
# same paired boards should still bet / raise.

REAL_HAND_PAIRED_BOARD_CASES = (
    # Q♣Q♥ on Q♣Q♦: set of Qs on the paired board → bet
    (["Qc", "Qh"], ["5c", "Qc", "Qd"]),
    # 4d4h on 4♣4♠: set of 4s → bet
    (["4d", "4h"], ["5h", "4c", "4s"]),
    # TdTh on T♣T♥: set of Ts → bet
    (["Td", "Th"], ["4s", "Tc", "Th"]),
)

REAL_HAND_PAIRED_BOARD_RIVER_CASES = (
    # A♣Q♣ on Q♦J♠6♦A♠J♦ river: two pair (Aces & Queens) → raise
    (["Ac", "Qc"], ["Qd", "Js", "6d", "As", "Jd"]),
    # 4dTd on 4♠T♣T♥2♣4♦ river: two pair (Fours & Tens) → raise
    (["4d", "Td"], ["4s", "Tc", "Th", "2c", "4d"]),
)


# ── Test helpers (mirror test_from_fielding.py) ─────────────────────────


@pytest.fixture
def choose_s2base():
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
    """Build a minimal postflop scenario the strategy can act on."""
    seats = [make_seat(1, HERO, hole)]
    for index in range(villains):
        seats.append(make_seat(2 + index, f"v{index}"))

    return {
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
    }, seats[0]


def choose_action_for(hole, board, choose_action_fn, **kwargs):
    table, hero = make_table(hole, board, **kwargs)
    return choose_action_fn(table, hero)


def action_for(hole, board, choose_action_fn, **kwargs):
    return choose_action_for(hole, board, choose_action_fn, **kwargs)[0]


def assert_no_value_bet(action):
    assert action != "bet", (
        f"bot lead-bet with no real hand on paired board (action={action!r})"
    )


def assert_does_not_fold(action):
    assert action != "fold", (
        f"bot folded a hand that should continue (action={action!r})"
    )


def assert_folds(action):
    assert action == "fold", f"bot should have folded (action={action!r})"


def assert_value_action(action):
    assert action in {"raise", "call", "bet"}, (
        f"real hand should continue or build (action={action!r})"
    )


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("hole,board", NO_LEAD_BARREL_CASES)
def test_no_three_street_barrel_with_no_pair_on_paired_board(
    choose_s2base, hole, board
):
    """Bot should not lead-bet flop with no real hand on a paired board.

    Each of these hands lost 100s of chips barrel-betting three streets
    on a paired board where villain's range is heavy with sets / two
    pair. The correct play is to check (give up) rather than turn J-high
    or bottom pair into an expensive multi-street bluff.
    """
    action = action_for(
        hole,
        board,
        choose_s2base,
        street="Flop",
        pot=120,
        call=0,
        available=CHECK_BET_AVAILABLE,
    )
    assert_no_value_bet(action)


def test_aa_defends_single_flop_barrel_on_dry_board(choose_s2base):
    """AA on 7-3-J must not fold to a single flop barrel of ~30% pot.

    Reference: cmqlko7d4juznf6jtv0yrh85r from s2v013 telemetry.
    AA has ~88% equity vs any flop-betting range. The "Rank 1 below
    price, folding" rule over-folds overpairs.
    """
    hole, board = AA_DEFEND_CASE
    action = action_for(
        hole,
        board,
        choose_s2base,
        street="Flop",
        pot=427,
        call=196,
        available=DEFAULT_AVAILABLE,
    )
    assert_does_not_fold(action)


@pytest.mark.parametrize("hole,board", NO_RIVER_CALL_PLAYED_BOARD_CASES)
def test_river_call_discipline_on_paired_board(choose_s2base, hole, board):
    """Hero must fold river on paired boards when dominated or playing board.

    Reference hands from s2v013 telemetry: Q-T on Q-J-A-J calls river 49
    at 27% pot (dominated by any Ax); 6-J on double-paired 4-T-T-2-4
    plays the board T-T-4-4-2 and calls river 102 at 28% pot.

    Tests use a 4-way pot to reproduce the leak (villains=3, matching
    the original telemetry's active_players count).
    """
    pot = 257 if "4d" in board else 134
    call = 102 if "4d" in board else 49
    action = action_for(
        hole,
        board,
        choose_s2base,
        street="River",
        pot=pot,
        call=call,
        available=DEFAULT_AVAILABLE,
        villains=3,
    )
    assert_folds(action)


@pytest.mark.parametrize("hole,board", REAL_HAND_PAIRED_BOARD_CASES)
def test_set_still_values_on_paired_board_flop(choose_s2base, hole, board):
    """A set on a paired board should still bet flop.

    Positive control: locks in that the "don't barrel with no pair"
    guard does not over-correct into folding made hands.
    """
    action = action_for(
        hole,
        board,
        choose_s2base,
        street="Flop",
        pot=120,
        call=0,
        available=CHECK_BET_AVAILABLE,
    )
    assert_value_action(action)


@pytest.mark.parametrize("hole,board", REAL_HAND_PAIRED_BOARD_RIVER_CASES)
def test_two_pair_still_raises_on_paired_board_river(choose_s2base, hole, board):
    """Two pair on a paired board should still raise for value/protection.

    Positive control: two pair on a paired board has the board pair
    counterfeited only by quads; against any reasonable range it is
    the best hand.
    """
    pot = 257 if "4d" in board else 134
    call = 102 if "4d" in board else 49
    action = action_for(
        hole,
        board,
        choose_s2base,
        street="River",
        pot=pot,
        call=call,
        available=DEFAULT_AVAILABLE,
        villains=2,
    )
    assert_value_action(action)
