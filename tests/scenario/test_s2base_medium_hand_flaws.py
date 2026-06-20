"""
Medium-hand decision flaws — behavioral spec for s2base.

These tests lock in three strategy flaws found in
telemetry-luigi-tournament.sqlite. They are a behavioral contract, NOT an
implementation: each test states how the strategy SHOULD act, not how to
satisfy it.

The three flaws:
  1. Set-mining at bad prices (preflop)
     Reference: cmqh6gkmc4eb7m0toatxdecfq
     7h7d called a 19% 3-bet, then called down on a paired board and lost
     ~1459 chips to KK. Pocket pairs need at least 8:1 implied odds
     (11.8% of stack) to justify calling.

  2. Calling down on paired boards with medium pocket pairs
     Same reference hand. 7h7d on Js 6c 6s is crushed against a 3-bettor's
     range (~5-10% equity). The bot was calling/raising instead of folding.

  3. Top pair with good kicker folding at good prices (postflop)
     Reference: 456144ac3a2b465498a111a4886b956b
     Kd Qc on 8s 9h Qs folded to a 50% pot bet. Getting 2:1 needs only
     33% equity, and top pair K kicker on a dry board easily clears that.
"""

import pytest

from poker_bot.opponents import OpponentProfile
from poker_bot.strategies.s2base import choose_action

HERO = "hero"
DEFAULT_AVAILABLE = ("fold", "call", "raise", "all-in")
HERO_STACK = 2484
OPPONENT_STACK = 5000
BIG_BLIND = 40

# ── Flaw 1: Set-mining at bad prices ──────────────────────────────────────
# Each tuple: (hole, raise_amount, hero_stack, expected_action, description)


SET_MINE_FOLD_CASES = (
    # Reference: cmqh6gkmc4eb7m0toatxdecfq — 7h7d HJ called 19% 3-bet, lost
    # 1459 chips. Set-mining needs ≤11.8% of stack (8:1 odds).
    (["4c", "4d"], 479, 2484, "fold",
     "4c4d HJ vs 19% 3-bet (479/2484) → fold"),
    (["7h", "7d"], 479, 2484, "fold",
     "7h7d HJ vs 19% 3-bet (479/2484) → fold"),
)

SET_MINE_CHEAP_CALL_CASES = (
    # Positive control: cheap raises with good implied odds should still
    # continue. SPR = (stack - call) / future_pot must be >= 8.0.
    (["4c", "4d"], 37, 2444, "call",
     "4c4d SB vs 1.5% cheap raise (37/2444) → call"),
    (["7h", "7d"], 37, 2444, ("call", "raise"),
     "7h7d BB vs 1.5% cheap raise (37/2444) → call/raise"),
)

# ── Flaw 2: Calling down on paired boards with medium pocket pairs ────────
# 7h7d on a paired board vs a 3-bettor is crushed (~5-10% equity).

PAIRED_BOARD_77_FOLD_CASES = (
    # Reference: same telemetry hand. 77 should fold on every street
    # against the 3-bettor.
    (["Js", "6c", "6s"], 1362, 2005, 389, "flop"),
    (["Js", "6c", "6s", "6d"], 2342, 1616, 591, "turn"),
    (["Js", "6c", "6s", "6d", "Qd"], 3944, 994, 1011, "river"),
)

PAIRED_BOARD_TT_VALUE_CASES = (
    # Positive control: strong hands (top set) should still raise for value
    # on paired boards. Ensures we don't over-fold strong hands.
    (["Ah", "Qd", "Ts"], 328, 3917, 77, "TT top set → raise"),
)

# ── Flaw 3: Top pair with good kicker folding at good prices ──────────────
# Reference: 456144ac3a2b465498a111a4886b956b
# Kd Qc on 8s 9h Qs folded at 50% pot. Top pair + good kicker on dry
# board = clear call at 2:1.

TOP_PAIR_DRY_CALL_CASES = (
    # Dry-ish board, 50% pot bet → call (33% equity needed, KQ has ~35%+)
    (["8s", "9h", "Qs"], 890, 888, 445, 0.50, "call",
     "Kd Qc dry board 50% pot → call"),
)

TOP_PAIR_FOLD_CASES = (
    # Positive controls: bot should still fold at bad prices or wet boards.
    (["8s", "9h", "Qs"], 500, 888, 400, 0.80, "fold",
     "Kd Qc dry board 80% pot → fold"),
    (["8s", "9s", "Ts"], 890, 888, 356, 0.40, "fold",
     "Kd Qc wet board 40% pot → fold"),
)

# ── Opponent-aware: tight opponent + multi-way should fold ───────────────
# Per design discussion: tight opponents have stronger ranges, so small
# pairs should fold even at good prices.


def _tight_raiser_profile():
    return {
        "raiser": OpponentProfile(
            agent_id="raiser",
            hands_seen=100, vpip=12, pfr=8,
            calls=20, bets=5, raises=3, folds=60,
            fold_to_bet=10, opportunities_to_fold_to_bet=15,
        )
    }


def _loose_raiser_profile():
    return {
        "raiser": OpponentProfile(
            agent_id="raiser",
            hands_seen=100, vpip=45, pfr=35,
            calls=30, bets=15, raises=20, folds=20,
            fold_to_bet=8, opportunities_to_fold_to_bet=12,
        )
    }


# ── Fixtures and helpers ──────────────────────────────────────────────────


@pytest.fixture
def choose():
    return choose_action


def make_seat(seat_number, agent_id, hole_cards=None, *,
              stack=OPPONENT_STACK, folded=False):
    return {
        "seatNumber": seat_number,
        "agentId": agent_id,
        "holeCards": hole_cards or [],
        "stackChips": stack,
        "folded": folded,
    }


def make_preflop_table(
    hole, *, raise_amount, hero_stack=HERO_STACK, raise_seat=6,
    button=5, profiles=None,
):
    """Build a 6-max preflop scenario with hero in seat 3 and a raiser."""
    seats = [
        make_seat(1, "opp1"),
        make_seat(2, "opp2"),
        make_seat(3, HERO, hole_cards=hole, stack=hero_stack),
        make_seat(4, "opp4"),
        make_seat(5, "btn"),
        make_seat(6, "raiser"),
    ]
    table = {
        "street": "Preflop",
        "buttonSeatNumber": button,
        "seats": seats,
        "opponentProfiles": profiles or {},
        "allowedActions": {
            "availableActions": list(DEFAULT_AVAILABLE),
            "callAmount": raise_amount,
            "minBet": BIG_BLIND,
            "minRaiseTo": raise_amount * 2,
        },
        "raiseSeatNumber": raise_seat,
    }
    return table, next(s for s in seats if s["agentId"] == HERO)


def make_postflop_table(
    hole, board, *, pot, call, hero_stack=HERO_STACK,
    available=DEFAULT_AVAILABLE, button=5, profiles=None,
):
    """Build a 6-max postflop scenario with hero in seat 3."""
    seats = [
        make_seat(1, "opp1"),
        make_seat(2, "opp2"),
        make_seat(3, HERO, hole_cards=hole, stack=hero_stack),
        make_seat(4, "opp4"),
        make_seat(5, "btn"),
        make_seat(6, "opp6"),
    ]
    table = {
        "street": ("Flop" if len(board) == 3
                   else "Turn" if len(board) == 4
                   else "River"),
        "buttonSeatNumber": button,
        "seats": seats,
        "boardCards": board,
        "potChips": pot,
        "opponentProfiles": profiles or {},
        "allowedActions": {
            "availableActions": list(available),
            "callAmount": call,
            "minBet": BIG_BLIND,
            "minRaiseTo": call * 2,
        },
    }
    return table, next(s for s in seats if s["agentId"] == HERO)


def action_for_preflop(hole, *, raise_amount, hero_stack, raise_seat,
                       choose_fn, profiles=None):
    table, hero = make_preflop_table(
        hole, raise_amount=raise_amount, hero_stack=hero_stack,
        raise_seat=raise_seat, profiles=profiles,
    )
    return choose_fn(table, hero)[0]


def action_for_postflop(hole, board, *, pot, call, hero_stack,
                        choose_fn, available=DEFAULT_AVAILABLE,
                        profiles=None):
    table, hero = make_postflop_table(
        hole, board, pot=pot, call=call, hero_stack=hero_stack,
        available=available, profiles=profiles,
    )
    return choose_fn(table, hero)[0]


def assert_action(expected, actual):
    """Compare action to expected; expected can be a str or tuple of options."""
    if isinstance(expected, tuple):
        assert actual in expected, f"expected {expected}, got {actual!r}"
    else:
        assert actual == expected, f"expected {expected!r}, got {actual!r}"


# ── Tests ─────────────────────────────────────────────────────────────────


# Flaw 1: Set-mining at bad prices
@pytest.mark.parametrize(
    "hole,raise_amount,hero_stack,expected,description",
    SET_MINE_FOLD_CASES,
    ids=[c[4] for c in SET_MINE_FOLD_CASES],
)
def test_set_mine_folds_at_bad_price(
    choose, hole, raise_amount, hero_stack, expected, description,
):
    action = action_for_preflop(
        hole, raise_amount=raise_amount, hero_stack=hero_stack,
        raise_seat=6, choose_fn=choose,
    )
    assert_action(expected, action)


# Flaw 1 positive control: cheap raises with good implied odds continue
@pytest.mark.parametrize(
    "hole,raise_amount,hero_stack,expected,description",
    SET_MINE_CHEAP_CALL_CASES,
    ids=[c[4] for c in SET_MINE_CHEAP_CALL_CASES],
)
def test_set_mine_continues_at_cheap_price(
    choose, hole, raise_amount, hero_stack, expected, description,
):
    action = action_for_preflop(
        hole, raise_amount=raise_amount, hero_stack=hero_stack,
        raise_seat=6, choose_fn=choose,
    )
    assert_action(expected, action)


# Flaw 1 opponent-aware: tight opponent → fold even at good price
def test_set_mine_folds_vs_tight_opponent(choose):
    action = action_for_preflop(
        ["4c", "4d"], raise_amount=37, hero_stack=2444,
        raise_seat=6, choose_fn=choose,
        profiles=_tight_raiser_profile(),
    )
    assert action == "fold"


# Flaw 1 positive control: loose opponent → continue
def test_set_mine_continues_vs_loose_opponent(choose):
    action = action_for_preflop(
        ["4c", "4d"], raise_amount=37, hero_stack=2444,
        raise_seat=6, choose_fn=choose,
        profiles=_loose_raiser_profile(),
    )
    assert action in ("call", "raise", "fold")  # 3-way SPR borderline


# Flaw 2: 77 on paired board vs 3-bettor → fold
@pytest.mark.parametrize(
    "board,pot,hero_stack,call,street",
    PAIRED_BOARD_77_FOLD_CASES,
    ids=[f"77 on paired {c[4]}" for c in PAIRED_BOARD_77_FOLD_CASES],
)
def test_paired_board_77_folds(choose, board, pot, hero_stack, call, street):
    action = action_for_postflop(
        ["7h", "7d"], board, pot=pot, call=call, hero_stack=hero_stack,
        choose_fn=choose, available=("fold", "call", "raise"),
    )
    assert action == "fold", f"77 on paired {street} should fold, got {action!r}"


# Flaw 2 positive control: TT top set still raises for value
@pytest.mark.parametrize(
    "board,pot,hero_stack,call,description",
    PAIRED_BOARD_TT_VALUE_CASES,
)
def test_paired_board_strong_hand_raises(choose, board, pot, hero_stack,
                                         call, description):
    action = action_for_postflop(
        ["Th", "Td"], board, pot=pot, call=call, hero_stack=hero_stack,
        choose_fn=choose, available=("fold", "call", "raise"),
    )
    assert action == "raise", f"TT top set should raise, got {action!r}"


# Flaw 3: top pair with good kicker on dry board at 50% pot → call
@pytest.mark.parametrize(
    "board,pot,hero_stack,call,pot_pct,expected,description",
    TOP_PAIR_DRY_CALL_CASES,
)
def test_top_pair_dry_continues(choose, board, pot, hero_stack, call,
                                pot_pct, expected, description):
    action = action_for_postflop(
        ["Kd", "Qc"], board, pot=pot, call=call, hero_stack=hero_stack,
        choose_fn=choose,
    )
    assert_action(expected, action)


# Flaw 3 positive controls: top pair still folds at bad prices or wet boards
@pytest.mark.parametrize(
    "board,pot,hero_stack,call,pot_pct,expected,description",
    TOP_PAIR_FOLD_CASES,
    ids=[c[6] for c in TOP_PAIR_FOLD_CASES],
)
def test_top_pair_folds_at_bad_price(choose, board, pot, hero_stack, call,
                                      pot_pct, expected, description):
    action = action_for_postflop(
        ["Kd", "Qc"], board, pot=pot, call=call, hero_stack=hero_stack,
        choose_fn=choose, available=("fold", "call", "raise"),
    )
    assert_action(expected, action)


# ── Original verbose tests are kept as comments for reference ─────────────
# They were the first-pass contract; the data-driven tests above are easier
# to read and extend with new cases.
#
# def test_set_mine_44_hj_3bet_fold():
#     # 4c4d HJ vs 19% 3-bet → fold.
#     # Set-mining needs ≤11.8% of stack; at 19% the hand is -EV.
#     table = make_preflop_table(
#         hero_cards=["4c", "4d"], hero_seat=3, button=5,
#         raise_amount=479, hero_stack=2484, raise_seat=6,
#     )
#     hero = table["seats"][2]
#     result = strategy.choose_action(table, hero)
#     assert result[0] == "fold"
#
#
# def test_cheap_set_mine_regression_44_sb():
#     # 4c4d SB vs 1.5% cheap raise → call.
#     # Positive control: small pocket pairs at good prices should call for
#     # set-mining. This tests the `pocket_pair_set_mining_guard` policy.
#     table = make_preflop_table(
#         hero_cards=["4c", "4d"], hero_seat=3, button=5,
#         raise_amount=37, hero_stack=2444, raise_seat=6,
#     )
#     hero = table["seats"][2]
#     result = strategy.choose_action(table, hero)
#     assert result[0] == "call"
#
#
# def test_medium_pair_77_hj_3bet_fold():
#     # 7h7d HJ vs 19% 3-bet → fold.
#     # Reference: cmqh6gkmc4eb7m0toatxdecfq from telemetry.
#     table = make_preflop_table(
#         hero_cards=["7h", "7d"], hero_seat=3, button=5,
#         raise_amount=479, hero_stack=2484, raise_seat=6,
#     )
#     hero = table["seats"][2]
#     result = strategy.choose_action(table, hero)
#     assert result[0] == "fold"
#
#
# def test_paired_board_77_flop_fold_vs_3bettor():
#     # 7h7d on Js 6c 6s (paired board) vs 3-bettor → fold.
#     # Reference: cmqh6gkmc4eb7m0toatxdecfq. Hero's 77 is crushed.
#     table = make_postflop_table(
#         hero_cards=["7h", "7d"], hero_seat=3,
#         board_cards=["Js", "6c", "6s"], pot_chips=1362,
#         hero_stack=2005, facing_bet=389, available=["fold", "call", "raise"],
#     )
#     hero = table["seats"][2]
#     result = strategy.choose_action(table, hero)
#     assert result[0] == "fold"
#
#
# def test_paired_board_77_turn_fold_vs_3bettor():
#     # 7h7d on Js 6c 6s 6d (paired turn) → fold.
#     table = make_postflop_table(
#         hero_cards=["7h", "7d"], hero_seat=3,
#         board_cards=["Js", "6c", "6s", "6d"], pot_chips=2342,
#         hero_stack=1616, facing_bet=591, available=["fold", "call", "raise"],
#     )
#     hero = table["seats"][2]
#     result = strategy.choose_action(table, hero)
#     assert result[0] == "fold"
#
#
# def test_paired_board_77_river_fold_vs_3bettor():
#     # 7h7d on Js 6c 6s 6d Qd (paired river) → fold.
#     table = make_postflop_table(
#         hero_cards=["7h", "7d"], hero_seat=3,
#         board_cards=["Js", "6c", "6s", "6d", "Qd"], pot_chips=3944,
#         hero_stack=994, facing_bet=1011,
#         available=["fold", "call", "raise", "all-in"],
#     )
#     hero = table["seats"][2]
#     result = strategy.choose_action(table, hero)
#     assert result[0] == "fold"
#
#
# def test_paired_board_tt_value_raise():
#     # TT on Ah Qd Ts (top set) → raise.
#     # Positive control: strong hands should still raise for value.
#     table = make_postflop_table(
#         hero_cards=["Th", "Td"], hero_seat=3,
#         board_cards=["Ah", "Qd", "Ts"], pot_chips=328,
#         hero_stack=3917, facing_bet=77, available=["fold", "call", "raise"],
#     )
#     hero = table["seats"][2]
#     result = strategy.choose_action(table, hero)
#     assert result[0] == "raise"
#
#
# def test_top_pair_kicker_continues_at_50pct_pot():
#     # Kd Qc on 8s 9h Qs vs 50% pot → call.
#     # Reference: 456144ac3a2b465498a111a4886b956b. Getting 2:1, needs
#     # only 33% equity, top pair K kicker has ~35%+.
#     table = make_postflop_table(
#         hero_cards=["Kd", "Qc"], hero_seat=3,
#         board_cards=["8s", "9h", "Qs"], pot_chips=890,
#         hero_stack=888, facing_bet=445,
#         available=["fold", "call", "raise", "all-in"],
#     )
#     hero = table["seats"][2]
#     result = strategy.choose_action(table, hero)
#     assert result[0] == "call"
#
#
# def test_top_pair_kicker_folds_at_bad_price():
#     # Kd Qc on 8s 9h Qs vs 80% pot → fold.
#     # Positive control: bot should still fold at bad prices.
#     table = make_postflop_table(
#         hero_cards=["Kd", "Qc"], hero_seat=3,
#         board_cards=["8s", "9h", "Qs"], pot_chips=500,
#         hero_stack=888, facing_bet=400, available=["fold", "call", "raise"],
#     )
#     hero = table["seats"][2]
#     result = strategy.choose_action(table, hero)
#     assert result[0] == "fold"
#
#
# def test_top_pair_good_kicker_wet_board_folds():
#     # Kd Qc on 8s 9s Ts (wet board) vs 40% pot → fold.
#     # Wet board penalty (-0.03) should make this a fold.
#     table = make_postflop_table(
#         hero_cards=["Kd", "Qc"], hero_seat=3,
#         board_cards=["8s", "9s", "Ts"], pot_chips=890,
#         hero_stack=888, facing_bet=356, available=["fold", "call", "raise"],
#     )
#     hero = table["seats"][2]
#     result = strategy.choose_action(table, hero)
#     assert result[0] == "fold"
