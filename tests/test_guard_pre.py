"""Tests for guard_pre.py — the 4 pre-decision guards.

The rails are default-shadow; these tests exercise guard decision logic,
so they force every guard active via POKER_GUARD_ACTIVATE.
"""

import pytest

from poker_bot.guards.context import GuardContext
from poker_bot.guards.guard_pre import guard_pre, guard_rail

HERO = "hero"


@pytest.fixture(autouse=True)
def _activate_all_guards(monkeypatch):
    monkeypatch.setenv("POKER_GUARD_ACTIVATE", "*")


def test_guard_pre_alias_matches_registry():
    assert guard_pre is guard_rail


def _ctx(hole, board, *, call=0, pot=400, players=2, street="River", stack=2000):
    seats = [
        {
            "seatNumber": i,
            "agentId": HERO if i == 1 else f"opp{i}",
            "holeCards": hole if i == 1 else [],
            "stackChips": stack,
            "currentBetChips": 0,
            "folded": False,
            "hasFolded": False,
        }
        for i in range(1, players + 1)
    ]
    actions = (
        ["fold", "call", "raise", "all-in"]
        if call > 0
        else ["fold", "check", "bet", "all-in"]
    )
    table = {
        "street": street,
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": 1,
        "seats": seats,
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": actions,
            "callAmount": call,
            "callChips": call,
            "raiseRange": {"min": max(call * 2, 4), "max": 4000},
        },
    }
    return GuardContext.build(table, seats[0])


class TestSliverShoveGuard:
    def test_calls_at_sliver_odds(self):
        ctx = _ctx(["4C", "5D"], ["7S", "9D", "KH", "2C", "3H"], call=50, pot=1000)
        result = guard_rail.run_pre(ctx)
        assert result is not None
        assert result[0][0] == "call"
        assert result[1] == "sliver_shove_guard"

    def test_silent_above_floor(self):
        ctx = _ctx(["4C", "5D"], ["7S", "9D", "KH", "2C", "3H"], call=200, pot=400)
        result = guard_rail.run_pre(ctx)
        # made_rank=0 so spr won't fire; sliver is 33% > 10%
        assert result is None or result[1] != "sliver_shove_guard"

    def test_river_only(self):
        ctx = _ctx(
            ["4C", "5D"], ["7S", "9D", "KH", "2C"], call=50, pot=1000, street="Turn"
        )
        result = guard_rail.run_pre(ctx)
        assert result is None or result[1] != "sliver_shove_guard"


class TestRoyalFlushPredecision:
    def test_aks_preflop_calls(self):
        ctx = _ctx(["AH", "KH"], [], call=5, pot=10, street="Preflop")
        result = guard_rail.run_pre(ctx)
        assert result is not None
        assert result[0][0] == "call"
        assert result[1] == "royal_flush_predecision"

    def test_non_aks_preflop_silent(self):
        ctx = _ctx(["AH", "KD"], [], call=5, pot=10, street="Preflop")
        result = guard_rail.run_pre(ctx)
        assert result is None or result[1] != "royal_flush_predecision"

    def test_royal_flush_possible_postflop(self):
        # AH KH on QH JH TH 2C -> royal flush possible
        ctx = _ctx(
            ["AH", "KH"], ["QH", "JH", "TH", "2C"], call=50, pot=200, street="Turn"
        )
        result = guard_rail.run_pre(ctx)
        assert result is not None
        assert result[0][0] == "call"
        assert result[1] == "royal_flush_predecision"


class TestBoardMadeHandGuard:
    def test_playing_the_board_checks(self):
        # 2C 3D on AS KS QS JH TH (royal flush on board) -> board-made
        ctx = _ctx(["2C", "3D"], ["AS", "KS", "QS", "JH", "TH"], call=0, pot=200)
        result = guard_rail.run_pre(ctx)
        assert result is not None
        assert result[0][0] == "check"
        assert result[1] == "board_made_hand_guard"

    def test_real_hand_silent(self):
        ctx = _ctx(["AH", "KH"], ["7S", "9D", "KH", "2C", "3H"], call=0, pot=200)
        result = guard_rail.run_pre(ctx)
        assert result is None or result[1] != "board_made_hand_guard"

    def test_folds_expensive_bet(self):
        # Board-made hand facing a 50% pot bet -> fold
        ctx = _ctx(["2C", "3D"], ["AS", "KS", "QS", "JH", "TH"], call=200, pot=400)
        result = guard_rail.run_pre(ctx)
        assert result is not None
        assert result[0][0] == "fold"
        assert result[1] == "board_made_hand_guard"


class TestSprCommitmentLock:
    def test_calls_pot_committed_two_pair(self):
        # Two pair, very low SPR -> call
        ctx = _ctx(
            ["3C", "3D"], ["7S", "9D", "3H", "2C", "9H"], call=1900, pot=100, stack=2000
        )
        result = guard_rail.run_pre(ctx)
        assert result is not None
        assert result[0][0] == "call"
        assert result[1] == "spr_commitment_lock"

    def test_silent_for_one_pair(self):
        # One pair, not strong enough
        ctx = _ctx(["KH", "3C"], ["7S", "9D", "KH", "2C", "9H"], call=100, pot=400)
        result = guard_rail.run_pre(ctx)
        assert result is None or result[1] != "spr_commitment_lock"

    def test_silent_preflop(self):
        ctx = _ctx(["AH", "AH"], [], call=5, pot=10, street="Preflop")
        result = guard_rail.run_pre(ctx)
        assert result is None or result[1] != "spr_commitment_lock"
