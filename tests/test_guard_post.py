"""Tests for guard_post.py — the 6 ported hu009-hu012 guards."""

from poker_bot.guards.context import GuardContext
from poker_bot.guards.guard_post import guard_post, guard_rail
from poker_bot.hand_utils import OpponentProfile

HERO = "hero"
VILL = "vill"
FOLD_CALL = ("fold", "call", "raise", "all-in")
CHECK_BET = ("fold", "check", "bet", "all-in")


def test_guard_post_alias_matches_registry():
    assert guard_post is guard_rail


BLUFFY = OpponentProfile(
    agent_id=VILL,
    hands_seen=20,
    vpip=10,
    calls=3,
    bets=4,
    raises=2,
    folds=6,
    fold_to_bet=4,
    opportunities_to_fold_to_bet=8,
    showdowns=5,
    weak_aggressive_showdowns=2,
)
TIGHT = {
    "hands_seen": 20,
    "vpip": 3,
    "pfr": 2,
    "calls": 2,
    "bets": 1,
    "raises": 0,
    "folds": 14,
    "fold_to_bet": 10,
    "opportunities_to_fold_to_bet": 12,
}
VALUE = OpponentProfile(
    agent_id=VILL,
    hands_seen=20,
    vpip=3,
    calls=5,
    bets=1,
    raises=0,
    folds=12,
    fold_to_bet=8,
    opportunities_to_fold_to_bet=10,
    showdowns=5,
    weak_aggressive_showdowns=0,
)


def _ctx(
    hole,
    board,
    *,
    call=0,
    pot=400,
    players=2,
    profiles=None,
    street="River",
    stack=2000,
):
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
    actions = list(FOLD_CALL) if call > 0 else list(CHECK_BET)
    table = {
        "street": street,
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": 1,
        "seats": seats,
        "opponentProfiles": profiles or {},
        "allowedActions": {
            "availableActions": actions,
            "callAmount": call,
            "callChips": call,
            "raiseRange": {"min": max(call * 2, 4), "max": 4000},
        },
    }
    return GuardContext.build(table, seats[0])


class TestTwoPairPairedBoardOverfold:
    def test_genuine_two_pair_calls(self):
        ctx = _ctx(["4H", "4S"], ["9C", "KD", "TD", "9S"], call=150, pot=400)
        result = guard_rail.run_post(ctx, ("raise", 300, "core: raise"))
        assert result[0][0] == "call"
        assert result[1] == "two_pair_paired_board_overfold"

    def test_fragile_excluded_33_on_akkqq(self):
        # paired_board_pot_control fires: fragile two pair at 50% price -> fold
        ctx = _ctx(["3D", "3S"], ["AS", "KS", "TC", "KH", "QS"], call=3100, pot=3128)
        result = guard_rail.run_post(ctx, ("raise", 6200, "core: raise"))
        assert result[0][0] == "fold"
        assert result[1] == "paired_board_pot_control"

    def test_board_dominated_excluded(self):
        # paired_board_pot_control fires: board-dominated two pair
        ctx = _ctx(["5S", "5H"], ["KS", "KH", "QD", "QS", "3C"], call=100, pot=400)
        result = guard_rail.run_post(ctx, ("raise", 200, "core: raise"))
        assert result[0][0] == "fold"
        assert result[1] == "paired_board_pot_control"

    def test_6max_excluded(self):
        # paired_board_pot_control fires (all sizes)
        ctx = _ctx(["4H", "4S"], ["9C", "KD", "TD", "9S"], call=150, pot=400, players=6)
        result = guard_rail.run_post(ctx, ("raise", 300, "core: raise"))
        assert result[0][0] == "fold"
        assert result[1] == "paired_board_pot_control"

    def test_fold_action_not_overridden(self):
        ctx = _ctx(["4H", "4S"], ["9C", "KD", "TD", "9S"], call=150, pot=400)
        result = guard_rail.run_post(ctx, ("fold", None, "core: fold"))
        assert result[1] == "approved"


class TestFlopHuBluffcatch:
    def test_calls_vs_bluffy_dry_flop(self):
        ctx = _ctx(
            ["TC", "2D"],
            ["7S", "9D", "KH"],
            call=100,
            pot=400,
            profiles={VILL: BLUFFY},
            street="Flop",
        )
        result = guard_rail.run_post(ctx, ("fold", None, "core: fold"))
        assert result[0][0] == "call"
        assert result[1] == "flop_hu_bluffcatch"

    def test_silent_vs_value_heavy(self):
        ctx = _ctx(
            ["TC", "2D"],
            ["7S", "9D", "KH"],
            call=100,
            pot=400,
            profiles={VILL: VALUE},
            street="Flop",
        )
        result = guard_rail.run_post(ctx, ("fold", None, "core: fold"))
        assert result[1] == "approved"

    def test_turn_rank1_calls_vs_bluffy(self):
        ctx = _ctx(
            ["3C", "3D"],
            ["7S", "9D", "KH", "2C"],
            call=100,
            pot=400,
            profiles={VILL: BLUFFY},
            street="Turn",
        )
        result = guard_rail.run_post(ctx, ("fold", None, "core: fold"))
        assert result[0][0] == "call"

    def test_turn_high_card_excluded(self):
        ctx = _ctx(
            ["JC", "5D"],
            ["7S", "9D", "KH", "2C"],
            call=100,
            pot=400,
            profiles={VILL: BLUFFY},
            street="Turn",
        )
        result = guard_rail.run_post(ctx, ("fold", None, "core: fold"))
        assert result[1] == "approved"

    def test_wet_board_excluded(self):
        # 2C 3D on 7S 8S 9H (wet flop) has no draw -> no bluff-catch
        ctx = _ctx(
            ["2C", "3D"],
            ["7S", "8S", "9H"],
            call=100,
            pot=400,
            profiles={VILL: BLUFFY},
            street="Flop",
        )
        result = guard_rail.run_post(ctx, ("fold", None, "core: fold"))
        assert result[1] == "approved"


class TestRiverTwoPairFacingBetCall:
    def test_calls_on_paired_board(self):
        ctx = _ctx(["KH", "3C"], ["7S", "7D", "KH", "2C", "9H"], call=134, pot=403)
        result = guard_rail.run_post(ctx, ("fold", None, "core: fold"))
        assert result[0][0] == "call"
        assert result[1] == "river_two_pair_facing_bet_call"

    def test_fragile_excluded(self):
        ctx = _ctx(["3D", "3S"], ["AS", "KS", "TC", "KH", "QS"], call=134, pot=403)
        result = guard_rail.run_post(ctx, ("fold", None, "core: fold"))
        assert result[1] == "approved"

    def test_unpaired_board_excluded(self):
        ctx = _ctx(["KH", "3C"], ["7S", "9D", "KH", "2C", "3H"], call=134, pot=403)
        result = guard_rail.run_post(ctx, ("fold", None, "core: fold"))
        assert result[1] == "approved"

    def test_above_40pct_odds_excluded(self):
        ctx = _ctx(["KH", "3C"], ["7S", "7D", "KH", "2C", "9H"], call=300, pot=403)
        result = guard_rail.run_post(ctx, ("fold", None, "core: fold"))
        assert result[1] == "approved"
