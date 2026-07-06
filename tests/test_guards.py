"""Unit tests for the guards package: GuardContext + GuardRail registry."""

from poker_bot.guards.context import GuardContext
from poker_bot.guards.registry import GuardRail
from poker_bot.hand_utils import OpponentProfile


def _make_table(
    hole, board, *, call=0, pot=100, players=2, profiles=None, street="River"
):
    seats = [
        {
            "seatNumber": i,
            "agentId": "hero" if i == 1 else f"opp{i}",
            "holeCards": hole if i == 1 else [],
            "stackChips": 2000,
            "currentBetChips": 0,
            "folded": False,
            "hasFolded": False,
        }
        for i in range(1, players + 1)
    ]
    return {
        "street": street,
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": 1,
        "seats": seats,
        "opponentProfiles": profiles or {},
        "allowedActions": {
            "availableActions": ["fold", "call", "raise", "all-in"]
            if call > 0
            else ["fold", "check", "bet", "all-in"],
            "callAmount": call,
            "callChips": call,
            "raiseRange": {"min": max(call * 2, 4), "max": 4000},
        },
    }, seats[0]


class TestGuardContext:
    def test_basic_hand_eval(self):
        # AH KH on QH JH TH 2C 3D -> royal flush (straight flush)
        t, s = _make_table(["AH", "KH"], ["QH", "JH", "TH", "2C", "3D"])
        ctx = GuardContext.build(t, s)
        assert ctx.made_rank == 8
        assert ctx.hand_rank[0] == 8

    def test_two_pair_fragility(self):
        t, s = _make_table(["3D", "3S"], ["AS", "KS", "TC", "KH", "QS"])
        ctx = GuardContext.build(t, s)
        assert ctx.made_rank == 2
        assert ctx.is_fragile_two_pair is True

    def test_facing_bet(self):
        t, s = _make_table(["AH", "KD"], ["7S", "9D", "KH"], call=100, pot=400)
        ctx = GuardContext.build(t, s)
        assert ctx.facing_bet is True
        assert ctx.call_price == 100
        assert abs(ctx.pot_odds - 0.2) < 1e-6

    def test_not_facing_bet(self):
        t, s = _make_table(["AH", "KD"], ["7S", "9D", "KH"], call=0, pot=100)
        ctx = GuardContext.build(t, s)
        assert ctx.facing_bet is False
        assert ctx.pot_odds is None

    def test_heads_up_vs_6max(self):
        t_hu, s_hu = _make_table(["AH", "KD"], ["7S", "9D", "KH"], players=2)
        assert GuardContext.build(t_hu, s_hu).is_heads_up is True
        t_6m, s_6m = _make_table(["AH", "KD"], ["7S", "9D", "KH"], players=6)
        ctx_6m = GuardContext.build(t_6m, s_6m)
        assert ctx_6m.is_heads_up is False
        assert ctx_6m.num_active_opponents == 5

    def test_opponent_profile(self):
        p = OpponentProfile(
            agent_id="vill",
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
        t, s = _make_table(["AH", "KD"], ["7S", "9D", "KH"], profiles={"vill": p})
        ctx = GuardContext.build(t, s)
        assert ctx.opponent_hands_seen == 20
        assert ctx.opponent_is_bluffy is True
        assert abs(ctx.opponent_vpip - 0.5) < 1e-6

    def test_draws(self):
        t, s = _make_table(["AH", "2H"], ["7H", "9D", "KH"])
        ctx = GuardContext.build(t, s)
        assert ctx.has_flush_draw is True


class TestGuardRail:
    def test_pre_guard_fires(self):
        rail = GuardRail()

        @rail.register("test_pre", "pre", 0, ["hu"], "Test")
        def guard(ctx):
            if ctx.made_rank == 2:
                return ("fold", None, "test: fold two pair")
            return None

        t, s = _make_table(["3D", "3S"], ["AS", "KS", "TC", "KH", "QS"])
        ctx = GuardContext.build(t, s)
        result = rail.run_pre(ctx)
        assert result is not None
        decision, guard_id = result
        assert decision[0] == "fold"
        assert guard_id == "test_pre"

    def test_pre_guard_skips_non_matching(self):
        rail = GuardRail()

        @rail.register("test_pre", "pre", 0, ["hu"], "Test")
        def guard(ctx):
            return None

        t, s = _make_table(["AH", "KD"], ["7S", "9D", "KH"])
        ctx = GuardContext.build(t, s)
        assert rail.run_pre(ctx) is None

    def test_post_guard_overrides(self):
        rail = GuardRail()

        @rail.register("cap_raise", "post", 10, ["hu", "6max"], "Cap raises")
        def guard(ctx, proposed):
            if proposed[0] == "raise":
                return ("call", ctx.call_price, "capped to call")
            return None

        t, s = _make_table(["AH", "KD"], ["7S", "9D", "KH"], call=100, pot=400)
        ctx = GuardContext.build(t, s)
        result = rail.run_post(ctx, ("raise", 400, "core: raise"))
        assert result[0][0] == "call"
        assert result[1] == "cap_raise"

    def test_post_guard_approves(self):
        rail = GuardRail()

        @rail.register("noop", "post", 10, ["hu", "6max"], "No-op")
        def guard(ctx, proposed):
            return None

        t, s = _make_table(["AH", "KD"], ["7S", "9D", "KH"])
        ctx = GuardContext.build(t, s)
        result = rail.run_post(ctx, ("check", None, "core: check"))
        assert result[0] == ("check", None, "core: check")
        assert result[1] == "approved"

    def test_precedence_ordering(self):
        rail = GuardRail()
        order = []

        @rail.register("general", "post", 20, ["hu", "6max"], "General")
        def general(ctx, proposed):
            order.append("general")
            return None

        @rail.register("specific", "post", 5, ["hu", "6max"], "Specific")
        def specific(ctx, proposed):
            order.append("specific")
            return None

        t, s = _make_table(["AH", "KD"], ["7S", "9D", "KH"])
        ctx = GuardContext.build(t, s)
        rail.run_post(ctx, ("check", None, "core"))
        assert order == ["specific", "general"]

    def test_table_size_filtering(self):
        rail = GuardRail()
        fired = []

        @rail.register("hu_only", "pre", 0, ["hu"], "HU only")
        def hu_only(ctx):
            fired.append("hu")
            return None

        t, s = _make_table(["AH", "KD"], ["7S", "9D", "KH"], players=2)
        ctx = GuardContext.build(t, s)
        rail.run_pre(ctx)
        assert "hu" in fired

        fired.clear()
        t, s = _make_table(["AH", "KD"], ["7S", "9D", "KH"], players=6)
        ctx = GuardContext.build(t, s)
        rail.run_pre(ctx)
        assert "hu" not in fired
