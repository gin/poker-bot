"""Unit tests for the guards package: GuardContext + GuardRail registry."""

from poker_bot.guards.context import GuardContext
from poker_bot.guards.registry import GuardRail
from poker_bot.hand_utils import (
    OpponentProfile,
    is_tight_opponent,
    profile_vpip_frequency,
)


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
        ctx_hu = GuardContext.build(t_hu, s_hu)
        assert ctx_hu.is_heads_up is True
        assert ctx_hu.regime == "heads_up"
        assert ctx_hu.num_dealt_in == 2
        t_6m, s_6m = _make_table(["AH", "KD"], ["7S", "9D", "KH"], players=6)
        ctx_6m = GuardContext.build(t_6m, s_6m)
        assert ctx_6m.is_heads_up is False
        assert ctx_6m.regime == "full_table"
        assert ctx_6m.num_dealt_in == 6
        assert ctx_6m.num_active_opponents == 5

    def test_three_handed_regime(self):
        t, s = _make_table(["AH", "KD"], ["7S", "9D", "KH"], players=3)
        ctx = GuardContext.build(t, s)
        assert ctx.is_heads_up is False
        assert ctx.regime == "three_handed"
        assert ctx.num_dealt_in == 3

    def test_empty_numbered_seat_slot_does_not_inflate_active_counts(self):
        # Regression: a numbered seat with no agentId (an empty/unfilled
        # slot) must never count as a live player. active_seat_numbers and
        # active_opponents used to only check seat_is_live (folded/status),
        # not agentId, so an empty {agentId: None, seatNumber: 3} slot with
        # no dead status/folded flag was misread as a third live player in
        # an otherwise heads-up table -- num_active=3, num_active_opponents=2
        # instead of the true 2/1. count_dealt_in_players already required
        # a truthy agentId; active_seat_numbers/active_opponents now match.
        table = {
            "street": "River",
            "boardCards": ["7S", "9D", "KH"],
            "potChips": 100,
            "buttonSeatNumber": 1,
            "seats": [
                {
                    "seatNumber": 1,
                    "agentId": "hero",
                    "holeCards": ["AH", "KD"],
                    "stackChips": 2000,
                    "currentBetChips": 0,
                    "folded": False,
                    "hasFolded": False,
                },
                {
                    "seatNumber": 2,
                    "agentId": "villain",
                    "holeCards": [],
                    "stackChips": 2000,
                    "currentBetChips": 0,
                    "folded": False,
                    "hasFolded": False,
                },
                {
                    "seatNumber": 3,
                    "agentId": None,
                    "holeCards": [],
                    "stackChips": 0,
                    "currentBetChips": 0,
                },
            ],
            "opponentProfiles": {},
            "allowedActions": {
                "availableActions": ["fold", "check", "bet", "all-in"],
                "callAmount": 0,
                "callChips": 0,
                "raiseRange": {"min": 4, "max": 4000},
            },
        }
        ctx = GuardContext.build(table, table["seats"][0])
        assert ctx.regime == "heads_up"
        assert ctx.is_heads_up is True
        assert ctx.num_active == 2
        assert ctx.num_active_opponents == 1

    def test_opponent_profile(self):
        p = OpponentProfile(
            agent_id="vill",
            hands_seen=20,
            preflop_hands_seen=20,
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

    def test_canonical_mapping_preserves_trusted_explicit_vpip_frequency(self):
        profile = {
            "hands_seen": 20,
            "preflop_hands_seen": 20,
            "profile_stats_schema_version": 2,
            "profile_stats_provenance": "canonical",
            "vpip": 2,
            "vpip_frequency": 0.4,
        }

        assert profile_vpip_frequency(profile) == 0.4
        t, _ = _make_table(["AH", "KD"], ["7S", "9D", "KH"], profiles={"vill": profile})
        assert is_tight_opponent(t) is False

    def test_canonical_object_uses_preflop_denominator_for_vpip_frequency(self):
        profile = OpponentProfile(
            agent_id="vill",
            hands_seen=20,
            preflop_hands_seen=10,
            vpip=3,
        )

        assert profile_vpip_frequency(profile) == 0.3
        t, _ = _make_table(["AH", "KD"], ["7S", "9D", "KH"], profiles={"vill": profile})
        assert is_tight_opponent(t, dict_only=False) is False

    def test_legacy_untrusted_vpip_never_marks_guard_context_tight(self):
        profile = {
            "hands_seen": 20,
            "vpip": 0,
            "vpip_frequency": 0.0,
            "fold_to_bet": 20,
            "opportunities_to_fold_to_bet": 20,
            "calls": 1,
            "profile_stats_provenance": "legacy_untrusted",
        }
        t, s = _make_table(["AH", "KD"], ["7S", "9D", "KH"], profiles={"vill": profile})

        ctx = GuardContext.build(t, s)

        assert profile_vpip_frequency(profile) == 0.0
        assert is_tight_opponent(t, use_frequency_signal=True) is False
        assert ctx.opponent_vpip == 0.0
        assert ctx.opponent_is_tight is False

    def test_draws(self):
        t, s = _make_table(["AH", "2H"], ["7H", "9D", "KH"])
        ctx = GuardContext.build(t, s)
        assert ctx.has_flush_draw is True


class TestGuardRail:
    def test_pre_guard_fires(self):
        rail = GuardRail()

        @rail.register("test_pre", "pre", 0, ["heads_up"], "Test")
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

        @rail.register("test_pre", "pre", 0, ["heads_up"], "Test")
        def guard(ctx):
            return None

        t, s = _make_table(["AH", "KD"], ["7S", "9D", "KH"])
        ctx = GuardContext.build(t, s)
        assert rail.run_pre(ctx) is None

    def test_post_guard_overrides(self):
        rail = GuardRail()

        @rail.register("cap_raise", "post", 10, ["all"], "Cap raises")
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

        @rail.register("noop", "post", 10, ["all"], "No-op")
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

        @rail.register("general", "post", 20, ["all"], "General")
        def general(ctx, proposed):
            order.append("general")
            return None

        @rail.register("specific", "post", 5, ["all"], "Specific")
        def specific(ctx, proposed):
            order.append("specific")
            return None

        t, s = _make_table(["AH", "KD"], ["7S", "9D", "KH"])
        ctx = GuardContext.build(t, s)
        rail.run_post(ctx, ("check", None, "core"))
        assert order == ["specific", "general"]

    def test_regime_filtering(self):
        rail = GuardRail()
        fired = []

        @rail.register("hu_only", "pre", 0, ["heads_up"], "HU only")
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

    def test_three_handed_regime_filtering(self):
        rail = GuardRail()
        fired = []

        @rail.register("hu_only", "pre", 0, ["heads_up"], "HU only")
        def hu_only(ctx):
            fired.append("hu")
            return None

        @rail.register("full_table_only", "pre", 1, ["full_table"], "Full table only")
        def full_table_only(ctx):
            fired.append("full_table")
            return None

        @rail.register("unrestricted", "pre", 2, ["all"], "Unrestricted")
        def unrestricted(ctx):
            fired.append("all")
            return None

        t, s = _make_table(["AH", "KD"], ["7S", "9D", "KH"], players=3)
        ctx = GuardContext.build(t, s)
        assert ctx.regime == "three_handed"
        rail.run_pre(ctx)
        assert fired == ["all"]
