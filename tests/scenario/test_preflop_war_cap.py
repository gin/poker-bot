"""Scenario: preflop raise-war escalation cap (table cmrfzix0fzabei1x8nvtd01br).

Tournament s6-tight, 2026-07-11: hero JJ 3-bet ("anti-bully value backraise"),
then faced a 4-bet AND a cold 5-bet (QQ and AJ both re-raising) and JAMMED
for -3337. Live data across 3 DBs: AA/KK in 3+ raise wars average +894 to
+3541; QQ/JJ average -79, collapsing to -1504 in 5+ raise wars.

Rule: past 3 TOTAL preflop raises AND >= 2 distinct non-hero raisers (the
pool gate rejected total-count alone at -30 to -43 bb/100: sim bots 4-bet
junk, so a single re-raiser means nothing), only AA/KK keeps escalating;
everything else calls at a sliver price (<= 10%) or folds.
"""

from poker_bot.guards.context import GuardContext
from poker_bot.guards.guard_post import _preflop_war_level, guard_post


def _spot(hole, *, call, pot, raises, stack=3332):
    seat = {
        "seatNumber": 4,
        "agentId": "hero",
        "holeCards": hole,
        "stackChips": stack,
        "currentBetChips": 425,
    }
    history = [
        {"agentId": f"opp{i}", "action": "raise", "street": "Preflop"}
        for i in range(raises)
    ]
    table = {
        "street": "Preflop",
        "boardCards": [],
        "potChips": pot,
        "currentBet": call + 425,
        "bigBlindChips": 25,
        "actionHistory": history,
        "seats": [
            seat,
            {"seatNumber": 1, "agentId": "opp0", "stackChips": 9000,
             "currentBetChips": call + 425, "holeCards": []},
            {"seatNumber": 2, "agentId": "opp1", "stackChips": 8000,
             "currentBetChips": 650, "holeCards": []},
        ],
        "allowedActions": {
            "availableActions": ["fold", "call", "raise", "all-in"],
            "callAmount": call,
            "callChips": call,
            "minBet": 25,
            "minRaiseTo": (call + 425) * 2,
            "raiseRange": {"min": (call + 425) * 2, "max": stack},
        },
    }
    return GuardContext.build(table, seat)


class TestWarCounting:
    def test_counts_everyone_from_action_history(self):
        ctx = _spot(["Js", "Jc"], call=2049, pot=3534, raises=5)
        assert _preflop_war_level(ctx.table, "hero") == (5, 5)

    def test_hero_raises_count_toward_total_not_raisers(self):
        ctx = _spot(["Js", "Jc"], call=2049, pot=3534, raises=2)
        ctx.table["actionHistory"].append(
            {"agentId": "hero", "action": "raise", "street": "Preflop"}
        )
        assert _preflop_war_level(ctx.table, "hero") == (3, 2)

    def test_counts_recent_events_source(self):
        ctx = _spot(["Js", "Jc"], call=2049, pot=3534, raises=0)
        ctx.table["recentEvents"] = [
            {"type": "ActionTaken", "street": "Preflop",
             "summary": {"seatNumber": 1, "action": "raise", "amount": 225}},
            {"type": "ActionTaken", "street": "Preflop",
             "summary": {"seatNumber": 2, "action": "raise", "amount": 625}},
        ]
        assert _preflop_war_level(ctx.table, "hero") == (2, 2)

    def test_single_maniac_reraising_never_caps(self):
        # One opponent re-raising four times: distinct raisers == 1, so the
        # cap must not fire — folding value to a lone maniac burned the
        # pools at -30 to -43 bb/100.
        ctx = _spot(["Js", "Jc"], call=2049, pot=3534, raises=0)
        ctx.table["actionHistory"] = [
            {"agentId": "opp0", "action": "raise", "street": "Preflop"}
            for _ in range(4)
        ]
        decision, guard_id = guard_post.run_post(ctx, ("raise", 3337, "jj"))
        assert guard_id != "preflop_min_raise_war_cap"


class TestWarCap:
    def test_jj_jam_into_five_raise_war_becomes_fold(self):
        # The actual -3337 decision: JJ facing 2049 into 3534 (37% required)
        # at war level 5. Core proposed raise-to-3337 (all-in backraise).
        ctx = _spot(["Js", "Jc"], call=2049, pot=3534, raises=5)
        decision, guard_id = guard_post.run_post(
            ctx, ("raise", 3337, "anti-bully value backraise JJ")
        )
        assert guard_id == "preflop_min_raise_war_cap"
        assert decision[0] == "fold"

    def test_aa_keeps_escalating(self):
        ctx = _spot(["As", "Ac"], call=2049, pot=3534, raises=5)
        decision, guard_id = guard_post.run_post(
            ctx, ("raise", 3337, "anti-bully value backraise AA")
        )
        assert guard_id != "preflop_min_raise_war_cap"

    def test_kk_keeps_escalating(self):
        ctx = _spot(["Kd", "Kh"], call=2049, pot=3534, raises=4)
        decision, guard_id = guard_post.run_post(ctx, ("raise", 3337, "kk"))
        assert guard_id != "preflop_min_raise_war_cap"

    def test_early_war_untouched(self):
        # JJ's initial 3-bet (war level 2) is fine and stays.
        ctx = _spot(["Js", "Jc"], call=220, pot=265, raises=2)
        decision, guard_id = guard_post.run_post(ctx, ("raise", 425, "3bet"))
        assert guard_id != "preflop_min_raise_war_cap"

    def test_cheap_war_price_calls_instead_of_folding(self):
        # War 3+ but the remaining price is cheap (<= 22% required):
        # multiway set-mining continues as a call, never a re-raise.
        ctx = _spot(["Js", "Jc"], call=100, pot=3534, raises=4)
        decision, guard_id = guard_post.run_post(ctx, ("raise", 900, "jj"))
        assert guard_id == "preflop_min_raise_war_cap"
        assert decision[0] == "call"

    def test_core_calls_and_folds_stand(self):
        ctx = _spot(["Js", "Jc"], call=2049, pot=3534, raises=5)
        for proposed in (("call", 2049, "core call"), ("fold", None, "core fold")):
            decision, guard_id = guard_post.run_post(ctx, proposed)
            assert guard_id != "preflop_min_raise_war_cap"
