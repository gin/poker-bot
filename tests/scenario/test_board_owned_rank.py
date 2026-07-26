"""Scenario: board-owned rank blindness (live table cmrdg5l9soa536xly5wnzzge6).

2026-07-09 playground: hero AJo, board 6c Tc 6h 6s. made_hand_rank read the
BOARD's trips as hero strength ("Value betting rank 3", then "Profiled call
rank 3" for the full stack) while 'friend' slow-played TT into tens-full.
Net: -957, entire stack.

Two failures, both pinned here:
1. is_board_made_or_kicker_vulnerable was river-only (len(board) == 5), so
   the turn disaster was invisible to the whole guard/context layer.
2. board_made_hand_guard (now the one ACTIVE guard) must check back the
   value-bet spot and fold the overbet shove; spr_commitment_lock's
   predicate must not endorse board-owned ranks.
"""

from poker_bot.guards.context import GuardContext
from poker_bot.guards.guard_post import guard_post
from poker_bot.guards.guard_pre import guard_pre
from poker_bot.hand_utils import is_board_made_or_kicker_vulnerable

DISASTER_TURN = ["6c", "Tc", "6h", "6s"]
HERO_AJ = ["Jd", "As"]


def _spot(board, call, pot, stack=945):
    seat = {
        "seatNumber": 5,
        "agentId": "hero",
        "holeCards": HERO_AJ,
        "stackChips": stack,
        "currentBetChips": 0,
    }
    actions = ["fold", "call"] if call else ["fold", "check", "bet"]
    # Mirror the real table: 6 seated, 4 folded, hero vs 'friend'. The guard
    # is full_table-gated (HU pool rejected it); folded seats still count as
    # dealt-in, so the hand's regime stays full_table.
    folded = [
        {"seatNumber": n, "agentId": f"folded{n}", "stackChips": 500,
         "currentBetChips": 0, "holeCards": [], "folded": True}
        for n in (1, 2, 4, 6)
    ]
    table = {
        "street": {3: "Flop", 4: "Turn", 5: "River"}[len(board)],
        "boardCards": board,
        "potChips": pot,
        "currentBet": call,
        "bigBlindChips": 4,
        "seats": [
            seat,
            {
                "seatNumber": 3,
                "agentId": "friend",
                "stackChips": 100,
                "currentBetChips": call,
                "holeCards": [],
            },
            *folded,
        ],
        "allowedActions": {
            "availableActions": actions,
            "callAmount": call,
            "callChips": call,
            "minBet": 4,
        },
    }
    return GuardContext.build(table, seat)


class TestBoardOwnedDetector:
    def test_turn_board_trips_with_air_is_board_owned(self):
        assert is_board_made_or_kicker_vulnerable(HERO_AJ, DISASTER_TURN) is True

    def test_flop_paired_board_with_air_is_board_owned(self):
        assert is_board_made_or_kicker_vulnerable(HERO_AJ, ["6c", "Tc", "6h"]) is True

    def test_the_trapper_has_private_strength(self):
        assert is_board_made_or_kicker_vulnerable(["Td", "Th"], DISASTER_TURN) is False

    def test_hole_card_participation_is_private(self):
        assert (
            is_board_made_or_kicker_vulnerable(["Ad", "6d"], ["6c", "Tc", "6h"])
            is False
        )

    def test_air_on_unpaired_board_is_not_board_owned(self):
        assert (
            is_board_made_or_kicker_vulnerable(["Kd", "Qh"], ["Tc", "4h", "2s"])
            is False
        )


class TestGuardOnTheDisasterHand:
    """board_made_hand_guard is a stack-off veto ONLY: it folds the core's
    expensive calls/raise-ins (>= 40% required equity) with board-owned rank.
    Bets, checks, and folds always stand — the broader versions (bet vetoes,
    33% threshold) failed the counterfactual pool gate."""

    def test_turn_value_bet_stands(self):
        # The core's "bet 12" (16 chips of the -957) is within its rights;
        # the pool gate showed vetoing such bets taxes real matchups.
        ctx = _spot(DISASTER_TURN, call=0, pot=20)
        decision, guard_id = guard_post.run_post(ctx, ("bet", 12, "value rank 3"))
        assert guard_id != "board_made_hand_guard"

    def test_turn_stack_off_call_becomes_fold_vs_value_heavy_read(self):
        # The core actually proposed "call 8792" ("Profiled call rank 3");
        # 8792 into 8836 requires ~50%, and 'friend' (a slow-player) reads
        # value-heavy (low wasd) — exactly the read-gated fold.
        ctx = _spot(DISASTER_TURN, call=8792, pot=8836, stack=8792)
        ctx.opponent_wasd = 0.10  # value-heavy: big bets are real
        ctx.opponent_is_bluffy = False
        decision, guard_id = guard_post.run_post(
            ctx, ("call", 8792, "profiled call rank 3")
        )
        assert guard_id == "board_made_hand_guard"
        assert decision[0] == "fold"

    def test_no_read_or_bluffy_read_call_stands(self):
        # Without a confident low-bluff read the core's call stands: vs
        # habitual shovers these spots win or split (sim gate proved the
        # unconditional fold taxes -4 to -5 bb/100).
        ctx = _spot(DISASTER_TURN, call=8792, pot=8836, stack=8792)
        assert ctx.opponent_wasd is None
        decision, guard_id = guard_post.run_post(ctx, ("call", 8792, "core"))
        assert guard_id != "board_made_hand_guard"
        ctx.opponent_wasd = 0.60  # bluff-heavy: calling is the exploit
        decision, guard_id = guard_post.run_post(ctx, ("call", 8792, "core"))
        assert guard_id != "board_made_hand_guard"

    def test_core_folds_always_stand(self):
        # The pre-guard version preempted smarter core folds; the post-guard
        # must never touch a proposed fold.
        ctx = _spot(DISASTER_TURN, call=8792, pot=8836, stack=8792)
        decision, guard_id = guard_post.run_post(ctx, ("fold", None, "core fold"))
        assert guard_id == "approved"
        assert decision[0] == "fold"

    def test_spr_lock_no_longer_endorses_board_owned_rank(self, monkeypatch):
        # Even fully activated, spr_commitment_lock must pass on this spot.
        monkeypatch.setenv("POKER_GUARD_ACTIVATE", "*")
        result = guard_pre.run_pre(_spot(DISASTER_TURN, call=8792, pot=8836, stack=8792))
        if result is not None:
            assert result[1] != "spr_commitment_lock"

    def test_real_hands_untouched(self):
        # The trapper's TT on the same board: guard must not veto real value.
        seat = {
            "seatNumber": 5,
            "agentId": "hero",
            "holeCards": ["Td", "Th"],
            "stackChips": 945,
            "currentBetChips": 0,
        }
        base = _spot(DISASTER_TURN, call=0, pot=20)
        table = {**base.table, "seats": [seat, base.table["seats"][1]]}
        ctx = GuardContext.build(table, seat)
        decision, guard_id = guard_post.run_post(ctx, ("bet", 12, "boat value"))
        assert guard_id != "board_made_hand_guard"
