"""Scenario: anti-bully vs slow-play traps (table cmrfrslyinvmqi1x8qjeqm165).

Tournament s6-tight, 2026-07-10: batu_sabar raised 2s3s preflop, TURNED a
flush on Ks Qh 4s Qs and CHECK-RAISED it; hero AJo's "anti-bully bluff
catch" called 330 with ACE-HIGH (the board's queens read as 'medium'), then
"anti-bully value raise rank 2" 3-bet the river check-raise with two pair
into a three-flush paired board. Net -2660, the whole stack.

Live-data verdict on anti-bully by street (3 DBs combined): preflop +504
(n=70) and flop +199/+4133 EARN — keep them; turn fires averaged -732
(bluff catch) and -392 (value raise). The fixes are surgical:
1. medium/strong require a hole card to participate (board-owned ranks are
   shared strength, not bluff-catchers);
2. a check-raise (trap line) downgrades anti-bully: no raises, strong-only
   calls at a tighter price, no bluff catches.
"""

from poker_bot.strategies.survival_sixmax import (
    _hole_participates,
    _raiser_checked_this_street,
    _raiser_is_trap_prone,
    anti_bully_action,
)

TURN_BOARD = ["Ks", "Qh", "4s", "Qs"]
RIVER_BOARD = ["Ks", "Qh", "4s", "Qs", "Ac"]


def _table(board, call, pot, *, raiser_checked, hero_hole, street=None):
    hero = {
        "seatNumber": 5,
        "agentId": "hero",
        "holeCards": hero_hole,
        "stackChips": 2500,
        "currentBetChips": 0,
    }
    villain = {
        "seatNumber": 2,
        "agentId": "batu",
        "stackChips": 9000,  # trap winnings: triggers bully_context stack ratio
        "currentBetChips": call,
        "holeCards": [],
    }
    street = street or {3: "Flop", 4: "Turn", 5: "River"}[len(board)]
    history = []
    if raiser_checked:
        history.append({"agentId": "batu", "action": "check", "street": street})
    history.append({"agentId": "batu", "action": "raise", "street": street})
    return {
        "street": street,
        "boardCards": board,
        "potChips": pot,
        "currentBet": call,
        "actionHistory": history,
        "seats": [hero, villain],
        "allowedActions": {
            "availableActions": ["fold", "call", "raise"],
            "callAmount": call,
            "callChips": call,
            "minBet": 20,
            "minRaiseTo": call * 2,
            "raiseRange": {"min": call * 2, "max": 2500},
        },
    }


class TestHoleParticipation:
    def test_board_pair_is_not_private_strength(self):
        assert _hole_participates(["Jd", "Ad"], TURN_BOARD) is False

    def test_river_ace_pairs_hero(self):
        assert _hole_participates(["Jd", "Ad"], RIVER_BOARD) is True

    def test_pocket_pair_participates(self):
        assert _hole_participates(["9h", "9s"], TURN_BOARD) is True


class TestTrapLineDetection:
    def test_check_raise_detected(self):
        table = _table(TURN_BOARD, 401, 731, raiser_checked=True,
                       hero_hole=["Jd", "Ad"])
        assert _raiser_checked_this_street(table) is True

    def test_plain_raise_is_not_a_trap(self):
        table = _table(TURN_BOARD, 401, 731, raiser_checked=False,
                       hero_hole=["Jd", "Ad"])
        assert _raiser_checked_this_street(table) is False

    def test_recent_events_source_also_works(self):
        table = _table(TURN_BOARD, 401, 731, raiser_checked=False,
                       hero_hole=["Jd", "Ad"])
        table["recentEvents"] = [
            {"type": "ActionTaken", "street": "Turn",
             "summary": {"seatNumber": 2, "action": "check"}},
        ]
        assert _raiser_checked_this_street(table) is True


class TestTrapProneRead:
    """Mined per-opponent knob: wake-up aggression >= 70% real value by
    revealed cards -> even a PLAIN raise is treated as a trap."""

    def test_trap_prone_playbook_detected(self):
        table = _table(RIVER_BOARD, 400, 1061, raiser_checked=False,
                       hero_hole=["Ah", "Kd"])
        table["opponentProfiles"] = {
            "batu": {"hands_seen": 40, "playbook": {"trap_prone": True}}
        }
        assert _raiser_is_trap_prone(table) is True

    def test_plain_raise_from_trap_prone_is_never_value_raised(self):
        # Without the read this exact spot value-raises (see
        # test_untrapped_value_raise_survives); the mined read flips it.
        table = _table(RIVER_BOARD, 400, 1061, raiser_checked=False,
                       hero_hole=["Ah", "Kd"])
        table["opponentProfiles"] = {
            "batu": {"hands_seen": 40, "playbook": {"trap_prone": True}}
        }
        result = anti_bully_action(table, table["seats"][0])
        assert result is None or result[0] != "raise"

    def test_no_playbook_no_read(self):
        table = _table(RIVER_BOARD, 400, 1061, raiser_checked=False,
                       hero_hole=["Ah", "Kd"])
        table["opponentProfiles"] = {"batu": {"hands_seen": 40}}
        assert _raiser_is_trap_prone(table) is False


class TestDisasterDecisions:
    def test_turn_ace_high_bluff_catch_is_gone(self):
        # The actual -2660 turn decision: AJ (no pair, board QQ) facing the
        # check-raise to 401. Old code: "anti-bully bluff catch" call 330.
        table = _table(TURN_BOARD, 330, 731, raiser_checked=True,
                       hero_hole=["Jd", "Ad"])
        result = anti_bully_action(table, table["seats"][0])
        assert result is None  # base logic decides (and folds air)

    def test_turn_ace_high_never_catches_even_untrapped(self):
        # Board-owned 'medium' must not bluff-catch regardless of line.
        table = _table(TURN_BOARD, 150, 731, raiser_checked=False,
                       hero_hole=["Jd", "Ad"])
        result = anti_bully_action(table, table["seats"][0])
        assert result is None or "bluff catch" not in result[2]

    def test_river_check_raise_is_never_reraised(self):
        # The actual river decision: two pair (aces up) facing the
        # check-raise to 2198. Old code: 3-bet to 4578. Now: call at most.
        table = _table(RIVER_BOARD, 1795, 3662, raiser_checked=True,
                       hero_hole=["Jd", "Ad"])
        result = anti_bully_action(table, table["seats"][0])
        assert result is None or result[0] != "raise"

    def test_untrapped_value_raise_survives(self):
        # Plain aggression from a big stack, hero holds REAL two pair:
        # the +EV flop/river value-raise behavior is preserved.
        table = _table(RIVER_BOARD, 400, 1061, raiser_checked=False,
                       hero_hole=["Ah", "Kd"])
        result = anti_bully_action(table, table["seats"][0])
        assert result is not None
        assert result[0] == "raise"
        assert "value raise" in result[2]

    def test_preflop_defend_untouched(self):
        # Cheap defend (10 into 40 = 20% <= the 24% gate): the +EV preflop
        # variant (+504 across 70 live fires) must keep working.
        table = _table([], 10, 40, raiser_checked=False,
                       hero_hole=["Jd", "Ad"], street="Preflop")
        result = anti_bully_action(table, table["seats"][0])
        assert result is not None
        assert "anti-bully" in result[2]
