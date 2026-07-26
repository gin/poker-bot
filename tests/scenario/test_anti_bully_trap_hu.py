"""Scenario: port of the anti-bully slow-play-trap fix into the HU cores.

Reference incident (table cmrfrslyinvmqi1x8qjeqm165, 2026-07-10): batu_sabar
raised 2s3s preflop, TURNED a flush on Ks Qh 4s Qs and CHECK-RAISED it; the
survival_sixmax "anti-bully bluff catch" called with ACE-HIGH (the board's
queens read as 'medium'), then "anti-bully value raise rank 2" 3-bet the
river check-raise. Net -2660. survival_sixmax.py was fixed with two gates:
1. medium/strong require a hole card to PARTICIPATE in the made hand (a
   board-owned pair/trips is shared strength, not a bluff-catcher);
2. a check-raise (trap line) downgrades anti-bully: no value raise, tighter
   call price, no bluff catch.

hubase.py and hutight001.py embed their own near-copy of this logic
(`_sixmax_anti_bully_action`) with their OWN strength thresholds -- medium:
made_rank in {1, 2}; strong: made_rank >= 3 (vs survival_sixmax's medium:
made_rank == 1; strong: made_rank >= 2). This suite ports the SAME two
fixes on top of those thresholds and mirrors
tests/scenario/test_anti_bully_trap.py.
"""

import importlib

import pytest

MODULE_NAMES = ("poker_bot.strategies.hubase", "poker_bot.strategies.hutight001")

# board pair only -- AJ never touches a queen. made_hand_rank == 1 (pair of
# queens), but hero's hole cards don't participate.
TURN_BOARD = ["Ks", "Qh", "4s", "Qs"]
# flop trips fully in hero's hand (pocket 7s on a 7-7 board -> quads).
# made_hand_rank == 7 and participates -- clears hubase's strong >= 3 gate.
TRIPS_BOARD = ["7h", "7d", "2c"]


@pytest.fixture(params=MODULE_NAMES)
def strat(request):
    return importlib.import_module(request.param)


def _table(board, call, pot, *, raiser_checked, hero_hole, hero_stack=2500,
           villain_stack=9000, street=None, available=("fold", "call", "raise")):
    """HU table: villain is a big stack pressuring hero (triggers
    bully_context's stack-ratio branch) and is the one facing/raising.
    """
    hero = {
        "seatNumber": 1,
        "agentId": "hero",
        "holeCards": hero_hole,
        "stackChips": hero_stack,
        "currentBetChips": 0,
        "folded": False,
        "hasFolded": False,
    }
    villain = {
        "seatNumber": 2,
        "agentId": "villain",
        "holeCards": [],
        "stackChips": villain_stack,
        "currentBetChips": call,
        "folded": False,
        "hasFolded": False,
    }
    if street is None:
        street = {3: "Flop", 4: "Turn", 5: "River"}[len(board)]
    history = []
    if raiser_checked:
        history.append({"agentId": "villain", "action": "check", "street": street})
    history.append({"agentId": "villain", "action": "raise", "street": street})
    return {
        "street": street,
        "boardCards": board,
        "potChips": pot,
        "currentBet": call,
        "actionHistory": history,
        "seats": [hero, villain],
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": list(available),
            "callAmount": call,
            "callChips": call,
            "minBet": 2,
            "minRaiseTo": call * 2 if call else 4,
            "raiseRange": {"min": call * 2 if call else 4, "max": hero_stack},
        },
    }


class TestHoleParticipation:
    def test_board_pair_is_not_private_strength(self, strat):
        assert strat._sixmax_hole_participates(["Jd", "Ad"], TURN_BOARD) is False

    def test_river_ace_pairs_hero(self, strat):
        river_board = TURN_BOARD + ["Ac"]
        assert strat._sixmax_hole_participates(["Jd", "Ad"], river_board) is True

    def test_pocket_pair_participates(self, strat):
        assert strat._sixmax_hole_participates(["9h", "9s"], TURN_BOARD) is True


class TestTrapLineDetection:
    def test_check_raise_detected_via_action_history(self, strat):
        table = _table(TURN_BOARD, 401, 731, raiser_checked=True,
                       hero_hole=["Jd", "Ad"])
        assert strat._sixmax_raiser_checked_this_street(table) is True

    def test_plain_raise_is_not_a_trap(self, strat):
        table = _table(TURN_BOARD, 401, 731, raiser_checked=False,
                       hero_hole=["Jd", "Ad"])
        assert strat._sixmax_raiser_checked_this_street(table) is False

    def test_recent_events_source_also_works(self, strat):
        table = _table(TURN_BOARD, 401, 731, raiser_checked=False,
                       hero_hole=["Jd", "Ad"])
        table["actionHistory"] = []
        table["recentEvents"] = [
            {"type": "ActionTaken", "street": "Turn",
             "summary": {"seatNumber": 2, "action": "check"}},
        ]
        assert strat._sixmax_raiser_checked_this_street(table) is True


class TestDisasterDecisions:
    def test_board_pair_ace_high_no_longer_bluff_catches(self, strat):
        # Untrapped, isolating the participation gate: required (~0.17) is
        # comfortably under the old 0.24 bluff-catch threshold, and the old
        # `medium = made_rank in {1, 2} or top_pair` read this as medium
        # purely off the board's queens. The fix requires a hole card to
        # participate, so AJ (rank 1, no participation) must not catch.
        table = _table(TURN_BOARD, 150, 731, raiser_checked=False,
                       hero_hole=["Jd", "Ad"])
        result = strat._sixmax_anti_bully_action(table, table["seats"][0])
        assert result is None or "bluff catch" not in result[2]

    def test_board_pair_ace_high_gone_when_trapped_too(self, strat):
        # The actual disaster line: check-raise on the turn. Old code called
        # "anti-bully bluff catch". Now: no participation -> not even
        # medium, and trapped forbids the tiny-price fallback too.
        table = _table(TURN_BOARD, 100, 731, raiser_checked=True,
                       hero_hole=["Jd", "Ad"])
        result = strat._sixmax_anti_bully_action(table, table["seats"][0])
        assert result is None

    def test_check_raise_is_never_value_raised(self, strat):
        # Hero has real quads (participates, clears strong >= 3) but the
        # villain check-raised -- a slow-played monster, not a bully. The
        # value-raise branch must be skipped; at most a call survives.
        table = _table(TRIPS_BOARD, 40, 200, raiser_checked=True,
                       hero_hole=["7s", "7c"])
        result = strat._sixmax_anti_bully_action(table, table["seats"][0])
        assert result is None or result[0] != "raise"

    def test_untrapped_value_raise_survives(self, strat):
        # Same real quads, but plain aggression (no check first): the +EV
        # untrapped value-raise behavior must be preserved exactly.
        table = _table(TRIPS_BOARD, 40, 200, raiser_checked=False,
                       hero_hole=["7s", "7c"])
        result = strat._sixmax_anti_bully_action(table, table["seats"][0])
        assert result is not None
        assert result[0] == "raise"
        assert "value raise" in result[2]
