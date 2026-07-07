"""Liveness must work under BOTH seat schemas (2026-07-06 audit).

The arena sends seat `status` and no `folded` flags; the simulator sends
`folded` flags and no `status`. Reading only one convention made liveness
wrong in exactly one environment (live telemetry showed active == seated
on every arena turn/river decision). These tests pin the dual-schema
behavior of hand_utils and of the shipped cores' private copies.
"""

import pytest

from poker_bot import hand_utils
from poker_bot.strategies import hubase, s5base

MODULES = [hand_utils, hubase, s5base]


def _table(seats):
    return {"seats": seats}


@pytest.mark.parametrize("mod", MODULES, ids=lambda m: m.__name__)
class TestSeatIsLive:
    def test_simulator_schema_folded_flag(self, mod):
        assert mod.seat_is_live({"agentId": "a"}) is True
        assert mod.seat_is_live({"agentId": "a", "folded": True}) is False
        assert mod.seat_is_live({"agentId": "a", "hasFolded": True}) is False

    def test_arena_schema_status(self, mod):
        assert mod.seat_is_live({"agentId": "a", "status": "Active"}) is True
        assert mod.seat_is_live({"agentId": "a", "status": "Folded"}) is False
        assert mod.seat_is_live({"agentId": "a", "status": "SittingOut"}) is False
        assert mod.seat_is_live({"agentId": "a", "status": "Waiting"}) is False

    def test_unknown_status_stays_live(self, mod):
        # Blocklist semantics: an all-in player must stay counted as live.
        assert mod.seat_is_live({"agentId": "a", "status": "AllIn"}) is True


@pytest.mark.parametrize("mod", MODULES, ids=lambda m: m.__name__)
def test_active_opponents_arena_schema(mod):
    hero = {"agentId": "hero", "seatNumber": 1, "status": "Active"}
    table = _table(
        [
            hero,
            {"agentId": "o1", "seatNumber": 2, "status": "Active"},
            {"agentId": "o2", "seatNumber": 3, "status": "Folded"},
            {"agentId": "o3", "seatNumber": 4, "status": "Folded"},
            {"agentId": "o4", "seatNumber": 5, "status": "SittingOut"},
        ]
    )
    # Pre-fix this returned 4 (nobody ever "folds" in arena schema).
    assert mod.active_opponents(table, hero) == 1
    assert len(mod.live_opponent_seats(table, hero)) == 1
