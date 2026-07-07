"""Scenario: preflop call-war (guard_audit telemetry, 2026-07-06).

Observed in 6-max selfplay: hero call-fed preflop min-raise wars 2bb at a
time with non-premium hands until all-in (worst hands: ~50 calls of 20 chips
with A3s, losing the full stack). Root cause turned out to be the multi_core
router bug (simulator seats carry no 'status', so every decision went to the
heads-up core); with routing fixed the guard measured neutral and now runs
in SHADOW mode. These tests pin the guard's decision logic via its shadow
events, and pin that it does NOT override while shadowed.
"""

from poker_bot.guards.context import GuardContext
from poker_bot.guards.guard_pre import guard_rail
from poker_bot.guards.telemetry import clear_events, drain_events

HERO = "hero"
BLIND = 10


def _ctx(hole, *, committed, call, players=6, stack=1000):
    seats = [
        {
            "seatNumber": i,
            "agentId": HERO if i == 1 else f"opp{i}",
            "holeCards": hole if i == 1 else [],
            "stackChips": stack,
            "currentBetChips": committed if i == 1 else committed + call,
            "folded": False,
            "hasFolded": False,
        }
        for i in range(1, players + 1)
    ]
    table = {
        "street": "Preflop",
        "boardCards": [],
        "potChips": 0,  # simulator leaves potChips at 0 preflop (live bets)
        "bigBlindChips": BLIND,
        "buttonSeatNumber": 1,
        "seats": seats,
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": ["fold", "call", "raise", "all-in"],
            "callAmount": call,
            "callChips": call,
            "minBet": BLIND,
            "raiseRange": {"min": call * 2, "max": stack},
        },
    }
    return GuardContext.build(table, seats[0])


def setup_function(_func):
    clear_events()


def run_commit_cap(ctx):
    """Return the guard's shadow-fire event decision, or None if silent."""
    result = guard_rail.run_pre(ctx)
    # Shadowed: must never be the applied override.
    assert result is None or result[1] != "preflop_commit_cap"
    events = [e for e in drain_events() if e.guard_id == "preflop_commit_cap"]
    if not events:
        return None
    assert events[0].applied is False
    return (events[0].final_action, events[0].final_amount, events[0].reason)


class TestPreflopCommitCap:
    def test_medium_hand_folds_once_past_cap(self):
        # A3s (medium, score 53) with 6bb already committed, facing more:
        # the observed war hand.
        decision = run_commit_cap(_ctx(["AS", "3S"], committed=60, call=20))
        assert decision is not None
        assert decision[0] == "fold"

    def test_medium_hand_calls_below_cap(self):
        decision = run_commit_cap(_ctx(["AS", "3S"], committed=30, call=20))
        assert decision is None

    def test_air_folds_at_lower_cap(self):
        # 72o is air: cap is 2bb.
        decision = run_commit_cap(_ctx(["7S", "2D"], committed=20, call=20))
        assert decision is not None
        assert decision[0] == "fold"

    def test_premium_hand_is_never_capped(self):
        decision = run_commit_cap(_ctx(["AS", "AD"], committed=500, call=100))
        assert decision is None

    def test_not_facing_bet_never_fires(self):
        ctx = _ctx(["AS", "3S"], committed=60, call=0)
        assert run_commit_cap(ctx) is None

    def test_heads_up_is_exempt(self):
        # The guard is 6max-only; HU tables must be untouched.
        decision = run_commit_cap(
            _ctx(["AS", "3S"], committed=200, call=20, players=2)
        )
        assert decision is None
