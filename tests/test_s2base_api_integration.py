"""Tests for s2base.py's API-source-aware behavior.

When the local-vs-API merge (``apply_external_stats_merge`` in
``opponent_store.py``) overrides a profile with API data, the
strategy must:

1. Cap ``profile_confidence`` at 0.5 in
   ``opponent_exploit_context`` (Point 1 of
   ``PLAN_PROFILE_AGENTS.md``) so that exploit branches gated
   on >= 0.5 do not trigger.

2. Expose an ``any_api_source`` flag so other strategy code
   can react to API-derived data.

3. Use ``api_aggr_freq`` for aggression frequency reads when
   the merge overrode the profile (Point 2 of
   ``PLAN_PROFILE_AGENTS.md``) — the local frequency is 0.0
   with sparse samples and the API value is a better read.
"""

from poker_bot.opponents import OpponentProfile
from poker_bot.strategies.s2base import (
    opponent_exploit_context,
    profile_aggression_frequency_merged,
)


def _make_seat(agent_id, stack=1500, folded=False):
    return {
        "agentId": agent_id,
        "seatNumber": 0,
        "stackChips": stack,
        "holeCards": [],
        "currentBetChips": 0,
        "folded": folded,
        "hasFolded": folded,
    }


def _make_table(hero_seat, opponent_seats, profiles=None):
    seats = [hero_seat] + list(opponent_seats)
    return {
        "street": "Preflop",
        "boardCards": [],
        "potChips": 15,
        "buttonSeatNumber": 0,
        "seats": seats,
        "opponentProfiles": profiles or {},
    }


# ════════════════════════════════════════════════════════════════════════════
# Point 1: any_api_source flag
# ════════════════════════════════════════════════════════════════════════════


def test_any_api_source_false_when_no_api_used():
    hero = _make_seat("hero")
    opp1 = _make_seat("opp-1")
    opp2 = _make_seat("opp-2")
    profiles = {
        "opp-1": OpponentProfile(agent_id="opp-1", hands_seen=50, vpip=20, pfr=8),
        "opp-2": OpponentProfile(agent_id="opp-2", hands_seen=30, vpip=15, pfr=5),
    }
    table = _make_table(hero, [opp1, opp2], profiles)
    ctx = opponent_exploit_context(table, hero)
    assert ctx["any_api_source"] is False


def test_any_api_source_true_when_one_opponent_api_derived():
    hero = _make_seat("hero")
    opp1 = _make_seat("opp-1")
    opp2 = _make_seat("opp-2")
    api_opp = OpponentProfile(agent_id="opp-1", hands_seen=200, vpip=80, pfr=40)
    api_opp.api_source_used = True  # merge overrode this profile
    profiles = {
        "opp-1": api_opp,
        "opp-2": OpponentProfile(agent_id="opp-2", hands_seen=30, vpip=15, pfr=5),
    }
    table = _make_table(hero, [opp1, opp2], profiles)
    ctx = opponent_exploit_context(table, hero)
    assert ctx["any_api_source"] is True


def test_profile_confidence_capped_when_any_api_source():
    """When API is used, profile_confidence is min(0.5, raw)."""
    hero = _make_seat("hero")
    opp1 = _make_seat("opp-1")
    opp2 = _make_seat("opp-2")
    opp3 = _make_seat("opp-3")
    # All three "confident" by hands_seen >= 15, but one is API-derived
    api_opp = OpponentProfile(agent_id="opp-1", hands_seen=200, vpip=80, pfr=40)
    api_opp.api_source_used = True
    profiles = {
        "opp-1": api_opp,
        "opp-2": OpponentProfile(agent_id="opp-2", hands_seen=50, vpip=20, pfr=8),
        "opp-3": OpponentProfile(agent_id="opp-3", hands_seen=30, vpip=15, pfr=5),
    }
    table = _make_table(hero, [opp1, opp2, opp3], profiles)
    ctx = opponent_exploit_context(table, hero)
    # Raw would be 3/3 = 1.0; capped at 0.5
    assert ctx["profile_confidence"] == 0.5
    assert ctx["any_api_source"] is True


def test_profile_confidence_uncapped_when_no_api():
    """When no API is used, profile_confidence is the raw value."""
    hero = _make_seat("hero")
    opp1 = _make_seat("opp-1")
    opp2 = _make_seat("opp-2")
    profiles = {
        "opp-1": OpponentProfile(agent_id="opp-1", hands_seen=50, vpip=20, pfr=8),
        "opp-2": OpponentProfile(agent_id="opp-2", hands_seen=30, vpip=15, pfr=5),
    }
    table = _make_table(hero, [opp1, opp2], profiles)
    ctx = opponent_exploit_context(table, hero)
    # Raw is 2/2 = 1.0; uncapped
    assert ctx["profile_confidence"] == 1.0
    assert ctx["any_api_source"] is False


def test_profile_confidence_cap_respects_underscore():
    """When only 1/2 opponents are confident AND api used, cap is min(0.5, 0.5) = 0.5."""
    hero = _make_seat("hero")
    opp1 = _make_seat("opp-1")
    opp2 = _make_seat("opp-2")
    # opp-1 is "confident" by hands_seen (post-merge) but API-derived
    api_opp = OpponentProfile(agent_id="opp-1", hands_seen=200, vpip=80, pfr=40)
    api_opp.api_source_used = True
    # opp-2 is NOT confident (only 5 hands)
    profiles = {
        "opp-1": api_opp,
        "opp-2": OpponentProfile(agent_id="opp-2", hands_seen=5, vpip=2, pfr=1),
    }
    table = _make_table(hero, [opp1, opp2], profiles)
    ctx = opponent_exploit_context(table, hero)
    # Raw: 1/2 = 0.5; capped at 0.5 → still 0.5
    assert ctx["profile_confidence"] == 0.5


# ════════════════════════════════════════════════════════════════════════════
# Point 2: profile_aggression_frequency_merged
# ════════════════════════════════════════════════════════════════════════════


def test_aggression_frequency_uses_local_when_sufficient():
    """Local is reliable, return local even if API data is set."""
    p = OpponentProfile(
        agent_id="x", hands_seen=50, calls=10, bets=20, raises=5, folds=15
    )
    p.api_aggr_freq = 0.99  # would be huge, must be ignored
    p.api_source_used = False
    # Local: (20+5)/(10+20+5+15) = 25/50 = 0.5
    assert profile_aggression_frequency_merged(p) == 0.5


def test_aggression_frequency_falls_back_to_api_when_used():
    """When merge overrode (api_source_used=True), return api_aggr_freq."""
    p = OpponentProfile(agent_id="x", hands_seen=5, calls=0, bets=0, raises=0, folds=0)
    p.api_aggr_freq = 0.46
    p.api_source_used = True
    # Without merge, local would be 0/0 = 0.0
    assert profile_aggression_frequency_merged(p) == 0.46


def test_aggression_frequency_local_when_api_aggr_freq_unset():
    """If api_aggr_freq wasn't recorded (no playingStyle), use local."""
    p = OpponentProfile(agent_id="x", hands_seen=5, calls=2, bets=1, raises=0, folds=3)
    p.api_source_used = True
    # No api_aggr_freq → use local: (1+0)/(2+1+0+3) = 1/6 ≈ 0.167
    local = profile_aggression_frequency_merged(p)
    assert abs(local - 1 / 6) < 1e-9


def test_aggression_frequency_handles_none_profile():
    assert profile_aggression_frequency_merged(None) == 0.0


def test_aggression_frequency_zero_actions_with_no_api():
    """Sparse local with no API override → 0.0 (the previous behavior)."""
    p = OpponentProfile(agent_id="x", hands_seen=5)
    # No actions recorded, no API
    assert profile_aggression_frequency_merged(p) == 0.0
