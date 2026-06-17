"""Tests for the local-vs-API opponent stats merge (PLAN_OPPONENT_STATS.md).

The merge logic:
- If our local sample is large enough (LOCAL_MIN_HANDS=20), keep our
  observation (per user spec: trust local over API when we have
  enough data, even if it disagrees with the API).
- Otherwise, if the API has fresh, large-sample data, use it.
- The API does not provide per-action breakdowns (calls / bets /
  raises / folds / fold_to_bet) so those local counters are
  preserved.
"""

import datetime as dt

from poker_bot.opponent_store import (
    API_STALE_DAYS,
    LOCAL_MIN_HANDS,
    _parse_fetched_at,
    apply_external_stats_merge,
    connect,
    increment_hand_seen,
    load_profiles_for_agents,
    record_external_agent_stats,
)
from poker_bot.opponents import OpponentProfile, profile_from_mapping


def _today():
    return dt.datetime(2026, 6, 17, 12, 0, 0, tzinfo=dt.UTC)


# ════════════════════════════════════════════════════════════════════════════
# Threshold logic
# ════════════════════════════════════════════════════════════════════════════


def test_local_kept_when_sufficient_hands_seen():
    """If hands_seen >= LOCAL_MIN_HANDS, keep local even if API differs."""
    p = OpponentProfile(agent_id="x", hands_seen=50, vpip=25, pfr=10)
    p.api_stats = {"hands": 500, "vpip": 50, "pfr": 30}
    p.api_fetched_at = _today() - dt.timedelta(hours=1)
    apply_external_stats_merge(p, today=_today())
    assert p.vpip == 25, f"local vpip must be kept; got {p.vpip}"
    assert p.pfr == 10
    assert p.hands_seen == 50
    assert p.api_source_used is False


def test_local_kept_at_exact_threshold():
    """hands_seen == LOCAL_MIN_HANDS is still "sufficient" — keep local."""
    p = OpponentProfile(agent_id="x", hands_seen=LOCAL_MIN_HANDS, vpip=10)
    p.api_stats = {"hands": 100, "vpip": 30}
    p.api_fetched_at = _today()
    apply_external_stats_merge(p, today=_today())
    assert p.api_source_used is False
    assert p.vpip == 10


def test_api_used_when_local_below_threshold():
    """If local < LOCAL_MIN_HANDS, replace with API values."""
    p = OpponentProfile(agent_id="x", hands_seen=5, vpip=2, pfr=1)
    p.api_stats = {"hands": 200, "vpip": 50, "pfr": 25}
    p.api_fetched_at = _today() - dt.timedelta(hours=1)
    apply_external_stats_merge(p, today=_today())
    assert p.hands_seen == 200
    assert p.vpip == 100  # 50% of 200
    assert p.pfr == 50  # 25% of 200
    assert p.api_source_used is True


def test_no_api_data_no_merge():
    """If api_stats is None, no merge happens."""
    p = OpponentProfile(agent_id="x", hands_seen=5, vpip=2)
    p.api_stats = None
    apply_external_stats_merge(p, today=_today())
    assert p.hands_seen == 5
    assert p.vpip == 2
    assert p.api_source_used is False


def test_api_sample_too_small_ignored():
    """If the API has fewer than LOCAL_MIN_HANDS, fall back to local."""
    p = OpponentProfile(agent_id="x", hands_seen=5, vpip=2)
    p.api_stats = {"hands": 10, "vpip": 50}  # < LOCAL_MIN_HANDS
    p.api_fetched_at = _today()
    apply_external_stats_merge(p, today=_today())
    assert p.hands_seen == 5
    assert p.vpip == 2
    assert p.api_source_used is False


# ════════════════════════════════════════════════════════════════════════════
# Stale-API check (2-day threshold)
# ════════════════════════════════════════════════════════════════════════════


def test_stale_api_ignored():
    """API data older than API_STALE_DAYS is rejected."""
    p = OpponentProfile(agent_id="x", hands_seen=5)
    p.api_stats = {"hands": 200, "vpip": 50}
    p.api_fetched_at = _today() - dt.timedelta(days=API_STALE_DAYS + 1)
    apply_external_stats_merge(p, today=_today())
    assert p.api_source_used is False


def test_fresh_api_at_threshold_used():
    """At exactly API_STALE_DAYS old, the API is still used."""
    p = OpponentProfile(agent_id="x", hands_seen=5)
    p.api_stats = {"hands": 200, "vpip": 50}
    p.api_fetched_at = _today() - dt.timedelta(days=API_STALE_DAYS)
    apply_external_stats_merge(p, today=_today())
    assert p.api_source_used is True


def test_missing_fetched_at_ignored():
    """If the API has no timestamp at all, treat as stale (safer)."""
    p = OpponentProfile(agent_id="x", hands_seen=5)
    p.api_stats = {"hands": 200, "vpip": 50}
    p.api_fetched_at = None
    apply_external_stats_merge(p, today=_today())
    assert p.api_source_used is False


def test_naive_datetime_treated_as_utc():
    """A naive datetime from SQLite should be treated as UTC."""
    p = OpponentProfile(agent_id="x", hands_seen=5)
    p.api_stats = {"hands": 200, "vpip": 50}
    # 1 hour ago, but as a naive datetime (e.g. parsed from a string
    # without timezone info).
    p.api_fetched_at = (_today() - dt.timedelta(hours=1)).replace(tzinfo=None)
    apply_external_stats_merge(p, today=_today())
    assert p.api_source_used is True


# ════════════════════════════════════════════════════════════════════════════
# Format normalization (VPIP/PFR percent vs fraction; AF always 0-1)
# ════════════════════════════════════════════════════════════════════════════


def test_vpip_as_percent_normalized():
    """VPIP > 1.5 is treated as percent, divided by 100."""
    p = OpponentProfile(agent_id="x", hands_seen=5)
    p.api_stats = {"hands": 200, "vpip": 37, "pfr": 22}  # percent
    p.api_fetched_at = _today()
    apply_external_stats_merge(p, today=_today())
    assert p.vpip == 74  # 37% of 200
    assert p.pfr == 44  # 22% of 200


def test_vpip_as_fraction_preserved():
    """VPIP <= 1.5 is treated as fraction, used as-is."""
    p = OpponentProfile(agent_id="x", hands_seen=5)
    p.api_stats = {"hands": 200, "vpip": 0.37, "pfr": 0.22}  # fraction
    p.api_fetched_at = _today()
    apply_external_stats_merge(p, today=_today())
    assert p.vpip == 74
    assert p.pfr == 44


def test_af_always_zero_to_one():
    """AF is always 0-1 per arena; clamped to [0, 0.70] for local-style use."""
    p = OpponentProfile(agent_id="x", hands_seen=5)
    p.api_stats = {"hands": 200, "vpip": 30, "af": 0.4}
    p.api_fetched_at = _today()
    apply_external_stats_merge(p, today=_today())
    assert 0.0 <= p.api_aggr_freq <= 0.70


def test_af_above_one_clamped():
    """Defensive: if AF > 1 (unexpected), clamp to 0.70."""
    p = OpponentProfile(agent_id="x", hands_seen=5)
    p.api_stats = {"hands": 200, "vpip": 30, "af": 2.0}  # bad data
    p.api_fetched_at = _today()
    apply_external_stats_merge(p, today=_today())
    assert p.api_aggr_freq == 0.70


def test_af_falls_back_to_playing_style_label():
    """If AF is missing, use playingStyle.aggression label."""
    p = OpponentProfile(agent_id="x", hands_seen=5)
    p.api_stats = {
        "hands": 200,
        "vpip": 30,
        "playingStyle": {"aggression": "aggressive"},
    }
    p.api_fetched_at = _today()
    apply_external_stats_merge(p, today=_today())
    assert p.api_aggr_freq == 0.46


def test_missing_api_fields_dont_crash():
    """Partial API responses (only ``hands``) should not crash."""
    p = OpponentProfile(agent_id="x", hands_seen=5)
    p.api_stats = {"hands": 200}  # no vpip/pfr/af
    p.api_fetched_at = _today()
    apply_external_stats_merge(p, today=_today())
    assert p.hands_seen == 200
    assert p.vpip == 0  # unchanged from default
    assert p.api_source_used is True


# ════════════════════════════════════════════════════════════════════════════
# API error responses
# ════════════════════════════════════════════════════════════════════════════


def test_api_error_dict_never_stored():
    """fetch_and_record_agent_stats returns False on {"error": ...} and
    never writes the row, so the loader sees api_stats=None."""
    p = OpponentProfile(agent_id="x", hands_seen=5, vpip=2)
    p.api_stats = None  # what we'd see if API returned an error
    apply_external_stats_merge(p, today=_today())
    assert p.api_source_used is False
    assert p.vpip == 2  # local preserved


# ════════════════════════════════════════════════════════════════════════════
# Local ground truth preserved
# ════════════════════════════════════════════════════════════════════════════


def test_local_calls_bets_raises_folds_preserved():
    """The API doesn't break out per-action counts; keep local."""
    p = OpponentProfile(agent_id="x", hands_seen=5, calls=2, bets=1, raises=0, folds=3)
    p.api_stats = {"hands": 200, "vpip": 50, "pfr": 25}
    p.api_fetched_at = _today()
    apply_external_stats_merge(p, today=_today())
    assert p.calls == 2
    assert p.bets == 1
    assert p.raises == 0
    assert p.folds == 3


def test_local_fold_to_bet_preserved():
    """The API doesn't provide fold_to_bet; keep local even if sparse."""
    p = OpponentProfile(
        agent_id="x",
        hands_seen=5,
        fold_to_bet=1,
        opportunities_to_fold_to_bet=4,
    )
    p.api_stats = {"hands": 200, "vpip": 50, "pfr": 25}
    p.api_fetched_at = _today()
    apply_external_stats_merge(p, today=_today())
    assert p.fold_to_bet == 1
    assert p.opportunities_to_fold_to_bet == 4


# ════════════════════════════════════════════════════════════════════════════
# profile_from_mapping: timestamp parsing
# ════════════════════════════════════════════════════════════════════════════


def test_fetched_at_iso_with_z_suffix():
    data = {"name": "x", "api_fetched_at": "2026-06-17T12:00:00Z"}
    p = profile_from_mapping("x", data)
    assert p.api_fetched_at == dt.datetime(2026, 6, 17, 12, 0, 0, tzinfo=dt.UTC)


def test_fetched_at_sqlite_default_format():
    """SQLite current_timestamp → 'YYYY-MM-DD HH:MM:SS'."""
    data = {"name": "x", "api_fetched_at": "2026-06-17 12:00:00"}
    p = profile_from_mapping("x", data)
    assert p.api_fetched_at == dt.datetime(2026, 6, 17, 12, 0, 0, tzinfo=dt.UTC)


def test_fetched_at_unparseable_string():
    data = {"name": "x", "api_fetched_at": "not a timestamp"}
    p = profile_from_mapping("x", data)
    assert p.api_fetched_at is None


def test_fetched_at_missing():
    data = {"name": "x"}
    p = profile_from_mapping("x", data)
    assert p.api_fetched_at is None


# ════════════════════════════════════════════════════════════════════════════
# End-to-end through load_profiles_for_agents
# ════════════════════════════════════════════════════════════════════════════


def test_load_profiles_applies_merge_when_local_sparse(tmp_path):
    """End-to-end: small local + large API → API is used."""
    db_path = tmp_path / "gameplay.sqlite"
    conn = connect(db_path)
    increment_hand_seen(conn, "arena", "agent-A", handle="A")
    record_external_agent_stats(
        conn,
        platform="arena",
        agent_id="agent-A",
        competition_id="cmp-1",
        stats={"hands": 200, "vpip": 40, "pfr": 20, "af": 0.3},
    )
    conn.commit()

    profiles = load_profiles_for_agents(conn, "arena", ["agent-A"])
    p = profiles["agent-A"]
    assert p.hands_seen == 200
    assert p.vpip == 80  # 40% of 200
    assert p.pfr == 40  # 20% of 200
    assert p.api_source_used is True
    conn.close()


def test_load_profiles_keeps_local_when_sufficient(tmp_path):
    """End-to-end: local has enough hands, keep local."""
    db_path = tmp_path / "gameplay.sqlite"
    conn = connect(db_path)
    for _ in range(50):
        increment_hand_seen(conn, "arena", "agent-B")
    record_external_agent_stats(
        conn,
        platform="arena",
        agent_id="agent-B",
        competition_id="cmp-1",
        stats={"hands": 200, "vpip": 40, "pfr": 20},
    )
    conn.commit()

    profiles = load_profiles_for_agents(conn, "arena", ["agent-B"])
    p = profiles["agent-B"]
    assert p.hands_seen == 50  # local kept
    assert p.api_source_used is False
    conn.close()


def test_load_profiles_skips_stale_api(tmp_path):
    """End-to-end: stale API falls back to local (sparse)."""
    db_path = tmp_path / "gameplay.sqlite"
    conn = connect(db_path)
    increment_hand_seen(conn, "arena", "agent-C", handle="C")
    record_external_agent_stats(
        conn,
        platform="arena",
        agent_id="agent-C",
        competition_id="cmp-1",
        stats={"hands": 200, "vpip": 40, "pfr": 20},
    )
    # Backdate the fetched_at to 3 days ago
    stale_ts = (dt.datetime.now(dt.UTC) - dt.timedelta(days=3)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    conn.execute(
        """
        update opponent_external_stats
        set fetched_at = :ts
        where opponent_id = (
            select id from opponents where agent_id = 'agent-C'
        )
        """,
        {"ts": stale_ts},
    )
    conn.commit()

    profiles = load_profiles_for_agents(conn, "arena", ["agent-C"])
    p = profiles["agent-C"]
    assert p.api_source_used is False
    assert p.hands_seen == 1  # local kept
    conn.close()
