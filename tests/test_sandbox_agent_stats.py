"""Tests for the sandbox opponent-stats prior layer.

Covers the merge policy (mirrors opponent_store.apply_external_stats_merge),
snapshot lookup, live-fetch caching, and the tracker integration: API prior
governs while the local sample is thin, local observation wins at 20 hands.
"""

import json

import pytest

from poker_bot import sandbox_agent_stats as sas
from poker_bot import sandbox_opponent_tracker as tracker

API_STATS = {
    "agentId": "agent-1",
    "competitionId": "comp-1",
    "sampleSize": 11079,
    "vpip": 0.4928,
    "pfr": 0.1170,
    "af": 1.3804,
    "wtsd": 0.4918,
    "bluffPct": 0.5369,
    "playingStyle": {"label": "loose-measured"},
}


@pytest.fixture(autouse=True)
def _reset_state():
    sas.reset_cache()
    tracker._PROFILES.clear()
    tracker._SEEN_HAND_KEYS.clear()
    tracker._SEEN_EVENT_KEYS.clear()
    yield
    sas.reset_cache()
    tracker._PROFILES.clear()
    tracker._SEEN_HAND_KEYS.clear()
    tracker._SEEN_EVENT_KEYS.clear()


class TestMergePolicy:
    def base(self, hands=0):
        return {
            "hands_seen": hands,
            "vpip": 1,
            "pfr": 0,
            "vpip_frequency": 0.5,
            "pfr_frequency": 0.0,
            "aggression_frequency": 0.1,
        }

    def test_api_governs_thin_local_sample(self):
        out = sas.merge_stats_into(
            self.base(hands=3), local_hands_seen=3, api_stats=API_STATS
        )
        assert out["api_source_used"] is True
        assert out["hands_seen"] == 11079
        assert out["vpip_frequency"] == 0.493
        assert out["pfr_frequency"] == 0.117
        assert out["vpip"] == round(0.4928 * 11079)
        # AF (aggr/calls) converts to a frequency via af/(1+af).
        assert out["aggression_frequency"] == round(1.3804 / 2.3804, 3)
        # Proxy fields replace the tracker's confident local zeros:
        # call ~ (vpip - pfr) * 0.75, fold-to-bet ~ 0.9 - wtsd, wasd ~ bluffPct.
        assert out["call_frequency"] == round((0.4928 - 0.1170) * 0.75, 3)
        assert out["fold_to_bet_frequency"] == round(0.9 - 0.4918, 3)
        assert out["weak_aggressive_showdown_frequency"] == 0.537
        assert out["api_style"] == "loose-measured"

    def test_local_wins_at_threshold(self):
        base = self.base(hands=sas.LOCAL_MIN_HANDS)
        out = sas.merge_stats_into(
            dict(base), local_hands_seen=sas.LOCAL_MIN_HANDS, api_stats=API_STATS
        )
        assert out == base

    def test_small_api_sample_ignored(self):
        base = self.base(hands=0)
        out = sas.merge_stats_into(
            dict(base), local_hands_seen=0, api_stats={**API_STATS, "sampleSize": 5}
        )
        assert out == base

    def test_percent_scale_autodetected(self):
        out = sas.merge_stats_into(
            self.base(),
            local_hands_seen=0,
            api_stats={"sampleSize": 100, "vpip": 49.0},  # percent, not fraction
        )
        assert out["vpip_frequency"] == 0.49

    def test_non_dict_api_stats_is_noop(self):
        base = self.base()
        merged = sas.merge_stats_into(dict(base), local_hands_seen=0, api_stats=None)
        assert merged == base


class TestSnapshotAndLiveFallback:
    def test_snapshot_lookup_by_id_and_handle(self, tmp_path):
        snap = tmp_path / sas.SNAPSHOT_BASENAME
        snap.write_text(
            json.dumps(
                {
                    "competitionId": "comp-1",
                    "agents": {"agent-1": {**API_STATS, "handle": "villain"}},
                }
            )
        )
        sas.load_snapshot(snap)
        assert sas.get_prior(agent_id="agent-1")["sampleSize"] == 11079
        assert sas.get_prior(handle="villain")["sampleSize"] == 11079
        assert sas.get_prior(agent_id="unknown") is None  # no key -> no live fetch

    def test_live_fallback_fetches_once_and_caches_failure(
        self, tmp_path, monkeypatch
    ):
        snap = tmp_path / sas.SNAPSHOT_BASENAME
        snap.write_text(
            json.dumps({"competitionId": "comp-1", "apiKey": "k", "agents": {}})
        )
        sas.load_snapshot(snap)
        calls = []

        def fake_fetch(agent_id, competition_id, api_key, timeout=None):
            calls.append(agent_id)
            return None  # simulate network failure / unknown agent

        monkeypatch.setattr(sas, "fetch_agent_stats", fake_fetch)
        assert sas.get_prior(agent_id="agent-x") is None
        assert sas.get_prior(agent_id="agent-x") is None
        assert calls == ["agent-x"], "failure must be cached: one fetch max"


class TestSqlitePriorTable:
    def _make_db(self, path, *, api_key=None):
        import sqlite3
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from update_opponent_prior_db import open_prior_db

        conn = open_prior_db(path)
        conn.execute(
            "insert into agent_prior(agent_id, handle, competition_id, "
            "sample_size, stats_json, fetched_at) values (?, ?, ?, ?, ?, ?)",
            (
                "agent-1",
                "villain",
                "comp-1",
                11079,
                json.dumps(API_STATS),
                "2026-07-07T00:00:00+00:00",
            ),
        )
        conn.execute(
            "insert into meta(key, value) values('competitionId', 'comp-1')"
        )
        if api_key:
            conn.execute(
                "insert into meta(key, value) values('apiKey', ?)", (api_key,)
            )
        conn.commit()
        conn.close()
        assert sqlite3.connect(path).execute("select 1").fetchone()

    def test_sqlite_prior_loads_and_matches_by_handle(self, tmp_path):
        db = tmp_path / sas.PRIOR_DB_BASENAME
        self._make_db(db, api_key="k")
        snap = sas.load_snapshot(db)
        assert snap["competitionId"] == "comp-1"
        assert snap["apiKey"] == "k"
        assert sas.get_prior(agent_id="agent-1")["sampleSize"] == 11079
        assert sas.get_prior(handle="villain")["sampleSize"] == 11079
        assert sas.get_prior(handle="  VILLAIN ")["sampleSize"] == 11079

    def test_seat_id_shift_matches_by_suffix(self):
        # Real incident: table seats expose cmq6atdj70el0pd228w5czu4o for
        # account cmq5atdj70el0pd228w5czu4o — same cuid suffix, one segment
        # shifted. The prior lookup must bridge that namespace gap.
        account_id = "cmq5atdj70el0pd228w5czu4o"
        seat_id = "cmq6atdj70el0pd228w5czu4o"
        sas._snapshot = {"agents": {account_id: dict(API_STATS)}}
        assert sas.get_prior(agent_id=seat_id)["sampleSize"] == 11079

    def test_alias_id_matches_by_suffix(self):
        sas._snapshot = {
            "agents": {"agent-1": {**API_STATS, "aliasId": "zzz-alias-9876543210abcd"}}
        }
        assert sas.get_prior(agent_id="zzz-alias-9876543210abcd") is not None

    def test_env_credentials_enable_live_fetch_without_prior_file(
        self, monkeypatch
    ):
        # The [VPIP: 0%] incident: no prior file used to mean no live
        # fetching at all. Env credentials must now be enough.
        sas._snapshot = {}
        monkeypatch.setenv("ARENA_API_KEY", "env-key")
        monkeypatch.setenv("ARENA_COMPETITION_ID", "comp-env")
        seen = {}

        def fake_fetch(agent_id, competition_id, api_key, timeout=None):
            seen.update(agent=agent_id, comp=competition_id, key=api_key)
            return dict(API_STATS)

        monkeypatch.setattr(sas, "fetch_agent_stats", fake_fetch)
        assert sas.get_prior(agent_id="agent-9")["sampleSize"] == 11079
        assert seen == {"agent": "agent-9", "comp": "comp-env", "key": "env-key"}

    def test_prior_status_reports_sources(self, tmp_path):
        db = tmp_path / sas.PRIOR_DB_BASENAME
        self._make_db(db, api_key="k")
        sas.load_snapshot(db)
        status = sas.prior_status()
        assert "agents=1" in status
        assert "key=yes" in status


class TestUpdaterScript:
    def test_parse_roster_extracts_ids(self, tmp_path):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from update_opponent_prior_db import parse_roster

        roster = tmp_path / "opponents.md"
        roster.write_text(
            "cmq6atdj70el0pd228w5czu4o\n"
            "- cmqvg50so537et6mn1cbrl1vm  # bullet + comment\n"
            "not an id\n"
            "cmq6atdj70el0pd228w5czu4o\n"  # duplicate
        )
        assert parse_roster(roster) == [
            "cmq6atdj70el0pd228w5czu4o",
            "cmqvg50so537et6mn1cbrl1vm",
        ]


class TestTrackerIntegration:
    def _table(self):
        return {
            "tableId": "t1",
            "street": "Preflop",
            "currentBet": 20,
            "selfSeatNumber": 1,
            "seats": [
                {"seatNumber": 1, "agentId": "hero", "holeCards": ["AS", "KD"]},
                {"seatNumber": 2, "agentId": "agent-1", "agentHandle": "villain"},
            ],
            "recentEvents": [],
        }

    def test_prior_seeds_profile_until_local_takes_over(self, tmp_path):
        snap = tmp_path / sas.SNAPSHOT_BASENAME
        snap.write_text(
            json.dumps({"competitionId": "comp-1", "agents": {"agent-1": API_STATS}})
        )
        sas.load_snapshot(snap)

        table = tracker.enrich_table(self._table())
        profile = table["opponentProfiles"]["agent-1"]
        assert profile["api_source_used"] is True
        assert profile["vpip_frequency"] == 0.493

        # Once the local sample reaches the threshold, observation wins.
        tracker._PROFILES["agent-1"].hands_seen = sas.LOCAL_MIN_HANDS
        table = tracker.enrich_table(self._table())
        profile = table["opponentProfiles"]["agent-1"]
        assert "api_source_used" not in profile
        assert profile["hands_seen"] == sas.LOCAL_MIN_HANDS

    def test_seat_fallback_ids_never_hit_the_network(self, monkeypatch):
        sas._snapshot = {"competitionId": "comp-1", "apiKey": "k", "agents": {}}
        calls = []
        monkeypatch.setattr(
            sas, "fetch_agent_stats", lambda *a, **k: calls.append(a) or None
        )
        table = self._table()
        table["seats"][1] = {"seatNumber": 2}  # no agentId -> tracker uses seat-2
        tracker.enrich_table(table)
        assert calls == []
