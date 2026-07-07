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
        # AF 1.38 clamps to the 0.70 frequency ceiling, like live play.
        assert out["aggression_frequency"] == 0.70
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
