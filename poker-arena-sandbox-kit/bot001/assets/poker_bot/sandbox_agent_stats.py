"""Arena agent-stats client + prior layer for the sandbox bundle.

Python port of scripts/texas_agent_stats.sh (GET /api/arena/texas/agent-stats)
plus the prior-vs-observation merge policy from opponent_store's
apply_external_stats_merge, shaped for the in-bundle tracker:

- A bundle-time snapshot (assets/opponent_stats_prior.json, written by
  scripts/bundle_strategy.py --competition) provides long-run stats for
  known opponents with no network and no key exposure at decision time.
- If the snapshot embeds an API key, opponents missing from it are fetched
  live ONCE per match (the endpoint's own guidance: stats are stable —
  fetch once per opponent and reuse, don't poll per hand). Failures are
  cached so a dead network costs at most one timeout per opponent.
- merge_stats_into() applies the same rule as live play: the local read
  wins once we have LOCAL_MIN_HANDS observed hands; below that, a fresh
  API read with a real sample replaces vpip/pfr/hands_seen.

Stdlib-only by design (urllib, json) so it ships in any bundle.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_BASE = "https://arena.dev.fun/api/arena"
FETCH_TIMEOUT_SECONDS = 2.5  # act() has ~10s total; never spend it on stats
SNAPSHOT_BASENAME = "opponent_stats_prior.json"

# Same thresholds as opponent_store.apply_external_stats_merge.
LOCAL_MIN_HANDS = 20
API_MIN_SAMPLE = 20

_snapshot: dict | None = None  # lazily loaded; {} means "looked, none found"
_live_cache: dict[str, dict | None] = {}  # agent_id -> stats (None = failed)


def fetch_agent_stats(agent_id, competition_id, api_key, timeout=None):
    """GET /texas/agent-stats. Returns the stats dict, or None on any error."""
    if not (agent_id and competition_id and api_key):
        return None
    query = urllib.parse.urlencode(
        {"competitionId": competition_id, "agentId": agent_id}
    )
    request = urllib.request.Request(
        f"{API_BASE}/texas/agent-stats?{query}",
        headers={"x-arena-api-key": api_key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout or FETCH_TIMEOUT_SECONDS
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None
    if not isinstance(payload, dict) or "error" in payload:
        return None
    return payload


def _find_snapshot_file() -> Path | None:
    """Locate the snapshot relative to this module.

    In a bundle this file lives at assets/poker_bot/sandbox_agent_stats.py
    and the snapshot at assets/opponent_stats_prior.json. In the repo there
    is normally no snapshot (live play uses opponent_store instead).
    """
    here = Path(__file__).resolve()
    for base in (here.parent.parent, here.parent):
        candidate = base / SNAPSHOT_BASENAME
        if candidate.is_file():
            return candidate
    return None


def load_snapshot(path=None) -> dict:
    """Load (and cache) the prior snapshot. Returns {} when unavailable."""
    global _snapshot
    if _snapshot is not None and path is None:
        return _snapshot
    file = Path(path) if path else _find_snapshot_file()
    data: dict = {}
    if file is not None:
        try:
            loaded = json.loads(Path(file).read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}
    _snapshot = data
    return _snapshot


def reset_cache() -> None:
    """Test hook: forget the loaded snapshot and live-fetch results."""
    global _snapshot
    _snapshot = None
    _live_cache.clear()


def get_prior(agent_id=None, handle=None):
    """Long-run stats for an opponent: snapshot first, then one live fetch.

    Returns the raw agent-stats dict (vpip/pfr/af/... as fractions) or None.
    """
    snapshot = load_snapshot()
    agents = snapshot.get("agents") or {}
    if agent_id and agent_id in agents:
        return agents[agent_id]
    if handle:
        for stats in agents.values():
            if isinstance(stats, dict) and stats.get("handle") == handle:
                return stats

    if not agent_id:
        return None
    if agent_id in _live_cache:
        return _live_cache[agent_id]
    stats = fetch_agent_stats(
        agent_id, snapshot.get("competitionId"), snapshot.get("apiKey")
    )
    _live_cache[agent_id] = stats  # cache failures too: one timeout max
    return stats


def merge_stats_into(profile_dict, *, local_hands_seen, api_stats):
    """Apply the live-play merge policy to a tracker profile dict.

    Mirrors opponent_store.apply_external_stats_merge: once the local sample
    reaches LOCAL_MIN_HANDS the local read wins outright (it is fresher and
    tracks mid-match strategy changes); below that, an API read with
    sampleSize >= API_MIN_SAMPLE replaces vpip/pfr/hands_seen and the
    derived frequencies. Per-action counters (calls/bets/raises/folds,
    fold_to_bet) are always local — the API has no breakdown for them.
    """
    if not isinstance(api_stats, dict):
        return profile_dict
    if local_hands_seen >= LOCAL_MIN_HANDS:
        return profile_dict
    sample = int(api_stats.get("sampleSize") or api_stats.get("hands") or 0)
    if sample < API_MIN_SAMPLE:
        return profile_dict

    def fraction(value):
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return v / 100.0 if v > 1.5 else v

    vpip = fraction(api_stats.get("vpip"))
    pfr = fraction(api_stats.get("pfr"))
    if vpip is not None:
        profile_dict["vpip"] = int(round(vpip * sample))
        profile_dict["vpip_frequency"] = round(vpip, 3)
    if pfr is not None:
        profile_dict["pfr"] = int(round(pfr * sample))
        profile_dict["pfr_frequency"] = round(pfr, 3)

    # AF -> aggression frequency, clamped like the live merge (an AF of
    # 1.38 would otherwise read as an impossible 138% frequency).
    try:
        af = float(api_stats.get("af"))
    except (TypeError, ValueError):
        af = None
    if af is not None:
        profile_dict["aggression_frequency"] = round(min(0.70, max(0.0, af)), 3)

    profile_dict["hands_seen"] = sample
    profile_dict["api_source_used"] = True
    profile_dict["api_sample_size"] = sample
    style = api_stats.get("playingStyle")
    if isinstance(style, dict) and style.get("label"):
        profile_dict["api_style"] = str(style["label"])
    return profile_dict
