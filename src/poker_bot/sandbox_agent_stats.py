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

Priors are read from (first found wins):

1. assets/opponent_stats_prior.sqlite — the "rainbow table" written by
   scripts/update_opponent_prior_db.py and copied in by the bundler.
   Schema: agent_prior(agent_id, handle, stats_json, ...) + meta(key,value)
   for competitionId / apiKey.
2. assets/opponent_stats_prior.json — the older JSON snapshot format.
3. Nothing — observation-only, with live fetch still possible if
   ARENA_API_KEY / ARENA_COMPETITION_ID are in the environment.

Stdlib-only by design (urllib, json, sqlite3) so it ships in any bundle.
"""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_BASE = "https://arena.dev.fun/api/arena"
FETCH_TIMEOUT_SECONDS = 2.5  # act() has ~10s total; never spend it on stats
SNAPSHOT_BASENAME = "opponent_stats_prior.json"
PRIOR_DB_BASENAME = "opponent_stats_prior.sqlite"

# Ids exposed in table seats are shifted variants of account agent ids
# (one cuid segment off) but share the trailing characters — e.g. seat
# cmq6atdj70el0pd228w5czu4o is account cmq5atdj70el0pd228w5czu4o. Suffix
# matching is therefore the reliable join between what a table shows and
# what the stats API is keyed by.
SUFFIX_LEN = 14

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
    """Locate the prior table relative to this module (sqlite preferred).

    In a bundle this file lives at assets/poker_bot/sandbox_agent_stats.py
    and the prior at assets/opponent_stats_prior.{sqlite,json}. In the repo
    there is normally no prior file (live play uses opponent_store instead).
    """
    here = Path(__file__).resolve()
    for basename in (PRIOR_DB_BASENAME, SNAPSHOT_BASENAME):
        for base in (here.parent.parent, here.parent):
            candidate = base / basename
            if candidate.is_file():
                return candidate
    return None


def load_prior_db(path) -> dict:
    """Read the SQLite rainbow table into snapshot form. {} on any error."""
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return {}
    try:
        columns = {row[1] for row in conn.execute("pragma table_info(agent_prior)")}
        alias_col = "alias_id" if "alias_id" in columns else "null"
        agents: dict[str, dict] = {}
        for agent_id, alias_id, handle, stats_json in conn.execute(
            f"select agent_id, {alias_col}, handle, stats_json from agent_prior"
        ):
            try:
                stats = json.loads(stats_json)
            except (TypeError, ValueError):
                continue
            if not isinstance(stats, dict):
                continue
            if handle and "handle" not in stats:
                stats["handle"] = handle
            if alias_id and "aliasId" not in stats:
                stats["aliasId"] = alias_id
            agents[str(agent_id)] = stats
        meta = dict(conn.execute("select key, value from meta"))
        return {
            "competitionId": meta.get("competitionId"),
            "apiKey": meta.get("apiKey"),
            "fetchedAt": meta.get("fetchedAt"),
            "agents": agents,
        }
    except sqlite3.Error:
        return {}
    finally:
        conn.close()


def load_snapshot(path=None) -> dict:
    """Load (and cache) the prior table. Returns {} when unavailable."""
    global _snapshot
    if _snapshot is not None and path is None:
        return _snapshot
    file = Path(path) if path else _find_snapshot_file()
    data: dict = {}
    if file is not None:
        if file.suffix == ".sqlite":
            data = load_prior_db(file)
        else:
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


def _credentials():
    """(competition_id, api_key) from the prior table, else the environment.

    The env fallback matters: a bundle built WITHOUT a prior table used to
    silently disable live fetching entirely (the [VPIP: 0%] incident) —
    now it still works wherever credentials are available.
    """
    snapshot = load_snapshot()
    competition_id = snapshot.get("competitionId") or os.environ.get(
        "ARENA_COMPETITION_ID"
    )
    api_key = snapshot.get("apiKey") or os.environ.get("ARENA_API_KEY")
    return competition_id, api_key


def _agent_indexes():
    """(by_suffix, by_handle) lookup indexes over the loaded prior table."""
    snapshot = load_snapshot()
    cached = snapshot.get("_indexes")
    if cached is not None:
        return cached
    by_suffix: dict[str, dict] = {}
    by_handle: dict[str, dict] = {}
    for agent_id, stats in (snapshot.get("agents") or {}).items():
        if not isinstance(stats, dict):
            continue
        by_suffix[str(agent_id)[-SUFFIX_LEN:]] = stats
        alias = stats.get("aliasId")
        if alias:
            by_suffix[str(alias)[-SUFFIX_LEN:]] = stats
        handle = stats.get("handle")
        if handle:
            by_handle[str(handle).strip().lower()] = stats
    snapshot["_indexes"] = (by_suffix, by_handle)
    return snapshot["_indexes"]


def get_prior(agent_id=None, handle=None):
    """Long-run stats for an opponent: prior table first, then one live fetch.

    Lookup order: exact agent id, cuid suffix (covers the seat-id/account-id
    shift AND alias ids stored by the updater), then normalized handle.
    Returns the raw agent-stats dict (vpip/pfr/... as fractions) or None.
    """
    snapshot = load_snapshot()
    agents = snapshot.get("agents") or {}
    if agent_id and agent_id in agents:
        return agents[agent_id]
    by_suffix, by_handle = _agent_indexes()
    if agent_id:
        stats = by_suffix.get(str(agent_id)[-SUFFIX_LEN:])
        if stats is not None:
            return stats
    if handle:
        stats = by_handle.get(str(handle).strip().lower())
        if stats is not None:
            return stats

    if not agent_id:
        return None
    if agent_id in _live_cache:
        return _live_cache[agent_id]
    competition_id, api_key = _credentials()
    stats = fetch_agent_stats(agent_id, competition_id, api_key)
    _live_cache[agent_id] = stats  # cache failures too: one timeout max
    return stats


def prior_status() -> str:
    """One-line diagnostic for decision traces: what priors are available.

    Answers, from inside a running match, the questions we cannot check
    locally: did a prior table ship, how many agents, are live-fetch
    credentials present, and did any live fetches succeed.
    """
    snapshot = load_snapshot()
    source = _find_snapshot_file()
    competition_id, api_key = _credentials()
    hits = sum(1 for v in _live_cache.values() if v)
    return (
        f"prior:{source.name if source else 'none'}"
        f" agents={len(snapshot.get('agents') or {})}"
        f" key={'yes' if api_key else 'no'}"
        f" comp={'yes' if competition_id else 'no'}"
        f" live={hits}/{len(_live_cache)}"
    )


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

    # AF (aggressive actions / calls, 0..inf) -> aggression frequency
    # (aggressive actions / all actions): af/(1+af). Live play's merge
    # clamps raw AF instead — this conversion is the correct one; align
    # opponent_store when convenient.
    try:
        af = float(api_stats.get("af"))
    except (TypeError, ValueError):
        af = None
    if af is not None:
        profile_dict["aggression_frequency"] = round(
            min(0.70, max(0.0, af / (1.0 + af))), 3
        )

    # The tracker exports call_frequency / fold_to_bet_frequency computed
    # from LOCAL counters, which are ~zero while the API governs — a
    # "confident zero" ("8000 hands, never calls, never folds") that
    # poisons the cores' calling-station and tightness reads. The API has
    # no direct fields for these, but carries the signal — map proxies:
    #
    # - call_frequency ~ passivity: the vpip-pfr gap is voluntary money
    #   that went in WITHOUT raising. Scaled to the [0, 0.75] range the
    #   local counter version produces in practice.
    # - fold_to_bet ~ 1 - wtsd: players who rarely reach showdown are the
    #   ones folding to bets. Clamped to a sane band; heuristic — validate
    #   against live reads when local samples accumulate.
    # - weak_aggressive_showdown_frequency ~ bluffPct: both measure
    #   "aggression shown down weak" — the bluff-catching gate.
    if vpip is not None and pfr is not None:
        profile_dict["call_frequency"] = round(
            min(0.75, max(0.0, (vpip - pfr) * 0.75)), 3
        )
    wtsd = fraction(api_stats.get("wtsd"))
    if wtsd is not None:
        profile_dict["fold_to_bet_frequency"] = round(
            min(0.75, max(0.15, 0.9 - wtsd)), 3
        )
    bluff = fraction(api_stats.get("bluffPct"))
    if bluff is not None:
        profile_dict["weak_aggressive_showdown_frequency"] = round(
            min(1.0, max(0.0, bluff)), 3
        )

    profile_dict["hands_seen"] = sample
    profile_dict["api_source_used"] = True
    profile_dict["api_sample_size"] = sample
    style = api_stats.get("playingStyle")
    if isinstance(style, dict) and style.get("label"):
        profile_dict["api_style"] = str(style["label"])
    return profile_dict
