"""Sandbox entrypoint for profiler_bot.

Imports the validated hubase strategy from assets/ using importlib.
At runtime in the sandbox, assets/ is added to sys.path explicitly before
calling importlib so the bundled hubase.py can be found.
"""
import importlib
import os
import sys
from pathlib import Path

# assets/ is expected to be at ../assets relative to this file (harness/).
_expected_assets = Path(__file__).resolve().parent.parent / "assets"
if str(_expected_assets) not in sys.path:
    sys.path.insert(0, str(_expected_assets))
os.environ.setdefault("POKER_RANGE_STATE_DIR", "/tmp/poker-range-state")

# Fallback: search for hubase.py relative to this file.
if "hubase" not in sys.modules:
    for candidate in [
        Path(__file__).resolve().parent.parent / "assets",
        Path(__file__).resolve().parent / "hubase",
        Path(__file__).resolve().parent / "assets",
    ]:
        if (candidate / "hubase.py").exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            break

_hubase = importlib.import_module("hubase")
_hubase_act = _hubase.act

# Load both local trackers (in-memory + SQLite) for comparison
_enrich_local = None
_enrich_db = None
try:
    _tracker_local = importlib.import_module("sandbox_opponent_tracker")
    _enrich_local = _tracker_local.enrich_table
except Exception:
    pass

try:
    _tracker_db = importlib.import_module("sandbox_opponent_tracker_sqlite")
    _enrich_db = _tracker_db.enrich_table
except Exception:
    pass

# Load remote API tracker
_fetch_remote = None
try:
    _api_tracker = importlib.import_module("sandbox_opponent_tracker_api")
    _fetch_remote = _api_tracker.fetch_opponent_stats
except Exception:
    pass


# Cache remote stats per hand to avoid multiple API calls
_REMOTE_CACHE: dict[str, dict] = {}
_HAND_ID_CACHE: str | None = None


def _current_hand_id(table: dict) -> str:
    """Extract a unique hand identifier from the table."""
    for key in ("handId", "hand_id", "handNumber", "hand_number"):
        if table.get(key):
            return str(table[key])
    return str(table.get("tableId") or table.get("id") or "unknown")


def _build_prefix(local_profiles: dict, db_profiles: dict, remote_stats: dict) -> str:
    """Build message prefix with all four data sources.
    Format: [OpponentName | local:hands=N vpip=X f2b=Y wasd=Z | db:hands=N vpip=X | API: stats]
    """
    parts = []
    all_agent_ids = set(local_profiles.keys()) | set(db_profiles.keys()) | set(remote_stats.keys())
    
    for agent_id in all_agent_ids:
        local = local_profiles.get(agent_id, {})
        db = db_profiles.get(agent_id, {})
        remote = remote_stats.get(agent_id, {})
        
        # Get opponent name from any source
        name = local.get("name") or db.get("name") or remote.get("agentName") or agent_id
        
        # Local (in-memory) stats
        l_hands = local.get("hands_seen", 0)
        l_vpip = local.get("vpip", 0)
        l_f2b = local.get("fold_to_bet", 0)
        l_wasd = local.get("weak_aggressive_showdowns", 0)
        local_part = f"local:hands={l_hands} vpip={l_vpip} f2b={l_f2b} wasd={l_wasd}"
        
        # DB (SQLite) stats
        d_hands = db.get("hands_seen", 0)
        d_vpip = db.get("vpip", 0)
        db_part = f"db:hands={d_hands} vpip={d_vpip}"
        
        # Remote API stats
        if remote:
            r_hands = remote.get("hands_seen", 0)
            r_vpip = remote.get("vpip", 0)
            r_f2b = remote.get("fold_to_bet", 0)
            r_wasd = remote.get("weak_aggressive_showdowns", 0)
            r_wsdf = remote.get("weak_aggressive_showdown_frequency")
            api_part = f"API: hands={r_hands} vpip={r_vpip} f2b={r_f2b} wasd={r_wasd}"
            if isinstance(r_wsdf, (int, float)):
                api_part += f" wasdf={r_wsdf:.2f}"
        else:
            api_part = 'API: "NA"'
        
        # Combine: [Name | local:... | db:... | API:...]
        opponent_part = f"{name} | {local_part} | {db_part} | {api_part}"
        parts.append(opponent_part)
    
    return " | ".join(parts)


def act(table):
    """Sandbox contract: receive table dict, return action dict."""
    # Enrich with both local trackers
    if _enrich_local is not None:
        table = _enrich_local(table)
    if _enrich_db is not None:
        table = _enrich_db(table)
    
    # Fetch remote stats once per hand
    hand_id = _current_hand_id(table)
    global _HAND_ID_CACHE, _REMOTE_CACHE
    if hand_id != _HAND_ID_CACHE:
        _HAND_ID_CACHE = hand_id
        _REMOTE_CACHE.clear()
    
    remote_stats = {}
    if _fetch_remote is not None:
        profiles = table.get("opponentProfiles") or {}
        for agent_id in profiles:
            if agent_id not in _REMOTE_CACHE:
                _REMOTE_CACHE[agent_id] = _fetch_remote(agent_id) or {}
            remote_stats[agent_id] = _REMOTE_CACHE[agent_id]
    
    result = _hubase_act(table)
    
    out = {"action": result[0] if result else None}
    if isinstance(result, tuple) and len(result) > 1 and result[1] is not None:
        out["amount"] = int(result[1])
    
    msg = result[2] if isinstance(result, tuple) and len(result) > 2 else ""
    local_profiles = table.get("opponentProfiles") or {}
    db_profiles = table.get("opponentProfiles") or {}
    
    # Get separate profiles for local vs db if both trackers ran
    # Both trackers write to same key, so we need to capture before second enrich
    # For now, use what's available
    if msg:
        prefix = _build_prefix(local_profiles, db_profiles, remote_stats)
        full_msg = f"[{prefix}] {msg}" if prefix else msg
        # Truncate to 500 chars
        if len(full_msg) > 500:
            full_msg = full_msg[:497] + "..."
        out["message"] = full_msg
        out["reasoning_text"] = full_msg
    
    return out
