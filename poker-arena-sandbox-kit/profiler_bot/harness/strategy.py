"""Sandbox entrypoint for profiler_bot.

Imports the validated hubase strategy from assets/ using importlib.
At runtime in the sandbox, assets/ is added to sys.path explicitly before
calling importlib so the bundled hubase.py can be found.
"""

import importlib
import json
import os
import sys
import types
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

_hand_utils = importlib.import_module("hand_utils")
_poker_bot_pkg = sys.modules.setdefault("poker_bot", types.ModuleType("poker_bot"))
setattr(_poker_bot_pkg, "hand_utils", _hand_utils)
sys.modules.setdefault("poker_bot.hand_utils", _hand_utils)
_guards_pkg = sys.modules.setdefault(
    "poker_bot.guards", types.ModuleType("poker_bot.guards")
)
_guards_pkg.__path__ = [str(_expected_assets / "guards")]
setattr(_poker_bot_pkg, "guards", _guards_pkg)

_hubase = importlib.import_module("hubase")
_hubase_act = _hubase.act

_GuardContext = None
_guard_pre = None
_guard_post = None
try:
    _guard_context_mod = importlib.import_module("poker_bot.guards.context")
    _guard_pre_mod = importlib.import_module("poker_bot.guards.guard_pre")
    _guard_post_mod = importlib.import_module("poker_bot.guards.guard_post")
    _GuardContext = _guard_context_mod.GuardContext
    _guard_pre = _guard_pre_mod.guard_pre
    _guard_post = _guard_post_mod.guard_post
except Exception:
    pass

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
    all_agent_ids = (
        set(local_profiles.keys()) | set(db_profiles.keys()) | set(remote_stats.keys())
    )

    for agent_id in all_agent_ids:
        local = local_profiles.get(agent_id, {})
        db = db_profiles.get(agent_id, {})
        remote = remote_stats.get(agent_id, {})

        # Get opponent name from any source
        name = (
            local.get("name") or db.get("name") or remote.get("agentName") or agent_id
        )

        # Local (in-memory) stats
        l_hands = local.get("hands_seen", 0)
        l_vpip = local.get("vpip", 0)
        # l_f2b = local.get("fold_to_bet", 0)
        # l_wasd = local.get("weak_aggressive_showdowns", 0)
        # local_part = f"local:hands={l_hands} vpip={l_vpip} f2b={l_f2b} wasd={l_wasd}"
        local_part = f"mem:hands={l_hands} vpip={l_vpip}"

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
            api_part = f"api: hands={r_hands} vpip={r_vpip} f2b={r_f2b} wasd={r_wasd}"
            if isinstance(r_wsdf, (int, float)):
                api_part += f" wasdf={r_wsdf:.2f}"
        else:
            api_part = "api: 0"

        # Combine: [Name | local:... | db:... | API:...]
        opponent_part = f"{name} | {local_part} | {db_part} | {api_part}"
        parts.append(opponent_part)

    return " | ".join(parts)


def _seat_agent_id(seat: dict) -> str | None:
    value = seat.get("agentId") or seat.get("agent_id")
    if value:
        return str(value)
    seat_number = seat.get("seatNumber")
    if seat_number is not None:
        return f"seat-{seat_number}"
    return None


def _hero_seat(table: dict) -> dict:
    my_seat_num = table.get("actingSeatNumber") or table.get("selfSeatNumber")
    for seat in table.get("seats") or []:
        if seat.get("seatNumber") == my_seat_num:
            return seat
    return {
        "seatNumber": my_seat_num,
        "holeCards": table.get("holeCards", table.get("hero_cards", [])),
        "currentBetChips": table.get("currentBetChips", table.get("bet", 0)),
        "stackChips": table.get("stackChips", table.get("stack", 0)),
        "bet": table.get("bet", 0),
    }


def _opponent_agent_ids(table: dict) -> list[str]:
    hero = _hero_seat(table)
    hero_id = _seat_agent_id(hero)
    hero_seat_num = hero.get("seatNumber")
    agent_ids = []
    for seat in table.get("seats") or []:
        agent_id = _seat_agent_id(seat)
        if not agent_id:
            continue
        if hero_id and agent_id == hero_id:
            continue
        if hero_seat_num is not None and seat.get("seatNumber") == hero_seat_num:
            continue
        agent_ids.append(agent_id)
    return agent_ids


def _stat_to_count(value, sample_size: int, *, rate_hint: bool = False) -> int:
    if not isinstance(value, (int, float)):
        return 0
    numeric = float(value)
    if rate_hint or numeric <= 1:
        return max(0, int(round(numeric * sample_size)))
    if sample_size > 0 and numeric <= sample_size:
        return int(round(numeric))
    if numeric <= 100:
        return max(0, int(round((numeric / 100.0) * sample_size)))
    return int(round(numeric))


def _remote_sample_size(remote: dict) -> int:
    for key in ("hands_seen", "handsSeen", "hands", "sampleSize", "n"):
        value = remote.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return 0


def _merge_remote_stats(table: dict, remote_stats: dict) -> None:
    profiles = dict(table.get("opponentProfiles") or {})
    for agent_id, remote in remote_stats.items():
        if not isinstance(remote, dict) or not remote:
            continue
        profile = dict(profiles.get(agent_id) or {})
        sample_size = _remote_sample_size(remote)
        local_hands = int(profile.get("hands_seen") or 0)
        api_source_used = sample_size > local_hands
        merged_hands = max(local_hands, sample_size)

        profile.setdefault("agent_id", agent_id)
        profile["name"] = (
            profile.get("name") or remote.get("agentName") or remote.get("name")
        )
        profile["api_stats"] = remote
        profile["api_source_used"] = api_source_used
        profile["api_sample_size"] = sample_size
        profile["stats_json"] = json.dumps(remote, separators=(",", ":"))

        api_aggr = remote.get("af")
        if api_aggr is None:
            api_aggr = remote.get("aggression_frequency")
        if isinstance(api_aggr, (int, float)):
            profile["api_aggr_freq"] = float(api_aggr)
            if api_source_used:
                profile["aggression_frequency"] = float(api_aggr)

        if api_source_used and sample_size > 0:
            profile["hands_seen"] = merged_hands
            for remote_key, local_key, rate_hint in (
                ("vpip", "vpip", False),
                ("pfr", "pfr", False),
                ("fold_to_bet_frequency", "fold_to_bet", True),
                ("foldToBet", "fold_to_bet", False),
                (
                    "weak_aggressive_showdown_frequency",
                    "weak_aggressive_showdowns",
                    True,
                ),
                ("bluffPct", "weak_aggressive_showdowns", True),
            ):
                if remote_key in remote:
                    profile[local_key] = max(
                        int(profile.get(local_key) or 0),
                        _stat_to_count(
                            remote.get(remote_key), sample_size, rate_hint=rate_hint
                        ),
                    )
            if "fold_to_bet" in profile:
                profile["opportunities_to_fold_to_bet"] = max(
                    int(profile.get("opportunities_to_fold_to_bet") or 0),
                    sample_size,
                )
            if "weak_aggressive_showdowns" in profile:
                profile["showdowns"] = max(
                    int(profile.get("showdowns") or 0), sample_size
                )

        profiles[agent_id] = profile
    table["opponentProfiles"] = profiles


def _guard_context(table: dict):
    if _GuardContext is None:
        return None
    try:
        return _GuardContext.build(table, _hero_seat(table))
    except Exception:
        return None


def _run_pre_guard(table: dict):
    if _guard_pre is None:
        return None
    ctx = _guard_context(table)
    if ctx is None:
        return None
    try:
        return _guard_pre.run_pre(ctx)
    except Exception:
        return None


def _run_post_guard(table: dict, proposed: tuple):
    if _guard_post is None:
        return None
    ctx = _guard_context(table)
    if ctx is None:
        return None
    try:
        return _guard_post.run_post(ctx, proposed)
    except Exception:
        return None


def _allowed_actions(table: dict) -> list[str]:
    allowed = table.get("allowedActions") or {}
    return [str(action).lower() for action in allowed.get("availableActions") or []]


def _range_for(allowed: dict, action: str) -> tuple[int, int] | None:
    if action == "raise":
        range_data = allowed.get("raiseRange") or {}
        minimum = allowed.get("minRaiseTo") or range_data.get("min")
    elif action == "bet":
        range_data = allowed.get("betRange") or {}
        minimum = allowed.get("minBet") or range_data.get("min")
    else:
        return None
    maximum = range_data.get("max") or allowed.get("maxCommit")
    try:
        lo = int(minimum or 0)
        hi = int(maximum or 0)
    except (TypeError, ValueError):
        return None
    if lo <= 0:
        return None
    if hi <= 0:
        hi = lo
    return lo, max(lo, hi)


def _fallback_action(table: dict, reason: str) -> tuple[str, int | None, str]:
    allowed = table.get("allowedActions") or {}
    available = _allowed_actions(table)
    if "check" in available:
        return "check", None, reason
    if "call" in available:
        return "call", None, reason
    if "fold" in available:
        return "fold", None, reason
    for action in ("bet", "raise"):
        if action in available:
            amount_range = _range_for(allowed, action)
            if amount_range is not None:
                return action, amount_range[0], reason
    if available:
        return available[0], None, reason
    return "fold", None, reason


def _legalize_result(table: dict, result) -> tuple[str, int | None, str]:
    allowed = table.get("allowedActions") or {}
    available = _allowed_actions(table)
    if isinstance(result, dict):
        action = result.get("action")
        amount = result.get("amount")
        message = result.get("reasoning_text") or result.get("message") or ""
    elif isinstance(result, (tuple, list)):
        action = result[0] if len(result) > 0 else None
        amount = result[1] if len(result) > 1 else None
        message = result[2] if len(result) > 2 else ""
    elif isinstance(result, str):
        action = result
        amount = None
        message = ""
    else:
        return _fallback_action(table, "fallback: strategy returned no action")

    action = str(action or "").lower().replace("_", "-")
    if action == "all-in" and "all-in" not in available:
        action = "raise" if "raise" in available else "bet"
    if available and action not in available:
        return _fallback_action(table, f"fallback: illegal action {action or 'none'}")

    if action in {"fold", "check", "call"}:
        return action, None, str(message or "")
    if action in {"bet", "raise", "all-in"}:
        if action == "all-in":
            maximum = allowed.get("allInToAmount") or allowed.get("maxCommit")
            try:
                return action, int(maximum), str(message or "")
            except (TypeError, ValueError):
                return _fallback_action(table, "fallback: invalid all-in amount")
        amount_range = _range_for(allowed, action)
        if amount_range is None:
            return _fallback_action(table, f"fallback: no legal {action} range")
        lo, hi = amount_range
        try:
            sized = int(amount if amount is not None else lo)
        except (TypeError, ValueError):
            sized = lo
        return action, max(lo, min(sized, hi)), str(message or "")
    return _fallback_action(table, f"fallback: unsupported action {action or 'none'}")


def _output_from_result(table: dict, result) -> dict:
    action, amount, _message = _legalize_result(table, result)
    out = {"action": action}
    if amount is not None and action in {"bet", "raise", "all-in"}:
        out["amount"] = int(amount)
    return out


def act(table):
    """Sandbox contract: receive table dict, return action dict."""
    # Enrich with both local trackers
    local_profiles = {}
    db_profiles = {}
    if _enrich_local is not None:
        try:
            table = _enrich_local(table)
            local_profiles = dict(table.get("opponentProfiles") or {})
        except Exception:
            local_profiles = dict(table.get("opponentProfiles") or {})
    if _enrich_db is not None:
        try:
            table = _enrich_db(table)
            db_profiles = dict(table.get("opponentProfiles") or {})
        except Exception:
            db_profiles = dict(table.get("opponentProfiles") or {})
    if not local_profiles:
        local_profiles = dict(table.get("opponentProfiles") or {})
    if not db_profiles:
        db_profiles = dict(table.get("opponentProfiles") or {})

    # Fetch remote stats once per hand
    hand_id = _current_hand_id(table)
    global _HAND_ID_CACHE, _REMOTE_CACHE
    if hand_id != _HAND_ID_CACHE:
        _HAND_ID_CACHE = hand_id
        _REMOTE_CACHE.clear()

    remote_stats = {}
    if _fetch_remote is not None:
        agent_ids = set(table.get("opponentProfiles") or {}) | set(
            _opponent_agent_ids(table)
        )
        for agent_id in agent_ids:
            if agent_id not in _REMOTE_CACHE:
                try:
                    _REMOTE_CACHE[agent_id] = _fetch_remote(agent_id) or {}
                except Exception:
                    _REMOTE_CACHE[agent_id] = {}
            remote_stats[agent_id] = _REMOTE_CACHE[agent_id]
    try:
        _merge_remote_stats(table, remote_stats)
    except Exception:
        pass

    guard_id = None
    pre_result = _run_pre_guard(table)
    if pre_result is not None:
        result, guard_id = pre_result
        if len(result) < 3:
            result = (result[0], result[1] if len(result) > 1 else None, "")
    else:
        try:
            result = _hubase_act(table)
        except Exception:
            result = _fallback_action(table, "fallback: strategy exception")
        post_result = _run_post_guard(table, result)
        if post_result is not None:
            result, guard_id = post_result

    result = _legalize_result(table, result)
    out = _output_from_result(table, result)

    msg = result[2] if len(result) > 2 else ""
    if msg:
        prefix = _build_prefix(local_profiles, db_profiles, remote_stats)
        if guard_id:
            prefix = f"{prefix} | guard={guard_id}" if prefix else f"guard={guard_id}"
        full_msg = f"[{prefix}] {msg}" if prefix else msg
        # Truncate to 500 chars
        if len(full_msg) > 500:
            full_msg = full_msg[:497] + "..."
        # out["message"] = full_msg
        # out["reasoning_text"] = full_msg
        out["message"] = full_msg
        out["reasoning_text"] = full_msg

    return out
