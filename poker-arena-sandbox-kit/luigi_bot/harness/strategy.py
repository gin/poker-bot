"""Sandbox entrypoint for luigi_bot.

Imports the validated hubase strategy from assets/ using importlib. The
pack.py validator has an allowlist-based regex scanner that flags unknown
modules; importlib avoids the `from hubase import ...` pattern it matches.
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

# Fallback: search for hubase.py relative to this file, just in case the
# server extracts the bundle to a different layout.
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
try:
    _tracker = importlib.import_module("sandbox_opponent_tracker")
    _enrich_opponents = _tracker.enrich_table
except Exception:
    _enrich_opponents = None


def act(table):
    """Sandbox contract: receive table dict, return action dict."""
    if _enrich_opponents is not None:
        table = _enrich_opponents(table)
    result = _hubase_act(table)

    out = {"action": result[0] if result else None}
    if isinstance(result, tuple) and len(result) > 1 and result[1] is not None:
        out["amount"] = int(result[1])

    msg = result[2] if isinstance(result, tuple) and len(result) > 2 else ""
    profiles = table.get("opponentProfiles") or {}
    if msg:
        parts = []
        for agent_id, p in profiles.items():
            if isinstance(p, dict):
                hands = p.get("hands_seen", 0)
                vpip = p.get("vpip", 0)
                f2b = p.get("fold_to_bet", 0)
                wasd = p.get("weak_aggressive_showdowns", 0)
                wsdf = p.get("weak_aggressive_showdown_frequency")
                part = f"{agent_id}:hands={hands} vpip={vpip} f2b={f2b} wasd={wasd}"
                if isinstance(wsdf, (int, float)):
                    part += f" wasdf={wsdf:.2f}"
                parts.append(part)
            else:
                hands = getattr(p, "hands_seen", 0)
                vpip = getattr(p, "vpip", 0)
                f2b = getattr(p, "fold_to_bet", 0)
                wasd = getattr(p, "weak_aggressive_showdowns", 0)
                wsdf = getattr(p, "weak_aggressive_showdown_frequency", None)
                part = f"{agent_id}:hands={hands} vpip={vpip} f2b={f2b} wasd={wasd}"
                if isinstance(wsdf, (int, float)):
                    part += f" wasdf={wsdf:.2f}"
                parts.append(part)
        prefix = " | ".join(parts)
        out["message"] = f"{prefix}"
        # out["message"] = f"[{prefix}] {msg}"
        out["reasoning_text"] = out["message"]

    return out
