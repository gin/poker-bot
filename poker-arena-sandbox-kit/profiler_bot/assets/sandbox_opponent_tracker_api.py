"""Remote opponent stats fetcher for dev.fun API.

Calls GET /agent/{agentId}/stats?competitionId={competitionId}
with x-arena-api-key header. Caches responses indefinitely.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any


# Load API key from harness/.arena-credentials-luigi-hu
_CREDENTIALS_PATH = Path(__file__).resolve().parent.parent / "harness" / ".arena-credentials-luigi-hu"
_API_KEY: str | None = None
if _CREDENTIALS_PATH.exists():
    try:
        with open(_CREDENTIALS_PATH) as f:
            creds = json.load(f)
            _API_KEY = creds.get("apiKey")
    except Exception:
        pass


API_BASE = "https://api.dev.fun"
COMPETITION_ID = "cmr3n8tft01nilecm1u5jlny7"

_CACHE: dict[str, dict[str, Any]] = {}


def fetch_opponent_stats(agent_id: str) -> dict[str, Any]:
    """Fetch opponent stats from dev.fun API (cached indefinitely).
    
    Returns empty dict if API key missing, network error, or invalid response.
    """
    if agent_id in _CACHE:
        return _CACHE[agent_id]
    
    if not _API_KEY:
        return {}
    
    url = f"{API_BASE}/agent/{agent_id}/stats?competitionId={COMPETITION_ID}"
    req = urllib.request.Request(url, headers={"x-arena-api-key": _API_KEY})
    
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            if isinstance(data, dict):
                _CACHE[agent_id] = data
                return data
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        pass
    
    return {}
