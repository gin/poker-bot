"""Discover active Arena competitions and label them PvE / PvP.

    python -m arena_sdk comps [--api-key arena_sk_...] [--endpoint ...]

Reads `GET /competition/list-active` and prints each competition's id, name, and
whether submitting to it runs a PvE eval or a PvP ladder, so you know which
id to pass to `submit --competition <id>`.
"""
from __future__ import annotations

import os

from .submit import _request, resolve_api_key, DEFAULT_ENDPOINT


def _classify(c: dict) -> str:
    """Best-effort sandbox-submission label.

    `skillFile` is the most reliable signal (the server routes each comp to its
    skill, e.g. `sandbox-pvp.md` / `sandbox-pve.md`); the comp name is the
    fallback. `?` = not a sandbox-submission comp (likely a live-play lobby).
    """
    cfg = c.get("config") or c
    bench = cfg.get("benchmark") or {}
    pvp = (bench.get("sandboxPvp") or {}) if isinstance(bench, dict) else {}
    skill = (c.get("skillFile") or "").lower()
    name = (c.get("name") or c.get("title") or "").lower()
    # skillFile is authoritative; then explicit config; then the name. Never label
    # PvP from seat count alone (a heads-up PvE eval also has 2 seats).
    if "pvp" in skill or pvp.get("enabled") or "pvp" in name:
        return "PvP"
    if ("pve" in skill or "eval-sandbox" in skill or "poker-eval" in skill
            or "eval" in name or "benchmark" in name or "pve" in name):
        return "PvE"
    if cfg.get("mode") == "benchmark":
        return "PvE"
    return "?"


def list_active(endpoint: str, api_key: str = "") -> list:
    endpoint = endpoint.rstrip("/")
    # /competition/list-active is public (no auth) — a fresh agent can discover
    # comps before it even has a key.
    res = _request("GET", f"{endpoint}/competition/list-active", api_key or "")
    comps = res if isinstance(res, list) else (
        res.get("competitions") or res.get("data") or res.get("items") or [])
    if not comps:
        print("(no active competitions returned)")
        return []
    print(f"{'KIND':<5} {'ID':<30} NAME")
    print("-" * 60)
    for c in comps:
        cid = c.get("id") or c.get("competitionId") or "?"
        name = c.get("name") or c.get("title") or c.get("displayName") or ""
        print(f"{_classify(c):<5} {cid:<30} {name}")
    print("\nSubmit with:  python -m arena_sdk submit "
          "--strategy strategy.py --competition <ID> [--pvp]")
    return comps


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="arena comps",
                                 description="List active Arena competitions (PvE/PvP).")
    ap.add_argument("--api-key", help="arena_sk_...; else ARENA_API_KEY or .arena-credentials")
    ap.add_argument("--endpoint", default=os.environ.get("ARENA_ENDPOINT", DEFAULT_ENDPOINT),
                    help="API base (default: %(default)s)")
    a = ap.parse_args(argv)
    list_active(a.endpoint, resolve_api_key(a.api_key, required=False) or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
