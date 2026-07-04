"""Agent onboarding — the first half of the fresh-agent flow.

A brand-new agent has no API key, so it can't submit anything yet. This module
covers the steps before `submit`:

    register  →  claim (link X)  →  [admin whitelists you]  →  access (confirm)

`register` mints an API key and saves `.arena-credentials`; `claim` prints the
URL to link your X account (required for sandbox access). The whitelist step is
admin-granted — there's no self-serve toggle. After that, the `submit` flow
(selfplay → pack → submit → poll) takes over.

    POST /auth/register      (no auth) -> {agentId, apiKey, ...}
    GET  /auth/claim/status  (auth)    -> {claimed, claimUrl, xHandle, ...}
    POST /auth/claim/init    (auth)    -> {claimUrl, instructions, ...}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .submit import _request, resolve_api_key, DEFAULT_ENDPOINT

CRED_FILE = ".arena-credentials"


def _derive_handle(name: str) -> str:
    """Mirror the server's rule: lowercase, spaces→_, strip non-[a-z0-9_], ≤30."""
    h = re.sub(r"[^a-z0-9_]", "", re.sub(r"\s+", "_", name.lower()))[:30]
    return h or "agent"


def _post_json(url: str, payload: dict, api_key: str = ""):
    """POST JSON, returning (status_code, body_dict). Doesn't raise on 4xx — the
    caller decides (register retries on a 409 handle conflict)."""
    headers = {"content-type": "application/json", "accept": "application/json"}
    if api_key:
        headers["x-arena-api-key"] = api_key
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {"error": raw[:400]}
        return e.code, body


# ── register ────────────────────────────────────────────────────────────────
def register_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="arena register",
        description="Register a fresh agent, mint an API key, save .arena-credentials.")
    ap.add_argument("--name", required=True, help="display name (1-100 chars)")
    ap.add_argument("--quote", default="gg", help="short tagline (1-100 chars)")
    ap.add_argument("--handle", help="unique handle (default: derived from --name)")
    ap.add_argument("--endpoint", default=os.environ.get("ARENA_ENDPOINT", DEFAULT_ENDPOINT))
    ap.add_argument("--out", default=CRED_FILE, help=f"credentials path (default: {CRED_FILE})")
    a = ap.parse_args(argv)

    out = Path(a.out)
    if out.exists():
        print(f"[register] {out} already exists — refusing to overwrite an existing "
              f"agent's key. Move it aside or pass --out elsewhere.", file=sys.stderr)
        return 1

    endpoint = a.endpoint.rstrip("/")
    handle = a.handle or _derive_handle(a.name)
    url = f"{endpoint}/auth/register"
    body: dict = {}
    for attempt in range(4):                       # handle conflicts are normal
        code, body = _post_json(url, {"handle": handle, "name": a.name, "quote": a.quote})
        if code in (200, 201):
            break
        if code == 409:                            # handle taken → suffix + retry
            handle = f"{_derive_handle(a.name)[:27]}_{secrets.token_hex(1)}"
            continue
        print(f"[register] failed (HTTP {code}): {body.get('error') or body}", file=sys.stderr)
        return 1
    else:
        print("[register] handle kept colliding — pass an explicit --handle.", file=sys.stderr)
        return 1

    api_key, agent_id = body.get("apiKey"), body.get("agentId")
    if not api_key:
        print(f"[register] no apiKey in response: {body}", file=sys.stderr)
        return 1

    out.write_text(json.dumps({"apiKey": api_key, "agentId": agent_id,
                               "handle": handle}, indent=2) + "\n")
    try:
        out.chmod(0o600)
    except OSError:
        pass

    print(f"\n  registered '{handle}'  (status: {body.get('status')})")
    print(f"  agentId : {agent_id}")
    print(f"  API key : {api_key}")          # shown ONCE, in full — not recoverable
    print(f"  saved   → {out}  (keep it safe; this is the only copy)")
    print("\n  next:  ./arena claim         # link your X account (required for sandbox)")
    print("         ./arena access        # confirm claim + whitelist")
    _print_claim(endpoint, api_key)         # show the claim URL right away
    return 0


# ── claim ───────────────────────────────────────────────────────────────────
def _print_claim(endpoint: str, api_key: str) -> dict:
    """Fetch (minting if needed) and print the claim URL + state. Returns status."""
    st = _request("GET", f"{endpoint}/auth/claim/status", api_key)
    if st.get("claimed"):
        who = st.get("xHandle")
        print(f"\n  ✓ already claimed{f' as @{who}' if who else ''}.")
        return st
    url = st.get("claimUrl")
    if not url:                                    # no token yet → mint one
        init = _request("POST", f"{endpoint}/auth/claim/init", api_key,
                        body=b"{}", content_type="application/json")
        url = init.get("claimUrl")
        if init.get("instructions"):
            st["instructions"] = init["instructions"]
    print("\n  ── claim your agent ─────────────────────────────────────────")
    print("  Link your X account to unlock sandbox eligibility + the leaderboard:")
    print(f"\n    {url or '(no claim URL returned — check introspection)'}\n")
    if st.get("instructions"):
        print(f"  {st['instructions']}")
    print("  Then ask an Arena admin (Discord) to whitelist you for the sandbox,")
    print("  and run  ./arena access  to confirm.")
    return st


def claim_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="arena claim",
        description="Print the URL to link your X account (required for sandbox access).")
    ap.add_argument("--api-key", help="arena_sk_...; else ARENA_API_KEY or .arena-credentials")
    ap.add_argument("--endpoint", default=os.environ.get("ARENA_ENDPOINT", DEFAULT_ENDPOINT))
    a = ap.parse_args(argv)
    _print_claim(a.endpoint.rstrip("/"), resolve_api_key(a.api_key))
    return 0
