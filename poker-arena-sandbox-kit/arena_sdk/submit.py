"""Submit a strategy to the dev.fun Arena Sandbox and watch it score.

The SAME `strategy.py` you self-play locally is what you upload.
The server runs it in an isolated sandbox against the panel (PvE) or against other
submitted bots (PvP). PvE vs PvP is decided by the COMPETITION, not a flag here:
submit to an eval comp -> PvE; submit to a sandbox-PvP comp -> PvP ladder.

Endpoints (x-arena-api-key auth):
    GET  {endpoint}/submissions/settings        # preflight: access + settings
    POST {endpoint}/submissions                  # multipart: competitionId, file, template
    GET  {endpoint}/submissions/{id}             # poll status -> bb/100 (+ pvp{} TrueSkill)

Usage:
    python -m arena_sdk submit --strategy examples/poker/strategy.py \
        --competition <competitionId> [--api-key arena_sk_...] \
        [--assets weights/ | --harness dir/] [--pvp|--pve] [--replace]
    python -m arena_sdk submit --strategy examples/poker/strategy.py \
        --competition demo --dry-run            # offline — exercise the whole flow
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Callable, Optional

from .pack import build_bundle, BundleError

# Production endpoint. Override per-call with --endpoint / $ARENA_ENDPOINT.
DEFAULT_ENDPOINT = "https://arena.dev.fun/api/arena"
TERMINAL = {"Succeeded", "Failed", "Cancelled", "TimedOut"}


# ── credentials ─────────────────────────────────────────────────────────────
def resolve_api_key(explicit: Optional[str], *, required: bool = True) -> Optional[str]:
    if explicit:
        return explicit
    if os.environ.get("ARENA_API_KEY"):
        return os.environ["ARENA_API_KEY"]
    for cand in (Path.cwd() / ".arena-credentials",
                 Path.home() / ".arena-credentials"):
        if cand.exists():
            try:
                d = json.loads(cand.read_text())
            except Exception:
                continue
            for k in ("apiKey", "api_key", "arenaApiKey", "key"):
                if isinstance(d, dict) and d.get(k):
                    return str(d[k])
    if not required:
        return None
    raise SystemExit("no API key: pass --api-key, set ARENA_API_KEY, or create "
                     ".arena-credentials ({\"apiKey\": \"arena_sk_...\"})")


# ── HTTP ────────────────────────────────────────────────────────────────────
def _request(method: str, url: str, api_key: str, *, body: Optional[bytes] = None,
             content_type: Optional[str] = None, retries: int = 3) -> dict:
    headers = {"x-arena-api-key": api_key, "accept": "application/json"}
    if content_type:
        headers["content-type"] = content_type
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:400]
            if e.code < 500 or attempt == retries:
                raise SystemExit(f"{method} {url} -> HTTP {e.code}: {detail}")
            last = e
        except Exception as e:
            if attempt == retries:
                raise SystemExit(f"{method} {url} -> {e}")
            last = e
        time.sleep(min(4.0, 0.5 * (2 ** attempt)))
    raise SystemExit(f"{method} {url} failed: {last}")


def _make_mock(expect: Optional[str]) -> Callable:
    """Offline stand-in for the three submission endpoints (--dry-run).

    Mirrors the real shapes so the whole submit→poll path runs with zero network:
    Queued → Running → Succeeded, with a pvp{} block when --pvp is set.
    """
    state = {"polls": 0}

    def mock(method: str, url: str, api_key: str, *, body=None,
             content_type=None, retries: int = 3) -> dict:
        tail = url.rstrip("/").split("/api/arena/")[-1]
        if tail.endswith("submissions/settings"):
            return {"access": {"sandboxBenchmark": {
                        "whitelisted": True, "claimed": True, "code": None,
                        "message": "(dry-run) sandbox access ok"}}}
        if method == "POST" and tail.endswith("submissions"):
            return {"id": "dry-run-0001", "status": "Queued",
                    "template": "static-agent", "competitionId": "dry-run",
                    "targetHands": 20}
        if method == "GET" and tail.startswith("submissions/"):  # GET submissions/:id
            state["polls"] += 1
            if state["polls"] < 2:
                return {"id": "dry-run-0001", "status": "Running", "_simulated": True,
                        "completedHands": 10, "targetHands": 20}
            out = {"id": "dry-run-0001", "status": "Succeeded", "_simulated": True,
                   "completedHands": 20, "targetHands": 20,
                   "rawBbPer100": 120.0, "adjustedBbPer100": 8.10,
                   "traceObjectKey": "(dry-run) arena/sandbox-submissions/.../jobs.zip"}
            if expect == "pvp":
                out["pvp"] = {"botId": "dry-bot", "status": "Active",
                              "completedHands": 20, "targetHands": 5000,
                              "rawBbPer100": 120.0, "adjustedBbPer100": 8.10,
                              "trueskillMu": 25.0, "trueskillSigma": 8.33,
                              "trueskillScore": 0.0}
            return out
        raise SystemExit(f"[dry-run] unmocked request: {method} {url}")

    return mock


def _multipart(fields: dict, file_field: str, filename: str,
               file_bytes: bytes) -> tuple[str, bytes]:
    """Encode multipart/form-data with urllib (no requests dependency)."""
    boundary = f"----devfunsdk{uuid.uuid4().hex}"
    crlf = b"\r\n"
    out = bytearray()
    for name, value in fields.items():
        if value is None:
            continue
        safe = str(value).replace("\r", "").replace("\n", "")   # no header injection
        out += b"--" + boundary.encode() + crlf
        out += f'Content-Disposition: form-data; name="{name}"'.encode() + crlf + crlf
        out += safe.encode() + crlf
    out += b"--" + boundary.encode() + crlf
    out += (f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"').encode() + crlf
    ctype = "application/zip" if filename.endswith(".zip") else "text/x-python"
    out += f"Content-Type: {ctype}".encode() + crlf + crlf
    out += file_bytes + crlf
    out += b"--" + boundary.encode() + b"--" + crlf
    return f"multipart/form-data; boundary={boundary}", bytes(out)


# ── preflight ───────────────────────────────────────────────────────────────
def check_access(endpoint: str, api_key: str, *, req: Callable = _request) -> dict:
    s = req("GET", f"{endpoint}/submissions/settings", api_key)
    access = (s.get("access") or {}).get("sandboxBenchmark") or {}
    wl = access.get("whitelisted")
    if wl is False:
        print(f"[submit] ⚠ access not granted: {access.get('code')} — "
              f"{access.get('message') or 'agent must be claimed + whitelisted'}")
        print("[submit]   ask an admin in Discord for sandbox eval access.")
    elif wl is True:
        print("[submit] access ok (claimed + sandbox eval whitelisted)")
    return s


# ── submit + poll ───────────────────────────────────────────────────────────
def submit(strategy: Optional[str] = None, *, competition_id: str,
           api_key: Optional[str],
           endpoint: str = DEFAULT_ENDPOINT,
           assets: Optional[str] = None, harness: Optional[str] = None,
           expect: Optional[str] = None, watch: bool = True,
           poll_s: float = 5.0, dry_run: bool = False,
           replace: bool = False) -> dict:
    endpoint = endpoint.rstrip("/")
    req: Callable = _make_mock(expect) if dry_run else _request
    if dry_run:
        print("[submit] DRY RUN — no network; exercising the full flow offline.")
        api_key = api_key or "dry-run"
    else:
        api_key = resolve_api_key(api_key)  # works for direct library callers too

    settings = check_access(endpoint, api_key, req=req)
    denied = (((settings.get("access") or {}).get("sandboxBenchmark") or {})
              .get("whitelisted") is False)
    if denied and not dry_run:
        raise SystemExit("[submit] aborting before upload — sandbox access not granted "
                         "(a 403 would follow). Claim your agent + get whitelisted "
                         "(./arena access), then retry.")

    # Build + isolation-validate the bundle (catches a missing-sibling import
    # locally, before you spend a metered submission), then upload the zip.
    try:
        payload = build_bundle(strategy, harness=harness, assets=assets)
    except BundleError as e:
        raise SystemExit(f"[submit] bundle invalid: {e}")

    fields = {"competitionId": competition_id, "template": "static-agent"}
    if replace:
        # PvP: replace your current unfinished active bot (else 409). The backend
        # reads the multipart field `replace`; the new bot's TrueSkill restarts.
        fields["replace"] = "true"
    ctype, body = _multipart(fields, "file", "bundle.zip", payload)
    print(f"[submit] POST {endpoint}/submissions  comp={competition_id} "
          f"({len(payload)} bytes)")
    # POST is non-idempotent — never auto-retry. A retried create after a
    # transient 5xx/timeout would duplicate the submission and burn the PvP
    # daily limit. retries=0 = single attempt. Canonical path has NO trailing
    # slash (matches __introspection) — avoids any redirect that could drop the body.
    res = req("POST", f"{endpoint}/submissions", api_key,
              body=body, content_type=ctype, retries=0)
    sid = res.get("id")
    print(f"[submit] accepted: id={sid} status={res.get('status')} "
          f"targetHands={res.get('targetHands')}")
    if expect == "pvp":
        print("[submit] note: PvP daily limit is 3 valid/pending submissions per UTC day.")
    if not watch or not sid:
        return res
    return poll(sid, api_key=api_key, endpoint=endpoint, poll_s=poll_s, req=req)


def _fmt(v, spec=".2f"):
    return format(v, spec) if isinstance(v, (int, float)) else "?"


def _pvp_rating(pvp: dict):
    """PvP (score, mu, sigma). Reads the current backend names (trueskill*), with
    forward-compat fallbacks (scaleRating / rating / mu / sigma). Picks the first
    NON-None candidate, so a transitional response where the old key is present but
    null (alongside a new key) still renders real numbers, not `?`."""
    def first(*keys):
        for k in keys:
            v = pvp.get(k)
            if v is not None:
                return v
        return None
    return (first("trueskillScore", "scaleRating", "rating"),
            first("trueskillMu", "mu"),
            first("trueskillSigma", "sigma"))


def poll(submission_id: str, *, api_key: str, endpoint: str = DEFAULT_ENDPOINT,
         poll_s: float = 5.0, timeout_s: float = 7200.0,
         req: Callable = _request) -> dict:
    endpoint = endpoint.rstrip("/")
    url = f"{endpoint}/submissions/{submission_id}"
    t0 = time.time()
    last_line = ""
    while True:
        st = req("GET", url, api_key)
        status = st.get("status")
        done, target = st.get("completedHands"), st.get("targetHands")
        prog = f"{done}/{target}" if target else (str(done) if done is not None else "-")
        line = f"[poll] {status} | hands={prog}"
        if st.get("adjustedBbPer100") is not None:
            line += f" | adjBb/100={_fmt(st['adjustedBbPer100'], '+.2f')}"
        pvp = st.get("pvp")
        if pvp:
            line += f" | pvp={pvp.get('status')}"
            score, mu, sigma = _pvp_rating(pvp)
            if score is not None:
                line += (f" TrueSkill={_fmt(score, '.1f')} "
                         f"(mu={_fmt(mu, '.1f')} sigma={_fmt(sigma, '.1f')})")
        if st.get("_simulated"):
            line += "  ·(simulated, not your bot)"
        if line != last_line:
            print(line, flush=True)
            last_line = line
        if status in TERMINAL:
            _print_final(st)
            return st
        if time.time() - t0 > timeout_s:
            print(f"[poll] gave up after {timeout_s:.0f}s (still {status})")
            return st
        time.sleep(poll_s)


def _print_final(st: dict) -> None:
    status = st.get("status")
    mark = "✓" if status == "Succeeded" else "✗"
    print(f"\n{mark} {status}")
    if st.get("_simulated"):
        print("  (DRY RUN — numbers are simulated, NOT your bot's real score)")
    if st.get("errorCode"):
        print(f"  errorCode: {st['errorCode']}")     # stable machine code (1 of 14)
    if st.get("error"):
        print(f"  error: {st['error']}")
    if st.get("rawBbPer100") is not None:
        print(f"  raw bb/100      = {_fmt(st['rawBbPer100'], '+.2f')}")
    if st.get("adjustedBbPer100") is not None:
        print(f"  adjusted bb/100 = {_fmt(st['adjustedBbPer100'], '+.2f')}")
    pvp = st.get("pvp")
    if pvp:
        score, mu, sigma = _pvp_rating(pvp)
        print(f"  PvP bot {pvp.get('status')} | TrueSkill score = "
              f"{_fmt(score)} (mu={_fmt(mu)}, sigma={_fmt(sigma)}) "
              f"over {pvp.get('completedHands')} hands")
        # A top-level Succeeded does NOT mean the bot is healthy — a bot can
        # activate then end Failed. Surface pvp.error so it isn't missed.
        if pvp.get("error"):
            print(f"  pvp.error: {pvp['error']}")
        if pvp.get("status") in ("Failed", "Discarded"):
            print(f"  ⚠️ bot is not live (pvp.status={pvp.get('status')}) — "
                  "fix and resubmit")
    if st.get("traceObjectKey"):
        print(f"  trace: {st['traceObjectKey']}")


def access_main(argv=None) -> int:
    """`access` subcommand — check claim + sandbox eval whitelist (the 403 gate)."""
    import argparse
    ap = argparse.ArgumentParser(prog="arena access",
                                 description="Check your sandbox eval access.")
    ap.add_argument("--api-key", help="arena_sk_...; else ARENA_API_KEY or .arena-credentials")
    ap.add_argument("--endpoint", default=os.environ.get("ARENA_ENDPOINT", DEFAULT_ENDPOINT),
                    help="API base (default: %(default)s)")
    a = ap.parse_args(argv)
    s = check_access(a.endpoint.rstrip("/"), resolve_api_key(a.api_key))
    acc = (s.get("access") or {}).get("sandboxBenchmark") or {}
    if not acc.get("whitelisted"):
        print("[access] not ready — claim your agent (link your X account) and ask "
              "an admin in Discord to enable sandbox eval, then re-check.")
        return 1
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="arena submit",
                                 description="Submit strategy.py to the Arena Sandbox (PvE/PvP).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--strategy", help="a single self-contained strategy.py")
    src.add_argument("--harness", help="dir copied into harness/ (multi-file bot; needs strategy.py)")
    ap.add_argument("--competition", required=True, help="competitionId (decides PvE vs PvP)")
    ap.add_argument("--api-key", help="arena_sk_...; else ARENA_API_KEY or .arena-credentials")
    ap.add_argument("--endpoint", default=os.environ.get("ARENA_ENDPOINT", DEFAULT_ENDPOINT),
                    help="API base (default: %(default)s)")
    ap.add_argument("--assets", help="dir copied into assets/ (trained weights / lookup tables)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pvp", dest="expect", action="store_const", const="pvp",
                   help="treat the comp as PvP (3 submissions/UTC-day; --replace to swap)")
    g.add_argument("--pve", dest="expect", action="store_const", const="pve")
    ap.add_argument("--no-watch", dest="watch", action="store_false",
                    help="submit and exit without polling")
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise the full flow offline (no network, no API key)")
    ap.add_argument("--replace", action="store_true",
                    help="PvP: replace your current unfinished active bot (resets its TrueSkill)")
    a = ap.parse_args(argv)
    api_key = resolve_api_key(a.api_key, required=not a.dry_run)
    submit(a.strategy, competition_id=a.competition, api_key=api_key,
           endpoint=a.endpoint, assets=a.assets, harness=a.harness,
           expect=a.expect, watch=a.watch, dry_run=a.dry_run, replace=a.replace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
