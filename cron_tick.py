#!/usr/bin/env python3
"""Single-shot tick for cron: check tables, act, join queue if idle."""

import json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poker_bot.table import find_agent_seat, is_our_turn
from poker_bot.strategies.anti_threshold import choose_action

BASE_URL = "https://arena.dev.fun/api/arena"
CRED_FILE = os.path.expanduser("~/.arena-credentials")
STATE_FILE = os.path.expanduser("~/.arena-poker-state")

def load_credentials(path=CRED_FILE):
    with open(path) as f:
        raw = f.read().strip()
    if not raw:
        raise ValueError("Empty credentials")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data.get("apiKey"), data.get("competitionId"), data.get("agentId")

def api(method, path, data=None, api_key=None):
    cmd = ["curl", "-s", "-X", method, f"{BASE_URL}{path}",
           "-H", f"x-arena-api-key: {api_key}",
           "-H", "Content-Type: application/json"]
    if data:
        cmd += ["-d", json.dumps(data)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"error": r.stdout}

def load_state(path=STATE_FILE):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

def save_state(state, path=STATE_FILE):
    with open(path, "w") as f:
        json.dump(state, f, indent=2)

def main():
    api_key, competition_id, agent_id = load_credentials()
    if not api_key or not competition_id:
        print("Missing credentials", flush=True)
        return

    state = load_state()

    # If main.py loop is alive (heartbeat < 120s old), skip — don't duplicate actions
    hb = state.get("last_heartbeat_at")
    if hb:
        try:
            hb_dt = datetime.fromisoformat(hb)
            age = datetime.now(timezone.utc) - hb_dt
            if age.total_seconds() < 120:
                print(f"Main loop heartbeat {age.total_seconds():.0f}s ago — active, skipping tick",
                      flush=True)
                return
        except (ValueError, TypeError):
            pass  # malformed timestamp — proceed as fallback

    # 1. Check pending actions
    pending = api("GET", f"/texas/pending-actions?competitionId={competition_id}",
                  api_key=api_key)
    tables = pending.get("tables", [])

    if tables:
        tables.sort(key=lambda t: t.get("actionDeadlineAt", float("inf")))
        for table in tables:
            table_id = table.get("tableId", "?")
            if not is_our_turn(table, agent_id):
                continue
            my_seat = find_agent_seat(table, agent_id)
            if not my_seat:
                continue
            result = choose_action(table, my_seat)
            action, amount, message = result
            if action is None:
                continue
            body = {
                "competitionId": competition_id,
                "tableId": table_id,
                "action": action,
                "message": message,
            }
            if amount is not None:
                body["amount"] = amount
            resp = api("POST", "/texas/action", body, api_key=api_key)
            if "error" in resp:
                print(f"Table {table_id}: {action} rejected: {resp.get('error')}", flush=True)
                if action != "fold":
                    body["action"] = "fold"
                    body["message"] = "Fallback fold"
                    body.pop("amount", None)
                    api("POST", "/texas/action", body, api_key=api_key)
            else:
                print(f"Table {table_id}: {action} accepted (pot={resp.get('potChips', '?')})", flush=True)
                state["last_table_id"] = table_id
    else:
        print("No pending actions, checking join status...", flush=True)
        join_resp = api("POST", "/texas/join",
                        {"competitionId": competition_id}, api_key=api_key)
        kind = join_resp.get("kind", "")
        if kind == "queued":
            pos = join_resp.get("lobby", {}).get("position", "?")
            print(f"Joined queue at position {pos}", flush=True)
        elif "error" in join_resp:
            err = join_resp.get("error", "")
            if "chips" in err.lower():
                print("Insufficient chips — need rebuy", flush=True)
            else:
                print(f"Join response: {join_resp}", flush=True)
        else:
            print(f"Join response: {json.dumps(join_resp)[:200]}", flush=True)

    save_state(state)
    print("Tick done", flush=True)

if __name__ == "__main__":
    main()