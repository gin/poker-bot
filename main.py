#!/usr/bin/env python3
"""DevFun Arena Texas Hold'em poker loop."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# from poker_bot.strategies.simple import choose_action  # noqa: E402
# from poker_bot.strategies.profiled_counter_adaptive import choose_action  # noqa: E402
from poker_bot.strategies.anti_threshold import choose_action  # noqa: E402
from poker_bot.table import find_agent_seat, is_our_turn  # noqa: E402

BASE_URL = "https://arena.dev.fun/api/arena"
STATE_FILE = os.path.expanduser("~/.arena-poker-state")
CRED_FILE = os.path.expanduser("~/.arena-credentials")
DEFAULT_AGENT_ID = "cmpzvsdsavulpc7zaxq9t2j6c"


def load_credentials(path=CRED_FILE):
    with open(path) as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError(f"Credentials file {path} is empty")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()

    api_key = data.get("apiKey") or data.get("API_KEY")
    competition_id = data.get("competitionId") or data.get("COMPETITION_ID")
    agent_id = data.get("agentId") or data.get("AGENT_ID") or DEFAULT_AGENT_ID

    if not api_key:
        raise ValueError("Missing apiKey/API_KEY in credentials file")
    if not competition_id:
        raise ValueError("Missing competitionId/COMPETITION_ID in credentials file")

    return api_key, competition_id, agent_id


def make_headers(api_key):
    return [
        "-H",
        f"x-arena-api-key: {api_key}",
        "-H",
        "Content-Type: application/json",
    ]


def api(
    method, path, data=None, api_key=None, runner=subprocess.run, base_url=BASE_URL
):
    if api_key is None:
        raise ValueError("api_key is required")

    cmd = ["curl", "-s", "-X", method, f"{base_url}{path}"] + make_headers(api_key)
    if data:
        cmd += ["-d", json.dumps(data)]
    result = runner(cmd, capture_output=True, text=True, timeout=30)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": result.stdout}


def make_api_client(api_key, runner=subprocess.run):
    return lambda method, path, data=None: api(
        method, path, data=data, api_key=api_key, runner=runner
    )


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
    try:
        api_key, competition_id, agent_id = load_credentials()
    except Exception as exc:
        print(f"Failed loading credentials: {exc}", flush=True)
        sys.exit(1)

    api_fn = make_api_client(api_key)
    print("🃏 Poker playing loop starting...", flush=True)
    state = load_state()
    consecutive_empty = 0

    while True:
        try:
            # Heartbeat: mark main loop as alive for cron tick fallback
            state["last_heartbeat_at"] = datetime.now(timezone.utc).isoformat()
            save_state(state)

            # Poll for pending actions
            pending = api_fn(
                "GET", f"/texas/pending-actions?competitionId={competition_id}"
            )

            if "error" in pending and "tables" not in pending:
                print(f"Error polling: {pending}", flush=True)
                time.sleep(5)
                continue

            tables = pending.get("tables", [])

            if not tables:
                consecutive_empty += 1
                if consecutive_empty % 60 == 0:
                    print(
                        f"  ...waiting for table ({consecutive_empty} polls)",
                        flush=True,
                    )
                if consecutive_empty == 30:  # ~1 min of no tables — try rejoining
                    join_resp = api_fn("POST", "/texas/join", {"competitionId": competition_id})
                    kind = join_resp.get("kind", "")
                    if kind == "queued":
                        pos = join_resp.get("lobby", {}).get("position", "?")
                        print(f"  Joined queue at position {pos}", flush=True)
                    elif "error" in join_resp and "chips" in join_resp.get("error", "").lower():
                        print(f"  Bankroll empty, waiting...", flush=True)
                time.sleep(2)
                continue

            # Sort by earliest action deadline
            consecutive_empty = 0
            tables.sort(key=lambda t: t.get("actionDeadlineAt", float("inf")))

            for table in tables:
                table_id = table.get("tableId", "?")
                street = table.get("street", "?")
                pot = table.get("potChips", 0)

                print(f"\n📍 Table {table_id} | {street} | Pot: {pot}", flush=True)

                if not is_our_turn(table, agent_id):
                    acting = table.get("actingSeatNumber")
                    my_seat_num = (find_agent_seat(table, agent_id) or {}).get(
                        "seatNumber"
                    )
                    print(
                        f"  Not our turn (seat {acting}, we are {my_seat_num})",
                        flush=True,
                    )
                    continue

                my_seat = find_agent_seat(table, agent_id)
                result = choose_action(table, my_seat)
                if result[0] is None:
                    print("  No valid action found, skipping", flush=True)
                    continue

                action, amount, message = result
                print(
                    f"  → {action}"
                    + (f" {amount}" if amount else "")
                    + f" | {message}",
                    flush=True,
                )

                body = {
                    "competitionId": competition_id,
                    "tableId": table_id,
                    "action": action,
                    "message": message,
                }
                if amount is not None:
                    body["amount"] = amount

                resp = api_fn("POST", "/texas/action", body)

                if "error" in resp:
                    error = resp.get("error")
                    message = resp.get("message", "")
                    print(
                        f"  ✗ Action rejected: {error} - {message}",
                        flush=True,
                    )
                    if action != "fold":
                        body["action"] = "fold"
                        body["message"] = "Fallback fold after rejection"
                        body.pop("amount", None)
                        resp = api_fn("POST", "/texas/action", body)
                        if "error" not in resp:
                            print("  → Fallback: fold accepted", flush=True)
                else:
                    print("  ✓ Action accepted", flush=True)
                    new_street = resp.get("street", "?")
                    new_pot = resp.get("potChips", 0)
                    print(f"    Street: {new_street} | Pot: {new_pot}", flush=True)

                    state["last_table_id"] = table_id
                    save_state(state)

        except KeyboardInterrupt:
            print("\n🛑 Poker loop stopped", flush=True)
            break
        except Exception as e:
            print(f"Exception: {e}", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
