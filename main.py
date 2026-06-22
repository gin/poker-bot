#!/usr/bin/env python3
"""DevFun Arena Poker loop"""

import json
import os
import secrets
import string
import subprocess
import sys
import time
from datetime import UTC, datetime
from dataclasses import dataclass
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from poker_bot.arena_stats import (  # noqa: E402
    ArenaStatsFetcher,
    schedule_table_opponent_stats,
)
from poker_bot.opponent_store import (  # noqa: E402
    connect,
    create_telemetry_run,
    fill_hand_telemetry_outcome_from_delta,
    fill_hand_telemetry_outcome_from_replay,
    increment_hand_seen,
    load_profile,
    load_profiles_for_agents,
    record_decision_telemetry,
    record_observed_action,
    update_hand_telemetry_outcome,
)
from poker_bot.strategies.loader import load_strategy  # noqa: E402
from poker_bot.table import find_agent_seat, is_our_turn  # noqa: E402

from poker_bot.langfuse_tracing import (  # noqa: E402
    flush_langfuse,
    span_action_result,
    span_decision,
    span_strategy_call,
    span_table_process,
    trace_hand,
)

BASE_URL = "https://arena.dev.fun/api/arena"
STATE_FILE = os.environ.get(
    "POKER_BOT_STATE",
    os.path.expanduser("~/.arena-poker-state"),
)
CRED_FILE = os.environ.get(
    "POKER_BOT_CREDENTIALS",
    os.path.expanduser("~/.arena-credentials"),
)
STRATEGY_CONFIG_FILE = os.path.expanduser("~/.arena-poker-strategy")
STRATEGY_ENV_VAR = "POKER_BOT_STRATEGY"
DEFAULT_AGENT_ID = "cmpzvsdsavulpc7zaxq9t2j6c"
DEFAULT_STRATEGY_NAME = "flattened_v2"

# Raw arena table payload dump (diagnostic). When enabled, the unmodified table
# dict received from the arena is written to ./raw-data/ before any local
# normalization mutates it. Used to discover the real field names the arena uses
# (e.g. the dealer-button key, which the telemetry recorder currently fails to
# resolve, leaving button_seat_number / hero_position NULL in live data).
# Enabled by default; set POKER_BOT_RAW_DUMP=0 to disable. Capped per run so it
# cannot fill the disk.
RAW_DUMP_DIR = Path(
    os.environ.get(
        "POKER_BOT_RAW_DUMP_DIR",
        str(Path(__file__).resolve().parent / "raw-data"),
    )
)
RAW_DUMP_MAX_FILES = int(os.environ.get("POKER_BOT_RAW_DUMP_MAX", "200"))

PUBLIC_MESSAGE_ALPHABET = string.ascii_letters + string.digits
POLL_INTERVAL_SECONDS = 2
ERROR_POLL_INTERVAL_SECONDS = 5
JOIN_RETRY_SECONDS = 30
# How often the bot reconciles its decision_telemetry rows with the arena's
# /agent/{id}/replays endpoint while idle. Throttled so the API isn't hit
# on every 2s poll, but recent enough that offline analysis is near-real-time.
REPLAY_BACKFILL_INTERVAL_SECONDS = int(
    os.environ.get("POKER_BOT_REPLAY_BACKFILL_INTERVAL", "60")
)
REPLAY_BACKFILL_LIMIT = int(os.environ.get("POKER_BOT_REPLAY_BACKFILL_LIMIT", "50"))


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


def load_strategy_name(path=STRATEGY_CONFIG_FILE, environ=os.environ):
    env_strategy = environ.get(STRATEGY_ENV_VAR, "").strip()
    if env_strategy:
        return env_strategy

    try:
        with open(path) as f:
            raw = f.read().strip()
    except OSError:
        return DEFAULT_STRATEGY_NAME

    if not raw:
        return DEFAULT_STRATEGY_NAME

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        for line in raw.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
        return DEFAULT_STRATEGY_NAME

    if isinstance(data, str):
        return data.strip() or DEFAULT_STRATEGY_NAME
    if isinstance(data, dict):
        for key in ("strategy", "strategyName", "STRATEGY"):
            value = str(data.get(key, "")).strip()
            if value:
                return value
    return DEFAULT_STRATEGY_NAME


def load_configured_strategy(path=STRATEGY_CONFIG_FILE, environ=os.environ):
    strategy_name = load_strategy_name(path=path, environ=environ)
    return strategy_name, load_strategy(strategy_name)


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
    result = runner(cmd, capture_output=True, text=True, timeout=90)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": result.stdout}


def make_api_client(api_key, runner=subprocess.run):
    return lambda method, path, data=None: api(
        method, path, data=data, api_key=api_key, runner=runner
    )


def public_action_message(length=16):
    # return "".join(secrets.choice(PUBLIC_MESSAGE_ALPHABET) for _ in range(length))
    return "glhf"


def action_request_body(_competition_id, table_id, action, amount=None):
    body = {
        "tableId": table_id,
        "action": action,
        "message": public_action_message(),
    }
    if amount is not None:
        body["amount"] = amount
    return body


def load_state(path=STATE_FILE):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state, path=STATE_FILE):
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def update_queue_state(state, pending, now=None):
    now = now or datetime.now(UTC)
    participant = pending.get("participant") or {}
    lobby = pending.get("lobby")
    runner = pending.get("runner") or {}

    state["last_pending_at"] = now.isoformat()
    state["participant_chip_state"] = participant.get("chipState")
    state["participant_bankroll_chips"] = participant.get("bankrollChips")
    state["participant_table_chips"] = participant.get("tableChips")
    state["runner_active_table_count"] = runner.get("activeTableCount")
    state["runner_can_stop_safely"] = runner.get("canStopSafely")

    if lobby:
        state["lobby_position"] = lobby.get("position")
        state["lobby_total"] = lobby.get("total")
        state["lobby_joined_at"] = lobby.get("joinedAt")
    else:
        state["lobby_position"] = None
        state["lobby_total"] = None
        state["lobby_joined_at"] = None


def should_attempt_join(pending, state=None, now=None):
    state = state or {}
    now = time.time() if now is None else now
    if pending.get("tables"):
        return False
    if pending.get("lobby") is not None:
        return False

    participant = pending.get("participant") or {}
    runner = pending.get("runner") or {}
    chip_state = participant.get("chipState")
    if chip_state is not None and chip_state != "available":
        return False
    if int(runner.get("activeTableCount") or 0) != 0:
        return False

    last_attempt = float(state.get("last_join_attempt_epoch") or 0)
    return now - last_attempt >= JOIN_RETRY_SECONDS


def record_join_attempt(state, join_resp, now=None):
    now = time.time() if now is None else now
    state["last_join_attempt_epoch"] = now
    state["last_join_attempt_at"] = datetime.fromtimestamp(now, UTC).isoformat()
    state["last_join_response_kind"] = join_resp.get("kind")
    state["last_join_response_error"] = join_resp.get("error")
    state["last_join_response_message"] = join_resp.get("message")

    lobby = join_resp.get("lobby") or {}
    state["last_join_lobby_position"] = lobby.get("position")
    state["last_join_lobby_total"] = lobby.get("total")


def describe_idle_state(pending):
    participant = pending.get("participant") or {}
    runner = pending.get("runner") or {}
    lobby = pending.get("lobby")
    chip_state = participant.get("chipState") or "unknown"

    if chip_state == "locked_in_play" or int(runner.get("activeTableCount") or 0) > 0:
        return "chips locked in play, waiting for our turn or settlement"
    if lobby:
        position = lobby.get("position", "?")
        total = lobby.get("total", "?")
        return f"queued position {position}/{total}"
    if chip_state == "available":
        return "available for matchmaking"
    if chip_state == "busted":
        return "busted, rebuy required"
    return f"idle chip_state={chip_state}"


def is_not_turn_rejection(response):
    error = str(response.get("error") or "").lower()
    message = str(response.get("message") or "").lower()
    text = f"{error} {message}"
    return (
        ("not this agent" in text and "turn" in text)
        or "not your turn" in text
        or "not this player's turn" in text
    )


def table_action_skip_reason(table, agent_id):
    my_seat = find_agent_seat(table, agent_id)
    if my_seat is None:
        return "agent is not seated at table"
    if not is_our_turn(table, agent_id):
        acting = table.get("actingSeatNumber") or table.get("actingAgentId")
        my_seat_num = my_seat.get("seatNumber")
        return f"not our turn (acting {acting}, we are {my_seat_num})"
    available = (table.get("allowedActions") or {}).get("availableActions") or []
    if not available:
        return "no available actions in table snapshot"
    return None


def telemetry_enabled():
    return os.environ.get("POKER_BOT_TELEMETRY", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def init_live_telemetry(state, competition_id, strategy_name=DEFAULT_STRATEGY_NAME):
    if not telemetry_enabled():
        return None, None
    conn = connect(telemetry=True)
    if state.get("telemetry_strategy") != strategy_name:
        state.pop("telemetry_run_id", None)
        state["telemetry_decision_indexes"] = {}
        state["telemetry_strategy"] = strategy_name
    run_id = state.get("telemetry_run_id")
    run_id = create_telemetry_run(
        conn,
        strategy=strategy_name,
        opponent="arena",
        players=None,
        platform="arena",
        metadata_json=json.dumps({"competitionId": competition_id}),
        run_id=run_id,
    )
    state["telemetry_run_id"] = run_id
    state.setdefault("telemetry_decision_indexes", {})
    return conn, run_id


def _process_hand_identity(table, state):
    """Resolve a stable per-hand identifier and detect hand boundaries.

    The arena payload does not expose a handId field (verified empirically —
    ``startedAt`` is constant across hands at the same table, and no
    ``handId``/``handNumber`` key exists). Two hands at the same table would
    otherwise collide on the same ``hand_id`` in decision_telemetry, causing
    outcome backfills to overwrite each other.

    Hand identity is synthesized as ``f"{tableId}:{boundary_iso}"`` where
    ``boundary_iso`` is the timestamp at which we first observed the current
    hand at this table. A new hand is detected when:

    * this is the first snapshot we've ever seen at this ``tableId``, OR
    * ``street == "Preflop"`` is observed after we had already recorded a
      non-Preflop street for the previous hand (Flop/Turn/River/Showdown) —
      i.e. the previous hand has ended and a new one has begun.

    Returns:
        (current_hand_id, is_new_hand, prev_hand_id, prev_last_stack,
         prev_last_committed) — ``prev_*`` fields are valid only when
        ``is_new_hand`` is True; otherwise they are None.

    State shape (mutated in place):
        ``state["live_hand_state"][table_id]`` is a dict with keys
        ``hand_id``, ``last_street``, ``last_hero_stack``,
        ``last_hero_committed``.
    """
    table_id = str(table.get("tableId") or table.get("id") or "unknown-table")
    street = table.get("street")
    live = state.setdefault("live_hand_state", {})
    hand_info = live.get(table_id)

    # A new hand is detected when:
    #   - this is the first snapshot we've ever seen at this tableId, OR
    #   - we now see Preflop and the last_street we recorded was a non-Preflop
    #     street (Flop/Turn/River/Showdown), which means the previous hand has
    #     ended and a new one has begun.
    if hand_info is None or (
        street == "Preflop" and hand_info.get("last_street") not in (None, "Preflop")
    ):
        is_new_hand = True
    else:
        is_new_hand = False

    if is_new_hand:
        if hand_info is None:
            prev_hand_id = None
            prev_stack = None
            prev_committed = None
        else:
            prev_hand_id = hand_info.get("hand_id")
            prev_stack = hand_info.get("last_hero_stack")
            prev_committed = hand_info.get("last_hero_committed")
        boundary = datetime.now(UTC).isoformat()
        live[table_id] = {
            "hand_id": f"{table_id}:{boundary}",
            "last_street": street,
            "last_hero_stack": None,
            "last_hero_committed": None,
        }
        return live[table_id]["hand_id"], True, prev_hand_id, prev_stack, prev_committed

    # hand_info is guaranteed non-None here: we returned in the is_new_hand branch
    # when hand_info was None or needed replacement.
    assert hand_info is not None
    hand_info["last_street"] = street

    return hand_info["hand_id"], False, None, None, None


def _derive_hand_id(table, state):
    """Public (within main) hand-id resolver used by the telemetry writers.

    Idempotent for repeated calls within the same snapshot — calling this
    multiple times on the same ``table`` returns the same ``hand_id`` without
    re-triggering boundary detection.
    """
    hand_id, _, _, _, _ = _process_hand_identity(table, state)
    return hand_id


def _record_hand_stack_state(table, state, hero_agent_id):
    """Record hero's stackChips/totalCommittedChips under the current hand_id.

    These values are consumed by ``backfill_hand_outcome_from_stack_delta``
    on the next hand boundary to estimate the previous hand's hero_net_chips.
    Called once per snapshot, after ``_process_hand_identity`` has resolved
    the current hand_id.
    """
    table_id = str(table.get("tableId") or table.get("id") or "unknown-table")
    hand_info = state.get("live_hand_state", {}).get(table_id)
    if hand_info is None:
        return
    hero_seat = next(
        (s for s in table.get("seats", []) if s.get("agentId") == hero_agent_id),
        None,
    )
    if hero_seat is None:
        return
    hand_info["last_hero_stack"] = _safe_int(hero_seat.get("stackChips"))
    hand_info["last_hero_committed"] = _safe_int(hero_seat.get("totalCommittedChips"))


def record_live_decision(
    telemetry_conn,
    telemetry_run_id,
    state,
    table,
    my_seat,
    action,
    amount,
    message,
    strategy_name=DEFAULT_STRATEGY_NAME,
):
    if telemetry_conn is None or telemetry_run_id is None:
        return
    table_id = str(table.get("tableId") or table.get("id") or "unknown-table")
    indexes = state.setdefault("telemetry_decision_indexes", {})
    decision_index = int(indexes.get(table_id, 0))
    allowed = table.get("allowedActions", {})
    call_amount = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    try:
        record_decision_telemetry(
            telemetry_conn,
            run_id=telemetry_run_id,
            hand_id=_derive_hand_id(table, state),
            decision_index=decision_index,
            strategy=strategy_name,
            table=table,
            seat=my_seat,
            action=action,
            amount=amount,
            message=message,
            facing_bet=call_amount > 0,
            voluntary=action in {"call", "bet", "raise"},
        )
        indexes[table_id] = decision_index + 1
    except Exception as exc:
        print(f"  Telemetry write failed: {exc}", flush=True)


def record_live_opponents_seen(telemetry_conn, state, table, agent_id):
    if telemetry_conn is None:
        return 0
    table_id = str(table.get("tableId") or table.get("id") or "unknown-table")
    seen_key = f"opponents_seen:{table_id}"
    if state.get(seen_key):
        return 0

    recorded = 0
    try:
        for seat in table.get("seats", []):
            seat_agent_id = seat.get("agentId")
            if not seat_agent_id or seat_agent_id == agent_id:
                continue
            increment_hand_seen(
                telemetry_conn,
                "arena",
                seat_agent_id,
                handle=seat_display_name(seat),
            )
            recorded += 1
        state[seen_key] = True
    except Exception as exc:
        print(f"  Opponent telemetry write failed: {exc}", flush=True)
    return recorded


def normalize_live_table_metadata(table):
    """Normalize arena aliases into fields the strategies and telemetry expect."""
    if table.get("buttonSeatNumber") is None:
        for key in (
            "dealerButtonSeatNumber",
            "dealerSeatNumber",
            "buttonSeat",
            "dealerSeat",
            "button",
        ):
            value = table.get(key)
            if isinstance(value, dict):
                value = value.get("seatNumber")
            if value is not None:
                table["buttonSeatNumber"] = value
                break

    # Arena fallback — no explicit dealer field. Derive the button from blind
    # posters (same logic as opponent_store._button_seat_number). The seat whose
    # currentBetChips matches smallBlindChips is the SB; the button is the seat
    # before the SB in circular seat order. Reliable at preflop where blinds
    # are live; always best-effort.
    if table.get("buttonSeatNumber") is None:
        sb_chips = _safe_int(table.get("smallBlindChips"))
        bb_chips = _safe_int(table.get("bigBlindChips"))
        if (
            sb_chips is not None
            and bb_chips is not None
            and sb_chips > 0
            and bb_chips > 0
        ):
            seats = table.get("seats", [])
            for s in seats:
                bet = _safe_int(s.get("currentBetChips"))
                sb_seat = _safe_int(s.get("seatNumber"))
                if (
                    bet is not None
                    and sb_seat is not None
                    and bet == sb_chips
                    and bet < bb_chips
                ):
                    active = sorted(
                        sn
                        for seat in seats
                        if (sn := _safe_int(seat.get("seatNumber"))) is not None
                    )
                    if sb_seat in active and len(active) >= 2:
                        idx = active.index(sb_seat)
                        table["buttonSeatNumber"] = active[(idx - 1) % len(active)]
                    break

    return table


def seat_display_name(seat):
    return (
        seat.get("agentHandle")
        or seat.get("agentName")
        or seat.get("handle")
        or seat.get("name")
    )


def enrich_table_with_opponent_profiles(
    telemetry_conn,
    table,
    agent_id,
    api_fn=None,
    competition_id=None,
    platform="arena",
    *,
    deadline_at=None,
):
    """Load opponent profiles from SQLite and inject into ``table``.

    NOTE: this function used to do a synchronous API fetch for any
    opponent that didn't have stats yet. The 10-second action
    budget (PLAN_OPPONENT_STATS.md §9) forbids that — the action
    path must NEVER block on the API. The background
    ``ArenaStatsFetcher`` (initialized in ``initialize_bot`` and
    fed by ``schedule_table_opponent_stats``) handles all fetches
    asynchronously, so a profile without API data simply plays
    unprofiled this hand and the API data lands for the next one.

    The ``api_fn`` and ``competition_id`` parameters are retained
    for backward compatibility with the previous signature but
    are unused.
    """
    if telemetry_conn is None:
        return table
    agent_ids = [
        seat.get("agentId")
        for seat in table.get("seats", [])
        if seat.get("agentId") and seat.get("agentId") != agent_id
    ]
    if not agent_ids:
        return table

    # Optional deadline guard. If we are within 2s of the action
    # deadline, skip the DB read — the merge inside is fast but
    # not free, and on a contended table the SQLite WAL write
    # from the background fetcher can briefly contend. We still
    # return the table so the strategy runs with whatever profile
    # state the previous hand left.
    if deadline_at is not None:
        import time

        now = time.time()
        if now >= float(deadline_at) - 2.0:
            return table

    profiles = load_profiles_for_agents(telemetry_conn, platform, agent_ids)
    if profiles:
        table["opponentProfiles"] = profiles
    return table


def _normalize_action_name(raw_action):
    if raw_action is None:
        return None
    action = str(raw_action).strip().lower().replace("_", "-")
    if "raise" in action:
        return "raise"
    if "bet" in action:
        return "bet"
    if "call" in action:
        return "call"
    if "fold" in action:
        return "fold"
    if "check" in action:
        return "check"
    if "all-in" in action or "allin" in action:
        return "raise"
    return None


def _first_present(mapping, keys):
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _event_summary(event):
    summary = event.get("summary")
    if isinstance(summary, dict):
        return summary
    return {}


def _event_value(event, keys):
    value = _first_present(event, keys)
    if value is not None:
        return value
    return _first_present(_event_summary(event), keys)


def _event_agent_id(event, seat_by_number):
    agent_id = _event_value(
        event,
        (
            "agentId",
            "actorAgentId",
            "playerAgentId",
            "participantId",
            "agent_id",
        ),
    )
    if agent_id:
        return str(agent_id)

    seat_number = _event_value(
        event,
        ("seatNumber", "actorSeatNumber", "playerSeatNumber", "seat"),
    )
    if seat_number is None:
        return None
    try:
        return seat_by_number.get(int(seat_number))
    except (TypeError, ValueError):
        return None


def _event_amount(event, action=None):
    if action == "raise":
        keys = (
            "toAmount",
            "raiseTo",
            "totalBetChips",
            "amount",
            "chips",
            "betChips",
            "currentBetChips",
        )
    else:
        keys = (
            "amount",
            "chips",
            "betChips",
            "toAmount",
            "raiseTo",
            "totalBetChips",
            "currentBetChips",
        )
    amount = _event_value(event, keys)
    if amount is None:
        return None
    try:
        return int(amount)
    except (TypeError, ValueError):
        return None


def _event_message(event):
    message = _event_value(
        event,
        (
            "message",
            "chat",
            "tableTalk",
            "table_talk",
            "speech",
            "comment",
            "reasoning",
        ),
    )
    if message is None:
        return None
    message = str(message).strip()
    return message or None


def _safe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iter_live_action_events(table):
    history_keys = (
        "actionHistory",
        "actions",
        "handActions",
        "bettingActions",
        "actionLog",
        "events",
        "recentEvents",
    )
    for key in history_keys:
        events = table.get(key)
        if not isinstance(events, list):
            continue
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            yield f"{key}:{index}", event

    for seat in table.get("seats", []):
        raw_action = _first_present(
            seat, ("lastAction", "last_action", "action", "previousAction")
        )
        if raw_action is None:
            continue
        seat_number = seat.get("seatNumber")
        key = (
            f"seat:{seat_number}:{raw_action}:"
            f"{seat.get('currentBetChips')}:{seat.get('stackChips')}"
        )
        yield (
            key,
            {
                "agentId": seat.get("agentId"),
                "seatNumber": seat_number,
                "action": raw_action,
                "amount": seat.get("currentBetChips"),
                "message": _first_present(
                    seat,
                    (
                        "lastMessage",
                        "last_message",
                        "message",
                        "chat",
                        "tableTalk",
                        "table_talk",
                    ),
                ),
                "street": table.get("street"),
            },
        )


def _event_key(table, source_key, event, agent_id, action, amount):
    event_id = _event_value(event, ("id", "actionId", "eventId"))
    hand_id = str(table.get("handId") or table.get("tableId") or table.get("id"))
    if event_id is not None:
        return f"{hand_id}:{event_id}"
    sequence = _event_value(event, ("sequence", "seq"))
    timestamp = _event_value(event, ("occurredAt", "createdAt", "timestamp", "at"))
    street = _event_value(event, ("street",)) or table.get("street")
    if sequence is not None:
        return f"{hand_id}:seq:{sequence}:{agent_id}:{street}:{action}:{amount}"
    return f"{hand_id}:{source_key}:{agent_id}:{street}:{action}:{amount}:{timestamp}"


def record_live_observed_actions(telemetry_conn, state, table, hero_agent_id):
    if telemetry_conn is None:
        return 0

    seat_by_number = {}
    for seat in table.get("seats", []):
        seat_number = _safe_int(seat.get("seatNumber"))
        if seat_number is not None and seat.get("agentId"):
            seat_by_number[seat_number] = seat.get("agentId")
    seat_by_agent = {
        seat.get("agentId"): seat
        for seat in table.get("seats", [])
        if seat.get("agentId")
    }
    prior_keys = list(state.get("observed_action_event_keys", []))
    seen = set(prior_keys)
    newly_seen = []
    hand_id = _derive_hand_id(table, state)
    recorded = 0

    try:
        for source_key, event in _iter_live_action_events(table):
            agent_id = _event_agent_id(event, seat_by_number)
            if not agent_id or agent_id == hero_agent_id:
                continue
            action = _normalize_action_name(
                _event_value(event, ("action", "actionType"))
            )
            if action is None:
                action = _normalize_action_name(_first_present(event, ("type",)))
            if action is None:
                continue
            amount = _event_amount(event, action)
            dedupe_key = _event_key(table, source_key, event, agent_id, action, amount)
            if dedupe_key in seen:
                continue

            seat = seat_by_agent.get(agent_id, {})
            facing_bet = bool(
                _event_value(event, ("facingBet", "facing_bet", "facing_bet_bool"))
                or _event_value(event, ("callAmount", "callChips"))
            )
            pot = _safe_int(
                _event_value(event, ("potChips", "pot")) or table.get("potChips")
            )
            record_observed_action(
                telemetry_conn,
                platform="arena",
                agent_id=agent_id,
                handle=seat_display_name(seat)
                or _event_value(
                    event,
                    ("agentHandle", "agentName", "handle", "name"),
                ),
                hand_id=hand_id,
                street=_event_value(event, ("street",)) or table.get("street"),
                action=action,
                amount=amount,
                pot=pot,
                message=_event_message(event),
                facing_bet=facing_bet,
                stack_chips=seat.get("stackChips"),
                hero_stack_chips=(seat_by_agent.get(hero_agent_id) or {}).get(
                    "stackChips"
                ),
                voluntary=action in {"call", "bet", "raise"},
            )
            seen.add(dedupe_key)
            newly_seen.append(dedupe_key)
            recorded += 1
    except Exception as exc:
        print(f"  Opponent action telemetry write failed: {exc}", flush=True)

    if newly_seen:
        state["observed_action_event_keys"] = (prior_keys + newly_seen)[-2000:]
    return recorded


@dataclass
class BotContext:
    """Holds initialized bot components for the main loop."""

    api_key: str
    competition_id: str
    agent_id: str
    strategy_name: str
    choose_action: callable
    api_fn: callable
    state: dict
    telemetry_conn: any
    telemetry_run_id: str
    stats_fetcher: any


def initialize_bot() -> BotContext:
    """Initialize bot credentials, strategy, state, and telemetry."""
    try:
        api_key, competition_id, agent_id = load_credentials()
    except Exception as exc:
        print(f"Failed loading credentials: {exc}", flush=True)
        sys.exit(1)

    try:
        strategy_name, choose_action = load_configured_strategy()
    except Exception as exc:
        print(f"Failed loading strategy: {exc}", flush=True)
        sys.exit(1)

    api_fn = make_api_client(api_key)
    state = load_state()
    telemetry_conn, telemetry_run_id = init_live_telemetry(
        state,
        competition_id,
        strategy_name,
    )
    stats_fetcher = (
        ArenaStatsFetcher(api_fn, competition_id)
        if telemetry_conn is not None
        else None
    )
    save_state(state)

    return BotContext(
        api_key=api_key,
        competition_id=competition_id,
        agent_id=agent_id,
        strategy_name=strategy_name,
        choose_action=choose_action,
        api_fn=api_fn,
        state=state,
        telemetry_conn=telemetry_conn,
        telemetry_run_id=telemetry_run_id,
        stats_fetcher=stats_fetcher,
    )


def handle_idle_state(ctx: BotContext, pending: dict, consecutive_empty: int) -> int:
    """Handle idle state: log, attempt to join queue, and update state."""
    consecutive_empty += 1
    idle_description = describe_idle_state(pending)

    if consecutive_empty % 30 == 0:
        print(
            f"  ...idle: {idle_description} ({consecutive_empty} polls)",
            flush=True,
        )

    now = time.time()
    if should_attempt_join(pending, ctx.state, now=now):
        join_resp = ctx.api_fn(
            "POST", "/texas/join", {"competitionId": ctx.competition_id}
        )
        record_join_attempt(ctx.state, join_resp, now=now)
        kind = join_resp.get("kind", "")
        if kind == "queued":
            pos = join_resp.get("lobby", {}).get("position", "?")
            total = join_resp.get("lobby", {}).get("total", "?")
            print(f"  Joined queue at position {pos}/{total}", flush=True)
        elif kind == "waiting_for_settlement":
            print("  Waiting for table settlement", flush=True)
        elif "error" in join_resp:
            print(
                "  Join skipped/rejected: "
                f"{join_resp.get('error')} - "
                f"{join_resp.get('message', '')}",
                flush=True,
            )
    elif ctx.state.get("last_reported_idle_state") != idle_description:
        print(f"  Idle: {idle_description}", flush=True)
        ctx.state["last_reported_idle_state"] = idle_description

    # While idle, reconcile our decision_telemetry rows with the arena's
    # /agent/{id}/replays. Throttled so we don't hammer the API on every
    # 2 s poll. Runs on every idle case (lobby, available, busted) — the
    # exact idle reason doesn't matter; what matters is that we're not
    # actively playing a hand.
    last_replay_epoch = float(ctx.state.get("last_replay_backfill_epoch", 0) or 0)
    if now - last_replay_epoch >= REPLAY_BACKFILL_INTERVAL_SECONDS:
        replay_seen = backfill_outcomes_from_replays(
            ctx.telemetry_conn,
            ctx.telemetry_run_id,
            agent_id=ctx.agent_id,
            competition_id=ctx.competition_id,
            api_fn=ctx.api_fn,
        )
        ctx.state["last_replay_backfill_epoch"] = now
        ctx.state["last_replay_backfill_count"] = replay_seen
        if replay_seen:
            print(
                f"  Replay backfill: reconciled {replay_seen} hand(s) from /replays",
                flush=True,
            )

    save_state(ctx.state)
    time.sleep(POLL_INTERVAL_SECONDS)

    return consecutive_empty


def dump_raw_table(table: dict) -> None:
    """Persist the unmodified arena table payload for field-name discovery.

    Writes one JSON file per call to RAW_DUMP_DIR, capturing the table dict
    exactly as the arena returned it — before normalize_live_table_metadata
    renames any aliases. This lets us find the real dealer-button field (and any
    other arena keys the recorder misses) without guessing. Best-effort: any
    failure is swallowed so it can never disrupt live play. Capped at
    RAW_DUMP_MAX_FILES files per directory.
    """
    if os.environ.get("POKER_BOT_RAW_DUMP", "1").lower() in {"0", "false", "no"}:
        return
    try:
        RAW_DUMP_DIR.mkdir(parents=True, exist_ok=True)
        existing = sorted(RAW_DUMP_DIR.glob("table-*.json"))
        if len(existing) >= RAW_DUMP_MAX_FILES:
            return
        table_id = str(table.get("tableId") or table.get("id") or "unknown")
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in table_id)
        street = str(table.get("street") or "unknown")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%f")
        path = RAW_DUMP_DIR / f"table-{timestamp}-{safe_id}-{street}.json"
        payload = {
            "captured_at": datetime.now(UTC).isoformat(),
            "top_level_keys": sorted(table.keys()),
            "seat_keys": sorted(
                {k for seat in table.get("seats", []) for k in seat.keys()}
            ),
            "table": table,
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
    except Exception as exc:  # never let diagnostics break the bot
        print(f"  Raw table dump failed: {exc}", flush=True)


def backfill_outcomes_from_replays(
    telemetry_conn,
    run_id,
    *,
    agent_id,
    competition_id,
    api_fn,
    limit=REPLAY_BACKFILL_LIMIT,
):
    """Reconcile decision_telemetry with the arena's /agent/{id}/replays.

    Called while the bot is idle. Fetches recent settled hands for our
    agent from the arena and fills `hero_net_chips` + `won_hand` on any
    matching `decision_telemetry` rows that are still NULL. The fill is
    idempotent (UPDATE … WHERE hero_net_chips IS NULL), so Showdown and
    stack-delta backfills keep precedence when they fire first.

    Returns the number of replay entries seen (not the number of rows
    updated — they are usually the same but can differ if a single
    `tableId` matches multiple decision rows from multiple hands).

    Never raises: any API or DB failure is caught and logged so live
    play is never blocked.
    """
    if telemetry_conn is None or run_id is None or not agent_id:
        return 0
    path = f"/agent/{agent_id}/replays"
    if competition_id:
        path += f"?competitionId={competition_id}&limit={int(limit)}"
    else:
        path += f"?limit={int(limit)}"
    try:
        response = api_fn("GET", path)
    except Exception as exc:
        print(f"  Replay backfill GET failed: {exc}", flush=True)
        return 0
    if not isinstance(response, list):
        # Some failures return dicts with `error`. Don't crash; log and bail.
        if isinstance(response, dict) and response.get("error"):
            print(
                f"  Replay backfill error: {response.get('error')}",
                flush=True,
            )
        return 0

    seen = 0
    for entry in response:
        if not isinstance(entry, dict):
            continue
        table_id = entry.get("tableId") or entry.get("handId")
        chip_delta = entry.get("chipDelta")
        if table_id is None or chip_delta is None:
            continue
        try:
            fill_hand_telemetry_outcome_from_replay(
                telemetry_conn,
                run_id=run_id,
                table_id=str(table_id),
                chip_delta=int(chip_delta),
                won_hand=int(chip_delta) > 0,
            )
        except Exception as exc:
            print(f"  Replay backfill write failed: {exc}", flush=True)
            continue
        seen += 1
    return seen


def backfill_hand_outcome_from_stack_delta(
    telemetry_conn,
    run_id,
    *,
    prev_hand_id,
    prev_stack,
    cur_stack,
    cur_committed,
):
    """Estimate previous hand's hero_net_chips from a stack delta at the boundary.

    Called on the first snapshot of a new hand at a table where we observed
    the previous hand. The formula:

        prev_net = (cur_stack + cur_committed) - prev_stack

    treats the delta between the last seen stack of the previous hand and
    the first seen stack of the new hand as ``settlement_of_prev -
    committed_to_cur_so_far``. This holds whenever the previous hand's
    last-seen snapshot is before settlement, which is the common case.

    Uses ``fill_hand_telemetry_outcome_from_delta`` (only fills NULL
    columns) so the more accurate Showdown snapshot, if observed, is
    never overwritten.

    Returns True when an UPDATE was issued, False otherwise. Safe to call
    with any of the inputs None — it just no-ops.
    """
    if telemetry_conn is None or run_id is None:
        return False
    if not prev_hand_id or prev_stack is None or cur_stack is None:
        return False
    if cur_committed is None:
        cur_committed = 0
    net = int(cur_stack + cur_committed - prev_stack)
    try:
        fill_hand_telemetry_outcome_from_delta(
            telemetry_conn,
            run_id=run_id,
            hand_id=prev_hand_id,
            hero_net_chips=net,
            won_hand=net > 0,
        )
        return True
    except Exception as exc:
        print(f"  Stack-delta backfill failed: {exc}", flush=True)
        return False


def backfill_hand_outcome(telemetry_conn, run_id, state, table, hero_agent_id):
    """If the table is a completed hand with winners, backfill outcome columns.

    Called from process_single_table on every table poll. Best-effort: any
    failure is logged but never disrupts live play. Does nothing if the table
    has no winners (hand still in progress) or the outcome was already recorded
    (dedup by run_id + hand_id).
    """
    try:
        winners = table.get("winners")
        if not winners:
            return  # hand still in progress, nothing to backfill

        hand_id = _derive_hand_id(table, state)
        seats = table.get("seats", [])

        # Find hero's seat to extract per-hand net
        hero_seat = next((s for s in seats if s.get("agentId") == hero_agent_id), None)
        if hero_seat is None:
            return

        hero_payout = _safe_int(hero_seat.get("payoutChips")) or 0
        hero_committed = _safe_int(hero_seat.get("totalCommittedChips")) or 0
        hero_net_chips = hero_payout - hero_committed

        # Did hero win? Check if hero's seatNumber appears in the winners list
        hero_seat_num = hero_seat.get("seatNumber")
        won_hand = any(
            w.get("seatNumber") == hero_seat_num or w.get("agentId") == hero_agent_id
            for w in winners
        )

        # Final pot = sum of all payouts (what the pot awarded)
        final_pot = sum(_safe_int(s.get("payoutChips")) or 0 for s in seats)

        update_hand_telemetry_outcome(
            telemetry_conn,
            run_id=run_id,
            hand_id=hand_id,
            hero_net_chips=hero_net_chips,
            won_hand=won_hand,
            final_pot=final_pot,
        )
    except Exception as exc:
        print(f"  Hand outcome backfill failed: {exc}", flush=True)


def process_single_table(table: dict, ctx: BotContext) -> None:
    """Process a single table: stats, skip checks, strategy execution, and fallbacks."""
    dump_raw_table(table)  # capture BEFORE normalization mutates the dict
    normalize_live_table_metadata(table)
    table_id = table.get("tableId", "?")
    street = table.get("street", "?")
    pot = table.get("potChips", 0)

    print(f"\n📍 Table {table_id} | {street} | Pot: {pot}", flush=True)

    # Resolve hand identity and detect boundaries. Side effect: state now
    # holds the current hand_id for this tableId so every downstream
    # telemetry writer uses the same one.
    _, is_new_hand, prev_hand_id, prev_stack, _prev_committed = _process_hand_identity(
        table, ctx.state
    )
    _record_hand_stack_state(table, ctx.state, ctx.agent_id)
    hero_seat_for_delta = next(
        (s for s in table.get("seats", []) if s.get("agentId") == ctx.agent_id),
        None,
    )
    cur_stack = (
        _safe_int(hero_seat_for_delta.get("stackChips"))
        if hero_seat_for_delta
        else None
    )
    cur_committed = (
        _safe_int(hero_seat_for_delta.get("totalCommittedChips"))
        if hero_seat_for_delta
        else None
    )

    # Hand-outcome backfill (A: stack delta on every boundary, B: Showdown
    # when observed). Stack-delta runs first and never overwrites; Showdown
    # runs after and is canonical. Order matters: Showdown must win.
    if is_new_hand and ctx.telemetry_conn is not None:
        backfill_hand_outcome_from_stack_delta(
            ctx.telemetry_conn,
            ctx.telemetry_run_id,
            prev_hand_id=prev_hand_id,
            prev_stack=prev_stack,
            cur_stack=cur_stack,
            cur_committed=cur_committed,
        )

    if ctx.telemetry_conn is not None:
        backfill_hand_outcome(
            ctx.telemetry_conn,
            ctx.telemetry_run_id,
            ctx.state,
            table,
            ctx.agent_id,
        )

    queued_stats = schedule_table_opponent_stats(ctx.stats_fetcher, table, ctx.agent_id)
    if queued_stats:
        print(f"  Queued Arena stats for {queued_stats} opponent(s)", flush=True)

    recorded_opponents = record_live_opponents_seen(
        ctx.telemetry_conn, ctx.state, table, ctx.agent_id
    )
    recorded_actions = record_live_observed_actions(
        ctx.telemetry_conn, ctx.state, table, ctx.agent_id
    )
    if recorded_opponents or recorded_actions:
        save_state(ctx.state)

    skip_reason = table_action_skip_reason(table, ctx.agent_id)
    if skip_reason:
        print(f"  Skipping table: {skip_reason}", flush=True)
        return

    enrich_table_with_opponent_profiles(
        ctx.telemetry_conn,
        table,
        ctx.agent_id,
        platform="arena",
        deadline_at=table.get("actionDeadlineAt"),
    )
    my_seat = find_agent_seat(table, ctx.agent_id)

    # --- Langfuse tracing spans for the decision path ---
    hand_id = _derive_hand_id(table, ctx.state)
    num_seats = len([s for s in table.get("seats", []) if s.get("agentId")])
    num_opponents = max(0, num_seats - 1)

    with (
        trace_hand(
            hand_id=hand_id,
            table_id=table_id,
            street=street,
            strategy_name=ctx.strategy_name,
            agent_id=ctx.agent_id,
            competition_id=ctx.competition_id,
            pot_chips=pot,
            num_seats=num_seats,
        ),
        span_table_process(
            table_id=table_id,
            street=street,
            pot_chips=pot,
            num_opponents=num_opponents,
        ) as table_span,
    ):
            # Resolve decision context for the span
            call_amount = int(
                (table.get("allowedActions") or {}).get("callAmount")
                or (table.get("allowedActions") or {}).get("callChips")
                or 0
            )
            hero_stack = _safe_int(my_seat.get("stackChips")) if my_seat else None
            bb_chips = _safe_int(table.get("bigBlindChips")) or 0
            decision_index = int(
                ctx.state.get("telemetry_decision_indexes", {}).get(table_id, 0)
            )

            with (
                span_decision(
                    decision_index=decision_index,
                    street=street,
                    position="unknown",
                    hand_description=(
                        (my_seat or {}).get("handDescription")
                        or (my_seat or {}).get("hand")
                        or (my_seat or {}).get("cards")
                        or "unknown"
                    ),
                    facing_bet=call_amount > 0,
                    pot_chips=pot,
                    stack_chips=hero_stack or 0,
                    big_blind=bb_chips,
                ),
                span_strategy_call(ctx.strategy_name) as strat_span,
            ):
                    result = ctx.choose_action(table, my_seat)
                    if result[0] is not None and strat_span is not None:
                        strat_span.update(
                            output={
                                "action": result[0],
                                "amount": result[1],
                                "message": result[2],
                            }
                        )

            if result[0] is None:
                print("  No valid action found, skipping", flush=True)
                if table_span is not None:
                    table_span.update(
                        output={"result": "no_valid_action"}
                    )
                return

            action, amount, message = result
            print(
                f"  → {action}" + (f" {amount}" if amount else "") + f" | {message}",
                flush=True,
            )

            body = action_request_body(ctx.competition_id, table_id, action, amount)
            with span_action_result(action, amount, accepted=True) as action_span:
                resp = ctx.api_fn("POST", "/texas/action", body)

                if "error" in resp:
                    error = resp.get("error")
                    error_message = resp.get("message", "")
                    print(
                        f"  ✗ Action rejected: {error} - {error_message}",
                        flush=True,
                    )
                    if action_span is not None:
                        action_span.update(
                            output={"accepted": False, "error": error}
                        )

                    if is_not_turn_rejection(resp):
                        ctx.state["last_not_turn_rejection_at"] = (
                            datetime.now(UTC).isoformat()
                        )
                        ctx.state["last_not_turn_rejection_table_id"] = table_id
                        save_state(ctx.state)
                        return

                    if action != "fold":
                        if (
                            action == "call"
                            and "nothing to call" in str(error or "").lower()
                        ):
                            fallback_message = "Fallback check (nothing to call)"
                            body["action"] = "check"
                            body["message"] = public_action_message()
                            body.pop("amount", None)
                        else:
                            fallback_message = "Fallback fold after rejection"
                            body["action"] = "fold"
                            body["message"] = public_action_message()
                            body.pop("amount", None)

                        resp = ctx.api_fn("POST", "/texas/action", body)
                        if "error" not in resp:
                            print(
                                f"  → Fallback: {body['action']} accepted",
                                flush=True,
                            )
                            record_live_decision(
                                ctx.telemetry_conn,
                                ctx.telemetry_run_id,
                                ctx.state,
                                table,
                                my_seat,
                                body["action"],
                                None,
                                fallback_message,
                                ctx.strategy_name,
                            )
                            save_state(ctx.state)
                else:
                    print("  ✓ Action accepted", flush=True)
                    record_live_decision(
                        ctx.telemetry_conn,
                        ctx.telemetry_run_id,
                        ctx.state,
                        table,
                        my_seat,
                        action,
                        amount,
                        message,
                        ctx.strategy_name,
                    )
                    new_street = resp.get("street", "?")
                    new_pot = resp.get("potChips", 0)
                    print(f"    Street: {new_street} | Pot: {new_pot}", flush=True)

                    if table_span is not None:
                        table_span.update(
                            output={
                                "action": action,
                                "amount": amount,
                                "new_street": new_street,
                                "new_pot": new_pot,
                            }
                        )

                    ctx.state["last_table_id"] = table_id
                    save_state(ctx.state)


def main():
    ctx = initialize_bot()
    print("🃏 Poker playing loop starting...", flush=True)
    print(f"  Strategy: {ctx.strategy_name}", flush=True)

    consecutive_empty = 0

    while True:
        try:
            # Heartbeat: mark main loop as alive for cron tick fallback
            ctx.state["last_heartbeat_at"] = datetime.now(UTC).isoformat()
            save_state(ctx.state)

            # Poll for pending actions
            pending = ctx.api_fn(
                "GET", f"/texas/pending-actions?competitionId={ctx.competition_id}"
            )

            if "error" in pending and "tables" not in pending:
                print(f"Error polling: {pending}", flush=True)
                time.sleep(ERROR_POLL_INTERVAL_SECONDS)
                continue

            update_queue_state(ctx.state, pending)
            tables = pending.get("tables", [])

            if not tables:
                consecutive_empty = handle_idle_state(ctx, pending, consecutive_empty)
                continue

            # Sort by earliest action deadline
            consecutive_empty = 0
            ctx.state["last_reported_idle_state"] = None
            tables.sort(key=lambda t: t.get("actionDeadlineAt", float("inf")))

            for table in tables:
                process_single_table(table, ctx)

            save_state(ctx.state)
            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("\n🛑 Poker loop stopped", flush=True)
            if ctx.stats_fetcher is not None:
                ctx.stats_fetcher.close()
            flush_langfuse()
            break
        except Exception as e:
            print(f"Exception: {e}", flush=True)
            time.sleep(ERROR_POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
