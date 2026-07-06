"""SQLite-backed persistent opponent tracking for the Arena sandbox.

The in-memory tracker resets between hands because the sandbox may restart
Python workers. This module stores opponent profiles in a SQLite database
beside this asset module so the database travels with the large assets area
rather than the size-limited harness area.
"""

from __future__ import annotations

import os
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OpponentProfile:
    agent_id: str
    name: str | None = None
    hands_seen: int = 0
    vpip: int = 0
    pfr: int = 0
    calls: int = 0
    bets: int = 0
    raises: int = 0
    folds: int = 0
    fold_to_bet: int = 0
    opportunities_to_fold_to_bet: int = 0
    showdowns: int = 0
    weak_aggressive_showdowns: int = 0
    recent_actions: deque = field(default_factory=lambda: deque(maxlen=20))

    @property
    def aggression_frequency(self) -> float:
        actions = self.calls + self.bets + self.raises + self.folds
        return 0.0 if actions == 0 else (self.bets + self.raises) / actions

    @property
    def call_frequency(self) -> float:
        actions = self.calls + self.bets + self.raises + self.folds
        return 0.0 if actions == 0 else self.calls / actions

    @property
    def fold_to_bet_frequency(self) -> float:
        if self.opportunities_to_fold_to_bet == 0:
            return 0.0
        return self.fold_to_bet / self.opportunities_to_fold_to_bet

    @property
    def vpip_frequency(self) -> float:
        return 0.0 if self.hands_seen == 0 else self.vpip / self.hands_seen

    @property
    def pfr_frequency(self) -> float:
        return 0.0 if self.hands_seen == 0 else self.pfr / self.hands_seen

    @property
    def weak_aggressive_showdown_frequency(self) -> float:
        return (
            0.0
            if self.showdowns == 0
            else self.weak_aggressive_showdowns / self.showdowns
        )

    def label(self) -> str:
        if self.hands_seen < 5 and len(self.recent_actions) < 8:
            return "unknown"
        if self.weak_aggressive_showdown_frequency >= 0.35:
            return "bluffer"
        if self.vpip_frequency >= 0.45 and self.aggression_frequency >= 0.35:
            return "loose_aggressive"
        if self.vpip_frequency >= 0.45 and self.call_frequency >= 0.45:
            return "calling_station"
        if self.vpip_frequency <= 0.18 and self.pfr_frequency <= 0.08:
            return "patient_methodical"
        if self.pfr_frequency >= 0.22 and self.aggression_frequency >= 0.30:
            return "tight_aggressive"
        return "balanced"

    def as_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "label": self.label(),
            "hands_seen": self.hands_seen,
            "vpip": self.vpip,
            "pfr": self.pfr,
            "calls": self.calls,
            "bets": self.bets,
            "raises": self.raises,
            "folds": self.folds,
            "fold_to_bet": self.fold_to_bet,
            "opportunities_to_fold_to_bet": self.opportunities_to_fold_to_bet,
            "showdowns": self.showdowns,
            "weak_aggressive_showdowns": self.weak_aggressive_showdowns,
            "vpip_frequency": round(self.vpip_frequency, 3),
            "pfr_frequency": round(self.pfr_frequency, 3),
            "aggression_frequency": round(self.aggression_frequency, 3),
            "call_frequency": round(self.call_frequency, 3),
            "fold_to_bet_frequency": round(self.fold_to_bet_frequency, 3),
            "weak_aggressive_showdown_frequency": round(
                self.weak_aggressive_showdown_frequency, 3
            ),
            "recent_actions": list(self.recent_actions),
        }


_DB_PATH = os.environ.get(
    "PROFILER_BOT_DB_PATH",
    str(Path(__file__).resolve().parent / "opponents.db"),
)
Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
_CONN: sqlite3.Connection | None = None


def _conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is not None:
        return _CONN

    conn = sqlite3.connect(_DB_PATH, timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS opponent_profiles (
            agent_id TEXT PRIMARY KEY,
            name TEXT,
            hands_seen INTEGER DEFAULT 0,
            vpip INTEGER DEFAULT 0,
            pfr INTEGER DEFAULT 0,
            calls INTEGER DEFAULT 0,
            bets INTEGER DEFAULT 0,
            raises INTEGER DEFAULT 0,
            folds INTEGER DEFAULT 0,
            fold_to_bet INTEGER DEFAULT 0,
            opportunities_to_fold_to_bet INTEGER DEFAULT 0,
            showdowns INTEGER DEFAULT 0,
            weak_aggressive_showdowns INTEGER DEFAULT 0,
            last_updated TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen_hands (
            agent_id TEXT,
            hand_key TEXT,
            PRIMARY KEY (agent_id, hand_key)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen_events (
            agent_id TEXT,
            event_key TEXT,
            PRIMARY KEY (agent_id, event_key)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_seen_hands_agent ON seen_hands(agent_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_seen_events_agent ON seen_events(agent_id)"
    )
    _CONN = conn
    return _CONN


def _row_from_profile(profile: OpponentProfile) -> tuple:
    return (
        profile.agent_id,
        profile.name,
        profile.hands_seen,
        profile.vpip,
        profile.pfr,
        profile.calls,
        profile.bets,
        profile.raises,
        profile.folds,
        profile.fold_to_bet,
        profile.opportunities_to_fold_to_bet,
        profile.showdowns,
        profile.weak_aggressive_showdowns,
        None,
    )


def _profile_from_row(row) -> OpponentProfile:
    return OpponentProfile(
        agent_id=row[0],
        name=row[1],
        hands_seen=row[2],
        vpip=row[3],
        pfr=row[4],
        calls=row[5],
        bets=row[6],
        raises=row[7],
        folds=row[8],
        fold_to_bet=row[9],
        opportunities_to_fold_to_bet=row[10],
        showdowns=row[11],
        weak_aggressive_showdowns=row[12],
    )


def _load_profile(agent_id: str) -> OpponentProfile:
    conn = _conn()
    row = conn.execute(
        "SELECT * FROM opponent_profiles WHERE agent_id = ?", (agent_id,)
    ).fetchone()
    if row:
        return _profile_from_row(row)
    return OpponentProfile(agent_id=agent_id)


def _save_profile(profile: OpponentProfile) -> None:
    conn = _conn()
    conn.execute(
        """
        INSERT INTO opponent_profiles (
            agent_id, name, hands_seen, vpip, pfr, calls, bets, raises,
            folds, fold_to_bet, opportunities_to_fold_to_bet,
            showdowns, weak_aggressive_showdowns, last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(agent_id) DO UPDATE SET
            name = excluded.name,
            hands_seen = excluded.hands_seen,
            vpip = excluded.vpip,
            pfr = excluded.pfr,
            calls = excluded.calls,
            bets = excluded.bets,
            raises = excluded.raises,
            folds = excluded.folds,
            fold_to_bet = excluded.fold_to_bet,
            opportunities_to_fold_to_bet = excluded.opportunities_to_fold_to_bet,
            showdowns = excluded.showdowns,
            weak_aggressive_showdowns = excluded.weak_aggressive_showdowns,
            last_updated = excluded.last_updated
        """,
        _row_from_profile(profile),
    )


def _profile_for(agent_id: str, name: str | None = None) -> OpponentProfile:
    profile = _load_profile(agent_id)
    profile.name = profile.name or name
    return profile


def _table_events(table: dict) -> list:
    events = []
    for key in ("recentEvents", "events", "actionHistory"):
        value = table.get(key)
        if isinstance(value, list):
            events.extend(event for event in value if isinstance(event, dict))
    return events


def _hand_key(table: dict) -> str:
    for key in ("handId", "hand_id", "handNumber", "hand_number"):
        if table.get(key):
            return str(table[key])
    for event in _table_events(table):
        event_id = event.get("id") or event.get("eventId")
        if event_id:
            return f"{table.get('tableId') or table.get('id') or 'table'}:{event_id}"
        sequence = event.get("sequence")
        occurred = event.get("occurredAt") or event.get("createdAt")
        if sequence is not None or occurred is not None:
            return (
                f"{table.get('tableId') or table.get('id') or 'table'}:"
                f"{sequence}:{occurred}"
            )
    return str(table.get("tableId") or table.get("id") or "unknown-table")


def _event_key(
    table: dict, event: dict, summary: dict, agent_id: str, action: str
) -> str:
    hand = _hand_key(table)
    event_id = event.get("id") or event.get("eventId")
    if event_id:
        return f"{hand}:{event_id}"
    sequence = event.get("sequence")
    street = (
        _first(summary, event, "street") or event.get("street") or table.get("street")
    )
    amount = _first(summary, event, "toAmount", "amount")
    seat_number = _first(summary, event, "seatNumber", "seat")
    return f"{hand}:seq:{sequence}:{seat_number}:{agent_id}:{street}:{action}:{amount}"


def _event_agent_id(event: dict, summary: dict, seats: list) -> str | None:
    agent_id = _first(summary, event, "agentId", "agent_id")
    if agent_id:
        return str(agent_id)
    seat_number = _first(summary, event, "seatNumber", "seat")
    if seat_number is not None:
        seat = _seat_by_number(seats).get(seat_number)
        if seat:
            return _seat_agent_id(seat)
    name = _first(summary, event, "agentName", "agentHandle", "name")
    if name:
        for seat in seats:
            if name in {
                _seat_name(seat),
                seat.get("agentName"),
                seat.get("agentHandle"),
            }:
                return _seat_agent_id(seat)
    return None


def _is_hero_seat(seat: dict, hero_seat_number, hero_id: str | None) -> bool:
    if hero_id and _seat_agent_id(seat) == hero_id:
        return True
    return hero_seat_number is not None and seat.get("seatNumber") == hero_seat_number


def _seat_by_number(seats: list) -> dict:
    return {
        seat.get("seatNumber"): seat
        for seat in seats
        if seat.get("seatNumber") is not None
    }


def _seat_by_agent_id(seats: list) -> dict:
    return {_seat_agent_id(seat): seat for seat in seats if _seat_agent_id(seat)}


def _seat_agent_id(seat: dict) -> str | None:
    value = seat.get("agentId") or seat.get("agent_id")
    if value:
        return str(value)
    seat_number = seat.get("seatNumber")
    if seat_number is not None:
        return f"seat-{seat_number}"
    return None


def _seat_name(seat: dict) -> str | None:
    value = seat.get("agentHandle") or seat.get("agentName") or seat.get("name")
    return str(value) if value else None


def _first(*items):
    dicts = []
    keys = []
    for item in items:
        if isinstance(item, dict) and not keys:
            dicts.append(item)
        else:
            keys.append(item)
    for data in dicts:
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
    return None


def _is_hand_seen(conn: sqlite3.Connection, agent_id: str, hand_key: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM seen_hands WHERE agent_id = ? AND hand_key = ?",
        (agent_id, hand_key),
    ).fetchone()
    return row is not None


def _mark_hand_seen_db(conn: sqlite3.Connection, agent_id: str, hand_key: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen_hands (agent_id, hand_key) VALUES (?, ?)",
        (agent_id, hand_key),
    )


def _is_event_seen(conn: sqlite3.Connection, agent_id: str, event_key: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM seen_events WHERE agent_id = ? AND event_key = ?",
        (agent_id, event_key),
    ).fetchone()
    return row is not None


def _mark_event_seen_db(
    conn: sqlite3.Connection, agent_id: str, event_key: str
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO seen_events (agent_id, event_key) VALUES (?, ?)",
        (agent_id, event_key),
    )


def _mark_hand_seen(table: dict, profile: OpponentProfile) -> None:
    hand_key = _hand_key(table)
    conn = _conn()
    if _is_hand_seen(conn, profile.agent_id, hand_key):
        return
    _mark_hand_seen_db(conn, profile.agent_id, hand_key)
    profile.hands_seen += 1
    _save_profile(profile)


def _record_action(
    profile: OpponentProfile,
    action: str,
    *,
    street: str | None = None,
    facing_bet: bool = False,
    voluntary: bool = False,
) -> None:
    profile.recent_actions.append({"street": street, "action": action})
    if voluntary:
        profile.vpip += 1
    if action == "call":
        profile.calls += 1
    elif action == "bet":
        profile.bets += 1
    elif action in {"raise", "all-in"}:
        profile.raises += 1
        if street == "Preflop":
            profile.pfr += 1
    elif action == "fold":
        profile.folds += 1
        if facing_bet:
            profile.fold_to_bet += 1
    if facing_bet:
        profile.opportunities_to_fold_to_bet += 1


def _process_event(
    table: dict,
    seats: list,
    hero_seat_number,
    hero_id: str | None,
    event: dict,
) -> None:
    summary = event.get("summary") if isinstance(event.get("summary"), dict) else event
    action = str(_first(summary, event, "action") or "").lower().replace("_", "-")
    if action not in {"fold", "check", "call", "bet", "raise", "all-in"}:
        return
    agent_id = _event_agent_id(event, summary, seats)
    if not agent_id:
        return
    seat = _seat_by_agent_id(seats).get(agent_id)
    if seat and _is_hero_seat(seat, hero_seat_number, hero_id):
        return
    if hero_id and agent_id == hero_id:
        return
    conn = _conn()
    event_key = _event_key(table, event, summary, agent_id, action)
    if _is_event_seen(conn, agent_id, event_key):
        return
    _mark_event_seen_db(conn, agent_id, event_key)

    profile = _profile_for(
        agent_id, _first(summary, event, "agentName", "agentHandle", "name")
    )
    street = str(
        _first(summary, event, "street")
        or event.get("street")
        or table.get("street")
        or ""
    )
    facing_bet = bool(_first(summary, event, "facingBet", "facing_bet"))
    if not facing_bet:
        facing_bet = action == "fold" and int(table.get("currentBet") or 0) > 0
    voluntary = street == "Preflop" and action in {"call", "bet", "raise", "all-in"}
    _record_action(
        profile, action, street=street, facing_bet=facing_bet, voluntary=voluntary
    )
    _save_profile(profile)


def enrich_table(table: dict) -> dict:
    """Update persistent opponent profiles and attach them to ``table``."""
    seats = list(table.get("seats") or [])
    hero_seat_number = table.get("selfSeatNumber") or table.get("actingSeatNumber")
    hero_seat = _seat_by_number(seats).get(hero_seat_number) or {}
    hero_id = _seat_agent_id(hero_seat) if hero_seat else None

    conn = _conn()

    for seat in seats:
        if _is_hero_seat(seat, hero_seat_number, hero_id):
            continue
        agent_id = _seat_agent_id(seat)
        if not agent_id:
            continue
        _profile_for(agent_id, _seat_name(seat))

    for event in _table_events(table):
        _process_event(table, seats, hero_seat_number, hero_id, event)

    current = dict(table.get("opponentProfiles") or {})
    for seat in seats:
        if _is_hero_seat(seat, hero_seat_number, hero_id):
            continue
        agent_id = _seat_agent_id(seat)
        if not agent_id:
            continue
        profile = _load_profile(agent_id)
        current[agent_id] = profile.as_dict()

    table["opponentProfiles"] = current
    return table
