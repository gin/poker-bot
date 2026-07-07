"""In-bundle opponent tracking for the Arena static-agent sandbox.

The hosted sandbox does not provide the app-side SQLite opponent store. This
module keeps a small in-memory profile cache inside the strategy worker and
enriches each table snapshot with ``table["opponentProfiles"]`` dictionaries
compatible with the strategies' profile_value() helpers (dict or object).

Canonical copy — scripts/bundle_strategy.py ships this into every bundle's
assets/ (unless --no-tracker). Adopted from luigi_bot/assets/, which keeps
its own frozen copy as a shipped bundle. Stdlib-only by design.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

try:  # optional prior layer: bundle-time snapshot + once-per-match API fetch
    from poker_bot import sandbox_agent_stats as _agent_stats
except Exception:  # tracker still works observation-only
    _agent_stats = None


@dataclass
class OpponentProfile:
    agent_id: str
    name: str | None = None
    api_stats: dict | None = None
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
        out = {
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
        # API prior takes over while the local sample is thin; local counters
        # (self.hands_seen) stay untouched so observation keeps accumulating
        # and wins outright once it reaches the merge threshold.
        if _agent_stats is not None and self.api_stats:
            out = _agent_stats.merge_stats_into(
                out, local_hands_seen=self.hands_seen, api_stats=self.api_stats
            )
        return out


_PROFILES: dict[str, OpponentProfile] = {}
_SEEN_HAND_KEYS: set[str] = set()
_SEEN_EVENT_KEYS: set[str] = set()


def enrich_table(table: dict) -> dict:
    """Update local opponent profiles and attach them to ``table``.

    This mutates and returns ``table`` so existing strategy code sees the
    enriched field without changing its call signature.
    """
    seats = list(table.get("seats") or [])
    hero_seat_number = table.get("selfSeatNumber") or table.get("actingSeatNumber")
    hero_seat = _seat_by_number(seats).get(hero_seat_number) or {}
    hero_id = _seat_agent_id(hero_seat) if hero_seat else None

    for seat in seats:
        if _is_hero_seat(seat, hero_seat_number, hero_id):
            continue
        agent_id = _seat_agent_id(seat)
        if not agent_id:
            continue
        profile = _profile_for(agent_id, _seat_name(seat))
        profile.name = profile.name or _seat_name(seat)
        _mark_hand_seen(table, profile)

    for event in _table_events(table):
        _record_event(table, seats, hero_seat_number, hero_id, event)

    current = dict(table.get("opponentProfiles") or {})
    for seat in seats:
        if _is_hero_seat(seat, hero_seat_number, hero_id):
            continue
        agent_id = _seat_agent_id(seat)
        if agent_id in _PROFILES:
            current[agent_id] = _PROFILES[agent_id].as_dict()
    table["opponentProfiles"] = current
    return table


def _profile_for(agent_id: str, name: str | None = None) -> OpponentProfile:
    if agent_id not in _PROFILES:
        profile = OpponentProfile(agent_id=agent_id, name=name)
        if _agent_stats is not None:
            try:
                # Snapshot lookup, or ONE live fetch per opponent per match
                # (result cached, including failures). "seat-N" is the local
                # fallback for seats with no agentId — never a real agent, so
                # don't burn a network timeout on it.
                real_id = agent_id if not agent_id.startswith("seat-") else None
                profile.api_stats = _agent_stats.get_prior(
                    agent_id=real_id, handle=name
                )
            except Exception:
                profile.api_stats = None
        _PROFILES[agent_id] = profile
    return _PROFILES[agent_id]


def _mark_hand_seen(table: dict, profile: OpponentProfile) -> None:
    key = f"{profile.agent_id}:{_hand_key(table)}"
    if key in _SEEN_HAND_KEYS:
        return
    _SEEN_HAND_KEYS.add(key)
    profile.hands_seen += 1


def _record_event(
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
    event_key = _event_key(table, event, summary, agent_id, action)
    if event_key in _SEEN_EVENT_KEYS:
        return
    _SEEN_EVENT_KEYS.add(event_key)

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
