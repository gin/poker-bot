"""Helpers for extracting reusable table context."""

from poker_bot.opponents import OpponentProfile, profile_from_mapping


def active_opponent_seats(table, my_seat):
    my_id = (my_seat or {}).get("agentId")
    return [
        seat
        for seat in table.get("seats", [])
        if seat.get("agentId") != my_id
        and not seat.get("folded", False)
        and not seat.get("hasFolded", False)
    ]


def active_opponents(table, my_seat):
    return max(1, len(active_opponent_seats(table, my_seat)))


def table_profiles(table, my_seat):
    raw_profiles = table.get("opponentProfiles", {})
    my_id = (my_seat or {}).get("agentId")
    profiles = []
    for seat in table.get("seats", []):
        agent_id = seat.get("agentId")
        if not agent_id or agent_id == my_id:
            continue
        raw = raw_profiles.get(agent_id)
        if isinstance(raw, OpponentProfile):
            profiles.append(raw)
        elif isinstance(raw, dict):
            profiles.append(profile_from_mapping(agent_id, raw))
    return profiles


def table_agent_stats(table, my_seat):
    raw_stats = table.get("opponentAgentStats") or table.get("agentStats") or {}
    my_id = (my_seat or {}).get("agentId")
    stats = []

    if isinstance(raw_stats, list):
        candidates = raw_stats
    elif isinstance(raw_stats, dict):
        candidates = raw_stats.values()
    else:
        candidates = []

    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        agent_id = raw.get("agentId")
        if agent_id and agent_id != my_id:
            stats.append(raw)
    return stats


def call_amount(allowed):
    return int(allowed.get("callAmount") or allowed.get("callChips") or 0)


def pot_size(table):
    return int(table.get("potChips") or 0)
