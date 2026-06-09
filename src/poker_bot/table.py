"""Helpers for reading Arena table state."""


def find_agent_seat(table, agent_id):
    for seat in table.get("seats", []):
        if seat.get("agentId") == agent_id:
            return seat
    return None


def _safe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_our_turn(table, agent_id):
    my_seat = find_agent_seat(table, agent_id)
    if my_seat is None:
        return False

    acting_agent_id = table.get("actingAgentId") or table.get("actorAgentId")
    if acting_agent_id is not None:
        return str(acting_agent_id) == str(agent_id)

    acting = _safe_int(table.get("actingSeatNumber"))
    my_seat_number = _safe_int(my_seat.get("seatNumber"))
    if acting is None or my_seat_number is None:
        return False
    return acting == my_seat_number
