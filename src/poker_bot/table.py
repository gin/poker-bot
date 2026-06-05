"""Helpers for reading Arena table state."""


def find_agent_seat(table, agent_id):
    for seat in table.get("seats", []):
        if seat.get("agentId") == agent_id:
            return seat
    return None


def is_our_turn(table, agent_id):
    acting = table.get("actingSeatNumber")
    my_seat = find_agent_seat(table, agent_id)
    return acting == (my_seat or {}).get("seatNumber")
