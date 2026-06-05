from poker_bot.table import find_agent_seat, is_our_turn


def test_find_agent_seat_returns_matching_seat():
    table = {
        "seats": [
            {"agentId": "other", "seatNumber": 1},
            {"agentId": "my-agent", "seatNumber": 2},
        ]
    }

    seat = find_agent_seat(table, "my-agent")

    assert seat == {"agentId": "my-agent", "seatNumber": 2}


def test_find_agent_seat_returns_none_when_missing():
    assert find_agent_seat({"seats": []}, "my-agent") is None
    assert find_agent_seat({}, "my-agent") is None


def test_is_our_turn_matches_acting_seat():
    table = {
        "actingSeatNumber": 2,
        "seats": [
            {"agentId": "my-agent", "seatNumber": 2},
            {"agentId": "other", "seatNumber": 1},
        ],
    }

    assert is_our_turn(table, "my-agent")
    assert not is_our_turn(table, "other")
    assert not is_our_turn(table, "missing-agent")
