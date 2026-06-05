from poker_bot.strategies.simple import choose_action


def test_choose_action_strong_pair():
    table = {
        "street": "Preflop",
        "allowedActions": {
            "availableActions": ["raise", "call", "fold"],
            "callAmount": 100,
            "minRaiseTo": 300,
            "maxCommit": 1000,
        },
        "seats": [
            {
                "agentId": "my-agent",
                "holeCards": ["AH", "AD"],
                "stackChips": 1000,
                "currentBetChips": 0,
            }
        ],
        "potChips": 200,
    }

    action, amount, message = choose_action(table, table["seats"][0])

    assert action == "raise"
    assert isinstance(amount, int)
    assert "Strong hand" in message


def test_choose_action_returns_message_when_no_actions_available():
    action, amount, message = choose_action({"allowedActions": {}}, None)

    assert action is None
    assert amount is None
    assert message == "No legal actions available"
