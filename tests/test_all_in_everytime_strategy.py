from poker_bot.strategies.all_in_everytime import choose_action


def make_table(available_actions, **allowed_overrides):
    allowed = {
        "availableActions": available_actions,
        "callAmount": 75,
        "maxCommit": 1000,
    }
    allowed.update(allowed_overrides)
    return {
        "allowedActions": allowed,
        "seats": [
            {
                "agentId": "my-agent",
                "stackChips": 900,
                "currentBetChips": 100,
            }
        ],
    }


def test_all_in_raises_to_max_commit_when_raise_available():
    table = make_table(["fold", "call", "raise"])

    action, amount, message = choose_action(table, table["seats"][0])

    assert action == "raise"
    assert amount == 1000
    assert message == "All-in every time"


def test_all_in_bets_to_max_commit_when_bet_available():
    table = make_table(["fold", "check", "bet"])

    action, amount, message = choose_action(table, table["seats"][0])

    assert action == "bet"
    assert amount == 1000
    assert message == "All-in every time"


def test_all_in_calls_when_only_call_can_continue():
    table = make_table(["fold", "call"])

    action, amount, message = choose_action(table, table["seats"][0])

    assert action == "call"
    assert amount == 75
    assert message == "Calling all available chips"


def test_all_in_falls_back_to_check_when_no_bet_available():
    table = make_table(["fold", "check"])

    action, amount, message = choose_action(table, table["seats"][0])

    assert action == "check"
    assert amount is None
    assert message == "No bet available, checking"


def test_all_in_folds_when_seat_is_missing():
    table = make_table(["fold", "call"])

    action, amount, message = choose_action(table, None)

    assert action == "fold"
    assert amount is None
    assert message == "Fallback: seat not found"


def test_all_in_uses_seat_stack_when_max_commit_is_missing():
    table = make_table(["fold", "call", "raise"])
    table["allowedActions"].pop("maxCommit")

    action, amount, message = choose_action(table, table["seats"][0])

    assert action == "raise"
    assert amount == 1000
    assert message == "All-in every time"
