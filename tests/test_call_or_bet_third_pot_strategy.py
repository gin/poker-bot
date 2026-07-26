from poker_bot.strategies.call_or_bet_third_pot import choose_action


def make_table(available_actions, *, pot=300, **allowed_overrides):
    allowed = {
        "availableActions": available_actions,
        "callAmount": 75,
        "minBet": 20,
        "maxCommit": 1000,
    }
    allowed.update(allowed_overrides)
    return {"potChips": pot, "allowedActions": allowed}


def test_calls_when_call_is_available():
    table = make_table(["fold", "call", "raise"])

    action, amount, message = choose_action(table, {})

    assert (action, amount) == ("call", 75)
    assert message == "Calling every time"


def test_bets_one_third_of_the_pot_when_not_facing_a_bet():
    table = make_table(["check", "bet"], pot=300)

    action, amount, message = choose_action(table, {})

    assert (action, amount) == ("bet", 100)
    assert message == "Betting one-third of the pot"


def test_bet_respects_legal_minimum():
    table = make_table(["check", "bet"], pot=30, minBet=20)

    action, amount, _message = choose_action(table, {})

    assert (action, amount) == ("bet", 20)


def test_bet_respects_bet_range_maximum():
    table = make_table(
        ["check", "bet"], pot=600, betRange={"min": 20, "max": 150}
    )

    action, amount, _message = choose_action(table, {})

    assert (action, amount) == ("bet", 150)


def test_checks_when_call_and_bet_are_unavailable():
    table = make_table(["fold", "check"])

    action, amount, message = choose_action(table, {})

    assert (action, amount) == ("check", None)
    assert message == "No call or bet available, checking"
