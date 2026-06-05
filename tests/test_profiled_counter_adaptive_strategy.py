from poker_bot.strategies.profiled_counter_adaptive import choose_action


def make_seat(agent_id, cards, folded=False):
    return {
        "agentId": agent_id,
        "holeCards": cards,
        "stackChips": 900,
        "currentBetChips": 100,
        "folded": folded,
    }


def make_table(
    street="Flop",
    board=None,
    actions=None,
    seats=None,
    profiles=None,
    **allowed_overrides,
):
    allowed = {
        "availableActions": actions or ["fold", "check", "bet"],
        "callAmount": 100,
        "minBet": 50,
        "minRaiseTo": 300,
        "maxCommit": 1000,
    }
    allowed.update(allowed_overrides)
    return {
        "street": street,
        "boardCards": board or ["KH", "7D", "2C"],
        "potChips": 400,
        "currentBet": 200,
        "allowedActions": allowed,
        "seats": seats
        or [
            make_seat("hero", ["AS", "QS"]),
            make_seat("villain", []),
        ],
        "opponentProfiles": profiles or {},
    }


def test_profiled_strategy_tightens_preflop_multiway():
    hero = make_seat("hero", ["9S", "TD"])
    seats = [hero] + [make_seat(f"v{i}", []) for i in range(5)]
    table = make_table(
        street="Preflop",
        board=[],
        actions=["fold", "call"],
        seats=seats,
        callAmount=100,
    )

    action, amount, message = choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    assert "Profiled weak" in message


def test_profiled_strategy_pressures_patient_heads_up_opponent():
    hero = make_seat("hero", ["3S", "8D"])
    table = make_table(
        actions=["fold", "check", "bet"],
        seats=[hero, make_seat("nit", [])],
        profiles={
            "nit": {
                "hands_seen": 20,
                "vpip": 2,
                "pfr": 0,
                "folds": 12,
            }
        },
    )

    action, amount, message = choose_action(table, hero)

    assert action == "bet"
    assert amount >= 50
    assert "patient" in message


def test_profiled_strategy_value_bets_calling_station_multiway():
    hero = make_seat("hero", ["KS", "8D"])
    table = make_table(
        board=["KH", "8C", "2D"],
        actions=["fold", "check", "bet"],
        seats=[hero, make_seat("station", []), make_seat("other", [])],
        profiles={
            "station": {
                "hands_seen": 20,
                "vpip": 14,
                "calls": 12,
                "bets": 1,
                "raises": 1,
                "folds": 2,
            }
        },
    )

    action, amount, message = choose_action(table, hero)

    assert action == "bet"
    assert amount >= 50
    assert "value bet" in message


def test_profiled_strategy_bluff_catches_known_bluffer():
    hero = make_seat("hero", ["KS", "8D"])
    table = make_table(
        board=["KH", "7D", "2C"],
        actions=["fold", "call"],
        seats=[hero, make_seat("bluffer", [])],
        profiles={
            "bluffer": {
                "hands_seen": 20,
                "vpip": 12,
                "bets": 8,
                "raises": 6,
                "calls": 2,
                "folds": 2,
                "showdowns": 5,
                "weak_aggressive_showdowns": 3,
            }
        },
        callAmount=120,
    )

    action, amount, message = choose_action(table, hero)

    assert action == "call"
    assert amount == 120
    assert "Bluff-catching" in message
