from poker_bot.strategies.survival_sixmax import choose_action


def make_seat(seat_number=4, cards=None, stack=1800):
    return {
        "agentId": "hero",
        "seatNumber": seat_number,
        "holeCards": cards or ["AS", "KS"],
        "stackChips": stack,
        "currentBetChips": 0,
    }


def make_table(
    street="Preflop",
    board=None,
    actions=None,
    hero=None,
    profiles=None,
    **allowed,
):
    hero = hero or make_seat()
    seats = [
        {
            "agentId": f"villain-{index + 1}",
            "seatNumber": index + 1,
            "holeCards": [],
            "stackChips": 1800,
            "currentBetChips": 0,
        }
        for index in range(6)
    ]
    seats[hero["seatNumber"] - 1] = hero
    defaults = {
        "availableActions": actions or ["fold", "call", "raise"],
        "callAmount": 50,
        "callChips": 50,
        "minRaiseTo": 100,
        "minBet": 50,
        "maxCommit": hero["stackChips"] + hero.get("currentBetChips", 0),
        "raiseRange": {
            "min": 100,
            "max": hero["stackChips"] + hero.get("currentBetChips", 0),
        },
        "betRange": {
            "min": 50,
            "max": hero["stackChips"] + hero.get("currentBetChips", 0),
        },
    }
    defaults.update(allowed)
    return {
        "street": street,
        "boardCards": board or [],
        "potChips": 75,
        "currentBet": 50,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": defaults,
        "seats": seats,
        "opponentProfiles": profiles or {},
    }


def test_survival_sixmax_continues_premium_early_position():
    hero = make_seat(4, ["AS", "KS"])
    table = make_table(hero=hero)

    action, amount, message = choose_action(table, hero)

    assert action in {"call", "raise"}
    assert amount is not None
    assert "survival preflop" in message


def test_survival_sixmax_folds_weak_early_position_to_price():
    hero = make_seat(4, ["7S", "2D"])
    table = make_table(hero=hero, callAmount=150, callChips=150, currentBet=150)

    action, amount, message = choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    assert "survival preflop" in message


def test_survival_sixmax_avoids_expensive_multiway_pair_call():
    hero = make_seat(4, ["KS", "8D"])
    table = make_table(
        street="Flop",
        board=["KH", "7D", "2C"],
        actions=["fold", "call"],
        hero=hero,
        callAmount=350,
        callChips=350,
        currentBet=350,
    )
    table["potChips"] = 500

    action, amount, message = choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    assert "survival defense" in message


def test_survival_sixmax_backraises_premium_against_big_stack_bully():
    hero = make_seat(4, ["AS", "KS"], stack=1200)
    table = make_table(
        hero=hero,
        callAmount=250,
        callChips=250,
        currentBet=250,
        minRaiseTo=500,
        profiles={
            "villain-1": {
                "hands_seen": 20,
                "vpip": 14,
                "pfr": 8,
                "bets": 8,
                "raises": 8,
                "calls": 2,
                "folds": 2,
            }
        },
    )
    table["seats"][0]["stackChips"] = 2600
    table["seats"][0]["currentBetChips"] = 250

    action, amount, message = choose_action(table, hero)

    assert action == "raise"
    assert amount >= 500
    assert "anti-bully" in message


def test_survival_sixmax_bluff_catches_top_pair_against_known_bully():
    hero = make_seat(4, ["KS", "8D"], stack=1200)
    table = make_table(
        street="Flop",
        board=["KH", "7D", "2C"],
        actions=["fold", "call"],
        hero=hero,
        callAmount=250,
        callChips=250,
        currentBet=250,
        profiles={
            "villain-1": {
                "hands_seen": 20,
                "vpip": 14,
                "pfr": 8,
                "bets": 8,
                "raises": 8,
                "calls": 2,
                "folds": 2,
            }
        },
    )
    table["potChips"] = 600
    table["seats"][0]["stackChips"] = 2600
    table["seats"][0]["currentBetChips"] = 250

    action, amount, message = choose_action(table, hero)

    assert action == "call"
    assert amount == 250
    assert "anti-bully" in message
