from poker_bot.strategies.survival_lookup import choose_action, table_style


def make_seat(agent_id="hero", seat_number=4, cards=None, stack=1800):
    return {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": cards or ["AS", "KS"],
        "stackChips": stack,
        "currentBetChips": 0,
    }


def make_table(street="Preflop", board=None, actions=None, hero=None, profiles=None):
    hero = hero or make_seat()
    seats = [
        make_seat(f"villain-{index + 1}", index + 1, [], stack=1800)
        for index in range(6)
    ]
    seats[hero["seatNumber"] - 1] = hero
    return {
        "street": street,
        "boardCards": board or [],
        "potChips": 600,
        "currentBet": 250,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": actions or ["fold", "call", "raise"],
            "callAmount": 150,
            "callChips": 150,
            "minRaiseTo": 500,
            "minBet": 50,
            "maxCommit": hero["stackChips"],
            "raiseRange": {"min": 500, "max": hero["stackChips"]},
            "betRange": {"min": 50, "max": hero["stackChips"]},
        },
        "seats": seats,
        "opponentProfiles": profiles or {},
    }


def fold_seat(seat):
    updated = dict(seat)
    updated["folded"] = True
    return updated


def loose_aggressive_profile():
    return {
        "hands_seen": 20,
        "vpip": 14,
        "pfr": 8,
        "bets": 8,
        "raises": 8,
        "calls": 2,
        "folds": 2,
    }


def test_survival_lookup_detects_loose_aggressive_table():
    hero = make_seat()
    table = make_table(
        hero=hero,
        profiles={
            "villain-1": loose_aggressive_profile(),
            "villain-2": loose_aggressive_profile(),
            "villain-3": loose_aggressive_profile(),
        },
    )

    assert table_style(table, hero) == "loose_aggressive"


def test_survival_lookup_uses_default_lookup_policy_without_profiles():
    hero = make_seat(cards=["AS", "KS"])
    table = make_table(hero=hero)

    action, amount, message = choose_action(table, hero)

    assert action in {"call", "raise"}
    assert amount is not None
    assert "lookup unknown/profiled" in message


def test_survival_lookup_bluff_catches_loose_aggressive_table():
    hero = make_seat(cards=["KS", "8D"])
    table = make_table(
        street="Flop",
        board=["KH", "7D", "2C"],
        actions=["fold", "call"],
        hero=hero,
        profiles={
            "villain-1": loose_aggressive_profile(),
            "villain-2": loose_aggressive_profile(),
            "villain-3": loose_aggressive_profile(),
        },
    )

    action, amount, message = choose_action(table, hero)

    assert action == "call"
    assert amount == 150
    assert "lookup bluff catch" in message


def test_survival_lookup_defends_before_pressure_when_callable():
    hero = make_seat(cards=["KS", "8D"])
    table = make_table(
        street="Flop",
        board=["KH", "7D", "2C"],
        actions=["fold", "call", "raise"],
        hero=hero,
    )

    _action, _amount, message = choose_action(table, hero)

    assert "lookup unknown/profiled" in message


def test_survival_lookup_short_handed_does_not_force_preflop_pressure():
    hero = make_seat(cards=["AS", "KS"])
    table = make_table(actions=["fold", "call", "raise"], hero=hero)
    table["seats"][0] = fold_seat(table["seats"][0])
    table["seats"][1] = fold_seat(table["seats"][1])
    table["seats"][2] = fold_seat(table["seats"][2])

    _action, _amount, message = choose_action(table, hero)

    assert "lookup value pressure" not in message
    assert "lookup short_handed/profiled" in message


def test_survival_lookup_short_handed_pot_controls_made_hands():
    hero = make_seat(cards=["KS", "7S"])
    table = make_table(
        street="Flop",
        board=["KH", "7D", "2C"],
        actions=["fold", "call", "raise"],
        hero=hero,
    )
    table["seats"][0] = fold_seat(table["seats"][0])
    table["seats"][1] = fold_seat(table["seats"][1])
    table["seats"][2] = fold_seat(table["seats"][2])

    action, amount, message = choose_action(table, hero)

    assert action == "call"
    assert amount == 150
    assert "lookup short-handed pot-control" in message
