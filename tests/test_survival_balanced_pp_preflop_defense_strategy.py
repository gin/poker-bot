from poker_bot.strategies import survival_balanced_pp_preflop_defense as strategy
from poker_bot.strategies.loader import load_strategy


def make_seat(agent_id="hero", seat_number=6, cards=None, stack=1800):
    return {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": cards or ["AS", "KS"],
        "stackChips": stack,
        "currentBetChips": 0,
    }


def make_table(street="Preflop", board=None, actions=None, hero=None):
    hero = hero or make_seat()
    seats = [
        make_seat(f"villain-{index + 1}", index + 1, [], stack=1800)
        for index in range(6)
    ]
    seats[hero["seatNumber"] - 1] = hero
    return {
        "street": street,
        "boardCards": board or [],
        "potChips": 150,
        "currentBet": 100,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": actions or ["fold", "call", "raise"],
            "callAmount": 50,
            "callChips": 50,
            "minRaiseTo": 250,
            "minBet": 50,
            "maxCommit": hero["stackChips"],
            "raiseRange": {"min": 250, "max": hero["stackChips"]},
            "betRange": {"min": 50, "max": hero["stackChips"]},
        },
        "seats": seats,
        "opponentProfiles": {},
    }


def test_survival_balanced_pp_preflop_defense_loads_as_strategy():
    loaded = load_strategy("survival_balanced_pp_preflop_defense")

    assert loaded is strategy.choose_action


def test_preflop_defense_calls_playable_late_position_raise():
    hero = make_seat(cards=["KS", "9S"], seat_number=6)
    table = make_table(hero=hero)

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "call"
    assert amount == 50
    assert "balanced defend" in message


def test_preflop_defense_folds_weak_hand_to_raise():
    hero = make_seat(cards=["8S", "3D"], seat_number=6)
    table = make_table(hero=hero)

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    assert "preflop fold" in message


def test_preflop_defense_raises_premium_pair():
    hero = make_seat(cards=["AS", "AD"], seat_number=4)
    table = make_table(hero=hero)

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "raise"
    assert amount >= table["allowedActions"]["minRaiseTo"]
    assert "value/open raise" in message


def test_preflop_defense_selective_calls_when_not_facing_raise():
    hero = make_seat(cards=["QS", "9S"], seat_number=6)
    table = make_table(hero=hero)
    table["currentBet"] = 50
    table["allowedActions"]["callAmount"] = 25
    table["allowedActions"]["callChips"] = 25

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "call"
    assert amount == 25
    assert "selective call" in message


def test_preflop_defense_keeps_anti_bully_postflop_call(monkeypatch):
    monkeypatch.setattr(
        strategy.survival_lookahead,
        "choose_action",
        lambda _table, _seat: ("fold", None, "forced fold"),
    )
    hero = make_seat(cards=["KS", "QD"], seat_number=6, stack=1000)
    table = make_table(
        street="Flop",
        board=["KH", "7D", "2C"],
        actions=["fold", "call", "raise"],
        hero=hero,
    )
    table["potChips"] = 200
    table["currentBet"] = 40
    table["allowedActions"]["callAmount"] = 40
    table["allowedActions"]["callChips"] = 40
    table["seats"][0]["stackChips"] = 5000

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "call"
    assert amount == 40
    assert "anti-bully cheap medium defense" in message
