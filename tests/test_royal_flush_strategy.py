from poker_bot.strategies import royal_flush as strategy
from poker_bot.strategies.loader import load_strategy


def make_seat(agent_id="hero", seat_number=6, cards=None, stack=1800):
    return {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": cards or ["AS", "KS"],
        "stackChips": stack,
        "currentBetChips": 0,
    }


def make_table(street="Turn", board=None, actions=None, hero=None):
    hero = hero or make_seat()
    seats = [
        make_seat(f"villain-{index + 1}", index + 1, [], stack=1800)
        for index in range(6)
    ]
    seats[hero["seatNumber"] - 1] = hero
    return {
        "street": street,
        "boardCards": board or ["QS", "JS", "2D", "7C"],
        "potChips": 200,
        "currentBet": 80,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": actions or ["fold", "call", "raise"],
            "callAmount": 80,
            "callChips": 80,
            "minRaiseTo": 220,
            "minBet": 50,
            "maxCommit": hero["stackChips"],
            "raiseRange": {"min": 220, "max": hero["stackChips"]},
            "betRange": {"min": 50, "max": hero["stackChips"]},
        },
        "seats": seats,
        "opponentProfiles": {},
    }


def force_blueprint(monkeypatch, action):
    monkeypatch.setattr(
        strategy.survival_lookahead,
        "choose_action",
        lambda _table, _seat: (action, None, f"forced {action}"),
    )


def test_royal_flush_loads_as_strategy():
    assert load_strategy("royal_flush") is strategy.choose_action


def test_royal_flush_possible_with_two_suited_royal_hole_cards_preflop():
    assert strategy.royal_flush_possible(["AS", "KS"], [])


def test_royal_flush_possible_ignores_lone_royal_hole_card_preflop():
    assert not strategy.royal_flush_possible(["AS", "7D"], [])


def test_royal_flush_possible_with_hole_card_and_board_royals():
    assert strategy.royal_flush_possible(["AS", "7D"], ["KS", "QS", "2C"])


def test_royal_flush_possible_when_draw_can_complete_on_river():
    assert strategy.royal_flush_possible(["AS", "KS"], ["QS", "JS", "2D"])


def test_royal_flush_override_checks_instead_of_betting(monkeypatch):
    force_blueprint(monkeypatch, "bet")
    hero = make_seat(cards=["AS", "KS"])
    table = make_table(
        street="Flop",
        board=["QS", "JS", "2D"],
        actions=["fold", "check", "bet"],
        hero=hero,
    )
    table["currentBet"] = 0
    table["allowedActions"]["callAmount"] = 0
    table["allowedActions"]["callChips"] = 0

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "check"
    assert amount is None
    assert "royal flush possible" in message


def test_royal_flush_override_raises_when_facing_action(monkeypatch):
    force_blueprint(monkeypatch, "fold")
    hero = make_seat(cards=["AS", "KS"])
    table = make_table(
        street="Flop",
        board=["QS", "JS", "2D"],
        actions=["fold", "call", "raise"],
        hero=hero,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "raise"
    assert amount >= table["allowedActions"]["minRaiseTo"]
    assert "royal flush possible" in message


def test_royal_flush_override_calls_when_raise_unavailable(monkeypatch):
    force_blueprint(monkeypatch, "fold")
    hero = make_seat(cards=["AS", "KS"])
    table = make_table(
        street="Turn",
        board=["QS", "JS", "2D", "7C"],
        actions=["fold", "call"],
        hero=hero,
    )
    table["allowedActions"]["callAmount"] = 125
    table["allowedActions"]["callChips"] = 125

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "call"
    assert amount == 125
    assert "without raise" in message


def test_royal_flush_strategy_falls_back_without_royal_potential(monkeypatch):
    force_blueprint(monkeypatch, "check")
    hero = make_seat(cards=["9C", "4D"])
    table = make_table(
        street="Flop",
        board=["QS", "JS", "2D"],
        actions=["fold", "check"],
        hero=hero,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "check"
    assert amount is None
    assert "forced check" in message
