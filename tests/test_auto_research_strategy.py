from poker_bot.strategies import auto_research as strategy
from poker_bot.strategies.loader import load_strategy


def make_seat(agent_id="hero", seat_number=1, cards=None, stack=1800, folded=False):
    seat = {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": cards or ["AS", "7D"],
        "stackChips": stack,
        "currentBetChips": 0,
    }
    if folded:
        seat["folded"] = True
    return seat


def make_table(
    *,
    street="Preflop",
    board=None,
    actions=None,
    hero=None,
    seats=None,
    pot=150,
    current_bet=50,
    call_amount=50,
    min_raise_to=150,
):
    hero = hero or make_seat()
    if seats is None:
        seats = [hero, make_seat("villain", 2, cards=[])]
    return {
        "street": street,
        "boardCards": board or [],
        "potChips": pot,
        "currentBet": current_bet,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": actions or ["fold", "call", "raise"],
            "callAmount": call_amount,
            "callChips": call_amount,
            "minBet": 50,
            "minRaiseTo": min_raise_to,
            "maxCommit": hero["stackChips"],
            "raiseRange": {"min": min_raise_to, "max": hero["stackChips"]},
            "betRange": {"min": 50, "max": hero["stackChips"]},
        },
        "seats": seats,
        "opponentProfiles": {},
    }


def force_counter_baseline(monkeypatch, action, amount=None, message="forced"):
    monkeypatch.setattr(
        strategy.counter,
        "choose_action",
        lambda _table, _seat: (action, amount, message),
    )


def test_auto_research_loads_as_strategy():
    assert load_strategy("auto_research") is strategy.choose_action


def test_auto_research_widens_short_handed_open_range():
    hero = make_seat(cards=["AS", "7D"])
    table = make_table(hero=hero)

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "raise"
    assert amount >= table["allowedActions"]["minRaiseTo"]
    assert "widened open" in message


def test_auto_research_adds_thin_dry_board_probe(monkeypatch):
    force_counter_baseline(monkeypatch, "check", None, "counter check")
    hero = make_seat(cards=["AS", "7D"])
    table = make_table(
        street="Flop",
        board=["KH", "7D", "2C"],
        actions=["fold", "check", "bet"],
        hero=hero,
        pot=300,
        current_bet=0,
        call_amount=0,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "bet"
    assert amount >= table["allowedActions"]["minBet"]
    assert "thin dry-board probe" in message


def test_auto_research_cheap_bluff_catches_short_handed(monkeypatch):
    force_counter_baseline(monkeypatch, "fold", None, "counter fold")
    hero = make_seat(cards=["KS", "QD"], stack=1800)
    table = make_table(
        street="Flop",
        board=["KH", "8D", "2C"],
        actions=["fold", "call", "raise"],
        hero=hero,
        pot=500,
        current_bet=80,
        call_amount=80,
        min_raise_to=220,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "call"
    assert amount == 80
    assert "cheap bluff-catch" in message


def test_auto_research_keeps_counter_branch_for_six_player_hand(monkeypatch):
    force_counter_baseline(monkeypatch, "check", None, "counter six-max")
    hero = make_seat(cards=["AS", "7D"], seat_number=1)
    seats = [
        hero,
        make_seat("villain-2", 2, cards=[]),
        make_seat("villain-3", 3, cards=[]),
        make_seat("villain-4", 4, cards=[]),
        make_seat("villain-5", 5, cards=[]),
        make_seat("villain-6", 6, cards=[]),
    ]
    table = make_table(hero=hero, seats=seats)

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "check"
    assert amount is None
    assert "0:counter six-max" in message
