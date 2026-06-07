from poker_bot.strategies.loader import load_strategy
from poker_bot.strategies.survival_balanced_postflop_pressure import choose_action


def make_seat(agent_id="hero", seat_number=4, cards=None, stack=1800):
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
        "potChips": 75,
        "currentBet": 50,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": actions or ["fold", "call", "raise"],
            "callAmount": 50,
            "callChips": 50,
            "minRaiseTo": 150,
            "minBet": 50,
            "maxCommit": hero["stackChips"],
            "raiseRange": {"min": 150, "max": hero["stackChips"]},
            "betRange": {"min": 50, "max": hero["stackChips"]},
        },
        "seats": seats,
        "opponentProfiles": {},
    }


def fold_villains(table, seat_numbers):
    for seat in table["seats"]:
        if seat["seatNumber"] in seat_numbers:
            seat["folded"] = True


def test_survival_balanced_postflop_pressure_loads_as_strategy():
    assert load_strategy("survival_balanced_postflop_pressure") is choose_action


def test_postflop_pressure_probes_medium_hand_on_dry_board():
    hero = make_seat(cards=["8S", "8D"], seat_number=6)
    table = make_table(
        street="Flop",
        board=["KH", "7D", "2C"],
        actions=["fold", "check", "bet"],
        hero=hero,
    )
    table["potChips"] = 500
    table["currentBet"] = 0
    table["allowedActions"]["callAmount"] = 0
    table["allowedActions"]["callChips"] = 0
    fold_villains(table, {2, 3})

    action, amount, message = choose_action(table, hero)

    assert action == "bet"
    assert amount == 160
    assert "dry-board probe" in message


def test_postflop_pressure_defends_cheap_medium_hand_when_covered():
    hero = make_seat(cards=["8S", "8D"], seat_number=6, stack=1000)
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
    table["allowedActions"]["minRaiseTo"] = 120
    table["seats"][0]["stackChips"] = 5000

    action, amount, message = choose_action(table, hero)

    assert action == "call"
    assert amount == 40
    assert "anti-bully cheap medium defense" in message


def test_postflop_pressure_defends_cheap_good_draw_when_covered():
    hero = make_seat(cards=["9S", "8S"], seat_number=6, stack=1000)
    table = make_table(
        street="Flop",
        board=["7H", "6D", "2C"],
        actions=["fold", "call", "raise"],
        hero=hero,
    )
    table["potChips"] = 200
    table["currentBet"] = 40
    table["allowedActions"]["callAmount"] = 40
    table["allowedActions"]["callChips"] = 40
    table["allowedActions"]["minRaiseTo"] = 120
    table["seats"][0]["stackChips"] = 5000

    action, amount, message = choose_action(table, hero)

    assert action == "call"
    assert amount == 40
    assert "anti-bully cheap draw defense" in message


def test_postflop_pressure_still_folds_expensive_bully_pressure():
    hero = make_seat(cards=["8S", "8D"], seat_number=6, stack=1000)
    table = make_table(
        street="Flop",
        board=["KH", "7D", "2C"],
        actions=["fold", "call", "raise"],
        hero=hero,
    )
    table["potChips"] = 200
    table["currentBet"] = 400
    table["allowedActions"]["callAmount"] = 400
    table["allowedActions"]["callChips"] = 400
    table["allowedActions"]["minRaiseTo"] = 900
    table["seats"][0]["stackChips"] = 5000

    action, amount, message = choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    assert "anti-bully" not in message
