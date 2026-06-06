from poker_bot.strategies.loader import load_strategy
from poker_bot.strategies.survival_aggressive import choose_action


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


def fold_seat(seat):
    updated = dict(seat)
    updated["folded"] = True
    return updated


def test_survival_aggressive_loads_as_strategy():
    assert load_strategy("survival_aggressive") is choose_action


def test_survival_aggressive_folds_marginal_preflop():
    hero = make_seat(cards=["QD", "8C"], seat_number=6)
    table = make_table(hero=hero)

    action, amount, message = choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    assert "aggressive fold" in message


def test_survival_aggressive_raises_premium_preflop():
    hero = make_seat(cards=["AS", "KS"], seat_number=6)
    table = make_table(hero=hero)

    action, amount, message = choose_action(table, hero)

    assert action == "raise"
    assert amount is not None
    assert "aggressive open/value raise" in message


def test_survival_aggressive_continuation_bets_dry_board():
    hero = make_seat(cards=["AS", "QS"], seat_number=6)
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
    for index in range(4):
        table["seats"][index] = fold_seat(table["seats"][index])

    action, amount, message = choose_action(table, hero)

    assert action == "bet"
    assert amount >= 50
    assert "aggressive" in message
