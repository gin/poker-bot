from poker_bot.strategies import survival_balanced_pp_pd_pr_patch1 as strategy
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
        "boardCards": board or ["9S", "AH", "9H", "AC"],
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


def test_survival_balanced_pp_pd_pr_patch1_loads_as_strategy():
    assert load_strategy("survival_balanced_pp_pd_pr_patch1") is strategy.choose_action


def test_patch1_detects_board_dominated_two_pair():
    assert strategy.board_dominated_two_pair(
        ["JC", "8D"],
        ["9S", "AH", "9H", "AC"],
        rank=2,
    )


def test_patch1_does_not_mark_clean_two_pair_as_fragile():
    assert not strategy.fragile_rank_two(
        ["AS", "KS"],
        ["AH", "KD", "2C"],
        rank=2,
    )


def test_patch1_folds_board_dominated_two_pair_to_large_bet(monkeypatch):
    force_blueprint(monkeypatch, "call")
    hero = make_seat(cards=["JC", "8D"], stack=700)
    table = make_table(hero=hero, actions=["fold", "call", "all-in"])
    table["potChips"] = 1737
    table["currentBet"] = 997
    table["allowedActions"]["callAmount"] = 641
    table["allowedActions"]["callChips"] = 641

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    assert "board-dominated" in message


def test_patch1_checks_fragile_rank_two_when_checked_to(monkeypatch):
    force_blueprint(monkeypatch, "check")
    hero = make_seat(cards=["JC", "8D"])
    table = make_table(hero=hero, actions=["fold", "check", "bet"])
    table["currentBet"] = 0
    table["allowedActions"]["callAmount"] = 0
    table["allowedActions"]["callChips"] = 0

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "check"
    assert amount is None
    assert "forced check" in message


def test_patch1_raises_clean_two_pair_for_value(monkeypatch):
    force_blueprint(monkeypatch, "fold")
    hero = make_seat(cards=["AS", "KS"])
    table = make_table(
        street="Flop",
        board=["AH", "KD", "2C"],
        hero=hero,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "raise"
    assert amount >= table["allowedActions"]["minRaiseTo"]
    assert "protection raise rank 2" in message
