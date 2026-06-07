from poker_bot.strategies import survival_balanced_pp_pd_postflop_raise as strategy
from poker_bot.strategies.loader import load_strategy


def make_seat(agent_id="hero", seat_number=4, cards=None, stack=1800):
    return {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": cards or ["AS", "KS"],
        "stackChips": stack,
        "currentBetChips": 0,
    }


def make_table(street="Flop", board=None, actions=None, hero=None):
    hero = hero or make_seat()
    seats = [
        make_seat(f"villain-{index + 1}", index + 1, [], stack=2200)
        for index in range(6)
    ]
    seats[hero["seatNumber"] - 1] = hero
    return {
        "street": street,
        "boardCards": board or ["KH", "7D", "2C"],
        "potChips": 200,
        "currentBet": 40,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": actions or ["fold", "call", "raise"],
            "callAmount": 40,
            "callChips": 40,
            "minRaiseTo": 120,
            "minBet": 50,
            "maxCommit": hero["stackChips"],
            "raiseRange": {"min": 120, "max": hero["stackChips"]},
            "betRange": {"min": 50, "max": hero["stackChips"]},
        },
        "seats": seats,
        "opponentProfiles": {},
    }


def fold_villains(table, seat_numbers):
    for seat in table["seats"]:
        if seat["seatNumber"] in seat_numbers:
            seat["folded"] = True


def force_blueprint_fold(monkeypatch):
    monkeypatch.setattr(
        strategy.survival_lookahead,
        "choose_action",
        lambda _table, _seat: ("fold", None, "forced fallback fold"),
    )


def test_survival_balanced_pp_pd_postflop_raise_loads_as_strategy():
    loaded = load_strategy("survival_balanced_pp_pd_postflop_raise")

    assert loaded is strategy.choose_action


def test_postflop_raise_back_raises_trips_when_blueprint_would_fold(monkeypatch):
    force_blueprint_fold(monkeypatch)
    hero = make_seat(cards=["7C", "7H"])
    table = make_table(board=["7S", "2D", "9C"], hero=hero)

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "raise"
    assert amount >= table["allowedActions"]["minRaiseTo"]
    assert "value raise" in message


def test_postflop_raise_back_raises_rank_two_for_protection(monkeypatch):
    force_blueprint_fold(monkeypatch)
    hero = make_seat(cards=["JC", "JH"])
    table = make_table(board=["9S", "5H", "9D"], hero=hero)
    table["potChips"] = 43
    table["currentBet"] = 24
    table["allowedActions"]["callAmount"] = 24
    table["allowedActions"]["callChips"] = 24
    table["allowedActions"]["minRaiseTo"] = 41

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "raise"
    assert amount >= 41
    assert "protection" in message


def test_postflop_raise_back_calls_rank_two_without_raise_available(monkeypatch):
    force_blueprint_fold(monkeypatch)
    hero = make_seat(cards=["7C", "7H"])
    table = make_table(
        street="River",
        board=["TH", "8S", "3S", "4C", "3C"],
        actions=["fold", "call"],
        hero=hero,
    )
    table["potChips"] = 169
    table["currentBet"] = 56
    table["allowedActions"]["callAmount"] = 56
    table["allowedActions"]["callChips"] = 56

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "call"
    assert amount == 56
    assert "rank 2" in message


def test_postflop_raise_back_can_raise_dry_board_medium_hand(monkeypatch):
    force_blueprint_fold(monkeypatch)
    monkeypatch.setattr(strategy, "stable_mix_percent", lambda *_args: 0)
    hero = make_seat(cards=["KS", "QD"], seat_number=6)
    table = make_table(board=["KH", "7D", "2C"], hero=hero)
    table["potChips"] = 500
    table["currentBet"] = 80
    table["allowedActions"]["callAmount"] = 80
    table["allowedActions"]["callChips"] = 80
    table["allowedActions"]["minRaiseTo"] = 200
    fold_villains(table, {2, 3, 4, 5})

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "raise"
    assert amount >= 200
    assert "medium raise" in message


def test_postflop_raise_back_can_semi_bluff_raise_good_draw(monkeypatch):
    force_blueprint_fold(monkeypatch)
    monkeypatch.setattr(strategy, "stable_mix_percent", lambda *_args: 0)
    hero = make_seat(cards=["9S", "8S"], seat_number=6)
    table = make_table(board=["7H", "6D", "2C"], hero=hero)
    table["potChips"] = 500
    table["currentBet"] = 80
    table["allowedActions"]["callAmount"] = 80
    table["allowedActions"]["callChips"] = 80
    table["allowedActions"]["minRaiseTo"] = 200
    fold_villains(table, {2, 3, 4, 5})

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "raise"
    assert amount >= 200
    assert "semi-bluff" in message
