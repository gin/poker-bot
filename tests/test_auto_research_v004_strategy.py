from poker_bot.strategies import auto_research_v004 as strategy
from poker_bot.strategies.loader import load_strategy


class FakeDistributionDecision:
    def __init__(self, selected, options=(("check", 0.25), ("bet", 0.75))):
        self.selected = selected
        self.options = options
        self.total_weight = sum(weight for _value, weight in options)
        self.roll = 0.12

    @property
    def probabilities(self):
        return tuple(
            (value, weight / self.total_weight) for value, weight in self.options
        )

    def summary(self):
        return "/".join(
            f"{value}:{probability:.0%}" for value, probability in self.probabilities
        )


def make_profile(*, calls=120, folds_to_bet=90, opportunities=240):
    return {
        "hands_seen": 80,
        "calls": calls,
        "bets": 20,
        "raises": 20,
        "folds": 120,
        "fold_to_bet": folds_to_bet,
        "opportunities_to_fold_to_bet": opportunities,
    }


def make_seat(agent_id="hero", seat_number=1, cards=None, folded=False):
    seat = {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": cards or ["2D", "2S"],
        "stackChips": 1800,
        "currentBetChips": 0,
    }
    if folded:
        seat["folded"] = True
    return seat


def make_table(*, hero=None, board=None, available=None, seats=None, profiles=None):
    hero = hero or make_seat()
    return {
        "street": "Flop",
        "boardCards": board or ["3C", "6S", "7H"],
        "potChips": 15,
        "currentBet": 0,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": available or ["fold", "check", "bet"],
            "callAmount": 0,
            "callChips": 0,
            "minBet": 5,
            "maxCommit": hero["stackChips"],
            "betRange": {"min": 5, "max": hero["stackChips"]},
        },
        "seats": seats
        or [
            hero,
            make_seat("villain-2", 2, cards=[]),
            make_seat("villain-3", 3, cards=[]),
            make_seat("villain-4", 4, cards=[]),
            make_seat("villain-5", 5, cards=[]),
            make_seat("villain-6", 6, cards=[]),
        ],
        "opponentProfiles": profiles or {},
    }


def force_v003_bet(monkeypatch):
    monkeypatch.setattr(
        strategy.champion,
        "choose_action",
        lambda table, my_seat: ("bet", 5, "Thin value against simple"),
    )


def force_v003_fold(monkeypatch):
    monkeypatch.setattr(
        strategy.champion,
        "choose_action",
        lambda table, my_seat: ("fold", None, "Forced threshold fold"),
    )


def test_auto_research_v004_loads_as_strategy():
    assert load_strategy("auto_research_v004") is strategy.choose_action


def test_weak_underpair_wet_board_bet_is_checked(monkeypatch):
    force_v003_bet(monkeypatch)
    monkeypatch.setattr(
        strategy,
        "resolve_distribution",
        lambda *args, **kwargs: FakeDistributionDecision("check"),
    )
    hero = make_seat(cards=["2D", "2S"])
    table = make_table(hero=hero, board=["3C", "6S", "7H"])

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "check"
    assert amount is None
    assert "weak non-top pair" in message


def test_weak_underpair_wet_turn_bet_is_checked(monkeypatch):
    force_v003_bet(monkeypatch)
    monkeypatch.setattr(
        strategy,
        "resolve_distribution",
        lambda *args, **kwargs: FakeDistributionDecision("check"),
    )
    hero = make_seat(cards=["2D", "2S"])
    table = make_table(hero=hero, board=["3C", "6S", "7H", "5H"])
    table["street"] = "Turn"

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "check"
    assert amount is None
    assert "pot control" in message


def test_weak_underpair_wet_board_can_mix_back_to_bet(monkeypatch):
    force_v003_bet(monkeypatch)
    monkeypatch.setattr(
        strategy,
        "resolve_distribution",
        lambda *args, **kwargs: FakeDistributionDecision("bet"),
    )
    hero = make_seat(cards=["2D", "2S"])
    table = make_table(hero=hero, board=["3C", "6S", "7H"])

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "bet"
    assert amount == 5
    # assert message.startswith("v004 champion baseline")


def test_top_pair_value_bet_is_not_blocked(monkeypatch):
    force_v003_bet(monkeypatch)
    hero = make_seat(cards=["7D", "AS"])
    table = make_table(hero=hero, board=["3C", "6S", "7H"])

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "bet"
    assert amount == 5
    # assert message.startswith("v004 champion baseline")


def test_pair_plus_good_draw_bet_is_not_blocked(monkeypatch):
    force_v003_bet(monkeypatch)
    hero = make_seat(cards=["5D", "5S"])
    table = make_table(hero=hero, board=["3C", "4S", "8H"])

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "bet"
    assert amount == 5
    # assert message.startswith("v004 champion baseline")


def test_heads_up_weak_pair_bet_is_not_blocked(monkeypatch):
    force_v003_bet(monkeypatch)
    hero = make_seat(cards=["2D", "2S"])
    table = make_table(
        hero=hero,
        board=["3C", "6S", "7H"],
        seats=[
            hero,
            make_seat("villain-2", 2, cards=[]),
        ],
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "bet"
    assert amount == 5
    # assert message.startswith("v004 champion baseline")


def test_sticky_profiles_raise_weak_pair_check_probability():
    hero = make_seat(cards=["2D", "2S"])
    sticky_profiles = {
        f"villain-{index}": make_profile(calls=170, folds_to_bet=25)
        for index in range(2, 7)
    }
    high_fold_profiles = {
        f"villain-{index}": make_profile(calls=40, folds_to_bet=210)
        for index in range(2, 7)
    }
    sticky_summary = strategy.bayesian_pressure_summary(
        make_table(hero=hero, profiles=sticky_profiles),
        hero,
    )
    high_fold_summary = strategy.bayesian_pressure_summary(
        make_table(hero=hero, profiles=high_fold_profiles),
        hero,
    )

    assert strategy.weak_pair_check_probability(sticky_summary) > (
        strategy.weak_pair_check_probability(high_fold_summary)
    )


def test_mixed_pot_control_passes_weighted_pressure_to_chooser(monkeypatch):
    force_v003_bet(monkeypatch)
    captured = {}

    def fake_resolve_distribution(options, *args, **kwargs):
        captured["options"] = options
        captured["extra"] = kwargs["extra"]
        return FakeDistributionDecision("check", options)

    monkeypatch.setattr(strategy, "resolve_distribution", fake_resolve_distribution)
    hero = make_seat(cards=["2D", "2S"])
    table = make_table(hero=hero, board=["3C", "6S", "7H"])

    action, _amount, message = strategy.choose_action(table, hero)

    assert action == "check"
    assert captured["options"][0][0] == "check"
    assert 0.08 <= captured["options"][0][1] <= 0.84
    assert len(captured["extra"]) == 4
    assert "dist" in message


def test_threshold_pressure_response_can_mix_call(monkeypatch):
    force_v003_fold(monkeypatch)
    monkeypatch.setattr(
        strategy,
        "resolve_distribution",
        lambda *args, **kwargs: FakeDistributionDecision(
            "call",
            (("call", 0.55), ("fold", 0.45)),
        ),
    )
    hero = make_seat(cards=["7D", "2S"])
    table = make_table(
        hero=hero,
        board=["3C", "6S", "7H"],
        available=["fold", "call"],
        seats=[hero, make_seat("villain-2", 2, cards=[])],
    )
    table["potChips"] = 300
    table["currentBet"] = 50
    table["allowedActions"]["callAmount"] = 50
    table["allowedActions"]["callChips"] = 50

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "call"
    assert amount == 50
    assert "threshold-pressure defense" in message
    assert "call:55%/fold:45%" in message


def test_threshold_pressure_response_respects_mixed_fold(monkeypatch):
    force_v003_fold(monkeypatch)
    monkeypatch.setattr(
        strategy,
        "resolve_distribution",
        lambda *args, **kwargs: FakeDistributionDecision(
            "fold",
            (("call", 0.55), ("fold", 0.45)),
        ),
    )
    hero = make_seat(cards=["7D", "2S"])
    table = make_table(
        hero=hero,
        board=["3C", "6S", "7H"],
        available=["fold", "call"],
        seats=[hero, make_seat("villain-2", 2, cards=[])],
    )
    table["potChips"] = 300
    table["currentBet"] = 50
    table["allowedActions"]["callAmount"] = 50
    table["allowedActions"]["callChips"] = 50

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    # assert message.startswith("v004 champion baseline")
