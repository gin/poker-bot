from poker_bot.strategies import auto_research_v007 as strategy
from poker_bot.strategies.loader import load_strategy


def make_seat(agent_id="hero", seat_number=5, cards=None, folded=False):
    seat = {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": cards or ["KS", "JC"],
        "stackChips": 1800,
        "currentBetChips": 0,
    }
    if folded:
        seat["folded"] = True
    return seat


def make_table(
    *,
    hero=None,
    board=None,
    street="Turn",
    available=None,
    current_bet=34,
    call_amount=34,
    pot=110,
):
    hero = hero or make_seat()
    return {
        "street": street,
        "boardCards": board or ["KD", "9C", "4H", "2S"],
        "potChips": pot,
        "currentBet": current_bet,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": available or ["fold", "call", "raise"],
            "callAmount": call_amount,
            "callChips": call_amount,
            "minBet": 5,
            "minRaiseTo": max(current_bet + call_amount + 5, 10),
            "maxCommit": hero["stackChips"],
            "betRange": {"min": 5, "max": hero["stackChips"]},
            "raiseRange": {"min": 10, "max": hero["stackChips"]},
        },
        "seats": [
            make_seat("button", 1, cards=[]),
            make_seat("sb", 2, cards=[]),
            make_seat("bb", 3, cards=[]),
            make_seat("utg", 4, cards=[]),
            hero,
            make_seat("co", 6, cards=[]),
        ],
        "opponentProfiles": {},
    }


def force_v005(monkeypatch, decision):
    monkeypatch.setattr(strategy.champion, "choose_action", lambda *_args: decision)


def force_cfr(monkeypatch, call_probability=0.65):
    monkeypatch.setattr(
        strategy,
        "_kuhn_strategy",
        lambda: {"Q|b": {"call": call_probability}, "K|b": {"call": 1.0}},
    )


def test_auto_research_v007_loads_as_strategy():
    assert load_strategy("auto_research_v007") is strategy.choose_action


def add_value_heavy_profiles(table):
    table["opponentProfiles"] = {
        f"villain-{index}": {
            "hands_seen": 40,
            "call_frequency": 0.02,
            "aggression_frequency": 0.64,
            "fold_to_bet_frequency": 0.36,
        }
        for index in range(1, 6)
    }


def test_simple_profile_river_bluff_catch_can_call(monkeypatch):
    force_v005(monkeypatch, ("fold", None, "baseline river fold rank 1"))
    force_cfr(monkeypatch, 0.80)
    monkeypatch.setattr(
        strategy,
        "resolve_distribution",
        lambda *_args, **_kwargs: type(
            "Decision",
            (),
            {"selected": "call", "roll": 0.01, "summary": lambda self: "call:100%"},
        )(),
    )
    hero = make_seat(cards=["3S", "3D"])
    table = make_table(
        hero=hero,
        board=["5C", "2S", "6D", "AS", "9H"],
        street="River",
        call_amount=50,
        pot=100,
    )
    add_value_heavy_profiles(table)

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "call"
    assert amount == 50
    assert "simple-profile river bluff catch" in message


def test_simple_profile_river_bluff_catch_respects_price_band(monkeypatch):
    force_v005(monkeypatch, ("fold", None, "baseline river fold rank 1"))
    force_cfr(monkeypatch, 0.80)
    hero = make_seat(cards=["3S", "3D"])
    table = make_table(
        hero=hero,
        board=["5C", "2S", "6D", "AS", "9H"],
        street="River",
        call_amount=20,
        pot=100,
    )
    add_value_heavy_profiles(table)

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    # assert message.startswith("v007 champion baseline")


def test_baseline_wins_when_v005_already_adjusted(monkeypatch):
    force_v005(monkeypatch, ("call", 34, "v005 strong top-pair defense"))
    hero = make_seat(cards=["AS", "KC"])
    table = make_table(hero=hero)

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "call"
    assert amount == 34
    # assert message.startswith("v007 champion baseline")


def test_paired_board_range_fold_can_release_fragile_two_pair(monkeypatch):
    force_v005(monkeypatch, ("call", 139, "v005 capped paired-board aggression"))
    monkeypatch.setattr(
        strategy,
        "resolve_distribution",
        lambda *_args, **_kwargs: type(
            "Decision",
            (),
            {"selected": "fold", "roll": 0.01, "summary": lambda self: "fold:100%"},
        )(),
    )
    hero = make_seat(cards=["7S", "5H"])
    table = make_table(
        hero=hero,
        board=["AC", "AS", "QC", "QH"],
        street="Turn",
        call_amount=139,
        pot=225,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    assert "paired-board fold" in message
