from poker_bot.strategies import auto_research_v003 as strategy
from poker_bot.strategies.loader import load_strategy


def make_profile(*, calls=120, folds_to_bet=180, opportunities=240):
    return {
        "hands_seen": 100,
        "calls": calls,
        "bets": 20,
        "raises": 20,
        "folds": 160,
        "fold_to_bet": folds_to_bet,
        "opportunities_to_fold_to_bet": opportunities,
    }


def make_seat(agent_id="hero", seat_number=1, cards=None, folded=False):
    seat = {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": cards or ["AS", "KS"],
        "stackChips": 1800,
        "currentBetChips": 0,
    }
    if folded:
        seat["folded"] = True
    return seat


def make_table(*, hero=None, profiles=None):
    hero = hero or make_seat()
    seats = [
        hero,
        make_seat("villain-2", 2, cards=[]),
        make_seat("villain-3", 3, cards=[]),
        make_seat("villain-4", 4, cards=[]),
        make_seat("villain-5", 5, cards=[], folded=True),
        make_seat("villain-6", 6, cards=[], folded=True),
    ]
    return {
        "street": "Flop",
        "boardCards": ["7C", "4D", "2S"],
        "potChips": 300,
        "currentBet": 0,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": ["fold", "check", "bet"],
            "callAmount": 0,
            "callChips": 0,
            "minBet": 50,
            "maxCommit": hero["stackChips"],
            "betRange": {"min": 50, "max": hero["stackChips"]},
        },
        "seats": seats,
        "opponentProfiles": profiles
        or {f"villain-{index}": make_profile() for index in range(2, 7)},
    }


def test_auto_research_v003_loads_as_strategy():
    assert load_strategy("auto_research_v003") is strategy.choose_action


def test_active_opponents_ignore_empty_seats():
    hero = make_seat()
    table = make_table(hero=hero)
    table["seats"].append(
        {
            "agentId": None,
            "seatNumber": 7,
            "stackChips": 0,
            "currentBetChips": 0,
        }
    )

    assert strategy.active_opponents(table, hero) == 3


def test_high_fold_table_requires_profile_consensus():
    table = make_table()

    assert strategy.high_fold_to_bet_table(table)

    table["opponentProfiles"]["villain-2"] = make_profile(
        folds_to_bet=20,
        opportunities=240,
    )
    table["opponentProfiles"]["villain-3"] = make_profile(
        folds_to_bet=20,
        opportunities=240,
    )

    assert not strategy.high_fold_to_bet_table(table)


def test_high_fold_table_accepts_three_active_high_fold_profiles():
    table = make_table()
    table["opponentProfiles"] = {
        f"villain-{index}": make_profile() for index in range(2, 5)
    }

    assert strategy.high_fold_to_bet_table(table)


def test_range_mixed_dry_probe_can_bet_with_range_edge(monkeypatch):
    monkeypatch.setattr(strategy, "choose_weighted", lambda *args, **kwargs: "bet")
    base = ("check", None, "forced check")
    hero = make_seat(cards=["AS", "KS"])
    table = make_table(hero=hero)

    action, amount, message = strategy.range_mixed_dry_probe(table, hero, base)

    assert action == "bet"
    assert amount >= table["allowedActions"]["minBet"]
    assert "mixed range probe" in message


def test_range_mixed_dry_probe_respects_mixed_check(monkeypatch):
    monkeypatch.setattr(strategy, "choose_weighted", lambda *args, **kwargs: "check")
    base = ("check", None, "forced check")
    hero = make_seat(cards=["AS", "KS"])
    table = make_table(hero=hero)

    assert strategy.range_mixed_dry_probe(table, hero, base) is None


def test_range_sizing_uses_range_fields_when_flat_minimums_are_absent():
    hero = make_seat()
    table = make_table(hero=hero)
    allowed = table["allowedActions"]
    allowed["minBet"] = None
    allowed["betRange"] = {"min": 80, "max": 120}
    allowed["maxCommit"] = 500

    assert strategy.bet_amount(table, allowed, 0.50) == 120

    table["currentBet"] = 50
    table["bigBlindChips"] = 50
    allowed["minRaiseTo"] = None
    allowed["raiseRange"] = {"min": 150, "max": 220}

    assert strategy.raise_to_amount(table, allowed, 4.0) == 200
