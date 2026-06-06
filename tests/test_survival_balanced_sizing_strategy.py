from poker_bot.strategies.loader import load_strategy
from poker_bot.strategies.survival_balanced_sizing import (
    choose_action,
    profile_adjusted_thresholds,
)


def make_seat(agent_id="hero", seat_number=4, cards=None, stack=1800):
    return {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": cards or ["AS", "KS"],
        "stackChips": stack,
        "currentBetChips": 0,
    }


def make_table(street="Preflop", board=None, actions=None, hero=None, profiles=None):
    hero = hero or make_seat()
    seats = [
        make_seat(f"villain-{index + 1}", index + 1, [], stack=2200)
        for index in range(6)
    ]
    seats[hero["seatNumber"] - 1] = hero
    return {
        "street": street,
        "boardCards": board or [],
        "potChips": 600 if street != "Preflop" else 75,
        "currentBet": 250 if street != "Preflop" else 50,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": actions or ["fold", "call", "raise"],
            "callAmount": 150 if street != "Preflop" else 50,
            "callChips": 150 if street != "Preflop" else 50,
            "minRaiseTo": 500 if street != "Preflop" else 150,
            "minBet": 50,
            "maxCommit": hero["stackChips"],
            "raiseRange": {"min": 500, "max": hero["stackChips"]},
            "betRange": {"min": 50, "max": hero["stackChips"]},
        },
        "seats": seats,
        "opponentProfiles": profiles or {},
    }


def tight_profile():
    return {
        "hands_seen": 30,
        "vpip": 3,
        "pfr": 1,
        "bets": 2,
        "raises": 1,
        "calls": 1,
        "folds": 20,
    }


def loose_aggressive_profile():
    return {
        "hands_seen": 30,
        "vpip": 18,
        "pfr": 10,
        "bets": 12,
        "raises": 10,
        "calls": 3,
        "folds": 4,
    }


def test_survival_balanced_sizing_loads_as_strategy():
    assert load_strategy("survival_balanced_sizing") is choose_action


def test_profile_adjusted_thresholds_open_wider_against_tight_table():
    hero = make_seat(cards=["KS", "9S"], seat_number=6)
    default_table = make_table(hero=hero)
    tight_table = make_table(
        hero=hero,
        profiles={
            "villain-1": tight_profile(),
            "villain-2": tight_profile(),
            "villain-3": tight_profile(),
        },
    )

    default_raise, _default_call, _default_style = profile_adjusted_thresholds(
        default_table, hero
    )
    tight_raise, _tight_call, style = profile_adjusted_thresholds(tight_table, hero)

    assert style == "tight"
    assert tight_raise < default_raise


def test_survival_balanced_sizing_tightens_calls_against_lag_table():
    hero = make_seat(cards=["AD", "2C"], seat_number=6)
    table = make_table(
        hero=hero,
        profiles={
            "villain-1": loose_aggressive_profile(),
            "villain-2": loose_aggressive_profile(),
            "villain-3": loose_aggressive_profile(),
        },
    )

    action, _amount, message = choose_action(table, hero)

    assert action == "fold"
    assert "loose_aggressive" in message


def test_survival_balanced_sizing_defends_stack_pressure_with_top_pair():
    hero = make_seat(cards=["KS", "QD"], seat_number=4, stack=1000)
    table = make_table(
        street="Flop",
        board=["KH", "7D", "2C"],
        actions=["fold", "call"],
        hero=hero,
    )
    table["seats"][0]["stackChips"] = 2200
    table["allowedActions"]["callAmount"] = 125
    table["allowedActions"]["callChips"] = 125

    action, amount, message = choose_action(table, hero)

    assert action == "call"
    assert amount == 125
    assert "stack-pressure" in message


def test_survival_balanced_sizing_uses_mixed_value_size():
    hero = make_seat(cards=["KS", "QD"], seat_number=6)
    table = make_table(
        street="Flop",
        board=["KH", "7D", "2C"],
        actions=["fold", "check", "bet"],
        hero=hero,
    )
    table["currentBet"] = 0
    table["allowedActions"]["callAmount"] = 0
    table["allowedActions"]["callChips"] = 0

    action, amount, message = choose_action(table, hero)

    assert action == "bet"
    assert 50 <= amount <= hero["stackChips"]
    assert "sizing" in message
