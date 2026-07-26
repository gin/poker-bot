from poker_bot.strategies.loader import load_strategy
from poker_bot.strategies.survival_lookahead import choose_action


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
        make_seat(f"villain-{index + 1}", index + 1, [], stack=1800)
        for index in range(6)
    ]
    seats[hero["seatNumber"] - 1] = hero
    return {
        "street": street,
        "boardCards": board or [],
        "potChips": 600,
        "currentBet": 250,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": actions or ["fold", "call", "raise"],
            "callAmount": 150,
            "callChips": 150,
            "minRaiseTo": 500,
            "minBet": 50,
            "maxCommit": hero["stackChips"],
            "raiseRange": {"min": 500, "max": hero["stackChips"]},
            "betRange": {"min": 50, "max": hero["stackChips"]},
        },
        "seats": seats,
        "opponentProfiles": profiles or {},
    }


def loose_aggressive_profile():
    return {
        "hands_seen": 20,
        "preflop_hands_seen": 20,
        "profile_stats_schema_version": 2,
        "profile_stats_provenance": "canonical",
        "vpip": 14,
        "pfr": 8,
        "bets": 8,
        "raises": 8,
        "calls": 2,
        "folds": 2,
    }


def test_survival_lookahead_loads_as_strategy():
    assert load_strategy("survival_lookahead") is choose_action


def test_survival_lookahead_uses_blueprint_preflop():
    hero = make_seat(cards=["AS", "KS"])
    table = make_table(hero=hero)

    action, amount, message = choose_action(table, hero)

    assert action in {"call", "raise"}
    assert amount is not None
    assert message.startswith("lookahead blueprint:")


def test_survival_lookahead_defends_made_hand_when_blueprint_folds():
    hero = make_seat(cards=["KS", "7S"])
    table = make_table(
        street="Flop",
        board=["KH", "7D", "2C"],
        actions=["fold", "call", "raise"],
        hero=hero,
        profiles={
            "villain-1": loose_aggressive_profile(),
            "villain-2": loose_aggressive_profile(),
            "villain-3": loose_aggressive_profile(),
        },
    )
    table["allowedActions"]["callAmount"] = 280
    table["allowedActions"]["callChips"] = 280

    action, amount, message = choose_action(table, hero)

    assert action == "call"
    assert amount == 280
    assert message.startswith("lookahead loose_aggressive:")
