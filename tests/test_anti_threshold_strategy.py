from eval.selfplay import run_selfplay
from poker_bot.strategies.anti_threshold import choose_action


def make_seat(agent_id="hero", cards=None):
    return {
        "agentId": agent_id,
        "holeCards": cards or ["AS", "QS"],
        "stackChips": 900,
        "currentBetChips": 100,
    }


def make_table(street="Flop", board=None, actions=None, seats=None, **overrides):
    allowed = {
        "availableActions": actions or ["fold", "check", "bet"],
        "callAmount": 100,
        "minBet": 50,
        "minRaiseTo": 300,
        "maxCommit": 1000,
    }
    allowed.update(overrides)
    hero = make_seat()
    return {
        "street": street,
        "boardCards": board or ["KH", "7D", "2C"],
        "potChips": 400,
        "currentBet": 200,
        "allowedActions": allowed,
        "seats": seats or [hero, make_seat("villain", [])],
    }


def test_anti_threshold_defends_medium_hand_against_small_pressure():
    hero = make_seat(cards=["KS", "8D"])
    table = make_table(
        board=["KH", "7D", "2C"],
        actions=["fold", "call"],
        seats=[hero, make_seat("villain", [])],
        callAmount=100,
    )

    action, amount, message = choose_action(table, hero)

    assert action == "call"
    assert amount == 100
    assert "medium hand" in message


def test_anti_threshold_value_raises_strong_hand():
    hero = make_seat(cards=["KS", "8D"])
    table = make_table(
        board=["KH", "KD", "2C"],
        actions=["fold", "call", "raise"],
        seats=[hero, make_seat("villain", [])],
        callAmount=100,
        minRaiseTo=350,
    )

    action, amount, message = choose_action(table, hero)

    assert action == "raise"
    assert amount >= 350
    assert "value" in message


def test_anti_threshold_does_not_float_wet_board_bad_price():
    hero = make_seat(cards=["3S", "8D"])
    table = make_table(
        board=["9H", "TH", "JC"],
        actions=["fold", "call"],
        seats=[hero, make_seat("villain", [])],
        callAmount=250,
    )

    action, amount, message = choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    assert "Declining" in message


def test_anti_threshold_positive_average_against_threshold_pressure():
    seeds = [1, 2, 3, 4, 5]
    results = [
        run_selfplay(
            "anti_threshold",
            opponent_name="threshold_pressure",
            hands=1000,
            seed=seed,
        )
        for seed in seeds
    ]
    average_net = sum(result.net_chips for result in results) / len(results)

    assert average_net > 0
