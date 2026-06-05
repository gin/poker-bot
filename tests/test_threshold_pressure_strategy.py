from eval.selfplay import run_selfplay
from poker_bot.strategies.threshold_pressure import choose_action


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


def test_threshold_pressure_bets_dry_board_slightly_over_threshold():
    hero = make_seat(cards=["3S", "8D"])
    table = make_table(
        board=["KH", "7D", "2C"],
        actions=["fold", "check", "bet"],
        seats=[hero, make_seat("villain", [])],
    )

    action, amount, message = choose_action(table, hero)

    assert action == "bet"
    assert amount >= 50
    assert "defend threshold" in message


def test_threshold_pressure_does_not_bluff_wet_multiway_board():
    hero = make_seat(cards=["3S", "8D"])
    table = make_table(
        board=["9H", "TH", "JC"],
        actions=["fold", "check", "bet"],
        seats=[hero, make_seat("v1", []), make_seat("v2", [])],
    )

    action, amount, message = choose_action(table, hero)

    assert action == "check"
    assert amount is None
    assert "checking" in message


def test_threshold_pressure_value_raises_strong_hand():
    hero = make_seat(cards=["KS", "8D"])
    table = make_table(
        board=["KH", "KD", "2C"],
        actions=["fold", "call", "raise"],
        seats=[hero, make_seat("villain", [])],
        minRaiseTo=350,
    )

    action, amount, message = choose_action(table, hero)

    assert action == "raise"
    assert amount >= 350
    assert "value raise" in message


def test_threshold_pressure_folds_marginal_hand_to_value_heavy_line():
    hero = make_seat(cards=["3S", "8D"])
    table = make_table(
        board=["KH", "7D", "2C"],
        actions=["fold", "call"],
        seats=[hero, make_seat("villain", [])],
        callAmount=350,
    )

    action, amount, message = choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    assert "value-heavy" in message


def test_threshold_pressure_positive_average_against_profiled_counter():
    seeds = [1, 2, 3, 4, 5]
    results = [
        run_selfplay(
            "threshold_pressure",
            opponent_name="profiled_counter_adaptive",
            hands=200,
            seed=seed,
        )
        for seed in seeds
    ]
    average_net = sum(result.net_chips for result in results) / len(results)

    assert average_net > 0
