from eval.selfplay import run_selfplay
from poker_bot.strategies.counter_adaptive import choose_action


def make_table(street="Flop", board=None, actions=None, **allowed_overrides):
    allowed = {
        "availableActions": actions or ["fold", "check", "bet"],
        "callAmount": 100,
        "minBet": 50,
        "minRaiseTo": 300,
        "maxCommit": 1000,
    }
    allowed.update(allowed_overrides)
    return {
        "street": street,
        "boardCards": board or ["KH", "7D", "2C"],
        "potChips": 400,
        "currentBet": 200,
        "allowedActions": allowed,
    }


def make_seat(cards):
    return {
        "agentId": "my-agent",
        "holeCards": cards,
        "stackChips": 900,
        "currentBetChips": 100,
    }


def test_counter_adaptive_pressure_bets_dry_board_when_checked_to():
    table = make_table(board=["KH", "7D", "2C"], actions=["fold", "check", "bet"])

    action, amount, message = choose_action(table, make_seat(["3S", "8D"]))

    assert action == "bet"
    assert amount >= 50
    assert "Small pressure" in message


def test_counter_adaptive_folds_marginal_hand_to_adaptive_aggression():
    table = make_table(
        board=["KH", "7D", "2C"],
        actions=["fold", "call"],
        callAmount=300,
    )

    action, amount, message = choose_action(table, make_seat(["3S", "8D"]))

    assert action == "fold"
    assert amount is None
    assert "Overfolding" in message


def test_counter_adaptive_value_bets_top_pair():
    table = make_table(board=["KH", "7D", "2C"], actions=["fold", "check", "bet"])

    action, amount, message = choose_action(table, make_seat(["KS", "8D"]))

    assert action == "bet"
    assert amount >= 50
    assert "Thin value" in message


def test_counter_adaptive_raises_strong_made_hand():
    table = make_table(
        board=["KH", "KD", "2C"],
        actions=["fold", "call", "raise"],
        callAmount=100,
        minRaiseTo=350,
    )

    action, amount, message = choose_action(table, make_seat(["KS", "8D"]))

    assert action == "raise"
    assert amount >= 350
    assert "Value raising" in message


def test_counter_adaptive_selfplay_positive_average_against_adaptive():
    seeds = [1, 2, 3, 4, 5]
    results = [
        run_selfplay("counter_adaptive", opponent_name="adaptive", hands=200, seed=seed)
        for seed in seeds
    ]
    average_net = sum(result.net_chips for result in results) / len(results)

    assert average_net > 0
