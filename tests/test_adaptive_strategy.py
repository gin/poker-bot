from eval.selfplay import run_selfplay
from poker_bot.strategies.adaptive import (
    RANK_VALUES,
    choose_action,
    has_top_pair_good_kicker,
    top_pair_defense_price_cap,
    top_pair_kicker_value,
)


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


def make_seat(cards, stack=900, current_bet=100):
    return {
        "agentId": "my-agent",
        "holeCards": cards,
        "stackChips": stack,
        "currentBetChips": current_bet,
    }


def test_adaptive_raises_premium_preflop_hand():
    table = make_table(
        street="Preflop",
        board=[],
        actions=["fold", "call", "raise"],
        minRaiseTo=150,
    )

    action, amount, message = choose_action(table, make_seat(["AS", "AD"]))

    assert action == "raise"
    assert amount >= 150
    assert "Premium preflop" in message


def test_adaptive_checks_weak_hand_when_check_is_available():
    table = make_table(actions=["fold", "check", "bet"])

    action, amount, message = choose_action(table, make_seat(["3S", "8D"]))

    assert action == "check"
    assert amount is None
    assert message == "Not advantageous, checking"


def test_adaptive_bets_top_pair_for_value():
    table = make_table(board=["KH", "7D", "2C"], actions=["fold", "check", "bet"])

    action, amount, message = choose_action(table, make_seat(["KS", "9D"]))

    assert action == "bet"
    assert amount >= 50
    assert "Thin value" in message


def test_adaptive_identifies_top_pair_good_kicker():
    hole_cards = ["AS", "KC"]
    board_cards = ["TD", "9D", "2H", "KS"]

    assert top_pair_kicker_value(hole_cards, board_cards) == RANK_VALUES["A"]
    assert has_top_pair_good_kicker(hole_cards, board_cards)
    assert top_pair_defense_price_cap(
        hole_cards,
        board_cards,
        street="Turn",
        active_opponents=5,
    ) > 34 / (84 + 34)


def test_adaptive_folds_marginal_hand_to_bad_price():
    table = make_table(
        board=["AH", "KD", "QC"],
        actions=["fold", "call"],
        callAmount=500,
    )

    action, amount, message = choose_action(table, make_seat(["3S", "8D"]))

    assert action == "fold"
    assert amount is None
    assert "below price" in message


def test_adaptive_selfplay_positive_average_against_simple():
    seeds = [1, 2, 3, 4, 5]
    results = [run_selfplay("adaptive", hands=200, seed=seed) for seed in seeds]
    average_net = sum(result.net_chips for result in results) / len(results)

    assert average_net > 0
