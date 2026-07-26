"""Benchmark-driven regressions for board-owned paired-rank probes."""

import pytest

from poker_bot.strategies.adaptive import choose_action


def _table(board_cards, **history):
    table = {
        "street": {3: "Flop", 4: "Turn", 5: "River"}[len(board_cards)],
        "boardCards": board_cards,
        "potChips": 100,
        "currentBet": 0,
        "allowedActions": {
            "availableActions": ["fold", "check", "bet"],
            "callAmount": 0,
            "minBet": 4,
            "maxCommit": 1_000,
        },
    }
    table.update(history)
    return table


def _seat(hole_cards):
    return {
        "agentId": "hero",
        "seatNumber": 1,
        "holeCards": hole_cards,
        "stackChips": 1_000,
    }


@pytest.mark.parametrize(
    ("hole_cards", "board_cards"),
    [
        (["Qd", "2s"], ["8h", "8c", "9d"]),
        (["Qd", "2s"], ["8h", "8c", "9d", "Kc"]),
        (["Ks", "9c"], ["3s", "3h", "Ad"]),
        (["Ks", "9c"], ["3s", "3h", "Ad", "7h"]),
    ],
)
def test_board_owned_pair_first_probe_is_pressure_not_thin_value(
    hole_cards, board_cards
):
    action, amount, message = choose_action(_table(board_cards), _seat(hole_cards))

    assert action == "bet"
    assert amount == 38
    assert message == "Pressure probing board-owned rank"
    assert "Thin value against simple" not in message


@pytest.mark.parametrize(
    ("hole_cards", "board_cards", "history_key", "history"),
    [
        (
            ["Qd", "2s"],
            ["8h", "8c", "9d", "Kc"],
            "actionHistory",
            [{"agentId": "hero", "action": "bet", "street": "Flop"}],
        ),
        (
            ["Qd", "2s"],
            ["8h", "8c", "9d", "Kc", "7h"],
            "action_history",
            [{"seatNumber": 1, "action": "bet", "street": "Flop"}],
        ),
        (
            ["Ks", "9c"],
            ["3s", "3h", "Ad", "7h"],
            "recentEvents",
            [{"street": "Flop", "summary": {"agentId": "hero", "action": "bet"}}],
        ),
        (
            ["Ks", "9c"],
            ["3s", "3h", "Ad", "7h", "8d"],
            "recentEvents",
            [{"street": "Flop", "summary": {"seatNumber": 1, "action": "bet"}}],
        ),
    ],
)
def test_board_owned_pair_checks_after_hero_flop_bet(
    hole_cards, board_cards, history_key, history
):
    action, amount, message = choose_action(
        _table(board_cards, **{history_key: history}), _seat(hole_cards)
    )

    assert action == "check"
    assert amount is None
    assert message == "Board-owned rank after prior bet, checking"


def test_top_pair_retains_thin_value_action():
    action, amount, message = choose_action(
        _table(['Qs', '9d', '4h']), _seat(['Qd', 'Tc'])
    )

    assert action == "bet"
    assert amount == 38
    assert message == "Thin value against simple"


def test_private_pocket_pair_retains_thin_value_action():
    action, amount, message = choose_action(
        _table(['8h', '5d', '2c']), _seat(['6s', '6d'])
    )

    assert action == "bet"
    assert amount == 38
    assert message == "Thin value against simple"


def test_private_two_pair_retains_strong_value_action():
    action, amount, message = choose_action(
        _table(['9d', '8c', '2h']), _seat(['9s', '8s'])
    )

    assert action == "bet"
    assert amount == 38
    assert message == "Value betting rank 2"
