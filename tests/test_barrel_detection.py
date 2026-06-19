"""Tests for opponent barrel detection."""

from poker_bot.strategies.s2base import (
    opponent_barrels_current_street,
    opponent_barrels_streets,
)


def make_table(action_history, street="River", n_players=4):
    seats = []
    for i in range(1, n_players + 1):
        if i == 1:
            seats.append(
                {
                    "agentId": "hero",
                    "seatNumber": i,
                    "holeCards": ["7h", "7d"],
                    "stackChips": 1000,
                    "currentBetChips": 0,
                    "folded": False,
                    "hasFolded": False,
                }
            )
        elif i == 2:
            seats.append(
                {
                    "agentId": "villain",
                    "seatNumber": i,
                    "holeCards": [],
                    "stackChips": 1000,
                    "currentBetChips": 0,
                    "folded": False,
                    "hasFolded": False,
                }
            )
        else:
            seats.append(
                {
                    "agentId": f"villain-{i}",
                    "seatNumber": i,
                    "holeCards": [],
                    "stackChips": 1000,
                    "currentBetChips": 0,
                    "folded": True,
                    "hasFolded": True,
                }
            )
    return {
        "street": street,
        "boardCards": ["Js", "6c", "3s", "6d", "Qd"],
        "potChips": 1000,
        "seats": seats,
        "actionHistory": action_history,
    }


def test_opponent_barrels_streets_detects_triple_barrel():
    table = make_table(
        [
            {"agentId": "villain", "action": "bet", "street": "Flop"},
            {"agentId": "villain", "action": "bet", "street": "Turn"},
            {"agentId": "villain", "action": "bet", "street": "River"},
        ]
    )
    hero = table["seats"][0]

    assert opponent_barrels_streets(table, hero) is True


def test_opponent_barrels_streets_requires_all_three_streets():
    table = make_table(
        [
            {"agentId": "villain", "action": "bet", "street": "Flop"},
            {"agentId": "villain", "action": "bet", "street": "Turn"},
        ]
    )
    hero = table["seats"][0]

    assert opponent_barrels_streets(table, hero) is False


def test_opponent_barrels_current_street_detects_river_bet():
    table = make_table(
        [
            {"agentId": "villain", "action": "bet", "street": "Flop"},
            {"agentId": "villain", "action": "bet", "street": "Turn"},
            {"agentId": "villain", "action": "bet", "street": "River"},
        ],
        street="River",
    )
    hero = table["seats"][0]

    assert opponent_barrels_current_street(table, hero) is True


def test_opponent_barrels_current_street_ignores_other_streets():
    table = make_table(
        [
            {"agentId": "villain", "action": "bet", "street": "Flop"},
            {"agentId": "villain", "action": "bet", "street": "Turn"},
        ],
        street="River",
    )
    hero = table["seats"][0]

    assert opponent_barrels_current_street(table, hero) is False


def test_opponent_barrels_streets_ignores_hero_actions():
    table = make_table(
        [
            {"agentId": "hero", "action": "bet", "street": "Flop"},
            {"agentId": "hero", "action": "bet", "street": "Turn"},
            {"agentId": "hero", "action": "bet", "street": "River"},
        ]
    )
    hero = table["seats"][0]

    assert opponent_barrels_streets(table, hero) is False
