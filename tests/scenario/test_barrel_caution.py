"""Scenario tests for triple-barrel caution."""

from poker_bot.strategies import s2base as strategy


def make_river_table(
    action_history,
    hero_cards=None,
    pot_chips=3944,
    facing_bet=1011,
    street="River",
    opponent_vpip=0.5,
    opponent_pfr=0.3,
):
    if hero_cards is None:
        hero_cards = ["7h", "7d"]
    board_cards = (
        ["Js", "6c", "3s", "2d", "Qd"]
        if hero_cards != ["Td", "Tc"]
        else ["Ah", "Qd", "Ts"]
    )
    return {
        "street": street,
        "boardCards": board_cards,
        "potChips": pot_chips,
        "buttonSeatNumber": 4,
        "actionHistory": action_history,
        "seats": [
            {
                "agentId": "hero",
                "seatNumber": 1,
                "holeCards": hero_cards,
                "stackChips": 2000,
                "currentBetChips": 0,
                "folded": False,
                "hasFolded": False,
            },
            {
                "agentId": "villain",
                "seatNumber": 2,
                "holeCards": [],
                "stackChips": 2000,
                "currentBetChips": facing_bet,
                "folded": False,
                "hasFolded": False,
            },
            {
                "agentId": "villain-3",
                "seatNumber": 3,
                "holeCards": [],
                "stackChips": 2000,
                "currentBetChips": 0,
                "folded": True,
                "hasFolded": True,
            },
            {
                "agentId": "villain-4",
                "seatNumber": 4,
                "holeCards": [],
                "stackChips": 2000,
                "currentBetChips": 0,
                "folded": True,
                "hasFolded": True,
            },
        ],
        "opponentProfiles": {
            "villain": {
                "hands_seen": 50,
                "vpip": int(round(opponent_vpip * 50)),
                "pfr": int(round(opponent_pfr * 50)),
            }
        },
        "allowedActions": {
            "availableActions": ["fold", "call", "raise"],
            "callAmount": facing_bet,
            "callChips": facing_bet,
        },
    }


def test_triple_barrel_with_top_pair_good_kicker_calls_on_river():
    """Kd Qc on Js 6c 3s 2d Qd vs triple barrel → don't fold.

    Hero has top pair with King kicker on the river. Even against a triple
    barrel, this is strong enough to continue because we beat bluffs and
    thinner value hands.
    """
    table = make_river_table(
        action_history=[
            {"agentId": "villain", "action": "bet", "street": "Flop"},
            {"agentId": "villain", "action": "bet", "street": "Turn"},
            {"agentId": "villain", "action": "bet", "street": "River"},
        ],
        hero_cards=["Kd", "Qc"],
        pot_chips=3944,
        facing_bet=1011,
    )
    hero = table["seats"][0]

    result = strategy.profiled_choose_action(table, hero)

    assert result is not None
    assert result[0] != "fold"


def test_double_barrel_with_medium_hand_can_call_on_river():
    """Kd Qc on Js 6c 3s 2d Qd vs double barrel → fold.

    Without the triple-barrel signal, the medium hand still folds at this price
    because the required equity is too high.
    """
    table = make_river_table(
        action_history=[
            {"agentId": "villain", "action": "bet", "street": "Flop"},
            {"agentId": "villain", "action": "bet", "street": "Turn"},
        ],
        hero_cards=["Kd", "Qc"],
        pot_chips=3944,
        facing_bet=1011,
    )
    hero = table["seats"][0]

    result = strategy.profiled_choose_action(table, hero)

    assert result is not None
    assert result[0] == "fold"


def test_triple_barrel_with_strong_hand_not_fold_if_opponent_is_loose():
    """TT on Ah Qd Ts vs triple barrel from loose opponent → continue.

    A loose opponent (high VPIP/PFR) triple-barreling is more likely to include
    bluffs and thinner value hands. Hero should not fold strong hands.
    """
    table = make_river_table(
        action_history=[
            {"agentId": "villain", "action": "bet", "street": "Flop"},
            {"agentId": "villain", "action": "bet", "street": "Turn"},
            {"agentId": "villain", "action": "bet", "street": "River"},
        ],
        hero_cards=["Td", "Tc"],
        pot_chips=74,
        facing_bet=29,
        opponent_vpip=0.45,
        opponent_pfr=0.30,
    )
    hero = table["seats"][0]

    result = strategy.profiled_choose_action(table, hero)

    assert result is not None
    assert result[0] != "fold"


def test_triple_barrel_with_strong_hand_fold_if_opponent_is_tight():
    """TT on Ah Qd Ts vs triple barrel from tight opponent → fold.

    A tight opponent (low VPIP/PFR) triple-barreling is heavily weighted toward
    strong value hands. Hero should fold even strong one-pair hands.
    """
    table = make_river_table(
        action_history=[
            {"agentId": "villain", "action": "bet", "street": "Flop"},
            {"agentId": "villain", "action": "bet", "street": "Turn"},
            {"agentId": "villain", "action": "bet", "street": "River"},
        ],
        hero_cards=["Td", "Tc"],
        pot_chips=74,
        facing_bet=29,
        opponent_vpip=0.12,
        opponent_pfr=0.08,
    )
    hero = table["seats"][0]

    result = strategy.profiled_choose_action(table, hero)

    assert result is not None
    assert result[0] == "fold"
