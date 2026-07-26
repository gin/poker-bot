"""Telemetry regressions for expensive anti-bully calls on fragile hands."""

from poker_bot.strategies import s5base
from poker_bot.strategies.survival_sixmax import anti_bully_action


def _table(
    hole_cards,
    board_cards,
    *,
    call,
    pot,
    stack,
    history=None,
    available_actions=None,
):
    hero = {
        "seatNumber": 5,
        "agentId": "hero",
        "holeCards": hole_cards,
        "stackChips": stack,
        "currentBetChips": 0,
        "folded": False,
        "hasFolded": False,
    }
    villain = {
        "seatNumber": 2,
        "agentId": "bully",
        "holeCards": [],
        "stackChips": 5_000,
        "currentBetChips": call,
        "folded": False,
        "hasFolded": False,
    }
    return {
        "street": {3: "Flop", 4: "Turn", 5: "River"}[len(board_cards)],
        "boardCards": board_cards,
        "potChips": pot,
        "currentBet": call,
        "buttonSeatNumber": 2,
        "actingSeatNumber": 5,
        "selfSeatNumber": 5,
        "actionHistory": history or [],
        "seats": [hero, villain],
        "opponentProfiles": {
            "bully": {"api_stats": {"playingStyle": {"label": "loose_aggressive"}}}
        },
        "allowedActions": {
            "availableActions": available_actions or ["fold", "call", "raise"],
            "callAmount": call,
            "callChips": call,
            "minBet": 20,
            "minRaiseTo": call * 2,
            "maxCommit": stack,
            "raiseRange": {"min": call * 2, "max": stack},
        },
    }


def test_tt_effective_all_in_paired_board_folds_end_to_end():
    table = _table(
        ["Ts", "Th"],
        ["2h", "Kh", "4d", "Kd", "3h"],
        call=696,
        pot=1_567,
        stack=567,
        history=[{"agentId": "bully", "action": "raise", "street": "River"}],
    )
    hero = table["seats"][0]

    anti_bully = anti_bully_action(table, hero)
    action, _amount, message = s5base.choose_action(table, hero)

    assert 696 >= 567
    assert anti_bully is None or anti_bully[2] != "anti-bully continue rank 2"
    assert action == "fold"
    assert "effective all-in fold" in message


def test_jj_expensive_one_pair_bluff_catch_is_not_called():
    table = _table(
        ["Jd", "Jh"],
        ["Tc", "Kh", "6d", "3d"],
        call=606,
        pot=1_439,
        stack=993,
        history=[
            {"agentId": "bully", "action": "raise", "street": "Flop"},
            {"agentId": "bully", "action": "bet", "street": "Turn"},
        ],
    )
    hero = table["seats"][0]

    decision = anti_bully_action(table, hero)

    assert 606 > 993 / 2
    assert decision is None or decision[0] != "call"


def test_cheap_one_pair_bluff_catch_remains_callable():
    table = _table(
        ["Kh", "Qd"],
        ["Ks", "7c", "2d"],
        call=100,
        pot=600,
        stack=1_000,
        history=[{"agentId": "bully", "action": "bet", "street": "Flop"}],
        available_actions=["fold", "call"],
    )

    decision = anti_bully_action(table, table["seats"][0])

    assert decision is not None
    assert decision[0] == "call"
    assert decision[1] == 100
    assert "bluff catch" in decision[2]


def test_private_set_keeps_strong_continuation_path():
    table = _table(
        ["Kh", "Kd"],
        ["Ks", "7c", "2d"],
        call=300,
        pot=1_000,
        stack=500,
        history=[{"agentId": "bully", "action": "bet", "street": "Flop"}],
        available_actions=["fold", "call"],
    )

    decision = anti_bully_action(table, table["seats"][0])

    assert decision is not None
    assert decision[0] == "call"
    assert decision[1] == 300
    assert "continue rank 3" in decision[2]
