"""Scenario tests for vulnerable_flush_guard.

These tests exercise the guard through the public `choose_action()` API. They do
not call `vulnerable_flush_guard()` directly.
"""

from poker_bot.strategies import s2base as strategy


def make_table(
    hero_cards,
    board_cards,
    hero_action,
    pot_chips,
    facing_bet=0,
    call_amount_value=0,
    available=None,
    street="Flop",
    n_players=6,
):
    if available is None:
        available = ["fold", "call", "raise", "check"]
    seats = []
    for i in range(1, n_players + 1):
        if i == 1:
            seats.append(
                {
                    "agentId": "hero",
                    "seatNumber": i,
                    "holeCards": hero_cards,
                    "stackChips": 2000,
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
                    "stackChips": 2000,
                    "currentBetChips": call_amount_value,
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
                    "stackChips": 2000,
                    "currentBetChips": 0,
                    "folded": True,
                    "hasFolded": True,
                }
            )
    return {
        "street": street,
        "boardCards": board_cards,
        "potChips": pot_chips,
        "buttonSeatNumber": n_players,
        "facing_bet": facing_bet,
        "seats": seats,
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": available,
            "callAmount": call_amount_value,
            "callChips": call_amount_value,
            "minBet": 10,
            "minRaiseTo": call_amount_value * 2 if call_amount_value else 20,
        },
    }


def test_non_nut_flush_on_paired_board_calls_when_facing_bet_and_intends_to_raise():
    """Hero has QhJh on Js 6s 6h 2h 3h (paired board).

    Facing a bet, the guard should downgrade the intended raise to a call
    (bluff catch) rather than fold away what might still be the best hand.
    """
    table = make_table(
        hero_cards=["Qh", "Jh"],
        board_cards=["Js", "6s", "6h", "2h", "3h"],
        hero_action="raise",
        pot_chips=500,
        facing_bet=1,
        call_amount_value=100,
        available=["fold", "call", "raise"],
    )
    hero = table["seats"][0]

    result = strategy.choose_action(table, hero)

    assert result is not None
    assert result[0] == "call"
    assert result[2] == "non-nut flush on paired board: bluff catch"


def test_non_nut_flush_on_paired_board_checks_back_when_first_to_act():
    """Hero has QhJh on Js 6s 6h 2h 3h (paired board).

    First to act, the guard should check back to control the pot instead of
    betting a vulnerable Q-high flush.
    """
    table = make_table(
        hero_cards=["Qh", "Jh"],
        board_cards=["Js", "6s", "6h", "2h", "3h"],
        hero_action="bet",
        pot_chips=500,
        facing_bet=0,
        call_amount_value=0,
        available=["fold", "bet", "check"],
    )
    hero = table["seats"][0]

    result = strategy.choose_action(table, hero)

    assert result is not None
    assert result[0] == "check"
    assert result[2] == "non-nut flush on paired board: check back"


def test_non_nut_flush_on_paired_board_folds_to_large_bet():
    """Hero has QhJh on Js 6s 6h 2h 3h (paired board).

    Facing a large bet (>33% pot odds), the guard should fold.
    """
    table = make_table(
        hero_cards=["Qh", "Jh"],
        board_cards=["Js", "6s", "6h", "2h", "3h"],
        hero_action="call",
        pot_chips=300,
        facing_bet=1,
        call_amount_value=150,
        available=["fold", "call"],
    )
    hero = table["seats"][0]

    result = strategy.choose_action(table, hero)

    assert result is not None
    assert result[0] == "fold"
    assert result[2].startswith("non-nut flush on paired board: folded large bet")


def test_non_nut_flush_on_paired_board_continues_to_small_bet():
    """Hero has QhJh on Js 6s 6h 2h 3h (paired board).

    Facing a small bet (<33% pot odds), the guard should not intervene and the
    base action should continue.
    """
    table = make_table(
        hero_cards=["Qh", "Jh"],
        board_cards=["Js", "6s", "6h", "2h", "3h"],
        hero_action="call",
        pot_chips=500,
        facing_bet=1,
        call_amount_value=50,
        available=["fold", "call"],
    )
    hero = table["seats"][0]

    result = strategy.choose_action(table, hero)

    assert result is not None
    assert result[0] == "call"


def test_k_high_flush_on_paired_board_raises_normally():
    """Hero has KhQh on Js 6s 6h 2h 3h (paired board).

    K-high flush is strong enough to raise normally. The guard should not
    intervene.
    """
    table = make_table(
        hero_cards=["Kh", "Qh"],
        board_cards=["Js", "6s", "6h", "2h", "3h"],
        hero_action="raise",
        pot_chips=500,
        facing_bet=0,
        call_amount_value=0,
        available=["fold", "call", "raise"],
    )
    hero = table["seats"][0]

    result = strategy.choose_action(table, hero)

    assert result is not None
    assert result[0] == "raise"


def test_nut_flush_on_paired_board_does_not_trigger_guard():
    """Hero has AhKh on Js 6s 6h 2h 3h (paired board).

    Nut flush should not trigger the vulnerable flush guard.
    """
    table = make_table(
        hero_cards=["Ah", "Kh"],
        board_cards=["Js", "6s", "6h", "2h", "3h"],
        hero_action="raise",
        pot_chips=500,
        facing_bet=0,
        call_amount_value=0,
        available=["fold", "call", "raise"],
    )
    hero = table["seats"][0]

    result = strategy.choose_action(table, hero)

    assert result is not None
    assert result[0] == "raise"


def test_non_nut_flush_on_unpaired_board_does_not_trigger_guard():
    """Hero has KhQh on Js 6s 2h 3h 4h (unpaired board).

    Non-nut flush on an unpaired board should not trigger the guard.
    """
    table = make_table(
        hero_cards=["Kh", "Qh"],
        board_cards=["Js", "6s", "2h", "3h", "4h"],
        hero_action="raise",
        pot_chips=500,
        facing_bet=0,
        call_amount_value=0,
        available=["fold", "call", "raise"],
    )
    hero = table["seats"][0]

    result = strategy.choose_action(table, hero)

    assert result is not None
    assert result[0] == "raise"
