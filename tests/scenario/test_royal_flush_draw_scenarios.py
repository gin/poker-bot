import pytest

from poker_bot.strategies import royal_adaptive, royal_flush

STRATEGIES = [
    pytest.param("royal_flush", royal_flush.choose_action, id="royal-flush"),
    pytest.param("royal_adaptive", royal_adaptive.choose_action, id="royal-adaptive"),
]

ROYAL_DRAW_HANDS = [
    pytest.param(["AS", "KS"], ["QS", "JS", "2D"], id="suited-AK"),
    pytest.param(["KS", "QS"], ["AS", "7D", "2C"], id="suited-KQ"),
    pytest.param(["QS", "JS"], ["AS", "KS", "2D"], id="suited-QJ"),
]

BLOCKED_ROYAL_HANDS = [
    pytest.param(["AS", "KS"], ["9H", "7D", "2C"], id="blocked-suited-AK"),
    pytest.param(["KS", "QS"], ["9H", "7D", "2C"], id="blocked-suited-KQ"),
    pytest.param(["QS", "JS"], ["9H", "7D", "2C"], id="blocked-suited-QJ"),
]


def make_seat(agent_id="hero", seat_number=3, cards=None, stack=1800):
    return {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": cards or ["AS", "KS"],
        "stackChips": stack,
        "currentBetChips": 0,
    }


def make_table(
    *,
    hero,
    board,
    actions,
    current_bet=80,
    call_amount=80,
    pot=240,
):
    seats = [
        {"agentId": "villain-1", "seatNumber": 1, "stackChips": 2200},
        {"agentId": "villain-2", "seatNumber": 2, "stackChips": 2200},
        hero,
        {"agentId": "villain-4", "seatNumber": 4, "stackChips": 2200},
        {"agentId": "villain-5", "seatNumber": 5, "stackChips": 2200},
        {"agentId": "villain-6", "seatNumber": 6, "stackChips": 2200},
    ]
    return {
        "street": "Flop",
        "boardCards": board,
        "potChips": pot,
        "currentBet": current_bet,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": actions,
            "callAmount": call_amount,
            "callChips": call_amount,
            "minBet": 50,
            "minRaiseTo": 220,
            "maxCommit": hero["stackChips"],
            "raiseRange": {"min": 220, "max": hero["stackChips"]},
            "betRange": {"min": 50, "max": hero["stackChips"]},
        },
        "seats": seats,
        "opponentProfiles": {},
    }


@pytest.mark.parametrize(("strategy_name", "choose_action"), STRATEGIES)
@pytest.mark.parametrize(("hole_cards", "board"), ROYAL_DRAW_HANDS)
def test_suited_broadway_royal_draws_do_not_fold_when_facing_bet(
    strategy_name,
    choose_action,
    hole_cards,
    board,
):
    hero = make_seat(cards=hole_cards)
    table = make_table(
        hero=hero,
        board=board,
        actions=["fold", "call", "raise"],
    )

    action, amount, message = choose_action(table, hero)

    assert royal_flush.royal_flush_possible(hole_cards, board)
    assert action in {"call", "raise"}, strategy_name
    assert action != "fold"
    if action == "raise":
        assert amount >= table["allowedActions"]["minRaiseTo"]
    assert "royal" in message.lower()


@pytest.mark.parametrize(("strategy_name", "choose_action"), STRATEGIES)
@pytest.mark.parametrize(("hole_cards", "board"), ROYAL_DRAW_HANDS)
def test_suited_broadway_royal_draws_check_instead_of_fold_when_free(
    strategy_name,
    choose_action,
    hole_cards,
    board,
):
    hero = make_seat(cards=hole_cards)
    table = make_table(
        hero=hero,
        board=board,
        actions=["fold", "check", "bet"],
        current_bet=0,
        call_amount=0,
    )

    action, amount, message = choose_action(table, hero)

    assert royal_flush.royal_flush_possible(hole_cards, board)
    assert action == "check", strategy_name
    assert amount is None
    assert "royal" in message.lower()


@pytest.mark.parametrize(("strategy_name", "choose_action"), STRATEGIES)
@pytest.mark.parametrize(("hole_cards", "board"), BLOCKED_ROYAL_HANDS)
def test_suited_broadway_hands_can_fold_after_flop_blocks_royal(
    monkeypatch,
    strategy_name,
    choose_action,
    hole_cards,
    board,
):
    monkeypatch.setattr(
        royal_flush.survival_lookahead,
        "choose_action",
        lambda _table, _seat: ("fold", None, "forced fold without royal draw"),
    )
    hero = make_seat(cards=hole_cards)
    table = make_table(
        hero=hero,
        board=board,
        actions=["fold", "call", "raise"],
        current_bet=600,
        call_amount=600,
        pot=240,
    )

    action, amount, message = choose_action(table, hero)

    assert not royal_flush.royal_flush_possible(hole_cards, board)
    assert action == "fold", strategy_name
    assert amount is None
    assert "royal flush possible" not in message.lower()
