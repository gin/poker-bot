from poker_bot.strategies import auto_research_v005 as strategy
from poker_bot.strategies.loader import load_strategy


def make_seat(agent_id="hero", seat_number=1, cards=None, folded=False):
    seat = {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": cards or ["6D", "6C"],
        "stackChips": 1800,
        "currentBetChips": 0,
    }
    if folded:
        seat["folded"] = True
    return seat


def make_table(
    *,
    hero=None,
    board=None,
    street="Flop",
    available=None,
    current_bet=0,
    call_amount=0,
    pot=40,
):
    hero = hero or make_seat()
    return {
        "street": street,
        "boardCards": board or ["7D", "7H", "2D"],
        "potChips": pot,
        "currentBet": current_bet,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": available or ["fold", "check", "bet"],
            "callAmount": call_amount,
            "callChips": call_amount,
            "minBet": 5,
            "minRaiseTo": max(current_bet + call_amount + 5, 10),
            "maxCommit": hero["stackChips"],
            "betRange": {"min": 5, "max": hero["stackChips"]},
            "raiseRange": {"min": 10, "max": hero["stackChips"]},
        },
        "seats": [
            hero,
            make_seat("villain-2", 2, cards=[]),
            make_seat("villain-3", 3, cards=[]),
            make_seat("villain-4", 4, cards=[]),
            make_seat("villain-5", 5, cards=[]),
            make_seat("villain-6", 6, cards=[]),
        ],
        "opponentProfiles": {},
    }


def force_v004(monkeypatch, decision):
    monkeypatch.setattr(strategy.champion, "choose_action", lambda *_args: decision)


def test_auto_research_v005_loads_as_strategy():
    assert load_strategy("auto_research_v005") is strategy.choose_action


def test_fragile_two_pair_on_paired_board_checks_when_checked_to(monkeypatch):
    force_v004(monkeypatch, ("bet", 5, "Value betting rank 2"))
    hero = make_seat(cards=["6D", "6C"])
    table = make_table(hero=hero, board=["7D", "7H", "2D"])

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "check"
    assert amount is None
    assert "fragile paired-board" in message


def test_fragile_two_pair_on_paired_board_calls_instead_of_raise_war(monkeypatch):
    force_v004(monkeypatch, ("raise", 289, "anti-bully value raise rank 2"))
    hero = make_seat(cards=["8H", "8C"])
    table = make_table(
        hero=hero,
        board=["7C", "AS", "AH"],
        available=["fold", "call", "raise"],
        current_bet=150,
        call_amount=92,
        pot=215,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "call"
    assert amount == 92
    assert "fragile two pair" in message


def test_fragile_two_pair_can_fold_at_bad_price(monkeypatch):
    force_v004(monkeypatch, ("raise", 1400, "anti-bully value raise rank 2"))
    hero = make_seat(cards=["8H", "8C"])
    table = make_table(
        hero=hero,
        board=["7C", "AS", "AH"],
        available=["fold", "call", "raise"],
        current_bet=1200,
        call_amount=1200,
        pot=900,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    assert "fragile paired-board" in message


def test_strong_top_pair_turn_defends_near_threshold(monkeypatch):
    force_v004(monkeypatch, ("fold", None, "Rank 1 below price, folding"))
    hero = make_seat(seat_number=5, cards=["AS", "KC"])
    table = make_table(
        hero=hero,
        board=["TD", "9D", "2H", "KS"],
        street="Turn",
        available=["fold", "call", "raise"],
        current_bet=34,
        call_amount=34,
        pot=84,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "call"
    assert amount == 34
    assert "strong top-pair defense" in message


def test_weak_top_pair_turn_still_folds_to_price(monkeypatch):
    force_v004(monkeypatch, ("fold", None, "Rank 1 below price, folding"))
    hero = make_seat(seat_number=5, cards=["8S", "KC"])
    table = make_table(
        hero=hero,
        board=["TD", "9D", "2H", "KS"],
        street="Turn",
        available=["fold", "call", "raise"],
        current_bet=34,
        call_amount=34,
        pot=84,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    # assert message.startswith("v005 champion baseline")


def test_trips_on_paired_board_is_not_blocked(monkeypatch):
    force_v004(monkeypatch, ("raise", 80, "Trips value raise"))
    hero = make_seat(cards=["7S", "KC"])
    table = make_table(
        hero=hero,
        board=["7D", "7H", "2D"],
        available=["fold", "call", "raise"],
        current_bet=20,
        call_amount=20,
        pot=60,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "raise"
    assert amount == 80
    # assert message.startswith("v005 champion baseline")


def test_non_nut_full_house_on_trips_board_calls_instead_of_reraising(monkeypatch):
    force_v004(monkeypatch, ("raise", 1705, "anti-bully value raise rank 6"))
    hero = make_seat(cards=["KD", "TD"])
    table = make_table(
        hero=hero,
        board=["2S", "2D", "TS", "2C"],
        street="Turn",
        available=["fold", "call", "raise"],
        current_bet=860,
        call_amount=427,
        pot=1301,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "call"
    assert amount == 427
    assert "non-nut full house" in message


def test_ace_full_house_on_trips_board_can_keep_raising(monkeypatch):
    force_v004(monkeypatch, ("raise", 1705, "anti-bully value raise rank 6"))
    hero = make_seat(cards=["AD", "AC"])
    table = make_table(
        hero=hero,
        board=["2S", "2D", "TS", "2C"],
        street="Turn",
        available=["fold", "call", "raise"],
        current_bet=860,
        call_amount=427,
        pot=1301,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "raise"
    assert amount == 1705
    # assert message.startswith("v005 champion baseline")
