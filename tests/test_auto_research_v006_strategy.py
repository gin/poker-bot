from poker_bot.strategies import auto_research_v006 as strategy
from poker_bot.strategies.loader import load_strategy


def make_seat(agent_id="hero", seat_number=1, cards=None, folded=False):
    seat = {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": cards or ["AH", "KH"],
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
    pot=120,
):
    hero = hero or make_seat()
    return {
        "street": street,
        "boardCards": board or ["QH", "JH", "2C"],
        "potChips": pot,
        "currentBet": current_bet,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": available or ["check", "bet"],
            "callAmount": call_amount,
            "callChips": call_amount,
            "minBet": 10,
            "minRaiseTo": max(current_bet + call_amount + 10, 20),
            "maxCommit": hero["stackChips"],
            "betRange": {"min": 10, "max": hero["stackChips"]},
            "raiseRange": {"min": 20, "max": hero["stackChips"]},
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


def force_v005(monkeypatch, decision):
    monkeypatch.setattr(strategy.champion, "choose_action", lambda *_args: decision)


def test_auto_research_v006_loads_as_strategy():
    assert load_strategy("auto_research_v006") is strategy.choose_action


def test_local_search_skips_preflop(monkeypatch):
    force_v005(monkeypatch, ("raise", 80, "premium preflop"))
    hero = make_seat(cards=["AS", "AC"])
    table = make_table(
        hero=hero,
        board=[],
        street="Preflop",
        available=["fold", "call", "raise"],
        current_bet=20,
        call_amount=20,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "raise"
    assert amount == 80
    # assert message.startswith("v006 champion baseline")


def test_local_search_can_override_close_check_to_bet(monkeypatch):
    force_v005(monkeypatch, ("check", None, "pot control"))
    hero = make_seat(cards=["AH", "KH"])
    table = make_table(hero=hero, board=["QH", "JH", "2C"], pot=600)
    table["opponentProfiles"] = {"villain-2": {"fold_to_bet_frequency": 0.85}}

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "bet"
    assert amount is not None
    assert "local search override" in message


def test_local_search_respects_ev_edge_threshold(monkeypatch):
    force_v005(monkeypatch, ("check", None, "thin spot"))
    monkeypatch.setattr(strategy, "rollout_equity", lambda *_args, **_kwargs: 0.50)
    hero = make_seat(cards=["9H", "8H"])
    table = make_table(hero=hero, board=["QH", "7C", "2D"], pot=40)

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "check"
    assert amount is None
    # assert message.startswith("v006 champion baseline")


def test_local_search_can_override_bad_price_call_to_fold(monkeypatch):
    force_v005(monkeypatch, ("call", 260, "defend draw"))
    monkeypatch.setattr(strategy, "rollout_equity", lambda *_args, **_kwargs: 0.18)
    hero = make_seat(cards=["9H", "8H"])
    table = make_table(
        hero=hero,
        board=["QH", "7C", "2D"],
        available=["fold", "call", "raise"],
        current_bet=260,
        call_amount=260,
        pot=240,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    assert "local search override" in message
