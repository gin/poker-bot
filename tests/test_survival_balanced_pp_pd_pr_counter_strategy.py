from poker_bot.strategies import survival_balanced_pp_pd_pr_counter as strategy
from poker_bot.strategies.loader import load_strategy


def make_seat(agent_id="hero", seat_number=1, cards=None, stack=1800, folded=False):
    seat = {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": cards or ["AD", "5D"],
        "stackChips": stack,
        "currentBetChips": 0,
    }
    if folded:
        seat["folded"] = True
    return seat


def make_table(
    *,
    street="Preflop",
    board=None,
    actions=None,
    hero=None,
    seats=None,
    pot=150,
    current_bet=50,
    call_amount=50,
    min_raise_to=150,
):
    hero = hero or make_seat()
    if seats is None:
        seats = [hero, make_seat("villain", 2, cards=[])]
    return {
        "street": street,
        "boardCards": board or [],
        "potChips": pot,
        "currentBet": current_bet,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": actions or ["fold", "call", "raise"],
            "callAmount": call_amount,
            "callChips": call_amount,
            "minBet": 50,
            "minRaiseTo": min_raise_to,
            "maxCommit": hero["stackChips"],
            "raiseRange": {"min": min_raise_to, "max": hero["stackChips"]},
            "betRange": {"min": 50, "max": hero["stackChips"]},
        },
        "seats": seats,
        "opponentProfiles": {},
    }


def force_patch1_blueprint(monkeypatch, action, amount=None):
    monkeypatch.setattr(
        strategy.patch1.survival_lookahead,
        "choose_action",
        lambda _table, _seat: (action, amount, f"forced {action}"),
    )


def test_counter_strategy_loads():
    assert load_strategy("survival_balanced_pp_pd_pr_counter") is strategy.choose_action


def test_short_handed_preflop_pressure_raises_borderline_suited_ace():
    hero = make_seat(cards=["AD", "5D"])
    table = make_table(hero=hero)

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "raise"
    assert amount >= table["allowedActions"]["minRaiseTo"]
    assert "short-handed open pressure" in message


def test_dry_board_pressure_bets_when_patch1_would_check(monkeypatch):
    force_patch1_blueprint(monkeypatch, "check")
    hero = make_seat(cards=["AD", "5D"])
    table = make_table(
        street="Flop",
        board=["KH", "7D", "2C"],
        actions=["fold", "check", "bet"],
        hero=hero,
        pot=300,
        current_bet=0,
        call_amount=0,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "bet"
    assert amount >= table["allowedActions"]["minBet"]
    assert "dry-board pressure" in message


def test_fragile_rank_two_fold_protection_is_preserved(monkeypatch):
    force_patch1_blueprint(monkeypatch, "call")
    hero = make_seat(cards=["JC", "8D"], stack=700)
    table = make_table(
        street="Turn",
        board=["9S", "AH", "9H", "AC"],
        actions=["fold", "call", "raise"],
        hero=hero,
        pot=1737,
        current_bet=997,
        call_amount=641,
        min_raise_to=1500,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    assert "board-dominated" in message


def test_six_player_hand_stays_on_sixmax_branch_after_folds(monkeypatch):
    hero = make_seat(cards=["2C", "7D"], seat_number=1)
    seats = [
        hero,
        make_seat("villain-2", 2, cards=[]),
        make_seat("villain-3", 3, cards=[], folded=True),
        make_seat("villain-4", 4, cards=[], folded=True),
        make_seat("villain-5", 5, cards=[], folded=True),
        make_seat("villain-6", 6, cards=[], folded=True),
    ]
    table = make_table(
        street="Flop",
        board=["KS", "8H", "3D"],
        actions=["fold", "check", "bet"],
        hero=hero,
        seats=seats,
        current_bet=0,
        call_amount=0,
    )
    monkeypatch.setattr(
        strategy.survival_sixmax,
        "choose_action",
        lambda _table, _seat: ("check", None, "sixmax branch"),
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "check"
    assert amount is None
    assert "counter six-max" in message
