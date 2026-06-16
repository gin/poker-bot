from poker_bot.strategies import survival_balanced_pp_pd_pr_patch1 as strategy
from poker_bot.strategies.loader import load_strategy


def make_seat(agent_id="hero", seat_number=4, cards=None, stack=1800):
    return {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": cards or ["AS", "KS"],
        "stackChips": stack,
        "currentBetChips": 0,
    }


def make_table(
    street="Flop",
    board=None,
    actions=None,
    hero=None,
    pot=200,
    current_bet=40,
    call_amount=40,
    min_raise_to=120,
):
    hero = hero or make_seat()
    seats = [
        make_seat(f"villain-{index + 1}", index + 1, [], stack=2200)
        for index in range(6)
    ]
    seats[hero["seatNumber"] - 1] = hero
    return {
        "street": street,
        "boardCards": board or ["KH", "7D", "2C"],
        "potChips": pot,
        "currentBet": current_bet,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": actions or ["fold", "call", "raise"],
            "callAmount": call_amount,
            "callChips": call_amount,
            "minRaiseTo": min_raise_to,
            "minBet": 50,
            "maxCommit": hero["stackChips"],
            "raiseRange": {"min": min_raise_to, "max": hero["stackChips"]},
            "betRange": {"min": 50, "max": hero["stackChips"]},
        },
        "seats": seats,
        "opponentProfiles": {},
    }


def fold_to_heads_up(table, hero):
    for seat in table["seats"]:
        if seat["agentId"] != hero["agentId"] and seat["seatNumber"] != 1:
            seat["folded"] = True


def force_blueprint(monkeypatch, action):
    monkeypatch.setattr(
        strategy.survival_lookahead,
        "choose_action",
        lambda _table, _seat: (action, None, f"forced fallback {action}"),
    )


def test_strategy_loads():
    assert load_strategy("survival_balanced_pp_pd_pr_patch1") is strategy.choose_action


def test_fold_board_dominated_two_pair_on_double_paired_ace_board(monkeypatch):
    force_blueprint(monkeypatch, "call")
    hero = make_seat(cards=["JC", "8D"], stack=620)
    table = make_table(
        street="Turn",
        board=["9S", "AH", "9H", "AC"],
        actions=["fold", "call", "all-in"],
        hero=hero,
        pot=1737,
        current_bet=997,
        call_amount=641,
        min_raise_to=0,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    assert "board-dominated" in message


def test_check_board_dominated_two_pair_when_checked_to(monkeypatch):
    force_blueprint(monkeypatch, "check")
    hero = make_seat(cards=["JC", "8D"])
    table = make_table(
        street="Turn",
        board=["9S", "AH", "9H", "AC"],
        actions=["fold", "check", "bet"],
        hero=hero,
        pot=200,
        current_bet=0,
        call_amount=0,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "check"
    assert amount is None
    assert "forced fallback check" in message


def test_raise_clean_unpaired_two_pair_for_value(monkeypatch):
    force_blueprint(monkeypatch, "fold")
    hero = make_seat(cards=["AS", "KS"], stack=1800)
    table = make_table(
        street="Flop",
        board=["AH", "KD", "2C"],
        hero=hero,
        pot=300,
        current_bet=80,
        call_amount=80,
        min_raise_to=220,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "raise"
    assert 220 <= amount <= hero["stackChips"]
    assert "protection raise rank 2" in message


def test_raise_trips_for_value_even_on_paired_board(monkeypatch):
    force_blueprint(monkeypatch, "fold")
    hero = make_seat(cards=["7C", "7H"], stack=1800)
    table = make_table(
        street="Flop",
        board=["7S", "2D", "9C"],
        hero=hero,
        pot=240,
        current_bet=70,
        call_amount=70,
        min_raise_to=180,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "raise"
    assert 180 <= amount <= hero["stackChips"]
    assert "value raise rank 3" in message


def test_fold_overpriced_river_trips_on_tripled_board(monkeypatch):
    force_blueprint(monkeypatch, "call")
    hero = make_seat(cards=["QD", "KH"], stack=33150)
    table = make_table(
        street="River",
        board=["3D", "8H", "3H", "3S", "5H"],
        actions=["fold", "call", "all-in"],
        hero=hero,
        pot=38013,
        current_bet=37730,
        call_amount=37636,
        min_raise_to=0,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    assert "overpriced river trips" in message


def test_call_cheap_river_trips_on_tripled_board(monkeypatch):
    force_blueprint(monkeypatch, "call")
    hero = make_seat(cards=["QD", "KH"], stack=1800)
    table = make_table(
        street="River",
        board=["3D", "8H", "3H", "3S", "5H"],
        actions=["fold", "call", "all-in"],
        hero=hero,
        pot=1000,
        current_bet=100,
        call_amount=100,
        min_raise_to=0,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "call"
    assert amount == 100
    assert "overpriced river trips" not in message


def test_call_but_do_not_raise_overpair_on_paired_board(monkeypatch):
    force_blueprint(monkeypatch, "fold")
    hero = make_seat(cards=["JC", "JH"], stack=1800)
    table = make_table(
        street="Flop",
        board=["9S", "5H", "9D"],
        hero=hero,
        pot=240,
        current_bet=40,
        call_amount=40,
        min_raise_to=120,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "call"
    assert amount == 40
    assert "paired-board rank-2 pot control" in message


def test_fold_fragile_rank_two_on_paired_board_at_bad_price(monkeypatch):
    force_blueprint(monkeypatch, "call")
    hero = make_seat(cards=["QC", "JD"], stack=1800)
    table = make_table(
        street="Flop",
        board=["AH", "AD", "QS"],
        hero=hero,
        pot=200,
        current_bet=120,
        call_amount=120,
        min_raise_to=300,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "fold"
    assert amount is None
    assert "fragile paired-board" in message


def test_call_top_pair_at_clear_price(monkeypatch):
    force_blueprint(monkeypatch, "fold")
    hero = make_seat(cards=["KS", "QD"], stack=1800)
    table = make_table(
        street="Flop",
        board=["KH", "7D", "2C"],
        hero=hero,
        pot=300,
        current_bet=75,
        call_amount=75,
        min_raise_to=200,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "call"
    assert amount == 75
    assert "medium defense" in message


def test_semi_bluff_raise_open_ended_draw_heads_up_when_mixed_in(monkeypatch):
    force_blueprint(monkeypatch, "fold")
    monkeypatch.setattr(strategy, "stable_mix_percent", lambda *_args: 0)
    hero = make_seat(cards=["9S", "8S"], seat_number=6, stack=1800)
    table = make_table(
        street="Flop",
        board=["7H", "6D", "2C"],
        hero=hero,
        pot=500,
        current_bet=80,
        call_amount=80,
        min_raise_to=200,
    )
    fold_to_heads_up(table, hero)

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "raise"
    assert 200 <= amount <= hero["stackChips"]
    assert "semi-bluff raise" in message


def test_check_clear_air_when_checked_to(monkeypatch):
    force_blueprint(monkeypatch, "check")
    hero = make_seat(cards=["3S", "8D"], stack=1800)
    table = make_table(
        street="Flop",
        board=["AH", "KD", "7C"],
        actions=["fold", "check", "bet"],
        hero=hero,
        pot=200,
        current_bet=0,
        call_amount=0,
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "check"
    assert amount is None
    assert "forced fallback check" in message
