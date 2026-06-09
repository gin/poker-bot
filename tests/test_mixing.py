from poker_bot.mixing import (
    choose_weighted,
    mix_percent,
    mix_value,
    resolve_distribution,
    should_take,
)


def make_table(board=None, table_id="hand-1"):
    return {
        "tableId": table_id,
        "street": "Flop",
        "boardCards": board or ["AS", "7D", "2C"],
        "potChips": 300,
        "currentBet": 0,
        "buttonSeatNumber": 1,
        "actingSeatNumber": 2,
        "allowedActions": {
            "availableActions": ["fold", "check", "bet"],
            "callAmount": 0,
            "minBet": 50,
        },
    }


def make_seat(cards=None):
    return {
        "agentId": "hero",
        "seatNumber": 2,
        "holeCards": cards or ["KS", "QD"],
    }


def test_mix_value_is_deterministic_for_same_state():
    table = make_table()
    seat = make_seat()

    first = mix_value("probe", table, seat, strategy="auto_research")
    second = mix_value("probe", table, seat, strategy="auto_research")

    assert first == second
    assert 0 <= first < 1
    assert mix_percent("probe", table, seat, strategy="auto_research") == int(
        first * 100
    )


def test_mix_value_changes_when_state_changes():
    seat = make_seat()

    first = mix_value("probe", make_table(board=["AS", "7D", "2C"]), seat)
    second = mix_value("probe", make_table(board=["AS", "7D", "3C"]), seat)

    assert first != second


def test_should_take_handles_extreme_probabilities():
    table = make_table()
    seat = make_seat()

    assert not should_take(0, "any", table, seat)
    assert should_take(1, "any", table, seat)


def test_choose_weighted_is_deterministic_and_ignores_zero_weights():
    table = make_table()
    seat = make_seat()
    options = [("fold", 0), ("call", 1), ("raise", 2)]

    first = choose_weighted(options, "mixed-action", table, seat)
    second = choose_weighted(options, "mixed-action", table, seat)

    assert first == second
    assert first in {"call", "raise"}


def test_resolve_distribution_exposes_probabilities_and_roll():
    table = make_table()
    seat = make_seat()

    decision = resolve_distribution(
        [("check", 1), ("bet", 3)],
        "mixed-action",
        table,
        seat,
    )

    assert decision.selected in {"check", "bet"}
    assert 0 <= decision.roll < 1
    assert decision.probability_for("check") == 0.25
    assert decision.probability_for("bet") == 0.75
    assert decision.summary() == "check:25%/bet:75%"
