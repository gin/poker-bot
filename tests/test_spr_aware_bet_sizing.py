from poker_bot.strategies import s2base as strategy
from poker_bot.strategies.s2base import (
    effective_spr,
    spr_band,
    spr_bet_fraction,
    spr_raise_pressure,
)


def test_effective_spr_uses_remaining_stack_after_call():
    table = {"potChips": 100, "currentBet": 0}
    my_seat = {"stackChips": 500}
    assert effective_spr(table, my_seat, call_amount=100) == 2

    table = {"potChips": 100, "currentBet": 0}
    my_seat = {"stackChips": 500}
    assert effective_spr(table, my_seat, call_amount=0) == 5


def test_spr_band_categorizes_low_medium_high():
    assert spr_band(1.5) == "low"
    assert spr_band(2.0) == "medium"
    assert spr_band(4.9) == "medium"
    assert spr_band(5.0) == "high"
    assert spr_band(None) == "unknown"


def test_spr_bet_fraction_adjusts_by_spr():
    assert spr_bet_fraction(1.5, strong=True) == 0.30
    assert spr_bet_fraction(1.5, strong=False) == 0.25
    assert spr_bet_fraction(3.0, strong=True) == 0.50
    assert spr_bet_fraction(3.0, strong=False) == 0.40
    assert spr_bet_fraction(8.0, strong=True) == 0.65
    assert spr_bet_fraction(8.0, strong=False) == 0.55


def test_spr_raise_pressure_adjusts_by_spr():
    assert spr_raise_pressure(1.5, strong=True) == 0.85
    assert spr_raise_pressure(1.5, strong=False) == 0.65
    assert spr_raise_pressure(3.0, strong=True) == 0.70
    assert spr_raise_pressure(3.0, strong=False) == 0.55
    assert spr_raise_pressure(8.0, strong=True) == 0.55
    assert spr_raise_pressure(8.0, strong=False) == 0.45


def test_high_spr_pot_control_checks_marginal_hand():
    table = {
        "street": "Flop",
        "boardCards": ["Ah", "Kd", "2c"],
        "potChips": 100,
        "seats": [
            {
                "agentId": "hero",
                "seatNumber": 1,
                "holeCards": ["3h", "4d"],
                "stackChips": 1000,
                "currentBetChips": 0,
                "folded": False,
                "hasFolded": False,
            },
            {
                "agentId": "villain",
                "seatNumber": 2,
                "holeCards": [],
                "stackChips": 1000,
                "currentBetChips": 20,
                "folded": False,
                "hasFolded": False,
            },
        ],
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": ["fold", "call", "check"],
            "callAmount": 20,
            "callChips": 20,
            "minBet": 10,
            "minRaiseTo": 40,
        },
    }
    hero = table["seats"][0]
    result = strategy.profiled_choose_action(table, hero)
    assert result[0] == "check"
