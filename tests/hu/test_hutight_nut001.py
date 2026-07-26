"""Behavioural coverage for the sole-nuts all-in ``hutight001`` variant."""

from __future__ import annotations

import pytest

from poker_bot.guards.nut_all_in import (
    commits_all_chips,
    has_all_in_permission,
    has_current_sole_nuts,
    veto_non_nut_all_in,
)
from poker_bot.strategies.hutight_nut001 import choose_action

HERO = "hero"


def _seat(hole_cards, *, stack=1_000, current_bet=0):
    return {
        "seatNumber": 1,
        "agentId": HERO,
        "holeCards": hole_cards,
        "stackChips": stack,
        "currentBetChips": current_bet,
        "folded": False,
        "hasFolded": False,
    }


def _table(
    street,
    board_cards,
    *,
    available,
    call=0,
    pot=200,
    raise_max=1_000,
):
    return {
        "street": street,
        "boardCards": board_cards,
        "potChips": pot,
        "seats": [],
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": available,
            "callAmount": call,
            "callChips": call,
            "minBet": 20,
            "minRaiseTo": max(40, call * 2),
            "raiseRange": {"min": max(40, call * 2), "max": raise_max},
            "betRange": {"min": 20, "max": raise_max},
        },
    }


@pytest.mark.parametrize(
    ("hole_cards", "board_cards", "expected"),
    [
        (["AD", "2D"], ["4D", "8D", "QD"], True),
        (["KD", "JD"], ["4D", "8D", "QD"], False),
        (["QS", "QC"], ["QD", "QH", "8D"], True),
        (["2D", "3C"], ["AH", "KD", "QC", "JS", "TC"], False),
    ],
)
def test_current_sole_nuts_uses_all_legal_opponent_hands(
    hole_cards, board_cards, expected
):
    assert has_current_sole_nuts(hole_cards, board_cards) is expected


def test_preflop_permission_is_aces_only():
    assert has_all_in_permission(["AS", "AH"], []) is True
    assert has_all_in_permission(["KS", "KH"], []) is False


def test_all_in_call_without_permission_folds():
    seat = _seat(["KS", "KH"])
    table = _table(
        "Preflop", [], available=["fold", "call", "raise"], call=1_000
    )

    result = veto_non_nut_all_in(table, seat, ("call", 1_000, "base call"))

    assert result == (
        "fold",
        None,
        "base call [nut all-in veto]: folding non-nut stack-off",
    )


def test_all_in_raise_without_permission_calls_when_safe():
    seat = _seat(["KS", "KH"])
    table = _table(
        "Preflop", [], available=["fold", "call", "raise"], call=200
    )

    result = veto_non_nut_all_in(table, seat, ("raise", 1_000, "base raise"))

    assert result == (
        "call",
        200,
        "base raise [nut all-in veto]: calling without committing stack",
    )


def test_open_all_in_without_permission_checks():
    seat = _seat(["KS", "KH"])
    table = _table(
        "Preflop", [], available=["fold", "check", "bet", "raise"], call=0
    )

    result = veto_non_nut_all_in(table, seat, ("bet", 1_000, "base bet"))

    assert result == (
        "check",
        None,
        "base bet [nut all-in veto]: checking without sole nuts",
    )


def test_full_stack_raise_accounts_for_already_committed_chips():
    seat = _seat(["KS", "KH"], stack=900, current_bet=100)
    table = _table("Preflop", [], available=["fold", "call", "raise"], call=200)

    assert commits_all_chips(table, seat, ("raise", 1_000, "base raise")) is True
    assert commits_all_chips(table, seat, ("raise", 999, "base raise")) is False


def test_nut_all_in_and_ordinary_action_are_preserved():
    seat = _seat(["AD", "2D"])
    table = _table(
        "Flop", ["4D", "8D", "QD"], available=["fold", "call", "raise"], call=200
    )
    all_in = ("raise", 1_000, "base raise")
    ordinary = ("call", 200, "base call")

    assert veto_non_nut_all_in(table, seat, all_in) == all_in
    assert veto_non_nut_all_in(table, seat, ordinary) == ordinary


def _war_table(hole_cards):
    hero = _seat(hole_cards, stack=1_800)
    villain = {
        "seatNumber": 2,
        "agentId": "villain",
        "holeCards": [],
        "stackChips": 1_800,
        "currentBetChips": 0,
        "folded": False,
        "hasFolded": False,
    }
    table = _table(
        "Preflop",
        [],
        available=["fold", "call", "raise", "all-in"],
        call=200,
        pot=300,
        raise_max=1_800,
    )
    table.update(
        {
            "seats": [hero, villain],
            "buttonSeatNumber": 1,
            "actionHistory": [
                event
                for _ in range(3)
                for event in (
                    {"agentId": "villain", "action": "raise", "street": "Preflop"},
                    {"agentId": HERO, "action": "raise", "street": "Preflop"},
                )
            ],
        }
    )
    return table, hero


def test_wrapper_preserves_aces_but_vetoes_legacy_king_shove():
    aa_table, aa_hero = _war_table(["AS", "AH"])
    kk_table, kk_hero = _war_table(["KS", "KH"])

    aa_action, aa_amount, _aa_message = choose_action(aa_table, aa_hero)
    kk_action, kk_amount, kk_message = choose_action(kk_table, kk_hero)

    assert (aa_action, aa_amount) == ("all-in", 1_800)
    assert (kk_action, kk_amount) == ("call", 200)
    assert "nut all-in veto" in kk_message
