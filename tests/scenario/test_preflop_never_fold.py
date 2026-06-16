"""Preflop never-fold scenarios for the s2v001 base strategy.

These tests confirm that, on an unopened preflop, the s2v001 base strategy
never chooses ``fold`` for the following premium/broadway hands across all
six positions (UTG, HJ, CO, BTN, SB, BB):

    AA, KK, QQ, JJ, AKs, AQs, AJs, KQs, KJs, QJs
"""

from __future__ import annotations

import pytest

from poker_bot.strategies.s2base import choose_action

HERO = "hero-agent"
STARTING_STACK = 1000
BIG_BLIND = 2


PREMIUM_HANDS = [
    (["As", "Ad"], "AA"),
    (["Kc", "Kh"], "KK"),
    (["Qc", "Qd"], "QQ"),
    (["Jc", "Jh"], "JJ"),
    (["As", "Ks"], "AKs"),
    (["Ah", "Qh"], "AQs"),
    (["Ac", "Jc"], "AJs"),
    (["Kd", "Qd"], "KQs"),
    (["Kh", "Jh"], "KJs"),
    (["Qs", "Js"], "QJs"),
]


# (seat_number, position_label). Seats are relative to the button (seat 1).
POSITIONS = [
    (4, "UTG"),
    (5, "HJ"),
    (6, "CO"),
    (1, "BTN"),
    (2, "SB"),
    (3, "BB"),
]


def _opponent_seat(seat_number: int) -> dict:
    if seat_number == 2:
        return {
            "seatNumber": seat_number,
            "agentId": f"opponent-{seat_number}",
            "holeCards": [],
            "stackChips": STARTING_STACK - 1,
            "currentBetChips": 1,
            "folded": False,
            "hasFolded": False,
        }
    if seat_number == 3:
        return {
            "seatNumber": seat_number,
            "agentId": f"opponent-{seat_number}",
            "holeCards": [],
            "stackChips": STARTING_STACK - 2,
            "currentBetChips": 2,
            "folded": False,
            "hasFolded": False,
        }
    return {
        "seatNumber": seat_number,
        "agentId": f"opponent-{seat_number}",
        "holeCards": [],
        "stackChips": STARTING_STACK,
        "currentBetChips": 0,
        "folded": False,
        "hasFolded": False,
    }


def _hero_seat(seat_number: int, hole_cards: list[str]) -> dict:
    if seat_number == 2:
        return {
            "seatNumber": seat_number,
            "agentId": HERO,
            "holeCards": hole_cards,
            "stackChips": STARTING_STACK - 1,
            "currentBetChips": 1,
            "folded": False,
            "hasFolded": False,
        }
    if seat_number == 3:
        return {
            "seatNumber": seat_number,
            "agentId": HERO,
            "holeCards": hole_cards,
            "stackChips": STARTING_STACK - 2,
            "currentBetChips": 2,
            "folded": False,
            "hasFolded": False,
        }
    return {
        "seatNumber": seat_number,
        "agentId": HERO,
        "holeCards": hole_cards,
        "stackChips": STARTING_STACK,
        "currentBetChips": 0,
        "folded": False,
        "hasFolded": False,
    }


def make_preflop_unopened_table(hero_seat: int, hero_hole: list[str]):
    seat_dict = {seat: _opponent_seat(seat) for seat, _label in POSITIONS}
    seat_dict[hero_seat] = _hero_seat(hero_seat, hero_hole)
    seats = [seat_dict[seat] for seat, _label in POSITIONS]

    if hero_seat == 3:
        call_amount = 0
        available = ["fold", "check", "raise"]
    else:
        call_amount = BIG_BLIND
        available = ["fold", "call", "raise"]

    hero_entry = seat_dict[hero_seat]
    max_commit = hero_entry["stackChips"] + hero_entry["currentBetChips"]

    allowed = {
        "availableActions": available,
        "callAmount": call_amount,
        "callChips": call_amount,
        "minBet": BIG_BLIND,
        "maxCommit": max_commit,
        "raiseRange": {"min": BIG_BLIND * 2, "max": max_commit},
        "betRange": {"min": BIG_BLIND, "max": max_commit},
    }

    table = {
        "street": "Preflop",
        "boardCards": [],
        "potChips": BIG_BLIND + 1,
        "currentBet": BIG_BLIND,
        "bigBlindChips": BIG_BLIND,
        "buttonSeatNumber": 1,
        "seats": seats,
        "opponentProfiles": {},
        "allowedActions": allowed,
    }
    return table, hero_entry


@pytest.mark.parametrize("hero_hole,hand_name", PREMIUM_HANDS)
@pytest.mark.parametrize("seat_number,position_name", POSITIONS)
def test_preflop_unopened_never_fold(hero_hole, hand_name, seat_number, position_name):
    table, hero = make_preflop_unopened_table(seat_number, hero_hole)

    action, _amount, _message = choose_action(table, hero)

    assert action != "fold", (
        f"{hand_name} from {position_name} should not fold on an unopened preflop"
    )
