"""Regression coverage for paired-board two pair facing an effective all-in."""

import pytest

from poker_bot.strategies import multi_core_tight as multi_core
from poker_bot.strategies import s5base


def _table(
    hole_cards,
    board_cards,
    *,
    call,
    pot,
    stack,
    available_actions=None,
    dealt_in_players=2,
):
    additional_players = [
        {
            "seatNumber": seat_number,
            "agentId": f"opponent-{seat_number}",
            "holeCards": [],
            "stackChips": 5_000,
            "currentBetChips": 0,
            "folded": False,
            "hasFolded": False,
        }
        for seat_number in (1, 3, 4, 6)[: dealt_in_players - 2]
    ]
    hero = {
        "seatNumber": 5,
        "agentId": "hero",
        "holeCards": hole_cards,
        "stackChips": stack,
        "currentBetChips": 0,
        "folded": False,
        "hasFolded": False,
    }
    villain = {
        "seatNumber": 2,
        "agentId": "villain",
        "holeCards": [],
        "stackChips": 5_000,
        "currentBetChips": call,
        "folded": False,
        "hasFolded": False,
    }
    return {
        "street": {3: "Flop", 4: "Turn", 5: "River"}[len(board_cards)],
        "boardCards": board_cards,
        "potChips": pot,
        "currentBet": call,
        "buttonSeatNumber": 2,
        "actingSeatNumber": 5,
        "selfSeatNumber": 5,
        "actionHistory": [{"agentId": "villain", "action": "raise", "street": "Turn"}],
        "seats": [hero, villain, *additional_players],
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": available_actions or ["fold", "call", "all-in"],
            "callAmount": call,
            "callChips": call,
            "minBet": 20,
            "minRaiseTo": call * 2,
            "maxCommit": stack,
            "raiseRange": {"min": call * 2, "max": stack},
        },
    }


def test_ac_kc_paired_board_effective_all_in_folds_end_to_end():
    table = _table(
        ["Ac", "Kc"],
        ["4h", "9s", "Ah", "9c"],
        call=3_850,
        pot=3_879,
        stack=861,
        dealt_in_players=6,
    )
    hero = table["seats"][0]

    assert multi_core.count_dealt_in_players(table) == 6
    action, _amount, message = multi_core.choose_action(table, hero)


    assert "[full_ring]" in message
    assert action == "fold"
    assert "paired-board two pair effective all-in fold" in message


def test_ac_kc_top_pair_plus_board_pair_is_rank_two():
    hole_cards = ["Ac", "Kc"]
    board_cards = ["4h", "9s", "Ah", "9c"]

    assert s5base.made_hand_rank(hole_cards, board_cards) == 2
    assert s5base.has_top_pair_or_better(hole_cards, board_cards)


def test_paired_board_two_pair_below_stack_is_not_effective_all_in_fold():
    table = _table(
        ["Ac", "Kc"],
        ["4h", "9s", "Ah", "9c"],
        call=860,
        pot=3_879,
        stack=861,
    )

    assert s5base.effective_all_in_paired_board_fold(table, table["seats"][0]) is None


@pytest.mark.parametrize(
    ("hole_cards", "expected_rank"),
    [(["9h", "Kc"], 3), (["9h", "Ac"], 6)],
    ids=["trips", "full-house"],
)
def test_paired_board_trips_and_full_house_are_not_rank_two_folds(
    hole_cards, expected_rank
):
    table = _table(
        hole_cards,
        ["4h", "9s", "Ah", "9c"],
        call=3_850,
        pot=3_879,
        stack=861,
    )
    hero = table["seats"][0]

    assert s5base.made_hand_rank(hole_cards, table["boardCards"]) == expected_rank
    assert s5base.effective_all_in_paired_board_fold(table, hero) is None


def test_unpaired_board_two_pair_is_not_effective_all_in_fold():
    table = _table(
        ["Ac", "Kc"],
        ["4h", "Ks", "Ah", "7c"],
        call=3_850,
        pot=3_879,
        stack=861,
    )
    hero = table["seats"][0]

    assert s5base.made_hand_rank(hero["holeCards"], table["boardCards"]) == 2
    assert not s5base.board_has_pair(table["boardCards"])
    assert s5base.effective_all_in_paired_board_fold(table, hero) is None
