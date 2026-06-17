"""Scenario tests for the board-illusion river exploit."""
## NOTE: check to see if newer s2 strat passes

from __future__ import annotations

import pytest

from poker_bot.strategies import s2v001_board_illusion as strategy_variant

# from poker_bot.strategies import s2baseog as strategy_variant
# from poker_bot.strategies import s2v004 as strategy_variant
from poker_bot.strategies.exploits import board_illusion
from poker_bot.strategies.exploits.board_illusion import apply_board_illusion

HERO = "hero-agent"


@pytest.fixture(autouse=True)
def clear_board_illusion_state():
    board_illusion.clear_state()
    yield
    board_illusion.clear_state()


def make_seat(
    seat_number: int,
    agent_id: str,
    *,
    hole_cards: list[str] | None = None,
    stack: int = 1000,
    current_bet: int = 0,
    folded: bool = False,
):
    return {
        "seatNumber": seat_number,
        "agentId": agent_id,
        "holeCards": hole_cards or [],
        "stackChips": stack,
        "currentBetChips": current_bet,
        "folded": folded,
        "hasFolded": False,
    }


def make_table(
    *,
    street: str,
    board: list[str],
    hero_hole: list[str],
    hand_id: str = "hand-1",
    hero_agent: str = HERO,
    villains: int = 1,
    call: int = 0,
    available: list[str] | None = None,
    max_commit: int = 1000,
    min_bet: int = 2,
):
    seats = [make_seat(1, hero_agent, hole_cards=hero_hole, current_bet=0)]
    for index in range(villains):
        seats.append(make_seat(2 + index, f"villain-{index}"))

    allowed = {
        "availableActions": list(available or []),
        "callAmount": call,
        "callChips": call,
        "minBet": min_bet,
        "maxCommit": max_commit,
        "raiseRange": {"min": max(call * 2, min_bet), "max": max_commit},
        "betRange": {"min": min_bet, "max": max_commit},
    }

    return {
        "handId": hand_id,
        "tableId": hand_id,
        "street": street,
        "boardCards": board,
        "potChips": 100,
        "buttonSeatNumber": 1,
        "seats": seats,
        "allowedActions": allowed,
    }, seats[0]


def assert_no_override(decision) -> None:
    assert decision is None


def test_shoves_top_range_when_facing_river_bet():
    board = ["5d", "6h", "7s", "8c", "9d"]
    table, hero = make_table(
        street="River",
        board=board,
        hero_hole=["Th", "Kd"],
        call=100,
        available=["fold", "call", "raise"],
        max_commit=1000,
    )

    action, amount, message = apply_board_illusion(table, hero, ("call", 100, "base"))

    assert action == "raise"
    assert amount == 1000
    assert message == "board-illusion: top-range river shove"


def test_does_not_shove_top_range_multiway_with_three_opponents():
    board = ["5d", "6h", "7s", "8c", "9d"]
    table, hero = make_table(
        street="River",
        board=board,
        hero_hole=["Th", "Kd"],
        villains=3,
        call=100,
        available=["fold", "call", "raise"],
        max_commit=1000,
    )

    assert_no_override(apply_board_illusion(table, hero, ("call", 100, "base")))


def test_does_not_shove_when_not_facing_bet():
    board = ["5d", "6h", "7s", "8c", "9d"]
    table, hero = make_table(
        street="River",
        board=board,
        hero_hole=["Th", "Kd"],
        call=0,
        available=["fold", "check", "bet"],
        max_commit=1000,
    )

    assert_no_override(apply_board_illusion(table, hero, ("check", None, "base")))


def test_bluffs_river_only_after_flop_and_turn_barrels():
    river_board = ["2d", "3h", "4s", "5c", "7d"]
    flop_table, hero = make_table(
        street="Flop",
        board=["2d", "3h", "4s"],
        hero_hole=["9h", "Kc"],
        available=["fold", "check", "bet"],
    )
    turn_table, _hero = make_table(
        street="Turn",
        board=["2d", "3h", "4s", "5c"],
        hero_hole=["9h", "Kc"],
        available=["fold", "check", "bet"],
    )
    river_table, _hero = make_table(
        street="River",
        board=river_board,
        hero_hole=["9h", "Kc"],
        call=0,
        available=["fold", "check", "bet"],
    )

    assert apply_board_illusion(flop_table, hero, ("bet", 2, "base")) is None
    assert apply_board_illusion(turn_table, hero, ("bet", 2, "base")) is None

    action, amount, message = apply_board_illusion(
        river_table,
        hero,
        ("check", None, "base"),
    )

    assert action == "bet"
    assert amount == 2
    assert message == "board-illusion: third barrel bottom range"


def test_does_not_bluff_river_without_flop_barrel():
    turn_table, hero = make_table(
        street="Turn",
        board=["2d", "3h", "4s", "5c"],
        hero_hole=["9h", "Kc"],
        available=["fold", "check", "bet"],
    )
    river_table, _hero = make_table(
        street="River",
        board=["2d", "3h", "4s", "5c", "7d"],
        hero_hole=["9h", "Kc"],
        call=0,
        available=["fold", "check", "bet"],
    )

    assert apply_board_illusion(turn_table, hero, ("bet", 2, "base")) is None

    assert_no_override(
        apply_board_illusion(
            river_table,
            hero,
            ("check", None, "base"),
        ),
    )


def test_does_not_bluff_river_without_turn_barrel():
    flop_table, hero = make_table(
        street="Flop",
        board=["2d", "3h", "4s"],
        hero_hole=["9h", "Kc"],
        available=["fold", "check", "bet"],
    )
    turn_table, _hero = make_table(
        street="Turn",
        board=["2d", "3h", "4s", "5c"],
        hero_hole=["9h", "Kc"],
        available=["fold", "check", "bet"],
    )
    river_table, _hero = make_table(
        street="River",
        board=["2d", "3h", "4s", "5c", "7d"],
        hero_hole=["9h", "Kc"],
        call=0,
        available=["fold", "check", "bet"],
    )

    assert apply_board_illusion(flop_table, hero, ("bet", 2, "base")) is None
    assert apply_board_illusion(turn_table, hero, ("check", None, "base")) is None

    assert_no_override(
        apply_board_illusion(
            river_table,
            hero,
            ("check", None, "base"),
        ),
    )


def test_does_not_bluff_river_multiway_with_three_opponents():
    flop_table, hero = make_table(
        street="Flop",
        board=["2d", "3h", "4s"],
        hero_hole=["9h", "Kc"],
        available=["fold", "check", "bet"],
    )
    turn_table, _hero = make_table(
        street="Turn",
        board=["2d", "3h", "4s", "5c"],
        hero_hole=["9h", "Kc"],
        available=["fold", "check", "bet"],
    )
    river_table, _hero = make_table(
        street="River",
        board=["2d", "3h", "4s", "5c", "7d"],
        hero_hole=["9h", "Kc"],
        villains=3,
        call=0,
        available=["fold", "check", "bet"],
    )

    assert apply_board_illusion(flop_table, hero, ("bet", 2, "base")) is None
    assert apply_board_illusion(turn_table, hero, ("bet", 2, "base")) is None

    assert_no_override(
        apply_board_illusion(
            river_table,
            hero,
            ("check", None, "base"),
        ),
    )


def test_does_not_bluff_river_if_facing_bet():
    flop_table, hero = make_table(
        street="Flop",
        board=["2d", "3h", "4s"],
        hero_hole=["9h", "Kc"],
        available=["fold", "check", "bet"],
    )
    turn_table, _hero = make_table(
        street="Turn",
        board=["2d", "3h", "4s", "5c"],
        hero_hole=["9h", "Kc"],
        available=["fold", "check", "bet"],
    )
    river_table, _hero = make_table(
        street="River",
        board=["2d", "3h", "4s", "5c", "7d"],
        hero_hole=["9h", "Kc"],
        call=100,
        available=["fold", "call", "raise"],
    )

    assert apply_board_illusion(flop_table, hero, ("bet", 2, "base")) is None
    assert apply_board_illusion(turn_table, hero, ("bet", 2, "base")) is None

    assert_no_override(
        apply_board_illusion(
            river_table,
            hero,
            ("check", None, "base"),
        ),
    )


def test_state_resets_when_hand_id_changes():
    flop_table, hero = make_table(
        street="Flop",
        board=["2d", "3h", "4s"],
        hero_hole=["9h", "Kc"],
        hand_id="hand-a",
        available=["fold", "check", "bet"],
    )
    turn_table, _hero = make_table(
        street="Turn",
        board=["2d", "3h", "4s", "5c"],
        hero_hole=["9h", "Kc"],
        hand_id="hand-a",
        available=["fold", "check", "bet"],
    )
    river_table, river_hero = make_table(
        street="River",
        board=["2d", "3h", "4s", "5c", "7d"],
        hero_hole=["9h", "Kc"],
        hand_id="hand-b",
        call=0,
        available=["fold", "check", "bet"],
    )

    assert apply_board_illusion(flop_table, hero, ("bet", 2, "base")) is None
    assert apply_board_illusion(turn_table, hero, ("bet", 2, "base")) is None

    assert_no_override(
        apply_board_illusion(
            river_table,
            river_hero,
            ("check", None, "base"),
        ),
    )


def test_state_resets_when_agent_changes():
    flop_table, hero = make_table(
        street="Flop",
        board=["2d", "3h", "4s"],
        hero_hole=["9h", "Kc"],
        hand_id="hand-a",
        available=["fold", "check", "bet"],
    )
    turn_table, _hero = make_table(
        street="Turn",
        board=["2d", "3h", "4s", "5c"],
        hero_hole=["9h", "Kc"],
        hand_id="hand-a",
        available=["fold", "check", "bet"],
    )
    river_table, other_hero = make_table(
        street="River",
        board=["2d", "3h", "4s", "5c", "7d"],
        hero_hole=["9h", "Kc"],
        hand_id="hand-a",
        hero_agent="other-agent",
        call=0,
        available=["fold", "check", "bet"],
    )

    assert apply_board_illusion(flop_table, hero, ("bet", 2, "base")) is None
    assert apply_board_illusion(turn_table, hero, ("bet", 2, "base")) is None

    assert_no_override(
        apply_board_illusion(
            river_table,
            other_hero,
            ("check", None, "base"),
        ),
    )


def test_does_not_bluff_river_with_one_pair():
    flop_table, hero = make_table(
        street="Flop",
        board=["2d", "3h", "4s"],
        hero_hole=["9h", "Kc"],
        available=["fold", "check", "bet"],
    )
    turn_table, _hero = make_table(
        street="Turn",
        board=["2d", "3h", "4s", "5c"],
        hero_hole=["9h", "Kc"],
        available=["fold", "check", "bet"],
    )
    river_table, river_hero = make_table(
        street="River",
        board=["2d", "3h", "4s", "5c", "7d"],
        hero_hole=["9h", "9c"],
        call=0,
        available=["fold", "check", "bet"],
    )

    assert apply_board_illusion(flop_table, hero, ("bet", 2, "base")) is None
    assert apply_board_illusion(turn_table, hero, ("bet", 2, "base")) is None

    assert_no_override(
        apply_board_illusion(
            river_table,
            river_hero,
            ("check", None, "base"),
        ),
    )


def test_strategy_variant_uses_base_decision_when_exploit_does_not_override(
    monkeypatch,
):
    base_decision = ("call", 100, "base")

    monkeypatch.setattr(
        strategy_variant.base_strategy,
        "choose_action",
        lambda table, my_seat: base_decision,
    )
    monkeypatch.setattr(strategy_variant, "apply_board_illusion", lambda *_args: None)

    assert strategy_variant.choose_action({"street": "River"}, {}) is base_decision


def test_strategy_variant_applies_exploit_override(monkeypatch):
    base_decision = ("check", None, "base")
    override = ("raise", 1000, "board-illusion override")

    def fake_choose_action(table, my_seat):
        return base_decision

    def fake_apply_board_illusion(table, my_seat, decision):
        assert decision is base_decision
        return override

    monkeypatch.setattr(
        strategy_variant.base_strategy,
        "choose_action",
        fake_choose_action,
    )
    monkeypatch.setattr(
        strategy_variant,
        "apply_board_illusion",
        fake_apply_board_illusion,
    )

    assert strategy_variant.choose_action({"street": "River"}, {}) is override


def test_strategy_variant_shoves_top_range_on_river_bet():
    board = ["5d", "6h", "7s", "8c", "9d"]
    table, hero = make_table(
        street="River",
        board=board,
        hero_hole=["Th", "Kd"],
        call=100,
        available=["fold", "call", "raise"],
        max_commit=1000,
    )

    action, amount, message = strategy_variant.choose_action(table, hero)

    assert action == "raise"
    assert amount == 1000
    assert message == "board-illusion: top-range river shove"


def test_strategy_variant_bluffs_after_monkeypatched_base_barrels(monkeypatch):
    def fake_choose_action(table, my_seat):
        if table["street"] == "Flop":
            return "bet", 2, "base flop bet"
        if table["street"] == "Turn":
            return "bet", 2, "base turn bet"
        return "check", None, "base river check"

    monkeypatch.setattr(
        strategy_variant.base_strategy,
        "choose_action",
        fake_choose_action,
    )

    flop_table, hero = make_table(
        street="Flop",
        board=["2d", "3h", "4s"],
        hero_hole=["9h", "Kc"],
        available=["fold", "check", "bet"],
    )
    turn_table, _hero = make_table(
        street="Turn",
        board=["2d", "3h", "4s", "5c"],
        hero_hole=["9h", "Kc"],
        available=["fold", "check", "bet"],
    )
    river_table, _hero = make_table(
        street="River",
        board=["2d", "3h", "4s", "5c", "7d"],
        hero_hole=["9h", "Kc"],
        call=0,
        available=["fold", "check", "bet"],
    )

    strategy_variant.choose_action(flop_table, hero)
    strategy_variant.choose_action(turn_table, hero)

    action, amount, message = strategy_variant.choose_action(river_table, hero)

    assert action == "bet"
    assert amount == 2
    assert message == "board-illusion: third barrel bottom range"
