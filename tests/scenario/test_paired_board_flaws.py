"""Scenario tests for s2v006 paired-board and set-mining flaws.

These tests lock in the two major leaks found in telemetry-luigi-tournament.sqlite:

1. **Set-mining at bad prices**: Calling preflop 3-bets with medium pocket
   pairs when the price-to-stack ratio is > 11.8% (8:1 odds). The 7h7d
   hand (cmqh6gkmc4eb7m0toatxdecfq) called a 3-bet to 479 with 2484 chips
   (19.3% of stack) and lost the entire stack. This should fold preflop.

2. **Calling down on paired boards with medium pocket pairs**: Once in the
   hand with 77 on a paired board against a 3-bettor, the strategy should
   recognize it's usually crushed and fold earlier. The telemetry hand called
   flop, turn, and river on Js 6c 3s 6d Qd against an opponent with KK.

The strategy under test is s2base.py.
"""

import pytest

from poker_bot.strategies import s2base as strategy


def make_seats(bets, stacks, hero_seat, hero_cards, button=1, n_players=6):
    """Build a 6-max seats list."""
    seats = []
    for index in range(n_players):
        seat_number = index + 1
        seat_bet = (bets[index] if bets else 0) or 0
        seats.append(
            {
                "agentId": f"villain-{seat_number}"
                if seat_number != hero_seat
                else "hero",
                "seatNumber": seat_number,
                "holeCards": hero_cards if seat_number == hero_seat else [],
                "stackChips": stacks[index] if stacks else 1500,
                "currentBetChips": seat_bet,
                "folded": seat_bet is None and seat_number != hero_seat,
                "hasFolded": seat_bet is None and seat_number != hero_seat,
            }
        )
    return seats


def make_preflop_table(
    hero_cards,
    hero_seat,
    button,
    raise_amount,
    hero_stack,
    raise_seat,
    n_players=6,
    raiser_stack=1500,
    raiser_vpip=0.5,
    raiser_pfr=0.3,
):
    """Build a preflop table where one player has raised."""
    seats = []
    for i in range(1, n_players + 1):
        if i == hero_seat:
            hero_blind = 0
            if hero_seat == 2:  # SB
                hero_blind = 5
            elif hero_seat == 3:  # BB
                hero_blind = 10
            seats.append(
                {
                    "agentId": "hero",
                    "seatNumber": i,
                    "holeCards": hero_cards,
                    "stackChips": hero_stack,
                    "currentBetChips": hero_blind,
                    "folded": False,
                    "hasFolded": False,
                }
            )
        elif i == raise_seat:
            seats.append(
                {
                    "agentId": "raiser",
                    "seatNumber": i,
                    "holeCards": [],
                    "stackChips": raiser_stack,
                    "currentBetChips": raise_amount,
                    "folded": False,
                    "hasFolded": False,
                }
            )
        else:
            seats.append(
                {
                    "agentId": f"villain-{i}",
                    "seatNumber": i,
                    "holeCards": [],
                    "stackChips": 1500,
                    "currentBetChips": 0,
                    "folded": True,
                    "hasFolded": True,
                }
            )

    call_amount = raise_amount - (seats[hero_seat - 1]["currentBetChips"] or 0)
    profiles = {
        "raiser": {
            "hands_seen": 50,
            "vpip": int(round(raiser_vpip * 50)),
            "pfr": int(round(raiser_pfr * 50)),
        }
    }

    return {
        "street": "Preflop",
        "boardCards": [],
        "potChips": raise_amount + 15,
        "buttonSeatNumber": button,
        "seats": seats,
        "opponentProfiles": profiles,
        "allowedActions": {
            "availableActions": ["fold", "call", "raise", "all-in"],
            "callAmount": call_amount,
            "callChips": call_amount,
            "minBet": 10,
            "minRaiseTo": max(call_amount * 2, raise_amount + raise_amount),
        },
    }


def make_postflop_table(
    hero_cards,
    hero_seat,
    board_cards,
    pot_chips,
    hero_stack,
    facing_bet,
    available=None,
    n_players=4,
):
    """Build a postflop table where hero is acting with facing_bet to call."""
    if available is None:
        available = ["fold", "call", "raise"]

    seats = []
    for i in range(1, n_players + 1):
        if i == hero_seat:
            seats.append(
                {
                    "agentId": "hero",
                    "seatNumber": i,
                    "holeCards": hero_cards,
                    "stackChips": hero_stack,
                    "currentBetChips": facing_bet,
                    "folded": False,
                    "hasFolded": False,
                }
            )
        else:
            seats.append(
                {
                    "agentId": f"villain-{i}",
                    "seatNumber": i,
                    "holeCards": [],
                    "stackChips": 1500,
                    "currentBetChips": 0,
                    "folded": False,
                    "hasFolded": False,
                }
            )

    button = (hero_seat - 1) if hero_seat > 1 else n_players
    return {
        "street": "Flop",
        "boardCards": board_cards,
        "potChips": pot_chips,
        "buttonSeatNumber": button,
        "seats": seats,
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": available,
            "callAmount": facing_bet,
            "callChips": facing_bet,
        },
    }


# ════════════════════════════════════════════════════════════════════════════
# Flaw 1: Set-mining at bad prices
# ════════════════════════════════════════════════════════════════════════════


def test_set_mine_77_hj_3bet_fold_vs_tight_opponent():
    """7h7d HJ vs 3-bet to 479 (19% of 2484 stack) from tight opponent → fold.

    Reference: cmqh6gkmc4eb7m0toatxdecfq from telemetry.
    The set-mining math requires price_to_stack <= 0.118 (8:1 odds).
    Here 479/2484 = 19.3%, which is way too expensive. Against a tight
    opponent's 3-bet, this should fold.
    """
    table = make_preflop_table(
        hero_cards=["7h", "7d"],
        hero_seat=4,  # HJ
        button=4,
        raise_amount=479,
        hero_stack=2484,
        raise_seat=6,  # BTN
        raiser_vpip=0.12,
        raiser_pfr=0.08,
    )
    hero = table["seats"][3]
    result = strategy.profiled_choose_action(table, hero)

    assert result is not None
    assert result[0] == "fold", (
        f"Expected fold for 7h7d HJ vs tight 19% 3-bet, got {result[0]}"
    )


def test_set_mine_77_hj_3bet_call_vs_loose_opponent():
    """7h7d HJ vs 3-bet to 479 (19% of 2484 stack) from loose opponent → continue.

    Same bad set-mining price, but against a loose opponent the 3-bet range is
    wider and more bluff-heavy. Hero can continue with a call or raise instead
    of auto-folding.
    """
    table = make_preflop_table(
        hero_cards=["7h", "7d"],
        hero_seat=4,  # HJ
        button=4,
        raise_amount=479,
        hero_stack=2484,
        raise_seat=6,  # BTN
        raiser_vpip=0.45,
        raiser_pfr=0.30,
    )
    hero = table["seats"][3]
    result = strategy.profiled_choose_action(table, hero)

    assert result is not None
    assert result[0] != "fold", (
        f"Expected call/raise for 7h7d HJ vs loose 19% 3-bet, got {result[0]}"
    )


def test_set_mine_77_bb_cheap_raise_call():
    """7h7d BB vs raise to 37 (1.5% of 2444 stack) → call.

    Regression test: cheap raises with good implied odds should still call.
    This ensures we don't over-fold pocket pairs preflop.
    """
    table = make_preflop_table(
        hero_cards=["7h", "7d"],
        hero_seat=3,  # BB
        button=5,
        raise_amount=37,
        hero_stack=2444,
        raise_seat=6,  # BTN
    )
    hero = table["seats"][2]
    result = strategy.profiled_choose_action(table, hero)

    assert result is not None
    assert result[0] in ("call", "raise"), (
        f"Expected call/raise for 7h7d BB vs cheap raise, got {result[0]}"
    )


# ════════════════════════════════════════════════════════════════════════════
# Flaw 2: Calling down on paired boards with medium pocket pairs
# ════════════════════════════════════════════════════════════════════════════


def test_paired_board_77_flop_fold_vs_3bettor():
    """7h7d on Js 6c 3s (paired board) vs 3-bettor → fold.

    Reference: cmqh6gkmc4eb7m0toatxdecfq from telemetry.
    Hero has 77 on a paired board (Js 6c 3s). The 3-bettor's range is
    heavily weighted toward overpairs (KK, QQ, JJ, AA). Hero's 77 is
    crushed. Should fold on the flop.
    """
    table = make_postflop_table(
        hero_cards=["7h", "7d"],
        hero_seat=3,
        board_cards=["Js", "6c", "6s"],
        pot_chips=1362,
        hero_stack=2005,
        facing_bet=389,
        available=["fold", "call", "raise"],
    )
    hero = table["seats"][2]
    result = strategy.profiled_choose_action(table, hero)

    assert result is not None
    assert result[0] == "fold", (
        f"Expected fold for 7h7d on paired board vs 3-bettor, got {result[0]}"
    )


def test_paired_board_77_turn_fold_vs_3bettor():
    """7h7d on Js 6c 3s 6d (paired board) vs 3-bettor → fold.

    Regression: even if we somehow called the flop, the turn paired board
    (6d) makes it even more likely the opponent has a full house or trips.
    Should fold on the turn.
    """
    table = make_postflop_table(
        hero_cards=["7h", "7d"],
        hero_seat=3,
        board_cards=["Js", "6c", "3s", "6d"],
        pot_chips=2342,
        hero_stack=1616,
        facing_bet=591,
        available=["fold", "call", "raise"],
    )
    hero = table["seats"][2]
    result = strategy.profiled_choose_action(table, hero)

    assert result is not None
    assert result[0] == "fold", (
        f"Expected fold for 7h7d on paired turn vs 3-bettor, got {result[0]}"
    )


def test_paired_board_77_river_fold_vs_3bettor():
    """7h7d on Js 6c 3s 6d Qd (paired board) vs 3-bettor → fold.

    The river paired board makes it even more likely the opponent has a
    full house or trips. Against a 3-bettor's range, 77 has maybe 5-10%
    equity. Calling 1011 into 3944 (25.6% pot) requires ~28% equity.
    Should fold.
    """
    table = make_postflop_table(
        hero_cards=["7h", "7d"],
        hero_seat=3,
        board_cards=["Js", "6c", "3s", "6d", "Qd"],
        pot_chips=3944,
        hero_stack=1025,
        facing_bet=1011,
        available=["fold", "call", "raise"],
    )
    hero = table["seats"][2]
    result = strategy.profiled_choose_action(table, hero)

    assert result is not None
    assert result[0] == "fold", (
        f"Expected fold for 7h7d on paired river vs 3-bettor, got {result[0]}"
    )


# ════════════════════════════════════════════════════════════════════════════
# Positive controls: hands that SHOULD call/raise
# ════════════════════════════════════════════════════════════════════════════


def test_paired_board_tt_value_raise():
    """TT on Ah Qd Ts (top set) → raise for value.

    Positive control: strong hands on paired boards should still bet/raise
    for value. This ensures we don't over-fold on paired boards.
    """
    table = make_postflop_table(
        hero_cards=["Td", "Tc"],
        hero_seat=3,
        board_cards=["Ah", "Qd", "Ts"],
        pot_chips=74,
        hero_stack=1164,
        facing_bet=29,
        available=["fold", "call", "raise"],
    )
    hero = table["seats"][2]
    result = strategy.profiled_choose_action(table, hero)

    assert result is not None
    assert result[0] in ("raise", "bet"), (
        f"Expected raise/bet for TT top set, got {result[0]}"
    )


def test_cheap_set_mine_regression_44_sb():
    """4c4d SB vs raise to 37 (1.5% of stack) → call.

    Positive control: cheap raises with good implied odds should still call.
    """
    table = make_preflop_table(
        hero_cards=["4c", "4d"],
        hero_seat=2,  # SB
        button=3,
        raise_amount=37,
        hero_stack=2444,
        raise_seat=6,
    )
    hero = table["seats"][1]
    result = strategy.profiled_choose_action(table, hero)

    assert result is not None
    assert result[0] == "call", (
        f"Expected call for 4c4d SB vs cheap raise, got {result[0]}"
    )
