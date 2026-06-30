"""Postflop war cap and paired board fold guard — hubase fixes.

Two patches:
1. postflop_marginal_hand_war_cap: fold rank 2 (two pair) when facing 4-bet
   at unfavorable pot odds (>33% price), instead of always calling.
2. medium_pair_paired_board_fold_guard: do NOT fold 77 when hero actually
   has a full house (77 on 3-3-7 board = 77733 full house, rank 6).
"""

import pytest

from poker_bot.strategies.hubase import choose_action

HERO = "hero"


def make_seat(seat_number, agent_id, hole_cards=None, *, folded=False):
    return {
        "seatNumber": seat_number,
        "agentId": agent_id,
        "holeCards": hole_cards or [],
        "stackChips": 2000,
        "currentBetChips": 0,
        "folded": folded,
        "hasFolded": False,
    }


def make_war_table(hole, board, pot, call, num_raises=3):
    """Build a Turn/River table where hero has raised num_raises times."""
    history = []
    for i in range(num_raises):
        history.append({"agentId": HERO, "action": "raise", "amount": 40, "street": "Turn"})
        history.append({"agentId": "villain", "action": "raise", "amount": 80, "street": "Turn"})

    seats = [make_seat(1, HERO, hole)]
    seats.append(make_seat(2, "villain"))

    table = {
        "street": "Turn",
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": 1,
        "seats": seats,
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": ["fold", "call", "raise", "all-in"],
            "callAmount": call,
            "callChips": call,
            "raiseRange": {"min": call * 2, "max": 2000},
            "betRange": {"min": 0, "max": 2000},
        },
        "actionHistory": history,
    }
    return table, seats[0]


def make_call_table(
    hole,
    board,
    *,
    street="River",
    pot=400,
    call=200,
    available=("fold", "call"),
):
    """Build a minimal facing-bet table."""
    seats = [make_seat(1, HERO, hole)]
    seats.append(make_seat(2, "villain"))

    table = {
        "street": street,
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": 1,
        "seats": seats,
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": list(available),
            "callAmount": call,
            "callChips": call,
            "raiseRange": {"min": call * 2, "max": 2000},
            "betRange": {"min": 0, "max": 2000},
        },
    }
    return table, seats[0]


# ────────────────────────────────────────────────────────────────────
# Test 1: Full house positive control — 77 should NOT fold when it
# actually makes a full house on a paired board.
# ────────────────────────────────────────────────────────────────────

def test_77_full_house_river_not_folded():
    """77 on 3-3-7-A-J board → full house (77733) — never fold."""
    table, hero = make_call_table(
        ["7C", "7D"], ["3C", "AD", "JS", "7S", "3D"],
        street="River", pot=400, call=200,
    )
    action, amount, msg = choose_action(table, hero)
    assert action != "fold", (
        f"77 with full house on paired board should NOT fold, got {action!r}: {msg}"
    )


def test_77_full_house_flop_not_folded():
    """77 on 7-3-3 board on the flop — full house (77733), should not fold."""
    table, hero = make_call_table(
        ["7C", "7D"], ["7H", "3S", "3D"],
        street="Flop", pot=100, call=50,
        available=("fold", "call"),
    )
    action, amount, msg = choose_action(table, hero)
    assert action != "fold", (
        f"77 making full house on flop should NOT fold, got {action!r}: {msg}"
    )


def test_88_full_house_not_folded():
    """88 on 8-2-2 board → full house (88822), should not fold."""
    table, hero = make_call_table(
        ["8C", "8D"], ["8H", "2S", "2D"],
        street="Flop", pot=100, call=60,
        available=("fold", "call"),
    )
    action, amount, msg = choose_action(table, hero)
    assert action != "fold", (
        f"88 with full house on flop should NOT fold, got {action!r}: {msg}"
    )


def test_77_no_full_house_on_paired_board_folds():
    """77 on 9-9-K board (nines on board, hero has 77) — two pair but board
    has trips potential + overpair. Should fold to a bet.
    Note: 77 on 9-9-K actually makes two pair (9s + 7s). Board has pair of 9s.
    Hero doesn't have a full house (no 9 in hand).
    """
    table, hero = make_call_table(
        ["7C", "7D"], ["9H", "KS", "9D"],
        street="Flop", pot=100, call=80,
        available=("fold", "call"),
    )
    action, amount, msg = choose_action(table, hero)
    assert action == "fold", (
        f"77 on 9-9-K (two pair, no full house) should fold, got {action!r}: {msg}"
    )


# ────────────────────────────────────────────────────────────────────
# Test 2: War cap — rank 2 should fold at >33% pot odds.
# ────────────────────────────────────────────────────────────────────

def test_war_cap_rank_2_folds_at_high_price():
    """Two pair (Q-Q-J-J on Q-J-7 board) facing 4-bet at 36% price → fold."""
    # pot=180, call=100 → price = 100/280 = 35.7% > 33% → fold
    table, hero = make_war_table(
        ["QC", "JD"], ["QD", "JC", "7H"],
        pot=180, call=100, num_raises=3,
    )
    action, amount, msg = choose_action(table, hero)
    assert action == "fold", (
        f"rank 2 facing 4-bet at >33% price should fold, got {action!r}: {msg}"
    )


def test_war_cap_rank_2_calls_at_low_price():
    """Two pair facing 4-bet at 14% price → call."""
    # pot=600, call=100 → price = 100/700 = 14.3% < 33% → call
    table, hero = make_war_table(
        ["QC", "JD"], ["QD", "JC", "7H"],
        pot=600, call=100, num_raises=3,
    )
    action, amount, msg = choose_action(table, hero)
    assert action == "call", (
        f"rank 2 facing 4-bet at <33% price should call, got {action!r}: {msg}"
    )


def test_war_cap_rank_3_not_affected():
    """Trips (rank 3+) facing 4-bet — not capped, allow raise."""
    history = []
    for i in range(3):
        history.append({"agentId": HERO, "action": "raise", "amount": 40, "street": "Turn"})
        history.append({"agentId": "villain", "action": "raise", "amount": 80, "street": "Turn"})

    seats = [make_seat(1, HERO, ["8D", "8C"]), make_seat(2, "villain")]
    table = {
        "street": "Turn", "boardCards": ["8S", "KH", "3D"],
        "potChips": 180, "buttonSeatNumber": 1, "seats": seats,
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": ["fold", "call", "raise", "all-in"],
            "callAmount": 100, "callChips": 100,
            "raiseRange": {"min": 200, "max": 2000},
            "betRange": {"min": 0, "max": 2000},
        },
        "actionHistory": history,
    }
    action, amount, msg = choose_action(table, seats[0])
    assert action == "raise", (
        f"trips facing 4-bet should raise, got {action!r}: {msg}"
    )
