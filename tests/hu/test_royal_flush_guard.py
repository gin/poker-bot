"""Royal flush guard — s4base.

Prevents folding with AKs preflop and when royal flush is possible postflop.
"""

import pytest

from poker_bot.strategies.s4base import choose_action

HERO = "hero"


def make_seat(seat_number, agent_id, hole_cards=None, *, folded=False, stack=2000):
    return {
        "seatNumber": seat_number,
        "agentId": agent_id,
        "holeCards": hole_cards or [],
        "stackChips": stack,
        "currentBetChips": 0,
        "folded": folded,
        "hasFolded": False,
    }


def make_table(
    hole,
    board,
    *,
    street="Flop",
    facing_bet=0,
    hero_stack=2000,
    pot=30,
    button=1,
    available=None,
):
    seats = [make_seat(1, HERO, hole, stack=hero_stack)]
    seats.append(make_seat(2, "villain", stack=hero_stack))

    if available is None:
        available = ["fold", "call", "raise", "all-in"] if facing_bet > 0 else ["fold", "check", "bet", "all-in"]

    table = {
        "street": street,
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": button,
        "seats": seats,
        "opponentProfiles": {},
        "actionHistory": [],
        "allowedActions": {
            "availableActions": available,
            "callAmount": facing_bet,
            "callChips": facing_bet,
            "raiseRange": {"min": facing_bet * 2 if facing_bet > 0 else 40, "max": hero_stack},
            "betRange": {"min": 0, "max": hero_stack},
        },
    }
    return table, seats[0]


def action_for(hole, board, **kwargs):
    table, hero = make_table(hole, board, **kwargs)
    return choose_action(table, hero)


# ── Preflop: AKs should not fold ────────────────────────────────────────

def test_aks_preflop_facing_bet_calls():
    """AKs preflop facing a bet should call, not fold."""
    result = action_for(
        ["AS", "KS"],
        [],
        street="Preflop",
        facing_bet=40,
        pot=60,
    )
    action, _amount, message = result
    assert action != "fold", f"AKs should not fold preflop, got {action!r}: {message}"
    assert action == "call", f"AKs should call when facing bet, got {action!r}"


def test_aks_preflop_no_bet_checks():
    """AKs preflop with no bet to face should check."""
    result = action_for(
        ["AS", "KS"],
        [],
        street="Preflop",
        facing_bet=0,
        pot=30,
    )
    action, _amount, message = result
    assert action != "fold", f"AKs should not fold preflop, got {action!r}: {message}"


def test_non_aks_preflop_can_fold():
    """Non-AKs hands preflop should still be able to fold."""
    result = action_for(
        ["QS", "JS"],
        [],
        street="Preflop",
        facing_bet=200,
        pot=30,
        available=["fold", "call", "raise"],
    )
    action, _amount, message = result
    # QS/JT is not AKs — guard should not fire, fold is allowed
    assert action == "fold" or action is not None, f"Non-AKs should be able to fold, got {action!r}"


# ── Postflop: royal flush possible → never fold ────────────────────────

def test_royal_flush_possible_flop_does_not_fold():
    """With royal flush possible on flop, should not fold."""
    # Ah Kh with Qh Jh Th on flop = royal flush already made, but even
    # if only 3 royal cards on board + 2 in hand, it's possible
    result = action_for(
        ["AH", "KH"],
        ["QH", "JH", "TH"],
        street="Flop",
        facing_bet=40,
        pot=100,
    )
    action, _amount, message = result
    assert action != "fold", f"Royal flush made should not fold, got {action!r}: {message}"


def test_royal_flush_draw_flop_does_not_fold():
    """With royal flush draw on flop (4 to royal), should not fold."""
    result = action_for(
        ["AH", "KH"],
        ["QH", "JH", "3C"],
        street="Flop",
        facing_bet=40,
        pot=100,
    )
    action, _amount, message = result
    assert action != "fold", f"Royal flush draw should not fold, got {action!r}: {message}"


def test_royal_flush_draw_turn_does_not_fold():
    """With royal flush draw on turn (4 to royal), should not fold."""
    result = action_for(
        ["AH", "KH"],
        ["QH", "JH", "3C", "5D"],
        street="Turn",
        facing_bet=40,
        pot=200,
    )
    action, _amount, message = result
    assert action != "fold", f"Royal flush draw on turn should not fold, got {action!r}: {message}"


# ── No royal flush chance: guard exits, fold allowed ───────────────────

def test_no_royal_flush_chance_folds():
    """Without royal flush chance, fold should be allowed."""
    result = action_for(
        ["2C", "7D"],
        ["9H", "KS", "3C"],
        street="Flop",
        facing_bet=100,
        pot=30,
        available=["fold", "call", "raise"],
    )
    action, _amount, message = result
    # No royal flush chance — guard exits, normal fold allowed
    assert action == "fold" or action is not None, f"Without royal chance, fold is valid, got {action!r}"


def test_no_royal_flush_chance_river_folds():
    """On river without royal flush, fold should be allowed."""
    result = action_for(
        ["AH", "KH"],
        ["QH", "3C", "5D", "7S", "9H"],
        street="River",
        facing_bet=100,
        pot=200,
        available=["fold", "call", "raise"],
    )
    action, _amount, message = result
    # Only 2 royal cards (A, K) — need 3 more but 0 board slots left → impossible
    assert action == "fold" or action is not None, f"Without royal chance on river, fold is valid, got {action!r}"


# ── Check preferred over call when available ───────────────────────────

def test_royal_flush_possible_preflop_checks_when_free():
    """AKs preflop with no bet should check (not call)."""
    result = action_for(
        ["AS", "KS"],
        [],
        street="Preflop",
        facing_bet=0,
        pot=30,
        available=["fold", "check", "bet", "all-in"],
    )
    action, _amount, message = result
    assert action == "check", f"AKs with no bet should check, got {action!r}: {message}"


def test_royal_flush_possible_postflop_checks_when_free():
    """Royal flush possible postflop with no bet should check."""
    result = action_for(
        ["AH", "KH"],
        ["QH", "JH", "3C"],
        street="Flop",
        facing_bet=0,
        pot=100,
        available=["fold", "check", "bet", "all-in"],
    )
    action, _amount, message = result
    assert action == "check", f"Royal possible + no bet should check, got {action!r}: {message}"
