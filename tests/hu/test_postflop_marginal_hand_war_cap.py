"""Postflop marginal hand war cap — behavioral spec for s3base.

Arena data (s3v005 run, 58 hands vs real bots) showed Ks 5d entering a
4-raise war on the turn with two pair, losing -684. Two pair on a wet
board is vulnerable to straights/flushes — when the opponent keeps
raising, the bot should cap its aggression.

The guard is hand-strength-aware: strong hands (set+) are allowed to
raise for value, but marginal hands (rank < 3) are capped after 3+ raises.

Tuning history:
- Threshold 2 (s3v012): 55% fold rate, 59% non-fold win rate — too aggressive
- Threshold 3 (s3v013): Only caps in extreme war situations (3+ raises)

Reference: PLAN_FIX_POSTFLOP_WAR.md
"""

import pytest

from poker_bot.strategies.s3base import choose_action

HERO = "hero"


def make_seat(seat_number, agent_id, hole_cards=None, *, folded=False, stack=2000, current_bet=0):
    return {
        "seatNumber": seat_number,
        "agentId": agent_id,
        "holeCards": hole_cards or [],
        "stackChips": stack,
        "currentBetChips": current_bet,
        "folded": folded,
        "hasFolded": False,
    }


def make_table(hole, board, *, facing_bet=0, action_history=None, pot=100):
    """Build a postflop table. action_history simulates prior raises this street."""
    seats = [make_seat(1, HERO, hole), make_seat(2, "villain")]

    if facing_bet > 0:
        available = ["fold", "call", "raise", "all-in"]
    else:
        available = ["fold", "check", "bet", "all-in"]

    table = {
        "street": "Turn",
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": 1,
        "seats": seats,
        "opponentProfiles": {},
        "actionHistory": action_history or [],
        "allowedActions": {
            "availableActions": available,
            "callAmount": facing_bet,
            "callChips": facing_bet,
            "raiseRange": {"min": facing_bet * 2, "max": 2000},
            "betRange": {"min": 0, "max": 2000},
        },
    }
    return table, seats[0]


# Ks 5d on turn (4d Kc 5s 3s): two pair (KK55) — the failing case from arena
TWO_PAIR_TURN = {
    "hole": ["Ks", "5d"],
    "board": ["4d", "Kc", "5s", "3s"],
}


# Three of a kind (set) on turn: should NOT be capped
SET_TURN = {
    "hole": ["Ks", "Kh"],
    "board": ["4d", "Kc", "5s", "3s"],
}


def action_for(hole, board, **kwargs):
    table, hero = make_table(hole, board, **kwargs)
    return choose_action(table, hero)[0]


def test_two_pair_cap_calls_after_3_raises():
    """Two pair with 3 prior raises should call (not raise) — the Ks 5d case."""
    # History: hero bet, villain raised, hero raised, villain raised, hero raised (3 hero raises)
    history = [
        {"agentId": "hero", "action": "bet", "street": "Turn"},
        {"agentId": "villain", "action": "raise", "street": "Turn"},
        {"agentId": "hero", "action": "raise", "street": "Turn"},
        {"agentId": "villain", "action": "raise", "street": "Turn"},
        {"agentId": "hero", "action": "raise", "street": "Turn"},
        {"agentId": "villain", "action": "raise", "street": "Turn"},
    ]
    action = action_for(
        TWO_PAIR_TURN["hole"], TWO_PAIR_TURN["board"],
        facing_bet=50, action_history=history, pot=200,
    )
    assert action == "call", (
        f"two pair with 3 prior raises should call, got {action!r}"
    )


def test_two_pair_still_raises_after_2_raises():
    """Two pair with only 2 prior raises should still be allowed to raise."""
    # History: hero bet, villain raised, hero raised (2 hero raises)
    history = [
        {"agentId": "hero", "action": "bet", "street": "Turn"},
        {"agentId": "villain", "action": "raise", "street": "Turn"},
        {"agentId": "hero", "action": "raise", "street": "Turn"},
        {"agentId": "villain", "action": "raise", "street": "Turn"},
    ]
    action = action_for(
        TWO_PAIR_TURN["hole"], TWO_PAIR_TURN["board"],
        facing_bet=50, action_history=history, pot=200,
    )
    # Only 2 prior raises — guard should NOT fire (threshold is 3)
    assert action == "raise", (
        f"two pair with 2 prior raises should raise, got {action!r}"
    )


def test_set_can_still_raise_after_3_raises():
    """Set (3 of a kind) with 3 prior raises should still be able to raise."""
    history = [
        {"agentId": "hero", "action": "bet", "street": "Turn"},
        {"agentId": "villain", "action": "raise", "street": "Turn"},
        {"agentId": "hero", "action": "raise", "street": "Turn"},
        {"agentId": "villain", "action": "raise", "street": "Turn"},
        {"agentId": "hero", "action": "raise", "street": "Turn"},
        {"agentId": "villain", "action": "raise", "street": "Turn"},
    ]
    action = action_for(
        SET_TURN["hole"], SET_TURN["board"],
        facing_bet=50, action_history=history, pot=200,
    )
    # Set should be allowed to raise — NOT capped
    assert action == "raise", (
        f"set with 3 prior raises should raise, got {action!r}"
    )


def test_two_pair_first_raise_allowed():
    """Two pair with only 1 prior raise should be able to raise (not capped)."""
    history = [
        {"agentId": "hero", "action": "bet", "street": "Turn"},
        {"agentId": "villain", "action": "raise", "street": "Turn"},
    ]
    action = action_for(
        TWO_PAIR_TURN["hole"], TWO_PAIR_TURN["board"],
        facing_bet=50, action_history=history, pot=200,
    )
    # Only 1 prior raise — guard should NOT fire
    assert action == "raise", (
        f"two pair with 1 prior raise should raise, got {action!r}"
    )