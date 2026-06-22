"""Postflop air double barrel guard — behavioral spec for s3base.

Arena data (s3v012 run, 38 hands vs real bots) showed 4 hands where the bot
called preflop with a marginal hand, missed the flop, and fired 2-3 barrels
before folding. Total loss: -251 chips.

The guard prevents double-barrelling with complete air (rank 0). First barrel
is allowed, but second barrel with air is converted to check.
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


def make_table(hole, board, *, action_history=None, pot=100, street="Turn"):
    """Build a postflop table with action history."""
    seats = [make_seat(1, HERO, hole), make_seat(2, "villain")]
    table = {
        "street": street,
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": 1,
        "seats": seats,
        "opponentProfiles": {},
        "actionHistory": action_history or [],
        "allowedActions": {
            "availableActions": ["fold", "check", "bet", "all-in"],
            "callAmount": 0,
            "callChips": 0,
            "raiseRange": {"min": 0, "max": 2000},
            "betRange": {"min": 0, "max": 2000},
        },
    }
    return table, seats[0]


def action_for(hole, board, **kwargs):
    table, hero = make_table(hole, board, **kwargs)
    return choose_action(table, hero)[0]


# As 7h on 5h 4s Tc Jh — complete air, already bet flop
AIR_DOUBLE_BARREL = {
    "hole": ["As", "7h"],
    "board": ["5h", "4s", "Tc", "Jh"],
}

# Ks 5d on 4d Kc 5s 3s — two pair (not air), should NOT be capped
TWO_PAIR_TURN = {
    "hole": ["Ks", "5d"],
    "board": ["4d", "Kc", "5s", "3s"],
}


def test_air_double_barrel_on_turn():
    """Air with a prior flop bet should check on turn (not bet)."""
    history = [
        {"agentId": "hero", "action": "bet", "street": "Flop"},
    ]
    action = action_for(
        AIR_DOUBLE_BARREL["hole"], AIR_DOUBLE_BARREL["board"],
        action_history=history, pot=50, street="Turn",
    )
    assert action == "check", (
        f"air with prior flop bet should check on turn, got {action!r}"
    )


def test_air_double_barrel_on_river():
    """Air with prior bets on flop and turn should check on river."""
    history = [
        {"agentId": "hero", "action": "bet", "street": "Flop"},
        {"agentId": "villain", "action": "call", "street": "Flop"},
        {"agentId": "hero", "action": "bet", "street": "Turn"},
        {"agentId": "villain", "action": "raise", "street": "Turn"},
    ]
    action = action_for(
        AIR_DOUBLE_BARREL["hole"], AIR_DOUBLE_BARREL["board"],
        action_history=history, pot=150, street="River",
    )
    assert action == "check", (
        f"air with prior bets should check on river, got {action!r}"
    )


def test_air_first_barrel_allowed():
    """Air on flop with no prior bets should be allowed to bet (first barrel)."""
    history = []
    action = action_for(
        AIR_DOUBLE_BARREL["hole"], AIR_DOUBLE_BARREL["board"][:3],  # Flop only
        action_history=history, pot=20, street="Flop",
    )
    # First barrel with air is allowed
    assert action == "bet", (
        f"air first barrel on flop should be allowed, got {action!r}"
    )


def test_two_pair_not_capped():
    """Two pair (rank 2) should NOT be capped by this guard."""
    history = [
        {"agentId": "hero", "action": "bet", "street": "Flop"},
    ]
    action = action_for(
        TWO_PAIR_TURN["hole"], TWO_PAIR_TURN["board"],
        action_history=history, pot=50, street="Turn",
    )
    # Two pair is not air — should be allowed to bet
    assert action == "bet", (
        f"two pair should not be capped, got {action!r}"
    )