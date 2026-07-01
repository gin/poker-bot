"""Preflop min-raise war cap — behavioral spec for s3base.

In min-raise wars, the bot and opponent keep min-raising each other preflop.
After 3+ raise-backs, the bot has committed a large fraction of its stack
and the SPR is very low. Continuing to raise at this point is -EV because:
1. The pot is already large relative to the remaining stack
2. Postflop play is trivial (SPR < 1.0)
3. The bot is often raising with marginal hands that should just call/fold

The guard should stop the bot from raising after 3+ preflop raise-backs.
Instead, it should call (if facing a bet) or check (if first to act).
"""

import pytest

from poker_bot.strategies.s3base import choose_action

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
    *,
    raise_count=0,
    facing_bet=0,
    hero_stack=2000,
    pot=30,
    button=1,
):
    """Build a preflop table with a history of raise-backs.

    raise_count: number of times hero has already raised this hand
    facing_bet: amount hero needs to call (0 = first to act)
    """
    seats = [make_seat(1, HERO, hole, stack=hero_stack)]
    seats.append(make_seat(2, "villain", stack=hero_stack))

    # Build action history simulating raise-backs
    history = []
    for i in range(raise_count):
        history.append({"agentId": "villain", "action": "raise", "street": "Preflop"})
        history.append({"agentId": HERO, "action": "raise", "street": "Preflop"})

    available = (
        ["fold", "call", "raise", "all-in"]
        if facing_bet > 0
        else ["fold", "check", "bet", "all-in"]
    )

    table = {
        "street": "Preflop",
        "boardCards": [],
        "potChips": pot,
        "buttonSeatNumber": button,
        "seats": seats,
        "opponentProfiles": {},
        "actionHistory": history,
        "allowedActions": {
            "availableActions": available,
            "callAmount": facing_bet,
            "callChips": facing_bet,
            "raiseRange": {"min": facing_bet * 2, "max": hero_stack},
            "betRange": {"min": 0, "max": hero_stack},
        },
    }
    return table, seats[0]


def action_for(hole, **kwargs):
    table, hero = make_table(hole, **kwargs)
    return choose_action(table, hero)[0]


def assert_does_not_raise(action):
    assert action not in ("raise", "bet"), (
        f"preflop war cap should stop raising: chose {action!r}"
    )


def assert_value_action(action):
    assert action in ("raise", "bet"), (
        f"normal preflop should still raise: chose {action!r}"
    )


@pytest.mark.parametrize("raise_count", [3, 4, 5])
@pytest.mark.parametrize("facing_bet", [0, 40, 100])
def test_preflop_war_cap_no_more_raises(raise_count, facing_bet):
    """After 3+ preflop raise-backs, stop raising.

    The bot should call or check, never raise again.
    """
    action = action_for(
        ["AS", "KD"],
        raise_count=raise_count,
        facing_bet=facing_bet,
        hero_stack=2000 - raise_count * 20,  # simulate stack depletion
        pot=30 + raise_count * 40,
    )
    assert_does_not_raise(action)


@pytest.mark.parametrize("raise_count", [0, 1, 2])
def test_preflop_normal_raise_still_fires(raise_count):
    """Before the cap (0-2 raise-backs), still raise normally."""
    action = action_for(
        ["AS", "KD"],
        raise_count=raise_count,
        facing_bet=40 if raise_count > 0 else 0,
        hero_stack=2000,
        pot=30,
    )
    assert_value_action(action)
