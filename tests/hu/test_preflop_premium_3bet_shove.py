"""Preflop premium 3bet shove — hubase fix.

After 3+ preflop raise-backs, premium hands (AA/KK/QQ, AKs/KQs) should go
all-in instead of capping or calling. Non-premium hands fall through to the
existing preflop_min_raise_war_cap (call/check).
"""

import pytest

from poker_bot.strategies.hubase import choose_action

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
    available=None,
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

    if available is None:
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
    return choose_action(table, hero)


# ── Premium hands: should shove all-in after 3+ raise-backs ─────────────


@pytest.mark.parametrize(
    "hole",
    [
        ["AS", "AH"],  # AA
        ["KS", "KH"],  # KK
        ["QS", "QH"],  # QQ
        ["AS", "KS"],  # AKs
        ["KS", "QS"],  # KQs
    ],
)
def test_premium_shoves_after_3_raises(hole):
    """Premium hands go all-in after 3+ preflop raise-backs."""
    result = action_for(
        hole,
        raise_count=3,
        facing_bet=200,
        hero_stack=1800,
        pot=300,
    )
    action, amount, message = result
    assert action == "all-in", f"Premium should shove all-in, got {action!r}: {message}"
    assert amount == 1800, f"All-in amount should be full stack (1800), got {amount}"


@pytest.mark.parametrize(
    "hole",
    [
        ["AS", "AH"],  # AA
        ["KS", "KH"],  # KK
        ["QS", "QH"],  # QQ
        ["AS", "KS"],  # AKs
        ["KS", "QS"],  # KQs
    ],
)
def test_premium_shoves_after_4_raises(hole):
    """Premium hands go all-in after 4+ preflop raise-backs too."""
    result = action_for(
        hole,
        raise_count=4,
        facing_bet=400,
        hero_stack=1600,
        pot=600,
    )
    action, amount, message = result
    assert action == "all-in", f"Premium should shove all-in, got {action!r}: {message}"
    assert amount == 1600


# ── Non-premium hands: should NOT shove, fall through to war cap ─────────


@pytest.mark.parametrize(
    "hole",
    [
        ["JS", "JH"],  # JJ
        ["TS", "TH"],  # TT
        ["AS", "KD"],  # AKo
        ["AS", "QD"],  # AQs
        ["9S", "9H"],  # 99
    ],
)
def test_non_premium_does_not_shove(hole):
    """Non-premium hands should not go all-in after 3+ raise-backs."""
    result = action_for(
        hole,
        raise_count=3,
        facing_bet=200,
        hero_stack=1800,
        pot=300,
    )
    action, _amount, message = result
    assert action != "all-in", (
        f"Non-premium should not shove, got {action!r}: {message}"
    )


# ── Before 3 raises: guard does not fire ────────────────────────────────


@pytest.mark.parametrize("raise_count", [0, 1, 2])
def test_guard_does_not_fire_early(raise_count):
    """Before 3 raise-backs, premium hands should still raise normally (not shove)."""
    result = action_for(
        ["AS", "AH"],
        raise_count=raise_count,
        facing_bet=200 if raise_count > 0 else 0,
        hero_stack=2000,
        pot=30 + raise_count * 40,
    )
    action, _amount, message = result
    # Should NOT be all-in from this guard (may be raise from other logic)
    # The guard itself returns None for raise_count < 3
    if raise_count > 0:
        assert action != "all-in" or "premium shove" not in message, (
            f"Guard should not fire before 3 raise-backs (raise_count={raise_count})"
        )


# ── All-in not available: fall back to raise=stack ──────────────────────


def test_premium_shove_fallback_to_raise():
    """If all-in is not available but raise is, raise full stack."""
    result = action_for(
        ["AS", "AH"],
        raise_count=3,
        facing_bet=200,
        hero_stack=1800,
        pot=300,
        available=["fold", "call", "raise"],
    )
    action, amount, message = result
    assert action == "raise", f"Should raise when all-in unavailable, got {action!r}"
    assert amount == 1800, f"Should raise full stack, got {amount}"
