"""SB preflop defend widening for suited hands — hubase fix.

In HU, the hero is always the button (SB). The SB was folding suited
hands at score 35-39 because the guard required wide_defense_ok
(profiled loose raiser), which is always False in selfplay.

Fix: add a score-based SB defend path that fires when score >= 35
and pot odds <= 40%, without requiring opponent profile.
"""

import pytest

from poker_bot.strategies.hubase import choose_action, preflop_score

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


def make_table(
    hole,
    *,
    call=5,
    current_bet=5,
    pot=10,
    available=("fold", "call", "raise", "all-in"),
    button=1,
    hero_seat=1,
):
    """Build a preflop table. Hero is always the button (SB) in HU."""
    seats = [make_seat(hero_seat, HERO, hole)]
    seats.append(make_seat(2 if hero_seat == 1 else 1, "villain"))

    table = {
        "street": "Preflop",
        "boardCards": [],
        "potChips": pot,
        "currentBet": current_bet,
        "buttonSeatNumber": button,
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


def action_for(hole, **kwargs):
    table, hero = make_table(hole, **kwargs)
    return choose_action(table, hero)[0]


# ────────────────────────────────────────────────────────────────────
# SB with suited hand at score 35-39 facing raise → call
# ────────────────────────────────────────────────────────────────────

def test_sb_suited_score36_facing_raise_calls():
    """5C 6C (score 36) from SB facing a raise → call."""
    assert preflop_score(["5C", "6C"]) == 36
    action = action_for(["5C", "6C"], call=5, current_bet=5, pot=10)
    assert action == "call", f"SB suited score 36 should call, got {action!r}"


def test_sb_suited_score35_facing_raise_calls():
    """3S 8S (score 35) from SB facing a raise → call."""
    assert preflop_score(["3S", "8S"]) == 35
    action = action_for(["3S", "8S"], call=5, current_bet=5, pot=10)
    assert action == "call", f"SB suited score 35 should call, got {action!r}"


def test_sb_suited_score39_facing_raise_calls():
    """9H 4H (score 39) from SB facing a raise → call."""
    assert preflop_score(["9H", "4H"]) == 39
    action = action_for(["9H", "4H"], call=5, current_bet=5, pot=10)
    assert action == "call", f"SB suited score 39 should call, got {action!r}"


def test_sb_suited_score37_facing_raise_calls():
    """2C 9C (score 37) from SB facing a raise → call."""
    assert preflop_score(["2C", "9C"]) == 37
    action = action_for(["2C", "9C"], call=5, current_bet=5, pot=10)
    assert action == "call", f"SB suited score 37 should call, got {action!r}"


# ────────────────────────────────────────────────────────────────────
# SB with junk below threshold → fold
# ────────────────────────────────────────────────────────────────────

def test_sb_junk_score30_facing_raise_folds():
    """2C 7D (score ~30) from SB facing a raise → fold."""
    score = preflop_score(["2C", "7D"])
    assert score < 35, f"Expected score < 35, got {score}"
    action = action_for(["2C", "7D"], call=5, current_bet=5, pot=10)
    assert action == "fold", f"SB junk score {score} should fold, got {action!r}"


# ────────────────────────────────────────────────────────────────────
# SB with suited hand facing expensive raise → fold
# ────────────────────────────────────────────────────────────────────

def test_sb_suited_score36_facing_expensive_raise_folds():
    """5C 6C (score 36) from SB facing 50% pot raise → fold."""
    # Large raise: pot=10, call=10 → 50% pot odds > 40% threshold
    action = action_for(["5C", "6C"], call=10, current_bet=10, pot=10)
    assert action == "fold", f"SB facing expensive raise should fold, got {action!r}"


# ────────────────────────────────────────────────────────────────────
# SB with suited hand at score 40+ → call (already worked before)
# ────────────────────────────────────────────────────────────────────

def test_sb_suited_score40_facing_raise_calls():
    """TC 9C (score 52 actually, but let's use a real score 40 hand)."""
    # 8S 9H = 8*3+9+8(suited)+5(consecutive) = 46. Hmm.
    # Let me find a score 40 hand: 9C 5C = 9*3+5+8 = 40
    score = preflop_score(["9C", "5C"])
    assert score == 40, f"Expected score 40, got {score}"
    action = action_for(["9C", "5C"], call=5, current_bet=5, pot=10)
    assert action == "call", f"SB suited score 40 should call, got {action!r}"
