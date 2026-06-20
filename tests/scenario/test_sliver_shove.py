"""Scenario tests for the sliver-shove exploit observed in the wild.

Reference: ``./OBSERVATION_TEST_CASE.md``.

The exploit builds a huge pot via a min-raise war preflop and then shoves the
remaining ~100 chips on the river, hoping the opponent's "call only with two
pair or better" rule fires and folds a clearly-callable made hand. The sliver
lays ~90:1 — the opponent needs only 1.1% equity to call, and *any* two live
cards clear that bar. Folding is -EV against any plausible range.

These tests lock in the defense half of the response (we do not fold
priced-in calls) and sketch the offensive half (when our bot is the shover,
the all-in still pays off when the price to villain is itself a sliver).

All tests target ``s2base.py`` via the full ``choose_action`` path — that is
the function that runs in the arena, so behavior under ``choose_action`` is
what matters in production. A handful of tests also pin
``profiled_choose_action`` behavior so leaks in the policy layer cannot
silently regress when a different policy chain takes over.
"""

from __future__ import annotations

import pytest

from poker_bot.strategies import s2base as strategy

# ── Scenario constants (from OBSERVATION_TEST_CASE.md) ────────────────────

HERO = ["Jh", "9c"]  # J9 offsuit, the canonical "undeniable" hand
BOARD = ["Js", "6d", "3c", "8h", "2s"]  # River, top pair Jacks for hero
POT = 9_000
CALL = 100  # villain's all-in: a sliver into the pot
POT_ODDS = CALL / (POT + CALL)  # ≈ 0.011, i.e. need ~1.1% equity to call


# ── Table builders ────────────────────────────────────────────────────────


def build_river_table(
    *,
    hero_cards=HERO,
    board_cards=BOARD,
    facing_bet=CALL,
    pot_chips=POT,
    hero_stack=1_000,
    villain_stack=0,
    opponent_vpip=None,
    opponent_pfr=None,
    action_history=None,
    button_seat=4,
):
    """Build a heads-up River scenario matching the OBSERVATION_TEST_CASE."""
    opponent_profiles = {}
    if opponent_vpip is not None and opponent_pfr is not None:
        vpip_value = int(round(opponent_vpip * 50))
        pfr_value = int(round(opponent_pfr * 50))
        opponent_profiles["villain"] = {
            "hands_seen": 50,
            "vpip": vpip_value,
            "pfr": pfr_value,
        }

    seats = [
        {
            "seatNumber": 1,
            "agentId": "hero",
            "stackChips": hero_stack,
            "holeCards": list(hero_cards),
            "currentBetChips": 0,
            "folded": False,
            "hasFolded": False,
        },
        {
            "seatNumber": 2,
            "agentId": "villain",
            "stackChips": villain_stack,
            "holeCards": [],
            "currentBetChips": facing_bet,
            "folded": False,
            "hasFolded": False,
        },
    ]

    return {
        "street": "River",
        "boardCards": list(board_cards),
        "potChips": pot_chips,
        "currentBet": facing_bet,
        "buttonSeatNumber": button_seat,
        "seats": seats,
        "opponentProfiles": opponent_profiles,
        "allowedActions": {
            "availableActions": ["fold", "call", "all-in"],
            "callAmount": facing_bet,
            "callChips": facing_bet,
        },
        "actionHistory": action_history if action_history is not None else [],
    }


# ── The exact OBSERVATION scenario ────────────────────────────────────────


def test_observation_exact_top_pair_calls_sliver():
    """J9o on Js 6d 3c 8h 2s facing 100 into 9,000 → call.

    Reference: OBSERVATION_TEST_CASE.md, Section 1 ("Min-raise war -> sliver
    shove"). Hero has top pair, villains sliver-shove the river. We are
    getting 90:1; the bot MUST call. A fold here forfeits a clearly
    profitable call.
    """
    table = build_river_table(
        action_history=[
            {"agentId": "villain", "action": "all-in", "street": "River"},
        ],
    )
    hero = table["seats"][0]

    action, _amount, message = strategy.choose_action(table, hero)

    assert action == "call", (
        f"FOLDED top pair for 100 into 9,000 — the sliver-shove leak. "
        f"message={message!r}"
    )


def test_observation_exact_action_history_min_raise_war_calls_sliver():
    """Same scenario with full preflop min-raise war history → call.

    Mirrors the live observation: opponents min-raise back and forth
    preflop to inflate the pot to ~9,000, then one shoves the last 100 on
    the river. The bot must call regardless of the noisy history.
    """
    table = build_river_table(
        action_history=[
            # Preflop min-raise war (each raise ~doubles the bet)
            {"agentId": "villain", "action": "raise", "street": "Preflop"},
            {"agentId": "hero", "action": "raise", "street": "Preflop"},
            {"agentId": "villain", "action": "raise", "street": "Preflop"},
            {"agentId": "hero", "action": "raise", "street": "Preflop"},
            {"agentId": "villain", "action": "raise", "street": "Preflop"},
            {"agentId": "hero", "action": "raise", "street": "Preflop"},
            {"agentId": "villain", "action": "raise", "street": "Preflop"},
            {"agentId": "hero", "action": "raise", "street": "Preflop"},
            {"agentId": "villain", "action": "raise", "street": "Preflop"},
            # Flop / turn check-check (pot stays at 9,000)
            {"agentId": "hero", "action": "check", "street": "Flop"},
            {"agentId": "villain", "action": "check", "street": "Flop"},
            {"agentId": "hero", "action": "check", "street": "Turn"},
            {"agentId": "villain", "action": "check", "street": "Turn"},
            # River sliver shove
            {"agentId": "villain", "action": "all-in", "street": "River"},
        ],
    )
    hero = table["seats"][0]

    action, _amount, message = strategy.choose_action(table, hero)

    assert action == "call", (
        f"Bot folded through a noisy min-raise-war history. message={message!r}"
    )


# ── Hand-strength sweep at the sliver price ───────────────────────────────


@pytest.mark.parametrize(
    "hero_cards, description",
    [
        (["Jh", "9c"], "top pair Jacks (J9o, the OBSERVATION hand)"),
        (["Ah", "Jc"], "top pair Jacks with A kicker"),
        (["Kc", "Js"], "top pair Jacks with K kicker"),
        (["Td", "Tc"], "pocket Tens overcards (set of Ts on the river)"),
        (["2d", "2h"], "pocket Twos → set of 2s"),
        (["Tc", "6h"], "middle pair 6s"),
        (["Jh", "2c"], "bottom pair 2s (paired board)"),
        (["7d", "7s"], "pocket 7s below top pair"),
        (["Ad", "Kd"], "two pair (Aces & Kings)"),
    ],
)
def test_any_pair_hand_calls_sliver(hero_cards, description):
    """Any pair or better must call 100 into 9,000 (≤ 2% equity needed).

    ``required = 0.011`` so we need at most 1.1% equity. Every pair has
    well above that against any reasonable sliver-shove range.
    """
    table = build_river_table(hero_cards=hero_cards)
    hero = table["seats"][0]

    action, _amount, message = strategy.choose_action(table, hero)

    assert action == "call", (
        f"{description}: folded priced-in call. required={POT_ODDS:.3f}. "
        f"message={message!r}"
    )


# ── Opponent profile sweep at the sliver price ────────────────────────────


@pytest.mark.parametrize(
    "opp_vpip, opp_pfr, profile_label",
    [
        (None, None, "no profile yet"),
        (0.55, 0.40, "loose-aggressive"),
        (0.50, 0.30, "loose-passive"),
        (0.30, 0.20, "average"),
        (0.12, 0.08, "tight"),
    ],
)
def test_top_pair_calls_sliver_against_any_profile(opp_vpip, opp_pfr, profile_label):
    """J9o top pair must call the sliver shove regardless of opponent profile.

    The price is 90:1. Even against a nit's 12% range, top pair has more
    equity than the 1.1% required.
    """
    table = build_river_table(
        opponent_vpip=opp_vpip,
        opponent_pfr=opp_pfr,
    )
    hero = table["seats"][0]

    action, _amount, message = strategy.choose_action(table, hero)

    assert action == "call", (
        f"top pair vs {profile_label}: folded the sliver. message={message!r}"
    )


# ── Action-history variants at the sliver price ───────────────────────────


@pytest.mark.parametrize(
    "history, label",
    [
        ([], "empty history"),
        (
            [{"agentId": "villain", "action": "bet", "street": "River"}],
            "river bet (current_barrel)",
        ),
        (
            [{"agentId": "villain", "action": "all-in", "street": "River"}],
            "river all-in (sliver shove)",
        ),
        (
            [
                {"agentId": "villain", "action": "bet", "street": "Flop"},
                {"agentId": "villain", "action": "bet", "street": "Turn"},
                {"agentId": "villain", "action": "bet", "street": "River"},
            ],
            "triple barrel",
        ),
        (
            [
                {"agentId": "villain", "action": "bet", "street": "Flop"},
                {"agentId": "villain", "action": "check", "street": "Turn"},
                {"agentId": "villain", "action": "all-in", "street": "River"},
            ],
            "flop barrel, turn check, river shove",
        ),
    ],
)
def test_top_pair_calls_sliver_with_any_history(history, label):
    """The price is what matters, not how villain got there.

    A "fold to a raise/all-in unless strong" rule leaks the entire pot when
    the raise happens to be a 100-chip sliver over a 9,000 pot. The bot must
    read the price, not the action shape.
    """
    table = build_river_table(action_history=history)
    hero = table["seats"][0]

    action, _amount, message = strategy.choose_action(table, hero)

    assert action == "call", (
        f"top pair vs {label}: folded the sliver. message={message!r}"
    )


# ── Price sweep at fixed hand ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "facing_bet, expected",
    [
        # Below ~10% of pot, the call is a no-brainer for any pair.
        (50, "call"),  # 0.55% pot odds
        (100, "call"),  # 1.10% (the OBSERVATION scenario)
        (200, "call"),  # 2.17%
        (500, "call"),  # 5.26%
        (900, "call"),  # 9.09%
        # Above ~25% of pot, even top pair needs to fold vs a wide range.
        (4_500, "fold"),  # 33.3% — top pair does not have 33% equity here
    ],
)
def test_top_pair_price_curve_at_river(facing_bet, expected):
    """J9o top pair has an inflection: cheap calls, expensive folds.

    The OBSERVATION specifically targets the *cheap* end of this curve.
    Locking in both endpoints prevents future fixes from overcorrecting in
    either direction.
    """
    table = build_river_table(facing_bet=facing_bet)
    hero = table["seats"][0]

    action, _amount, message = strategy.choose_action(table, hero)

    po = facing_bet / (POT + facing_bet)
    assert action == expected, (
        f"top pair facing {facing_bet}/{POT} (po={po:.3f}): "
        f"got {action}, expected {expected}. message={message!r}"
    )


# ── Negative controls ─────────────────────────────────────────────────────


def test_top_pair_does_not_call_when_price_is_truly_steep():
    """Same hand, but villain bet 4,500 into a 9,000 pot (33%).

    Even top pair does not have 33% equity vs a wide range on a Js 6d 3c 8h
    2s board. This is the control that proves the sliver-shove call is not
    just a flat "call everything" bug.
    """
    table = build_river_table(facing_bet=4_500)
    hero = table["seats"][0]

    action, _amount, message = strategy.choose_action(table, hero)

    assert action == "fold", (
        f"top pair facing 4,500/9,000 must fold. got {action}. message={message!r}"
    )


def test_no_call_when_no_facing_bet():
    """Edge: if there's no bet, the bot must not invent a call action.

    Sanity check that the sliver-shove tests aren't passing because of some
    default-call path.
    """
    table = build_river_table(facing_bet=0)
    table["allowedActions"] = {
        "availableActions": ["check"],
        "callAmount": 0,
        "callChips": 0,
    }
    hero = table["seats"][0]

    action, _amount, message = strategy.choose_action(table, hero)

    assert action in ("check", "fold"), (
        f"With no bet to face, got {action}. message={message!r}"
    )


# ── Policy-layer pinning (profiled_choose_action) ─────────────────────────
#
# ``profiled_choose_action`` is the layer that drives the policy decision.
# When the strategy runs in production it goes through ``choose_action``
# (which adds the survival defense fallback that currently saves the day
# for top-pair calls). Pinning the policy layer separately catches silent
# regressions when a different chain takes over (heads-up mode, patch1
# short-circuit, etc.).


def test_profiled_top_pair_calls_sliver_no_history():
    """profiled_choose_action must call top pair at the sliver price.

    Without any action history, ``profiled_choose_action`` is the right
    level to pin: it should not need a survival fallback to take a clearly
    priced-in call.
    """
    table = build_river_table(action_history=[])
    hero = table["seats"][0]

    action, _amount, message = strategy.profiled_choose_action(table, hero)

    assert action == "call", (
        f"profiled layer folded top pair at 1.1% pot odds. message={message!r}"
    )


def test_profiled_top_pair_calls_sliver_with_river_shove():
    """profiled_choose_action must also call when villain shoved the river.

    The bare-bones "fold medium hands to a river barrel" rule is exactly
    the leak the OBSERVATION describes. A priced-in shove (1.1% pot odds)
    must override it.
    """
    table = build_river_table(
        action_history=[{"agentId": "villain", "action": "all-in", "street": "River"}],
    )
    hero = table["seats"][0]

    action, _amount, message = strategy.profiled_choose_action(table, hero)

    assert action == "call", (
        f"profiled layer folded top pair to river sliver-shove. message={message!r}"
    )


# ── Pot-odds sanity (locks the math) ──────────────────────────────────────


def test_pot_odds_constant_matches_observation():
    """Sanity-check the math used throughout the test file.

    OBSERVATION_TEST_CASE computes ``pot_odds(call, pot) = call/(pot+call)``.
    For 100 into 9,000 that's 100/9100 ≈ 1.10%.
    """
    assert strategy.pot_odds(CALL, POT) == pytest.approx(POT_ODDS)
    assert strategy.pot_odds(CALL, POT) < 0.012  # well under 2%
    # Break-even equity = pot_odds.
    assert POT_ODDS < 0.05  # comfortably below "any two cards" ~5% equity floor
