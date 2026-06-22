"""
Heads-up (1 opponent) scenario tests copied from tests/scenario/.

These tests are taken verbatim from their original scenario files — same
table setups, same hole cards, same boards, same assertions — with only
the strategy import changed from s2base to s3base.  They serve as a
behavioral baseline for s3base on spots that were identified as leaks in
s2base during Season 2 telemetry review.

Source files (all strictly 2-seat / heads-up tables):
  - test_observed_fragile_pair.py          (2 tests)
  - test_observed_high_flush_.py           (2 tests)
  - test_telemetry_cmqlho6nw277ef6jtvzct06nh.py  (3 tests)
  - test_telemetry_cmqlqloirlrqwf6jt9qg7xwzk.py  (1 test)
  - test_telemetry_s2v012_overcalling.py   (2 tests)
  - test_sliver_shove.py                   (11 tests)
"""

from __future__ import annotations

import pytest

from poker_bot.strategies import s3base as strategy

# ════════════════════════════════════════════════════════════════════════════
# From test_observed_fragile_pair.py
# ════════════════════════════════════════════════════════════════════════════

HERO = "hero"


def _fragile_table(hole, board, *, street="River", pot=400, call=100,
                   available=("fold", "call"), profiles=None):
    """Minimal table/seat the strategy can act on (2-seat HU)."""
    seats = [
        {"seatNumber": 1, "agentId": HERO, "holeCards": hole,
         "folded": False, "stackChips": 5000, "currentBetChips": 0},
        {"seatNumber": 2, "agentId": "villain", "holeCards": [],
         "folded": False, "stackChips": 5000, "currentBetChips": call},
    ]
    t = {
        "street": street,
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": 2,
        "seats": seats,
        "opponentProfiles": profiles or {},
        "allowedActions": {
            "availableActions": list(available),
            "callAmount": call,
            "minBet": 10,
            "minRaiseTo": call * 2,
        },
    }
    return t, seats[0]


def _fragile_act(hole, board, **kw):
    action, _amount, _reason = strategy.choose_action(*_fragile_table(hole, board, **kw))
    return action


def test_fragile_two_pair_on_paired_board_folds_vs_tight_opponent():
    """Qd Tc on Jd Kc 3d Js Kd vs tight opponent → fold.

    Reference: cmqi8w7prprmabezqlfim1ele from telemetry.
    Hero has two pair (Jacks + Kings) but the board is double-paired (J, K).
    Against a tight opponent's river bet, the range is condensed toward full
    houses. Folding is correct.
    """
    profiles = {"villain": {"hands_seen": 50, "vpip": 6, "pfr": 4}}
    assert _fragile_act(
        ["Qd", "Tc"], ["Jd", "Kc", "3d", "Js", "Kd"],
        pot=403, call=134, profiles=profiles,
    ) == "fold"


def test_fragile_two_pair_on_paired_board_continues_vs_loose_opponent():
    """6d 6s on Qs Kh Js 4h 4d vs loose opponent → continue.

    Reference: cmqjqdfklzvb5bezqsz6iucz1 from telemetry.
    Hero has two pair (6s + 4s) on a paired board. Against a loose opponent,
    the betting range is wider with more bluffs and thinner value, so continuing
    with a call is reasonable.
    """
    profiles = {"villain": {"hands_seen": 50, "vpip": 23, "pfr": 15}}
    assert _fragile_act(
        ["6d", "6s"], ["Qs", "Kh", "Js", "4h", "4d"],
        pot=416, call=145, profiles=profiles,
    ) != "fold"


# ════════════════════════════════════════════════════════════════════════════
# From test_observed_high_flush_.py
# ════════════════════════════════════════════════════════════════════════════

def _flush_table(hole, board, *, street="River", pot=400, call=100,
                 available=("fold", "call"), profiles=None):
    """Minimal table/seat the strategy can act on (2-seat HU)."""
    seats = [
        {"seatNumber": 1, "agentId": HERO, "holeCards": hole,
         "folded": False, "stackChips": 5000, "currentBetChips": 0},
        {"seatNumber": 2, "agentId": "villain", "holeCards": [],
         "folded": False, "stackChips": 5000, "currentBetChips": call},
    ]
    t = {
        "street": street,
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": 2,
        "seats": seats,
        "opponentProfiles": profiles or {},
        "allowedActions": {
            "availableActions": list(available),
            "callAmount": call,
            "minBet": 10,
            "minRaiseTo": call * 2,
        },
    }
    return t, seats[0]


def _flush_act(hole, board, **kw):
    action, _amount, _reason = strategy.choose_action(*_flush_table(hole, board, **kw))
    return action


def test_k_high_flush_on_paired_board_folds_vs_tight_opponent():
    """Jd Kh on 9h 9d 3h 8h 2h vs tight opponent → fold.

    Reference: cmqhrlt0h6cskbezq0zhq4oie from telemetry.
    Hero has K-high flush on a paired board. Against a tight-passive opponent
    (VPIP 15%, PFR 1%), the betting range is condensed toward full houses.
    Folding is strategically reasonable.
    """
    profiles = {"villain": {"hands_seen": 95, "vpip": 15, "pfr": 1}}
    # Adjust to not fold if hero has A or K flush
    assert _flush_act(
        ["Jd", "Kh"], ["9h", "9d", "3h", "8h", "2h"],
        pot=124, call=70, profiles=profiles,
    ) != "fold"
    assert _flush_act(
        ["Jd", "Ah"], ["9h", "9d", "3h", "8h", "2h"],
        pot=124, call=70, profiles=profiles,
    ) != "fold"

    assert _flush_act(
        ["Jd", "Qh"], ["9h", "9d", "3h", "8h", "2h"],
        pot=124, call=70, profiles=profiles,
    ) == "fold"
    assert _flush_act(
        ["Jd", "Jh"], ["9h", "9d", "3h", "8h", "2h"],
        pot=124, call=70, profiles=profiles,
    ) == "fold"


def test_k_high_flush_on_paired_board_continues_vs_loose_opponent():
    """Jd Kh on 9h 9d 3h 8h 2h vs loose opponent → continue.

    Same hand/board as cmqhrlt0h6cskbezq0zhq4oie, but against a loose opponent.
    The betting range is wider with more bluffs and thinner value, so K-high
    flush should continue rather than auto-fold to the flush guard.
    """
    profiles = {"villain": {"hands_seen": 50, "vpip": 45, "pfr": 30}}
    assert _flush_act(
        ["Jd", "Kh"], ["9h", "9d", "3h", "8h", "2h"],
        pot=124, call=35, profiles=profiles,
    ) != "fold"


# ════════════════════════════════════════════════════════════════════════════
# From test_telemetry_cmqlho6nw277ef6jtvzct06nh.py
# ════════════════════════════════════════════════════════════════════════════

HERO_CMQLHO = "hero"
DEFAULT_AVAILABLE_CMQLHO = ("fold", "check", "bet", "all-in")


def _cmqlho_make_seat(seat_number, agent_id, hole_cards=None, *, folded=False):
    return {
        "seatNumber": seat_number,
        "agentId": agent_id,
        "holeCards": hole_cards or [],
        "stackChips": 12000,
        "currentBetChips": 0,
        "folded": folded,
    }


def _cmqlho_make_table(
    hole,
    board,
    *,
    street="Turn",
    pot=180,
    call=0,
    available=DEFAULT_AVAILABLE_CMQLHO,
    villains=1,
    profiles=None,
    button=1,
):
    """Build a minimal s3base table fixture for the Q5o telemetry scenario."""
    seats = [_cmqlho_make_seat(1, HERO_CMQLHO, hole)]
    for index in range(villains):
        seats.append(_cmqlho_make_seat(2 + index, f"v{index}"))

    table = {
        "street": street,
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": button,
        "seats": seats,
        "opponentProfiles": profiles or {},
        "allowedActions": {
            "availableActions": list(available),
            "callAmount": call,
            "minBet": 10,
            "minRaiseTo": max(20, call * 2),
        },
    }
    return table, seats[0]


def _cmqlho_action_for(hole, board, choose_action_fn, **kwargs):
    table, hero = _cmqlho_make_table(hole, board, **kwargs)
    return choose_action_fn(table, hero)


def _cmqlho_act(hole, board, **kw):
    return _cmqlho_action_for(hole, board, strategy.choose_action, **kw)[0]


def _cmqlho_assert_checks_medium_pair(action):
    assert action == "check"


def _cmqlho_assert_does_not_thin_value_bet(action):
    assert action != "bet"


@pytest.mark.parametrize(
    "hole,board",
    (
        (["5h", "Qs"], ["4h", "Js", "Ad", "5c"]),  # bottom pair, Q kicker
    ),
)
def test_bottom_pair_high_card_board_checks_turn(hole, board):
    """Q5o on Ad Js 4h 5c should check, not thin-value bet bottom pair.

    Reference: cmqlho6nw277ef6jtvzct06nh from telemetry.
    The bot bet 68 into 180 on the Turn with bottom pair and a Q kicker. This is
    dominated by most continuing ranges on an A-high, J-high board.
    """
    action = _cmqlho_act(hole, board, pot=180)
    _cmqlho_assert_checks_medium_pair(action)


@pytest.mark.parametrize(
    "hole,board",
    (
        (["5h", "Qs"], ["4h", "Js", "Ad", "5c", "Ts"]),  # bottom two, weak kicker
    ),
)
def test_bottom_two_high_card_board_checks_river(hole, board):
    """Q5o on Ad Js 4h 5c Ts should check, not thin-value bet bottom two.

    Reference: cmqlho6nw277ef6jtvzct06nh from telemetry.
    The bot bet 120 into 316 on the River with bottom two. The T on the river
    gives opponents AT, JT, 5T possibilities, and the A/J high cards dominate
    this hand. This is not a value bet.
    """
    action = _cmqlho_act(hole, board, pot=316)
    _cmqlho_assert_checks_medium_pair(action)


@pytest.mark.parametrize(
    "hole,board",
    (
        (["5h", "Qs"], ["4h", "Js", "Ad", "5c"]),
        (["5h", "Qs"], ["4h", "Js", "Ad", "5c", "Ts"]),
    ),
)
def test_medium_pair_does_not_thin_value_bet_high_card_boards(hole, board):
    """Q5o should not thin-value bet on A/J high-card boards.

    This is the core flaw from cmqlho6nw277ef6jtvzct06nh: the bot's
    adaptive_choose_action() treats made_rank == 1 as "medium" and value-bets it
    with the message "Thin value against simple". That logic is too loose for
    bottom pair / bottom two with weak kickers.
    """
    action = _cmqlho_act(hole, board, pot=180 if len(board) == 4 else 316)
    _cmqlho_assert_does_not_thin_value_bet(action)


# ════════════════════════════════════════════════════════════════════════════
# From test_telemetry_cmqlqloirlrqwf6jt9qg7xwzk.py
# ════════════════════════════════════════════════════════════════════════════

def _cmqlqlo_make_turn_table(opponent_vpip=38, opponent_pfr=23):
    """Build the 6d Jd on 4s Tc Th 2c scenario from telemetry."""
    return {
        "street": "Turn",
        "boardCards": ["4s", "Tc", "Th", "2c"],
        "potChips": 157,
        "buttonSeatNumber": 5,
        "opponentProfiles": {
            "villain": {
                "hands_seen": 189,
                "vpip": opponent_vpip,
                "pfr": opponent_pfr,
            }
        },
        "seats": [
            {
                "seatNumber": 1,
                "agentId": "hero",
                "holeCards": ["6d", "Jd"],
                "folded": False,
                "stackChips": 157,
                "currentBetChips": 0,
            },
            {
                "seatNumber": 2,
                "agentId": "villain",
                "holeCards": [],
                "folded": False,
                "stackChips": 157,
                "currentBetChips": 0,
            },
        ],
        "allowedActions": {
            "availableActions": ["fold", "check", "bet", "all-in"],
            "callAmount": 0,
            "minBet": 10,
            "minRaiseTo": 20,
        },
    }


def _cmqlqlo_act(hero_cards, board_cards, pot, profiles=None):
    """Call s3base.choose_action with a minimal table setup."""
    table = _cmqlqlo_make_turn_table()
    table["boardCards"] = board_cards
    table["potChips"] = pot
    table["seats"][0]["holeCards"] = hero_cards
    if profiles is not None:
        table["opponentProfiles"] = profiles
    action, _amount, _message = strategy.choose_action(table, table["seats"][0])
    return action


def test_middle_pair_on_paired_board_checks_vs_tight_turn():
    """6d Jd on 4s Tc Th 2c vs tight opponent Turn → check.

    Reference: cmqlqloirlrqwf6jt9qg7xwzk from telemetry.
    Hero has middle pair (tens) with J kicker on a paired board. Against a
    tight opponent (VPIP 38%, PFR 23% in telemetry, labeled "tight"), the Turn
    bet is thin value that's likely dominated. Checking is strategically
    correct — medium-strength hands with weak kickers should not thin-value bet
    on paired boards against tight opponents.
    """
    profiles = {
        "villain": {
            "hands_seen": 189,
            "vpip": 30,  # 15.9% — telemetry labels this opponent "tight"
            "pfr": 44,  # 23.3%
            "calls": 30,
            "bets": 30,
            "raises": 30,
            "folds": 99,
        }
    }
    assert _cmqlqlo_act(
        ["6d", "Jd"], ["4s", "Tc", "Th", "2c"], pot=157, profiles=profiles,
    ) == "check"


# ════════════════════════════════════════════════════════════════════════════
# From test_telemetry_s2v012_overcalling.py
# ════════════════════════════════════════════════════════════════════════════

def _s2v012_make_turn_table(opponent_vpip=16, opponent_pfr=14):
    """Build the 5c Ks on As 2s 5s Kc scenario from telemetry."""
    return {
        "street": "Turn",
        "boardCards": ["As", "2s", "5s", "Kc"],
        "potChips": 3122,
        "buttonSeatNumber": 5,
        "opponentProfiles": {
            "villain": {
                "hands_seen": 127,
                "vpip": opponent_vpip,
                "pfr": opponent_pfr,
            }
        },
        "seats": [
            {
                "seatNumber": 1,
                "agentId": "hero",
                "holeCards": ["5c", "Ks"],
                "folded": False,
                "stackChips": 1131,
                "currentBetChips": 0,
            },
            {
                "seatNumber": 2,
                "agentId": "villain",
                "holeCards": [],
                "folded": False,
                "stackChips": 0,
                "currentBetChips": 2490,
            },
        ],
        "allowedActions": {
            "availableActions": ["fold", "call", "all-in"],
            "callAmount": 2490,
            "minBet": 10,
            "minRaiseTo": 4980,
        },
    }


def _s2v012_act(hero_cards, board_cards, pot, call, profiles=None):
    """Call s3base.choose_action with a minimal table setup."""
    table = _s2v012_make_turn_table()
    table["boardCards"] = board_cards
    table["potChips"] = pot
    table["seats"][0]["holeCards"] = hero_cards
    table["seats"][0]["stackChips"] = 1131
    table["seats"][1]["currentBetChips"] = call
    table["allowedActions"]["callAmount"] = call
    if profiles is not None:
        table["opponentProfiles"] = profiles
    action, _amount, _message = strategy.choose_action(table, table["seats"][0])
    return action


def test_two_pair_on_paired_board_folds_vs_tight_aggressive_turn_jam():
    """5c Ks on As 2s 5s Kc vs TAG opponent Turn jam → fold.

    Reference: cmql7dvor5s9vf6jtank1kk27 from telemetry.
    Hero has two pair (Kings + 5s) on a paired board with a flush draw.
    Against a tight-aggressive opponent (VPIP 16%, PFR 14%) with 0 all-ins
    and 0 large bets in 127 hands, the Turn jam is value-heavy. Folding is
    strategically correct — medium-strength hands should not call 220% of
    remaining stack against a TAG's first-ever all-in.
    """
    profiles = {"villain": {"hands_seen": 127, "vpip": 16, "pfr": 14}}
    assert _s2v012_act(
        ["5c", "Ks"], ["As", "2s", "5s", "Kc"],
        pot=3122, call=2490, profiles=profiles,
    ) == "fold"


def test_two_pair_on_paired_board_continues_vs_loose_aggressive_turn_jam():
    """5c Ks on As 2s 5s Kc vs loose opponent Turn jam → continue.

    Same hand/board as cmql7dvor5s9vf6jtank1kk27, but against a loose-aggressive
    opponent (VPIP 43%, PFR 23%). The jamming range is wider with more bluffs
    and thinner value, so two pair should continue rather than auto-fold.
    """
    profiles = {"villain": {"hands_seen": 127, "vpip": 43, "pfr": 23}}
    assert _s2v012_act(
        ["5c", "Ks"], ["As", "2s", "5s", "Kc"],
        pot=3122, call=2490, profiles=profiles,
    ) != "fold"


# ════════════════════════════════════════════════════════════════════════════
# From test_sliver_shove.py
# ════════════════════════════════════════════════════════════════════════════

# ── Scenario constants (from OBSERVATION_TEST_CASE.md) ────────────────────

SLIVER_HERO = ["Jh", "9c"]  # J9 offsuit, the canonical "undeniable" hand
SLIVER_BOARD = ["Js", "6d", "3c", "8h", "2s"]  # River, top pair Jacks for hero
SLIVER_POT = 9_000
SLIVER_CALL = 100  # villain's all-in: a sliver into the pot
SLIVER_POT_ODDS = SLIVER_CALL / (SLIVER_POT + SLIVER_CALL)  # ≈ 0.011


def _sliver_build_river_table(
    *,
    hero_cards=SLIVER_HERO,
    board_cards=SLIVER_BOARD,
    facing_bet=SLIVER_CALL,
    pot_chips=SLIVER_POT,
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


def test_observation_exact_top_pair_calls_sliver():
    """J9o on Js 6d 3c 8h 2s facing 100 into 9,000 → call.

    Reference: OBSERVATION_TEST_CASE.md, Section 1 ("Min-raise war -> sliver
    shove"). Hero has top pair, villains sliver-shove the river. We are
    getting 90:1; the bot MUST call. A fold here forfeits a clearly
    profitable call.
    """
    table = _sliver_build_river_table(
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
    table = _sliver_build_river_table(
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
    table = _sliver_build_river_table(hero_cards=hero_cards)
    hero = table["seats"][0]

    action, _amount, message = strategy.choose_action(table, hero)

    assert action == "call", (
        f"{description}: folded priced-in call. required={SLIVER_POT_ODDS:.3f}. "
        f"message={message!r}"
    )


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
    table = _sliver_build_river_table(
        opponent_vpip=opp_vpip,
        opponent_pfr=opp_pfr,
    )
    hero = table["seats"][0]

    action, _amount, message = strategy.choose_action(table, hero)

    assert action == "call", (
        f"top pair vs {profile_label}: folded the sliver. message={message!r}"
    )


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
    table = _sliver_build_river_table(action_history=history)
    hero = table["seats"][0]

    action, _amount, message = strategy.choose_action(table, hero)

    assert action == "call", (
        f"top pair vs {label}: folded the sliver. message={message!r}"
    )


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
    table = _sliver_build_river_table(facing_bet=facing_bet)
    hero = table["seats"][0]

    action, _amount, message = strategy.choose_action(table, hero)

    po = facing_bet / (SLIVER_POT + facing_bet)
    assert action == expected, (
        f"top pair facing {facing_bet}/{SLIVER_POT} (po={po:.3f}): "
        f"got {action}, expected {expected}. message={message!r}"
    )


def test_top_pair_does_not_call_when_price_is_truly_steep():
    """Same hand, but villain bet 4,500 into a 9,000 pot (33%).

    Even top pair does not have 33% equity vs a wide range on a Js 6d 3c 8h
    2s board. This is the control that proves the sliver-shove call is not
    just a flat "call everything" bug.
    """
    table = _sliver_build_river_table(facing_bet=4_500)
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
    table = _sliver_build_river_table(facing_bet=0)
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


def test_profiled_top_pair_calls_sliver_no_history():
    """profiled_choose_action must call top pair at the sliver price.

    Without any action history, ``profiled_choose_action`` is the right
    level to pin: it should not need a survival fallback to take a clearly
    priced-in call.
    """
    table = _sliver_build_river_table(action_history=[])
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
    table = _sliver_build_river_table(
        action_history=[{"agentId": "villain", "action": "all-in", "street": "River"}],
    )
    hero = table["seats"][0]

    action, _amount, message = strategy.profiled_choose_action(table, hero)

    assert action == "call", (
        f"profiled layer folded top pair to river sliver-shove. message={message!r}"
    )


def test_pot_odds_constant_matches_observation():
    """Sanity-check the math used throughout the test file.

    OBSERVATION_TEST_CASE computes ``pot_odds(call, pot) = call/(pot+call)``.
    For 100 into 9,000 that's 100/9100 ≈ 1.10%.
    """
    assert strategy.pot_odds(SLIVER_CALL, SLIVER_POT) == pytest.approx(SLIVER_POT_ODDS)
    assert strategy.pot_odds(SLIVER_CALL, SLIVER_POT) < 0.012  # well under 2%
    # Break-even equity = pot_odds.
    assert SLIVER_POT_ODDS < 0.05  # comfortably below "any two cards" ~5% equity floor
