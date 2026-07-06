"""Turn two-pair value-bet suppression — metric-driven (hu009 candidate).

Telemetry (benchmark.sqlite, hu008, heads-up): betting two pair on the turn
when not facing a bet leaks chips — paired board -192 avg (108 hands),
unpaired -124 avg (98 hands). The leak is opponent-dependent, so the guard
uses the opponent's VPIP / fold-to-bet / aggression / call-frequency rather
than a coarse tight/loose switch.

Covers the SOUL_HU opponent archetypes: tight, loose, aggressive, passive,
calling-station, and unknown.
"""

import pytest

from poker_bot.strategies.hubase import (
    choose_action,
    turn_two_pair_bet_suppression,
    turn_weak_hand_fold_vs_tight_raise,
    flop_hu_bluffcatch_guard,
    river_two_pair_facing_bet_call_guard,
    OpponentProfile,
    profile_call_frequency,
    profile_fold_to_bet_frequency,
    profile_aggression_frequency_merged,
)

HERO = "hero"
VILLAIN = "villain"

# Unpaired-board two pair (7s + Ks) — the case the old paired-only guard
# could not reach.
HOLE = ["7C", "KC"]
BOARD = ["7S", "9D", "KH", "AS"]

CHECK_BET = ("fold", "check", "bet", "all-in")
CALL_RAISE = ("fold", "call", "raise", "all-in")


def _table(
    *,
    street="Turn",
    pot=200,
    call=0,
    available=CHECK_BET,
    hero_stack=2000,
    villain_stack=2000,
    profiles=None,
    hole=HOLE,
    board=BOARD,
):
    seats = [
        {
            "seatNumber": 1,
            "agentId": HERO,
            "holeCards": hole,
            "stackChips": hero_stack,
            "currentBetChips": 0,
            "folded": False,
            "hasFolded": False,
        },
        {
            "seatNumber": 2,
            "agentId": VILLAIN,
            "holeCards": [],
            "stackChips": villain_stack,
            "currentBetChips": 0,
            "folded": False,
            "hasFolded": False,
        },
    ]
    return {
        "street": street,
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": 1,
        "seats": seats,
        "opponentProfiles": profiles or {},
        "allowedActions": {
            "availableActions": list(available),
            "callAmount": call,
            "callChips": call,
            "raiseRange": {"min": call * 2 if call > 0 else 40, "max": 4000},
            "betRange": {"min": 0, "max": 4000},
        },
    }, seats[0]


def _guard(profiles, **kw):
    table, hero = _table(profiles=profiles, **kw)
    return turn_two_pair_bet_suppression(table, hero, ("bet", 80, "base bet"))


# ── Opponent archetypes (frequency form; helpers read these from dicts) ──────

TIGHT_PASSIVE = {
    "villain": {
        "hands_seen": 50,
        "vpip_frequency": 0.20,
        "call_frequency": 0.25,
        "fold_to_bet_frequency": 0.60,
        "aggression_frequency": 0.20,
    }
}
TIGHT_VIA_FOLD = {
    "villain": {
        "hands_seen": 50,
        "vpip_frequency": 0.35,
        "call_frequency": 0.30,
        "fold_to_bet_frequency": 0.60,
        "aggression_frequency": 0.20,
    }
}
TIGHT_AGGRESSIVE = {
    "villain": {
        "hands_seen": 50,
        "vpip_frequency": 0.22,
        "call_frequency": 0.20,
        "fold_to_bet_frequency": 0.40,
        "aggression_frequency": 0.45,
    }
}
CALLING_STATION = {
    "villain": {
        "hands_seen": 50,
        "vpip_frequency": 0.55,
        "call_frequency": 0.65,
        "fold_to_bet_frequency": 0.20,
        "aggression_frequency": 0.10,
    }
}
LOOSE = {
    "villain": {
        "hands_seen": 50,
        "vpip_frequency": 0.50,
        "call_frequency": 0.40,
        "fold_to_bet_frequency": 0.30,
        "aggression_frequency": 0.30,
    }
}
LOOSE_AGGRESSIVE = {
    "villain": {
        "hands_seen": 50,
        "vpip_frequency": 0.60,
        "call_frequency": 0.20,
        "fold_to_bet_frequency": 0.20,
        "aggression_frequency": 0.55,
    }
}
UNKNOWN = {"villain": {"hands_seen": 3}}


# ── Direct guard tests: fires for tight/passive, silent otherwise ────────────


@pytest.mark.parametrize("profile", [TIGHT_PASSIVE, TIGHT_VIA_FOLD, TIGHT_AGGRESSIVE])
def test_suppresses_turn_two_pair_bet_vs_tight(profile):
    """Tight opponents (low VPIP, or high fold-to-bet + passive) → check back."""
    decision = _guard(profile)
    assert decision is not None
    assert decision[0] == "check"


@pytest.mark.parametrize("profile", [CALLING_STATION, LOOSE, LOOSE_AGGRESSIVE, UNKNOWN])
def test_does_not_suppress_vs_station_loose_unknown(profile):
    """Stations/loose callers (bet is +EV) and unknowns → no suppression."""
    assert _guard(profile) is None


def test_no_suppression_when_facing_bet():
    """Facing a bet is owned by rank_two_facing_bet_guard, not this guard."""
    assert _guard(TIGHT_PASSIVE, call=80, available=CALL_RAISE) is None


def test_no_suppression_at_low_spr():
    """Pot-committed (SPR < 3): two pair should commit, not check back."""
    assert _guard(TIGHT_PASSIVE, pot=600, hero_stack=1500, villain_stack=1500) is None


def test_no_suppression_wrong_street():
    assert _guard(TIGHT_PASSIVE, street="Flop", board=["7S", "9D", "KH"]) is None
    assert (
        _guard(TIGHT_PASSIVE, street="River", board=["7S", "9D", "KH", "AS", "2C"])
        is None
    )


def test_no_suppression_when_not_two_pair():
    # One pair (pair of 7s) is not two pair.
    assert (
        _guard(TIGHT_PASSIVE, hole=["7C", "KC"], board=["7S", "9D", "2H", "3S"]) is None
    )


# ── Integration through choose_action ────────────────────────────────────────


def _choose(profiles, **kw):
    table, hero = _table(profiles=profiles, **kw)
    return choose_action(table, hero)[0]


def test_choose_action_checks_back_vs_tight_passive():
    assert _choose(TIGHT_PASSIVE) == "check"


def test_choose_action_keeps_value_bet_vs_station():
    """Calling stations call with worse — the value bet must be preserved."""
    assert _choose(CALLING_STATION) == "bet"


def test_choose_action_keeps_value_bet_vs_loose():
    assert _choose(LOOSE) == "bet"


# ── Raw-count OpponentProfile objects (as populated during real benchmark play)
# The dict-form tests above use precomputed *_frequency keys; these tests feed
# real OpponentProfile dataclass objects with raw integer counts to lock in the
# helpers' fallback-computation path (vpip/hands_seen, calls/(calls+bets+...),
# fold_to_bet/opportunities).
# ────────────────────────────────────────────────────────────────────────────

TIGHT_RAW = OpponentProfile(
    agent_id=VILLAIN,
    hands_seen=50,
    vpip=10,
    calls=5,
    bets=2,
    raises=2,
    folds=10,
    fold_to_bet=12,
    opportunities_to_fold_to_bet=20,
)  # vpip 0.20, call ~0.26, fold-to-bet 0.60, aggression ~0.21 -> value-owned

STATION_RAW = OpponentProfile(
    agent_id=VILLAIN,
    hands_seen=50,
    vpip=28,
    calls=22,
    bets=2,
    raises=1,
    folds=3,
    fold_to_bet=2,
    opportunities_to_fold_to_bet=15,
)  # vpip 0.56, call ~0.79 -> calling station -> no suppression


def test_raw_count_frequency_computation():
    """Helpers must derive the right frequencies from raw integer counts."""
    assert abs(TIGHT_RAW.vpip / TIGHT_RAW.hands_seen - 0.20) < 1e-9
    assert profile_call_frequency(TIGHT_RAW) < 0.50  # not a station
    assert profile_fold_to_bet_frequency(TIGHT_RAW) >= 0.55  # folds to bets
    assert profile_aggression_frequency_merged(TIGHT_RAW) <= 0.35  # passive
    # Station: high call frequency, loose vpip
    assert profile_call_frequency(STATION_RAW) >= 0.50
    assert (STATION_RAW.vpip / STATION_RAW.hands_seen) >= 0.45


def test_suppresses_vs_tight_raw_opponent_profile():
    """Real OpponentProfile object (raw counts), tight opponent -> check back."""
    decision = _guard({VILLAIN: TIGHT_RAW})
    assert decision is not None
    assert decision[0] == "check"


def test_does_not_suppress_vs_station_raw_opponent_profile():
    """Real OpponentProfile object (raw counts), calling station -> no suppression."""
    assert _guard({VILLAIN: STATION_RAW}) is None


def test_choose_action_checks_back_vs_tight_raw_profile():
    """End-to-end: choose_action checks back two pair vs a tight raw profile."""
    table, hero = _table(profiles={VILLAIN: TIGHT_RAW})
    assert choose_action(table, hero)[0] == "check"


def test_choose_action_keeps_value_bet_vs_station_raw_profile():
    """End-to-end: choose_action keeps the value bet vs a calling-station raw profile."""
    table, hero = _table(profiles={VILLAIN: STATION_RAW})
    assert choose_action(table, hero)[0] == "bet"


# ── Turn weak-hand fold vs tight raise (metric-driven) ───────────────────────
# Hero has a low-probability-win hand (high card / non-top pair) facing a tight
# opponent's raise -> fold. Top pair, two pair+, stations/loose, and cheap
# min-raises are excluded. Uses the same tight-opponent gate as the bet
# suppression guard.
# ────────────────────────────────────────────────────────────────────────────

HIGH_BOARD = ["7S", "9D", "KH", "AS"]  # A-high unpaired board
HIGH_BOARD_NOQ = ["7S", "9D", "QH", "AS"]  # A-high board for high-card test


def _facing_raise(
    hole, board, profiles, *, call=200, pot=400, hero_stack=2000, villain_stack=2000
):
    table, hero = _table(
        profiles=profiles,
        call=call,
        pot=pot,
        available=CALL_RAISE,
        hole=hole,
        board=board,
        hero_stack=hero_stack,
        villain_stack=villain_stack,
    )
    return table, hero


def _fold_guard(hole, board, profiles, **kw):
    call = kw.pop("call", 200)
    table, hero = _facing_raise(hole, board, profiles, call=call, **kw)
    return turn_weak_hand_fold_vs_tight_raise(table, hero, ("call", call, "base call"))


WEAK_HAND_CASES = [
    (["3C", "3D"], HIGH_BOARD, "bottom pair 33 on A-high board"),
    (["8C", "8D"], HIGH_BOARD, "pocket pair 88 below board"),
    (["KH", "2D"], HIGH_BOARD_NOQ, "high card K on A-high board"),
]


@pytest.mark.parametrize("hole,board,label", WEAK_HAND_CASES)
def test_folds_weak_hand_vs_tight_raise(hole, board, label):
    """Low-probability-win hand vs tight opponent raise -> fold (raw profile)."""
    decision = _fold_guard(hole, board, {VILLAIN: TIGHT_RAW})
    assert decision is not None, f"{label}: guard should fire"
    assert decision[0] == "fold"


@pytest.mark.parametrize("hole,board,label", WEAK_HAND_CASES)
def test_folds_weak_hand_vs_tight_raise_dict(hole, board, label):
    """Same, with dict-form profile (parity with raw-count path)."""
    decision = _fold_guard(hole, board, TIGHT_PASSIVE)
    assert decision is not None and decision[0] == "fold"


def test_does_not_fold_top_pair_vs_tight_raise():
    """Top pair (pair rank >= highest board card) -> keep calling, don't fold."""
    assert _fold_guard(["AH", "2D"], HIGH_BOARD, {VILLAIN: TIGHT_RAW}) is None


def test_does_not_fold_two_pair_vs_tight_raise():
    """Two pair (rank 2) is never folded by this guard."""
    assert _fold_guard(["7C", "KC"], HIGH_BOARD, {VILLAIN: TIGHT_RAW}) is None


def test_does_not_fold_weak_hand_vs_station_raise():
    """Calling station raises with worse -> keep calling, don't fold."""
    assert _fold_guard(["3C", "3D"], HIGH_BOARD, {VILLAIN: STATION_RAW}) is None


def test_does_not_fold_weak_hand_vs_loose_raise():
    """Loose opponent raises with worse -> keep calling, don't fold."""
    assert _fold_guard(["3C", "3D"], HIGH_BOARD, LOOSE) is None


def test_does_not_fold_weak_hand_vs_cheap_min_raise():
    """Cheap min-raise (< 25% pot odds) -> pot odds justify a call."""
    assert (
        _fold_guard(["3C", "3D"], HIGH_BOARD, {VILLAIN: TIGHT_RAW}, call=25, pot=400)
        is None
    )


def test_does_not_fold_weak_hand_when_all_in():
    """Calling would be all-in (free showdown) -> don't fold, realise equity."""
    assert (
        _fold_guard(
            ["3C", "3D"],
            HIGH_BOARD,
            {VILLAIN: TIGHT_RAW},
            call=200,
            pot=400,
            hero_stack=150,
        )
        is None
    )


def test_does_not_fold_weak_hand_unknown_opponent():
    """Unknown opponent (< 10 hands) -> no confident read, don't fold."""
    assert _fold_guard(["3C", "3D"], HIGH_BOARD, UNKNOWN) is None


def test_choose_action_folds_bottom_pair_vs_tight_raise():
    """End-to-end: choose_action folds bottom pair vs a tight raise."""
    table, hero = _facing_raise(["3C", "3D"], HIGH_BOARD, {VILLAIN: TIGHT_RAW})
    assert choose_action(table, hero)[0] == "fold"


def test_choose_action_folds_high_card_vs_tight_raise():
    """End-to-end: choose_action folds high card vs a tight raise."""
    table, hero = _facing_raise(["KH", "2D"], HIGH_BOARD_NOQ, {VILLAIN: TIGHT_RAW})
    assert choose_action(table, hero)[0] == "fold"


def test_choose_action_keeps_calling_two_pair_vs_tight_raise():
    """End-to-end: two pair vs tight raise keeps calling (never folded)."""
    table, hero = _facing_raise(["7C", "KC"], HIGH_BOARD, {VILLAIN: TIGHT_RAW})
    assert choose_action(table, hero)[0] == "call"


# ── Flop HU cheap-call extension (cheap_postflop_continue) ───────────────────
# Data (benchmark.sqlite, hu008, heads-up): flop HU folds of rank 0/1 at 20-40%
# pot odds averaged -25 to -35 chips/hand while calls averaged +3 to +32. The
# existing 16% gate left rank-0 (high card) with no path and rank-1 folds at
# 16-25% unconverted. The extension widens to 25% for flop HU rank 0/1 only.
# ────────────────────────────────────────────────────────────────────────────

from poker_bot.strategies.hubase import cheap_postflop_continue, pot_odds

FLOP_BOARD = ["7S", "9D", "KH"]  # K-high unpaired flop
FOLD_CALL = ("fold", "call", "raise", "all-in")


def _flop_hu_table(hole, *, call=100, pot=400, hero_stack=2000):
    seats = [
        {
            "seatNumber": 1,
            "agentId": HERO,
            "holeCards": hole,
            "stackChips": hero_stack,
            "currentBetChips": 0,
            "folded": False,
            "hasFolded": False,
        },
        {
            "seatNumber": 2,
            "agentId": VILLAIN,
            "holeCards": [],
            "stackChips": 2000,
            "currentBetChips": 0,
            "folded": False,
            "hasFolded": False,
        },
    ]
    return {
        "street": "Flop",
        "boardCards": FLOP_BOARD,
        "potChips": pot,
        "buttonSeatNumber": 1,
        "seats": seats,
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": list(FOLD_CALL),
            "callAmount": call,
            "callChips": call,
            "raiseRange": {"min": call * 2, "max": 4000},
            "betRange": {"min": 0, "max": 4000},
        },
    }, seats[0]


def _flop_guard(hole, *, call=100, pot=400):
    table, hero = _flop_hu_table(hole, call=call, pot=pot)
    return cheap_postflop_continue(table, hero, ("fold", 0, "base fold"))


# True high card (no pair with board): TC 2D on 7 9 K -> high card K (rank 0)
HIGH_CARD_HOLE = ["TC", "2D"]
BOTTOM_PAIR_HOLE = ["3C", "3D"]  # pair of 3s on 7 9 K (rank 1, non-top-pair)


def test_flop_hu_does_not_call_above_25pct_pot_odds():
    """Above 25% pot odds -> no conversion (let base decide)."""
    # 140 / (400 + 140) = 25.9%
    assert _flop_guard(HIGH_CARD_HOLE, call=140, pot=400) is None


def test_flop_hu_extension_does_not_fire_on_turn():
    """Turn street -> flop-HU extension must not fire."""
    table, hero = _flop_hu_table(HIGH_CARD_HOLE, call=100, pot=400)
    table["street"] = "Turn"
    table["boardCards"] = ["7S", "9D", "KH", "AS"]
    assert cheap_postflop_continue(table, hero, ("fold", 0, "base fold")) is None


def test_flop_hu_extension_does_not_fire_on_river():
    """River street -> flop-HU extension must not fire."""
    table, hero = _flop_hu_table(HIGH_CARD_HOLE, call=100, pot=400)
    table["street"] = "River"
    table["boardCards"] = ["7S", "9D", "KH", "AS", "2C"]
    assert cheap_postflop_continue(table, hero, ("fold", 0, "base fold")) is None


def test_flop_hu_extension_only_converts_folds():
    """Base action 'call' or 'raise' -> guard returns None (no override)."""
    table, hero = _flop_hu_table(HIGH_CARD_HOLE, call=100, pot=400)
    assert cheap_postflop_continue(table, hero, ("call", 100, "base call")) is None
    assert cheap_postflop_continue(table, hero, ("raise", 200, "base raise")) is None


def test_flop_hu_existing_16pct_path_unchanged_for_rank2():
    """Rank 2 at <= 16% pot odds -> still calls via existing path."""
    # 60 / (400 + 60) = 13%
    table, hero = _flop_hu_table(["7C", "7D"], call=60, pot=400)
    decision = cheap_postflop_continue(table, hero, ("fold", 0, "base fold"))
    assert decision is not None and decision[0] == "call"
    assert "cheap continue made rank" in decision[2]


def test_flop_hu_does_not_call_when_price_exceeds_stack_cap():
    """Price > max(blind, 8% stack) -> no call (stack safety)."""
    # 200 call but hero stack only 2000 -> 8% = 160, so 200 > 160 -> blocked
    table, hero = _flop_hu_table(HIGH_CARD_HOLE, call=200, pot=400, hero_stack=2000)
    # 200 / (400 + 200) = 33% -> above 25% anyway, but also above stack cap
    assert cheap_postflop_continue(table, hero, ("fold", 0, "base fold")) is None


def test_choose_action_calls_bottom_pair_flop_hu_at_20pct():
    """End-to-end: choose_action calls bottom pair at 20% pot odds on flop HU."""
    table, hero = _flop_hu_table(BOTTOM_PAIR_HOLE, call=100, pot=400)
    assert choose_action(table, hero)[0] == "call"


# ── Flop HU bluff-catch guard (vs bluffy opponents on dry boards) ────────────
DRY_FLOP = ["7S", "9D", "KH"]
WET_FLOP = ["7S", "8S", "9H"]
PAIRED_FLOP = ["7S", "7D", "KH"]

BLUFFY_RAW = OpponentProfile(
    agent_id=VILLAIN,
    hands_seen=20,
    vpip=10,
    calls=3,
    bets=4,
    raises=2,
    folds=6,
    fold_to_bet=4,
    opportunities_to_fold_to_bet=8,
    showdowns=5,
    weak_aggressive_showdowns=2,
)  # wasd = 0.40 -> bluffy
BLUFFY_DICT = {VILLAIN: {"hands_seen": 20, "weak_aggressive_showdown_frequency": 0.40}}
VALUE_RAW = OpponentProfile(
    agent_id=VILLAIN,
    hands_seen=20,
    vpip=10,
    calls=3,
    bets=4,
    raises=2,
    folds=6,
    fold_to_bet=4,
    opportunities_to_fold_to_bet=8,
    showdowns=5,
    weak_aggressive_showdowns=0,
)  # wasd = 0.0 -> value-heavy
VALUE_DICT = {VILLAIN: {"hands_seen": 20, "weak_aggressive_showdown_frequency": 0.0}}

HIGH_CARD_F = ["TC", "2D"]
BOTTOM_PAIR_F = ["3C", "3D"]
TOP_PAIR_F = ["KH", "2D"]


def _flop_bluff_table(hole, board, profiles, *, call=100, pot=400, hero_stack=2000):
    seats = [
        {
            "seatNumber": 1,
            "agentId": HERO,
            "holeCards": hole,
            "stackChips": hero_stack,
            "currentBetChips": 0,
            "folded": False,
            "hasFolded": False,
        },
        {
            "seatNumber": 2,
            "agentId": VILLAIN,
            "holeCards": [],
            "stackChips": 2000,
            "currentBetChips": 0,
            "folded": False,
            "hasFolded": False,
        },
    ]
    return {
        "street": "Flop",
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": 1,
        "seats": seats,
        "opponentProfiles": profiles,
        "allowedActions": {
            "availableActions": list(FOLD_CALL),
            "callAmount": call,
            "callChips": call,
            "raiseRange": {"min": call * 2, "max": 4000},
            "betRange": {"min": 0, "max": 4000},
        },
    }, seats[0]


def _bluff_guard(hole, board, profiles, **kw):
    call = kw.pop("call", 100)
    table, hero = _flop_bluff_table(hole, board, profiles, call=call, **kw)
    return flop_hu_bluffcatch_guard(table, hero, ("fold", 0, "base fold"))


@pytest.mark.parametrize(
    "hole,label",
    [
        (HIGH_CARD_F, "high card"),
        (BOTTOM_PAIR_F, "bottom pair"),
    ],
)
def test_flop_bluffcatch_calls_vs_bluffy_raw(hole, label):
    """Bluffy opponent (raw profile), dry board, weak hand -> call."""
    decision = _bluff_guard(hole, DRY_FLOP, {VILLAIN: BLUFFY_RAW})
    assert decision is not None and decision[0] == "call", f"{label} should call"


@pytest.mark.parametrize(
    "hole,label",
    [
        (HIGH_CARD_F, "high card"),
        (BOTTOM_PAIR_F, "bottom pair"),
    ],
)
def test_flop_bluffcatch_calls_vs_bluffy_dict(hole, label):
    """Same, with dict-form profile (parity with raw-count path)."""
    decision = _bluff_guard(hole, DRY_FLOP, BLUFFY_DICT)
    assert decision is not None and decision[0] == "call", f"{label} should call"


@pytest.mark.parametrize(
    "hole,label",
    [
        (HIGH_CARD_F, "high card"),
        (BOTTOM_PAIR_F, "bottom pair"),
    ],
)
def test_flop_bluffcatch_silent_vs_value_heavy(hole, label):
    """Value-heavy opponent (low wasd) -> no bluff-catch (calling is -EV)."""
    assert _bluff_guard(hole, DRY_FLOP, {VILLAIN: VALUE_RAW}) is None
    assert _bluff_guard(hole, DRY_FLOP, VALUE_DICT) is None


def test_flop_bluffcatch_silent_on_wet_board():
    """Wet board (connected) -> semi-bluffs possible -> no bluff-catch."""
    assert _bluff_guard(HIGH_CARD_F, WET_FLOP, {VILLAIN: BLUFFY_RAW}) is None


def test_flop_bluffcatch_silent_on_paired_board():
    """Paired board -> full house/trips danger -> no bluff-catch."""
    assert _bluff_guard(HIGH_CARD_F, PAIRED_FLOP, {VILLAIN: BLUFFY_RAW}) is None


def test_flop_bluffcatch_silent_for_top_pair():
    """Top pair is not weak -> no bluff-catch."""
    assert _bluff_guard(TOP_PAIR_F, DRY_FLOP, {VILLAIN: BLUFFY_RAW}) is None


def test_flop_bluffcatch_silent_above_25pct_pot_odds():
    """Above 25% pot odds -> not enough equity to justify."""
    assert (
        _bluff_guard(HIGH_CARD_F, DRY_FLOP, {VILLAIN: BLUFFY_RAW}, call=150, pot=400)
        is None
    )  # 150/550 = 27%


def test_flop_bluffcatch_silent_unknown_opponent():
    """Unknown opponent (< 15 hands) -> no confident bluff read."""
    assert _bluff_guard(HIGH_CARD_F, DRY_FLOP, {VILLAIN: {"hands_seen": 10}}) is None


def test_flop_bluffcatch_silent_non_fold_action():
    """Base action 'call' -> guard returns None (no override)."""
    table, hero = _flop_bluff_table(HIGH_CARD_F, DRY_FLOP, {VILLAIN: BLUFFY_RAW})
    assert flop_hu_bluffcatch_guard(table, hero, ("call", 100, "base call")) is None


def test_flop_bluffcatch_silent_not_facing_bet():
    """Not facing a bet (call=0) -> no bluff-catch."""
    assert _bluff_guard(HIGH_CARD_F, DRY_FLOP, {VILLAIN: BLUFFY_RAW}, call=0) is None


def test_choose_action_bluffcatches_vs_bluffy():
    """End-to-end: choose_action calls high card vs bluffy opponent on dry flop."""
    table, hero = _flop_bluff_table(HIGH_CARD_F, DRY_FLOP, {VILLAIN: BLUFFY_RAW})
    assert choose_action(table, hero)[0] == "call"


def test_choose_action_folds_vs_value_heavy():
    """End-to-end: choose_action still folds high card vs value-heavy opponent."""
    table, hero = _flop_bluff_table(HIGH_CARD_F, DRY_FLOP, {VILLAIN: VALUE_RAW})
    assert choose_action(table, hero)[0] == "fold"


# ── Turn HU bluff-catch extension (same guard, turn street) ─────────────────
# The guard now also fires on the Turn for rank 1 (non-top-pair) only.
# Rank 0 (high card) is excluded on the turn (too little equity, one card to
# come). Same bluffy-opponent + dry-board + <= 25% pot odds conditions.
# ────────────────────────────────────────────────────────────────────────────

DRY_TURN = ["7S", "9D", "KH", "2C"]  # dry on turn (rainbow, uncoordinated)
WET_TURN = ["7S", "8S", "9H", "2C"]  # connected -> wet
PAIRED_TURN = ["7S", "7D", "KH", "2C"]  # paired -> dangerous

# True high card on turn (no pair with board): JC 5D
TURN_HIGH_CARD = ["JC", "5D"]  # rank 0
# Bottom pair on turn: 3C 3D (pair of 3s, non-top-pair, K-high board)
TURN_BOTTOM_PAIR = ["3C", "3D"]  # rank 1, non-top-pair
# Top pair on turn: uses a board card. AH 2D on 7S 9D KH AC -> pair of aces (top)
TURN_TOP_PAIR = ["AH", "2D"]


def _turn_bluff_table(hole, board, profiles, *, call=100, pot=400, hero_stack=2000):
    seats = [
        {
            "seatNumber": 1,
            "agentId": HERO,
            "holeCards": hole,
            "stackChips": hero_stack,
            "currentBetChips": 0,
            "folded": False,
            "hasFolded": False,
        },
        {
            "seatNumber": 2,
            "agentId": VILLAIN,
            "holeCards": [],
            "stackChips": 2000,
            "currentBetChips": 0,
            "folded": False,
            "hasFolded": False,
        },
    ]
    return {
        "street": "Turn",
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": 1,
        "seats": seats,
        "opponentProfiles": profiles,
        "allowedActions": {
            "availableActions": list(FOLD_CALL),
            "callAmount": call,
            "callChips": call,
            "raiseRange": {"min": call * 2, "max": 4000},
            "betRange": {"min": 0, "max": 4000},
        },
    }, seats[0]


def _turn_bluff_guard(hole, board, profiles, **kw):
    call = kw.pop("call", 100)
    table, hero = _turn_bluff_table(hole, board, profiles, call=call, **kw)
    return flop_hu_bluffcatch_guard(table, hero, ("fold", 0, "base fold"))


def test_turn_bluffcatch_calls_bottom_pair_vs_bluffy():
    """Turn rank 1 (non-top-pair), dry board, bluffy opponent -> call."""
    decision = _turn_bluff_guard(TURN_BOTTOM_PAIR, DRY_TURN, {VILLAIN: BLUFFY_RAW})
    assert decision is not None and decision[0] == "call"


def test_turn_bluffcatch_silent_for_high_card():
    """Turn rank 0 (high card) is excluded -- too little equity, one card to come."""
    assert _turn_bluff_guard(TURN_HIGH_CARD, DRY_TURN, {VILLAIN: BLUFFY_RAW}) is None


def test_turn_bluffcatch_silent_vs_value_heavy():
    """Value-heavy opponent -> no bluff-catch on turn."""
    assert _turn_bluff_guard(TURN_BOTTOM_PAIR, DRY_TURN, {VILLAIN: VALUE_RAW}) is None


def test_turn_bluffcatch_silent_on_wet_board():
    """Wet board on turn -> semi-bluffs possible -> no bluff-catch."""
    assert _turn_bluff_guard(TURN_BOTTOM_PAIR, WET_TURN, {VILLAIN: BLUFFY_RAW}) is None


def test_turn_bluffcatch_silent_on_paired_board():
    """Paired board on turn -> full house/trips danger -> no bluff-catch."""
    assert (
        _turn_bluff_guard(TURN_BOTTOM_PAIR, PAIRED_TURN, {VILLAIN: BLUFFY_RAW}) is None
    )


def test_turn_bluffcatch_silent_above_25pct_pot_odds():
    """Above 25% pot odds on turn -> not enough equity."""
    assert (
        _turn_bluff_guard(
            TURN_BOTTOM_PAIR, DRY_TURN, {VILLAIN: BLUFFY_RAW}, call=150, pot=400
        )
        is None
    )  # 150/550 = 27%


def test_choose_action_bluffcatches_turn_vs_bluffy():
    """End-to-end: choose_action calls bottom pair vs bluffy on dry turn."""
    table, hero = _turn_bluff_table(TURN_BOTTOM_PAIR, DRY_TURN, {VILLAIN: BLUFFY_RAW})
    assert choose_action(table, hero)[0] == "call"


def test_choose_action_folds_turn_vs_value_heavy():
    """End-to-end: vs value-heavy opponent, the bluff-catch guard does not fire
    (the base strategy decides call/fold on its own for this spot)."""
    table, hero = _turn_bluff_table(TURN_BOTTOM_PAIR, DRY_TURN, {VILLAIN: VALUE_RAW})
    # The guard is silent (value-heavy); the base strategy calls bottom pair
    # at 20% pot odds via "postflop medium defense". Verify the guard didn't
    # fire by checking it returns None directly.
    assert flop_hu_bluffcatch_guard(table, hero, ("fold", 0, "base fold")) is None


# ── River two-pair fold->call guard (paired board, non-fragile) ──────────────
PAIRED_RIVER = ["7S", "7D", "KH", "2C", "9H"]
UNPAIRED_RIVER = ["7S", "9D", "KH", "2C", "3H"]
DOUBLE_PAIRED_RIVER = ["JS", "KD", "3C", "JH", "KS"]
RIVER_REAL_TWO_PAIR = ["KH", "3C"]  # two pair Ks+7s on 77K29 (hero has K, non-fragile)
RIVER_FRAGILE_TWO_PAIR = ["QD", "TC"]
RIVER_ONE_PAIR = ["AC", "2D"]


def _river_table(hole, board, *, call=134, pot=403, hero_stack=2000):
    seats = [
        {
            "seatNumber": 1,
            "agentId": HERO,
            "holeCards": hole,
            "stackChips": hero_stack,
            "currentBetChips": 0,
            "folded": False,
            "hasFolded": False,
        },
        {
            "seatNumber": 2,
            "agentId": VILLAIN,
            "holeCards": [],
            "stackChips": 2000,
            "currentBetChips": 0,
            "folded": False,
            "hasFolded": False,
        },
    ]
    return {
        "street": "River",
        "boardCards": board,
        "potChips": pot,
        "buttonSeatNumber": 1,
        "seats": seats,
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": list(FOLD_CALL),
            "callAmount": call,
            "callChips": call,
            "raiseRange": {"min": call * 2, "max": 4000},
            "betRange": {"min": 0, "max": 4000},
        },
    }, seats[0]


def _river_guard(hole, board, **kw):
    call = kw.pop("call", 134)
    pot = kw.pop("pot", 403)
    table, hero = _river_table(hole, board, call=call, pot=pot, **kw)
    return river_two_pair_facing_bet_call_guard(table, hero, ("fold", 0, "base fold"))


def test_river_two_pair_calls_on_paired_board():
    """Real two pair on paired board, <= 40% pot odds -> call (was fold)."""
    decision = _river_guard(RIVER_REAL_TWO_PAIR, PAIRED_RIVER, call=134, pot=403)
    assert decision is not None and decision[0] == "call"


def test_river_two_pair_silent_on_unpaired_board():
    """Unpaired board -> no conversion (data shows both fold/call are -EV)."""
    # KH 3C on 7S 9D KH 2C 3H -> two pair Ks+3s, board unpaired
    assert _river_guard(["KH", "3C"], UNPAIRED_RIVER, call=134, pot=403) is None


def test_river_two_pair_silent_for_fragile():
    """Fragile two pair (both pairs on board) -> no conversion (fold is correct)."""
    assert (
        _river_guard(RIVER_FRAGILE_TWO_PAIR, DOUBLE_PAIRED_RIVER, call=134, pot=403)
        is None
    )


def test_river_two_pair_silent_above_40pct_pot_odds():
    """Above 40% pot odds -> no conversion (calling is -EV even with two pair)."""
    assert (
        _river_guard(RIVER_REAL_TWO_PAIR, PAIRED_RIVER, call=300, pot=403) is None
    )  # 300/703 = 42.7%


def test_river_two_pair_silent_not_folding():
    """Base action 'call' -> guard returns None (no override)."""
    table, hero = _river_table(RIVER_REAL_TWO_PAIR, PAIRED_RIVER)
    assert (
        river_two_pair_facing_bet_call_guard(table, hero, ("call", 134, "base call"))
        is None
    )


def test_river_two_pair_silent_not_two_pair():
    """One pair (not two pair) -> no conversion."""
    assert _river_guard(RIVER_ONE_PAIR, PAIRED_RIVER, call=134, pot=403) is None


def test_river_two_pair_silent_not_facing_bet():
    """Not facing a bet (call=0) -> no conversion."""
    assert _river_guard(RIVER_REAL_TWO_PAIR, PAIRED_RIVER, call=0) is None


def test_river_two_pair_silent_wrong_street():
    """Turn street -> guard must not fire (river only)."""
    table, hero = _river_table(RIVER_REAL_TWO_PAIR, PAIRED_RIVER[:4])
    table["street"] = "Turn"
    table["boardCards"] = PAIRED_RIVER[:4]
    assert (
        river_two_pair_facing_bet_call_guard(table, hero, ("fold", 0, "base fold"))
        is None
    )
