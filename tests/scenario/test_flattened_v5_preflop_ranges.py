from poker_bot.strategies.flattened_v5 import (
    opponent_aware_preflop,
    preflop_three_bet,
    sixmax_range_preflop,
)


def make_preflop_table(
    seat_number,
    hole_cards,
    current_bet=0,
    call_amount=0,
    button_seat=1,
    seats=None,
):
    """Build a minimal 6-max preflop table for testing."""
    if seats is None:
        seats = [
            (1, "BTN"),
            (2, "SB"),
            (3, "BB"),
            (4, "UTG"),
            (5, "MP"),
            (6, "CO"),
        ]
    seat_objs = []
    for num, _pos in seats:
        agent = "hero" if num == seat_number else f"villain-{num}"
        seat_objs.append({
            "agentId": agent,
            "seatNumber": num,
            "holeCards": hole_cards if num == seat_number else ["2c", "7d"],
            "stackChips": 2000,
            "currentBetChips": 0,
        })
    return {
        "street": "Preflop",
        "boardCards": [],
        "potChips": 3,
        "currentBet": current_bet,
        "buttonSeatNumber": button_seat,
        "selfSeatNumber": seat_number,
        "actingSeatNumber": seat_number,
        "allowedActions": {
            "availableActions": ["fold", "call", "raise"],
            "callAmount": call_amount,
            "callChips": call_amount,
            "minBet": 2,
            "minRaiseTo": max(4, current_bet * 2) if current_bet else 4,
            "maxCommit": 2000,
            "betRange": {"min": 2, "max": 2000},
            "raiseRange": {"min": max(4, current_bet * 2) if current_bet else 4, "max": 2000},
        },
        "seats": seat_objs,
        "opponentProfiles": {},
    }


def _hero_seat(table):
    for seat in table["seats"]:
        if seat["agentId"] == "hero":
            return seat
    return table["seats"][0]


# ── Plan A: range-based opens ────────────────────────────────────────────────


def test_btns_opens_22_unopened():
    """BTN should open 22 in an unopened pot (in BTN open range)."""
    table = make_preflop_table(1, ["2c", "2d"])
    action, amount, _ = sixmax_range_preflop(table, _hero_seat(table))
    assert action == "raise", f"expected raise, got {action}"
    assert amount is not None


def test_btns_opens_a2s_unopened():
    table = make_preflop_table(1, ["Ac", "2c"])
    action, amount, _ = sixmax_range_preflop(table, _hero_seat(table))
    assert action == "raise"
    assert amount is not None


def test_co_opens_kqo_unopened():
    """KQo is in CO open range."""
    table = make_preflop_table(6, ["Kc", "Qc"])
    action, amount, _ = sixmax_range_preflop(table, _hero_seat(table))
    assert action == "raise"


def test_co_folds_72o_unopened():
    """72o is NOT in CO open range."""
    table = make_preflop_table(6, ["7c", "2d"])
    action, _, _ = sixmax_range_preflop(table, _hero_seat(table))
    assert action == "fold"


def test_utg_opens_aqo_unopened():
    """AQo is in UTG open range."""
    table = make_preflop_table(4, ["Ac", "Qc"])
    action, amount, _ = sixmax_range_preflop(table, _hero_seat(table))
    assert action == "raise"


def test_utg_folds_72o_unopened():
    """72o is NOT in UTG open range."""
    table = make_preflop_table(4, ["7c", "2d"])
    action, _, _ = sixmax_range_preflop(table, _hero_seat(table))
    assert action == "fold"


def test_bb_defends_suited_connector_vs_btn():
    """BB should defend 76s vs BTN open (in BB defend range)."""
    # BTN opens to 6, BB faces a call of 4
    table = make_preflop_table(3, ["7c", "6c"], current_bet=6, call_amount=4)
    action, _, _ = sixmax_range_preflop(table, _hero_seat(table))
    assert action in ("call", "raise"), f"expected defend, got {action}"


def test_bb_folds_72o_vs_btn():
    """BB should fold 72o vs BTN open (not in BB defend range)."""
    table = make_preflop_table(3, ["7c", "2d"], current_bet=6, call_amount=4)
    action, _, _ = sixmax_range_preflop(table, _hero_seat(table))
    assert action == "fold", f"expected fold, got {action}"


def test_premium_always_raises():
    """AA/KK/QQ/AKs should always raise for value."""
    for hole, label in [
        (["Ac", "Ad"], "AA"),
        (["Kc", "Kd"], "KK"),
        (["Ac", "Kc"], "AKs"),
    ]:
        for seat_num in (1, 4, 6):  # BTN, UTG, CO
            table = make_preflop_table(seat_num, hole)
            action, amount, _ = sixmax_range_preflop(table, _hero_seat(table))
            assert action == "raise", f"{label} from seat {seat_num}: expected raise, got {action}"


# ── Plan C: 3-bet branch ─────────────────────────────────────────────────────


def _open_table(hero_seat, hero_cards, open_size=6, call_amount=4):
    """Build a table where villain has open-raised to `open_size`."""
    table = make_preflop_table(hero_seat, hero_cards, current_bet=open_size, call_amount=call_amount)
    # Add a small open from the villain in seat 1 (BTN) so the table is realistic
    return table


def test_value_3bet_qq_vs_open():
    """QQ should 3-bet for value vs an open (base action = call)."""
    table = _open_table(3, ["Qc", "Qh"], open_size=6, call_amount=4)
    action, amount, _ = preflop_three_bet(table, _hero_seat(table), ("call", 4, "base call"))
    assert action == "raise", f"expected 3-bet, got {action}"
    assert amount is not None
    assert amount >= 15  # at least 2.5x the open


def test_value_3bet_aks_vs_open():
    table = _open_table(3, ["Ac", "Kc"], open_size=6, call_amount=4)
    action, amount, _ = preflop_three_bet(table, _hero_seat(table), ("call", 4, "base call"))
    assert action == "raise"


def test_value_3bet_ako_vs_open():
    table = _open_table(3, ["Ac", "Ko"], open_size=6, call_amount=4)
    action, amount, _ = preflop_three_bet(table, _hero_seat(table), ("call", 4, "base call"))
    assert action == "raise"


def test_bluff_3bet_a5s_from_btn():
    """A5s should 3-bet bluff from BTN vs an open (Ace blocker)."""
    table = make_preflop_table(1, ["Ac", "5c"], current_bet=6, call_amount=4)
    result = preflop_three_bet(table, _hero_seat(table), ("call", 4, "base call"))
    assert result is not None, "expected 3-bet bluff, got None"
    action, amount, _ = result
    assert action == "raise"
    assert amount is not None


def test_no_3bet_72o_vs_open():
    """72o should NOT 3-bet vs an open (returns None)."""
    table = _open_table(3, ["7c", "2d"], open_size=6, call_amount=4)
    result = preflop_three_bet(table, _hero_seat(table), ("call", 4, "base call"))
    assert result is None, f"expected None (no 3-bet), got {result}"


def test_3bet_fires_on_fold_base():
    """3-bet should also fire when base action is fold."""
    table = _open_table(3, ["Ac", "Kc"], open_size=6, call_amount=4)
    result = preflop_three_bet(table, _hero_seat(table), ("fold", None, "base fold"))
    assert result is not None
    action, _, _ = result
    assert action == "raise"


def test_3bet_does_not_fire_on_postflop():
    """3-bet branch should be a no-op postflop (returns None)."""
    table = make_preflop_table(3, ["Ac", "Kc"], current_bet=6, call_amount=4)
    table["street"] = "Flop"
    table["boardCards"] = ["Ah", "Kd", "3c"]
    result = preflop_three_bet(table, _hero_seat(table), ("call", 4, "base"))
    assert result is None


# ── Plan E: opponent-aware preflop override ────────────────────────────────


def _loose_aggressive_profiles():
    """Build a profile dict for a loose-aggressive table."""
    return {
        "villain-2": {
            "hands_seen": 50,
            "aggression_frequency": 0.7,
            "call_frequency": 0.15,
            "fold_to_bet_frequency": 0.3,
        },
        "villain-3": {
            "hands_seen": 50,
            "aggression_frequency": 0.7,
            "call_frequency": 0.15,
            "fold_to_bet_frequency": 0.3,
        },
    }


def _tight_profiles():
    """Build a profile dict for a tight/patient table."""
    return {
        "villain-2": {
            "hands_seen": 50,
            "aggression_frequency": 0.2,
            "call_frequency": 0.1,
            "fold_to_bet_frequency": 0.5,
            "label": "patient_methodical",
        },
        "villain-3": {
            "hands_seen": 50,
            "aggression_frequency": 0.2,
            "call_frequency": 0.1,
            "fold_to_bet_frequency": 0.5,
            "label": "patient_methodical",
        },
    }


def test_no_override_without_profiles():
    """No override when opponent profiles are empty."""
    table = make_preflop_table(1, ["2c", "2d"])
    table["opponentProfiles"] = {}
    result = opponent_aware_preflop(table, _hero_seat(table), ("fold", None, "base fold"))
    assert result is None


def test_steal_widens_vs_loose_aggressive():
    """vs loose_aggressive, BTN should steal wider (A2s, K9s, etc.)."""
    table = make_preflop_table(1, ["Ac", "2c"])  # BTN with A2s (score 52)
    table["opponentProfiles"] = _loose_aggressive_profiles()
    result = opponent_aware_preflop(table, _hero_seat(table), ("fold", None, "base fold"))
    assert result is not None
    action, amount, _ = result
    assert action == "raise"
    assert amount is not None


def test_no_steal_of_junk_vs_tight_table():
    """vs tight table, BTN should NOT steal 72o (junk below threshold)."""
    table = make_preflop_table(1, ["7c", "2d"])  # BTN with 72o (score 23)
    table["opponentProfiles"] = _tight_profiles()
    result = opponent_aware_preflop(table, _hero_seat(table), ("fold", None, "base fold"))
    # 72o score 23 is below all steal thresholds; should not fire
    assert result is None


def test_override_postflop_noop():
    """Override should be a no-op on postflop streets."""
    table = make_preflop_table(1, ["2c", "2d"])
    table["opponentProfiles"] = _loose_aggressive_profiles()
    table["street"] = "Flop"
    table["boardCards"] = ["Ah", "Kd", "3c"]
    result = opponent_aware_preflop(table, _hero_seat(table), ("fold", None, "base fold"))
    assert result is None


def test_override_does_not_fire_on_raise_base():
    """Override should only convert fold → raise, not change raise bases."""
    table = make_preflop_table(1, ["2c", "2d"])
    table["opponentProfiles"] = _loose_aggressive_profiles()
    result = opponent_aware_preflop(table, _hero_seat(table), ("raise", 6, "base raise"))
    assert result is None


def test_override_does_not_fire_on_call_base():
    """Override should not change call bases (only fold → raise)."""
    table = make_preflop_table(1, ["2c", "2d"])
    table["opponentProfiles"] = _loose_aggressive_profiles()
    result = opponent_aware_preflop(table, _hero_seat(table), ("call", 2, "base call"))
    assert result is None


def test_no_steal_from_early_position():
    """Override should not fire from UTG/MP (no steal thresholds defined)."""
    table = make_preflop_table(4, ["2c", "2d"])  # UTG
    table["opponentProfiles"] = _loose_aggressive_profiles()
    result = opponent_aware_preflop(table, _hero_seat(table), ("fold", None, "base fold"))
    assert result is None
