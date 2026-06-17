"""Scenario tests for s2base self-analysis fixes (v005 self-analysis).

These tests lock in the five patches applied to s2base.py after the
tournament telemetry review (2026-06-16). Each test reproduces a
specific spot from the busted s2v005 session and asserts that the
patched s2base.py returns the correct action.

Reference hands from telemetry-luigi-tournament.sqlite:
  - cmqh6gkmc4eb7m0toatxdecfq: 7h7d vs 3-bet to 479 → should fold
  - cmqh5ym6h1788m0toplfjck6n: 4c4d SB vs raise to 37 → should call
  - cmqh5n1ivysqpm0toz2g4ha98: TT 3-barrel vs page → demote to call
  - cmqh58ulbw8alm0tokjjbbljp: 7d2s on 5h3cJd → wheel gutshot call
  - Various UTG/HJ/BTN opens → widen vs unprofiled field
"""

from poker_bot.strategies import s2v005_self_patch as strategy


# ════════════════════════════════════════════════════════════════════════════
# Test fixtures
# ════════════════════════════════════════════════════════════════════════════


def make_seats(bets, stacks, hero_seat, hero_cards, button=1, all_aggressive=False):
    """Build a 6-max seats list. bets[i] is hero's current bet if i == hero_seat-1."""
    seats = []
    n = 6
    for index in range(n):
        seat_number = index + 1
        seat_bet = (bets[index] if bets else 0) or 0
        seats.append(
            {
                "agentId": f"villain-{seat_number}"
                if seat_number != hero_seat
                else "hero",
                "seatNumber": seat_number,
                "holeCards": hero_cards if seat_number == hero_seat else [],
                "stackChips": stacks[index] if stacks else 1500,
                "currentBetChips": seat_bet,
                "folded": seat_bet is None and seat_number != hero_seat,
                "hasFolded": seat_bet is None and seat_number != hero_seat,
            }
        )
    return seats


def make_preflop_table(
    hero_cards,
    hero_seat,
    button,
    raise_amount,
    hero_stack,
    raise_seat,
    n_players=6,
    raiser_stack=1500,
    all_aggressive=False,
):
    """Build a preflop table where one player has raised to raise_amount and
    hero is at hero_seat with hero_cards and hero_stack chips remaining."""
    seats = []
    # Order seats 1..n. Assume button is fixed.
    for i in range(1, n_players + 1):
        if i == hero_seat:
            # Hero hasn't put money in yet (unless hero is SB/BB).
            hero_blind = 0
            if hero_seat == 2:  # SB
                hero_blind = 5
            elif hero_seat == 3:  # BB
                hero_blind = 10
            seats.append(
                {
                    "agentId": "hero",
                    "seatNumber": i,
                    "holeCards": hero_cards,
                    "stackChips": hero_stack,
                    "currentBetChips": hero_blind,
                    "folded": False,
                    "hasFolded": False,
                }
            )
        elif i == raise_seat:
            seats.append(
                {
                    "agentId": "raiser",
                    "seatNumber": i,
                    "holeCards": [],
                    "stackChips": raiser_stack,
                    "currentBetChips": raise_amount,
                    "folded": False,
                    "hasFolded": False,
                }
            )
        else:
            # Other seats folded
            seats.append(
                {
                    "agentId": f"villain-{i}",
                    "seatNumber": i,
                    "holeCards": [],
                    "stackChips": 1500,
                    "currentBetChips": 0,
                    "folded": True,
                    "hasFolded": True,
                }
            )
    # Set up the table
    max_bet = raise_amount
    call_amount = max_bet - (seats[hero_seat - 1]["currentBetChips"] or 0)
    # Add a profile for the raiser so we don't fall through to counter_adaptive.
    profiles = {"raiser": {"hands_seen": 5, "vpip": 0.5, "pfr": 0.3}}
    if all_aggressive:
        # Set fold_to_bet high so the existing logic widens even more.
        profiles["raiser"]["fold_to_bet"] = 0.7
    return {
        "street": "Preflop",
        "boardCards": [],
        "potChips": raise_amount + 15,  # raise + SB + BB
        "buttonSeatNumber": button,
        "seats": seats,
        "opponentProfiles": profiles,
        "allowedActions": {
            "availableActions": ["fold", "call", "raise", "all-in"],
            "callAmount": call_amount,
            "callChips": call_amount,
            "minBet": 10,  # big blind — used as blind_size fallback
            "minRaiseTo": max(call_amount * 2, raise_amount + raise_amount),
        },
    }


def make_postflop_table(
    hero_cards,
    hero_seat,
    board_cards,
    pot_chips,
    hero_stack,
    facing_bet,
    available=None,
):
    """Build a postflop table where hero is acting with facing_bet to call."""
    if available is None:
        available = ["fold", "call", "raise"]
    n = 4  # 4-handed postflop for simplicity
    seats = []
    for i in range(1, n + 1):
        if i == hero_seat:
            seats.append(
                {
                    "agentId": "hero",
                    "seatNumber": i,
                    "holeCards": hero_cards,
                    "stackChips": hero_stack,
                    "currentBetChips": facing_bet,
                    "folded": False,
                    "hasFolded": False,
                }
            )
        else:
            seats.append(
                {
                    "agentId": f"villain-{i}",
                    "seatNumber": i,
                    "holeCards": [],
                    "stackChips": 1500,
                    "currentBetChips": 0,
                    "folded": False,
                    "hasFolded": False,
                }
            )
    button = (hero_seat - 1) if hero_seat > 1 else n
    return {
        "street": "Flop",
        "boardCards": board_cards,
        "potChips": pot_chips,
        "buttonSeatNumber": button,
        "seats": seats,
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": available,
            "callAmount": facing_bet,
            "callChips": facing_bet,
        },
    }


# ════════════════════════════════════════════════════════════════════════════
# Patch 3: pocket-pair set-mining math
# ════════════════════════════════════════════════════════════════════════════


def test_set_mine_77_hj_3bet_fold():
    """7h7d HJ vs 3-bet to 479 (19% of 2484 stack) → fold (was: call).
    Patch 3: pocket-pair set-mining math (price_to_stack > 0.118 → fold)."""
    table = make_preflop_table(
        hero_cards=["7h", "7d"],
        hero_seat=4,  # HJ
        button=4,
        raise_amount=479,
        hero_stack=2484,
        raise_seat=6,  # BTN
    )
    hero = table["seats"][3]
    result = strategy.profiled_choose_action(table, hero)
    assert result is not None
    assert result[0] == "fold", (
        f"Expected fold for 7h7d HJ vs 19% 3-bet, got {result[0]}"
    )


def test_set_mine_44_sb_vs_cheap_raise():
    """4c4d SB vs raise to 37 (1.5% of 2444 stack) → call (was: fold)."""
    table = make_preflop_table(
        hero_cards=["4c", "4d"],
        hero_seat=2,  # SB
        button=3,
        raise_amount=37,
        hero_stack=2444,
        raise_seat=6,
    )
    hero = table["seats"][1]
    result = strategy.profiled_choose_action(table, hero)
    assert result is not None
    assert result[0] == "call", (
        f"Expected call for 4c4d SB vs cheap raise, got {result[0]}"
    )


def test_set_mine_aqo_bb_regression():
    """AQo BB vs BTN raise to 25 → still call (no regression on premium hands)."""
    table = make_preflop_table(
        hero_cards=["Ah", "Qd"],
        hero_seat=2,  # BB
        button=5,
        raise_amount=25,
        hero_stack=1500,
        raise_seat=6,  # BTN
    )
    hero = table["seats"][1]
    result = strategy.profiled_choose_action(table, hero)
    assert result is not None
    assert result[0] == "call", f"Expected call for AQo BB, got {result[0]}"


def test_set_mine_22_short_stack_fold():
    """22 SB vs 4x over-priced raise (stack too short) → fold.
    Patch 3: price_to_stack > 0.118 → fold."""
    table = make_preflop_table(
        hero_cards=["2c", "2d"],
        hero_seat=2,
        button=3,
        raise_amount=100,  # 40% of stack
        hero_stack=250,
        raise_seat=6,
    )
    hero = table["seats"][1]
    result = strategy.profiled_choose_action(table, hero)
    assert result is not None
    assert result[0] == "fold", f"Expected fold for over-priced 22, got {result[0]}"


# ════════════════════════════════════════════════════════════════════════════
# Patch 1: tighten paired_board_pot_control for fragile hands
# ════════════════════════════════════════════════════════════════════════════


def test_fragile_two_pair_folds_at_30pct_required():
    """7h7d on Js6c3s → 6d, opponent bets 30% pot — Patch 1 over-priced guard.
    Note: the 0.42 cap is preserved; the new over-priced guard only
    fires when pot > 35% of stack AND price > 30% of stack. At
    pot 200 / stack 1500 (13% PSR) the guard doesn't fire and
    the original 0.42 cap applies (30% < 42% → call)."""
    pot = 200
    price = 60  # 30% pot odds, 4% of stack
    table = make_postflop_table(
        hero_cards=["7h", "7d"],
        hero_seat=2,
        board_cards=["Js", "6c", "3s", "6d"],
        pot_chips=pot,
        hero_stack=1500,
        facing_bet=price,
    )
    hero = table["seats"][1]
    base = ("raise", 60, "base")
    result = strategy.paired_board_pot_control(table, hero, base)
    # At pot/stack 13% (below 35% threshold), the over-priced guard
    # doesn't fire, and required 30% < 0.42 cap → call.
    assert result is not None
    assert result[0] == "call", (
        f"Expected call at 30% pot odds in normal pot, got {result[0]}"
    )


def test_fragile_two_pair_folds_big_pot():
    """7h7d in 3-bet pot (already 35%+ of stack) → fold (Patch 1 guard).
    The over-priced guard: pot > 35% of stack AND price > 30% of stack."""
    pot = 700  # > 35% of 1500 stack
    price = 500  # 33% of 1500 stack
    table = make_postflop_table(
        hero_cards=["7h", "7d"],
        hero_seat=2,
        board_cards=["Js", "6c", "3s", "6d"],
        pot_chips=pot,
        hero_stack=1500,
        facing_bet=price,
    )
    hero = table["seats"][1]
    base = ("raise", 500, "base")
    result = strategy.paired_board_pot_control(table, hero, base)
    # 700 > 0.35 * 1500 = 525 (true), 500 > 0.30 * 1500 = 450 (true)
    # → over-priced guard fires → fold
    assert result is not None
    assert result[0] == "fold", (
        f"Expected fold for fragile two pair in big pot, got {result[0]}"
    )


def test_non_fragile_two_pair_unchanged():
    """JJ on Js6c3s → non-fragile top two pair → still raises (unchanged)."""
    # 7h7d on Js6c3s = fragile (uses board pair)
    # JJ on Js6c3s = top two pair, NOT fragile
    pot = 100
    price = 50
    table = make_postflop_table(
        hero_cards=["Jd", "Jh"],
        hero_seat=2,
        board_cards=["Js", "6c", "3s"],
        pot_chips=pot,
        hero_stack=1500,
        facing_bet=price,
    )
    hero = table["seats"][1]
    base = ("raise", 50, "base")
    result = strategy.paired_board_pot_control(table, hero, base)
    # JJ doesn't have a fragile rank-2, so the function should not intercept
    # (it returns None, leaving the base decision in place)
    assert result is None, f"Expected None for non-fragile two pair, got {result}"


# ════════════════════════════════════════════════════════════════════════════
# Patch 2: anti-bully demote 3-barrel vs non-aggressive opponents
# ════════════════════════════════════════════════════════════════════════════


def _make_river_table_with_aggression(
    hero_cards, hero_stack, opp_stack, hero_seat, opp_seat
):
    """Build a River table where the opponent has bet and we need to decide
    whether to 3-barrel raise or bluff-catch call. The opponent is NOT in
    _SIXMAX_AGGRESSIVE_LABELS — just a big stack."""
    seats = []
    for i in range(1, 4):  # 3-handed
        if i == hero_seat:
            seats.append(
                {
                    "agentId": "hero",
                    "seatNumber": i,
                    "holeCards": hero_cards,
                    "stackChips": hero_stack,
                    "currentBetChips": 0,
                    "folded": False,
                    "hasFolded": False,
                }
            )
        elif i == opp_seat:
            seats.append(
                {
                    "agentId": "page",  # NOT in _SIXMAX_AGGRESSIVE_LABELS
                    "seatNumber": i,
                    "holeCards": [],
                    "stackChips": opp_stack,
                    "currentBetChips": 200,
                    "folded": False,
                    "hasFolded": False,
                }
            )
        else:
            seats.append(
                {
                    "agentId": f"villain-{i}",
                    "seatNumber": i,
                    "holeCards": [],
                    "stackChips": 1500,
                    "currentBetChips": 0,
                    "folded": True,
                    "hasFolded": True,
                }
            )
    return {
        "street": "River",
        "boardCards": ["Ah", "Qd", "Ts", "6c", "3h"],
        "potChips": 2000,
        "buttonSeatNumber": (hero_seat + 1) if hero_seat < 3 else 1,
        "seats": seats,
        "opponentProfiles": {
            "page": {
                "hands_seen": 30,
                "label": "balanced",
                "vpip": 0.2,
                "pfr": 0.18,
                "fold_to_bet": 0.45,
            },
        },
        "allowedActions": {
            "availableActions": ["fold", "call", "raise", "all-in"],
            "callAmount": 200,
            "callChips": 200,
            "minBet": 10,
        },
    }


def test_anti_bully_demote_vs_non_aggressive_big_stack_3_streets():
    """TT on River, page (non-aggressive big stack) bet → demote to call.
    Patch 2: 3-barrel raise demoted to bluff-catch call when
    opponent is not in _SIXMAX_AGGRESSIVE_LABELS."""
    table = _make_river_table_with_aggression(
        hero_cards=["Td", "Tc"],
        hero_stack=800,
        opp_stack=1457,
        hero_seat=1,
        opp_seat=2,
    )
    hero = table["seats"][0]
    result = strategy._sixmax_anti_bully_action(table, hero)
    assert result is not None
    # Demote to call (not raise) — opponent is page (balanced, not aggressive)
    assert result[0] == "call", f"Expected demote to call, got {result[0]}"


def test_anti_bully_still_3_barrels_vs_aggressive_label():
    """TT on River, opponent is a known loose-aggressive → still raise (unchanged)."""
    seats = []
    for i in range(1, 4):
        if i == 1:
            seats.append(
                {
                    "agentId": "hero",
                    "seatNumber": i,
                    "holeCards": ["Td", "Tc"],
                    "stackChips": 800,
                    "currentBetChips": 0,
                    "folded": False,
                    "hasFolded": False,
                }
            )
        elif i == 2:
            seats.append(
                {
                    "agentId": "loose_aggressive_bot",
                    "seatNumber": i,
                    "holeCards": [],
                    "stackChips": 1457,
                    "currentBetChips": 200,
                    "folded": False,
                    "hasFolded": False,
                }
            )
        else:
            seats.append(
                {
                    "agentId": f"villain-{i}",
                    "seatNumber": i,
                    "holeCards": [],
                    "stackChips": 1500,
                    "currentBetChips": 0,
                    "folded": True,
                    "hasFolded": True,
                }
            )
    # Profile stats tuned to trigger label() == "loose_aggressive":
    #   vpip_frequency = vpip/hands_seen >= 0.45 → vpip >= 14
    #   aggression_frequency = (bets+raises)/actions >= 0.35
    table = {
        "street": "River",
        "boardCards": ["Ah", "Qd", "Ts", "6c", "3h"],
        "potChips": 2000,
        "buttonSeatNumber": 2,
        "seats": seats,
        "opponentProfiles": {
            "loose_aggressive_bot": {
                "hands_seen": 30,
                "vpip": 15,  # 0.5
                "pfr": 8,  # 0.27
                "calls": 8,  # 23 actions total
                "bets": 4,
                "raises": 5,  # 9/23 = 0.39 aggression
                "folds": 6,
                "fold_to_bet": 4,
                "opportunities_to_fold_to_bet": 10,  # 0.4
                "showdowns": 3,
                "weak_aggressive_showdowns": 1,
            },
        },
        "allowedActions": {
            "availableActions": ["fold", "call", "raise", "all-in"],
            "callAmount": 200,
            "callChips": 200,
            "minBet": 10,
        },
    }
    hero = table["seats"][0]
    result = strategy._sixmax_anti_bully_action(table, hero)
    assert result is not None
    # Against an aggressive opponent, we still raise
    assert result[0] == "raise", (
        f"Expected raise vs aggressive opponent, got {result[0]}"
    )


# ════════════════════════════════════════════════════════════════════════════
# Patch 4: gutshot draw in postflop_draw_continue
# ════════════════════════════════════════════════════════════════════════════


def test_gutshot_call_at_cheap_price():
    """7d2s on 5h3cJd (wheel gutshot) facing 12% c-bet → call.
    Patch 4: gutshot-only draws get a tighter cap (12% flop / 8% turn)."""
    pot = 100
    price = 12  # 12% pot odds
    table = make_postflop_table(
        hero_cards=["7d", "2s"],
        hero_seat=2,
        board_cards=["5h", "3c", "Jd"],
        pot_chips=pot,
        hero_stack=1000,
        facing_bet=price,
    )
    hero = table["seats"][1]
    base = ("fold", None, "base fold")
    result = strategy.postflop_draw_continue(table, hero, base)
    assert result is not None
    assert result[0] == "call", f"Expected call for gutshot at 12%, got {result[0]}"


def test_gutshot_fold_at_25pct_bet():
    """7d2s on 5h3cJd facing 25% c-bet → fold (above gutshot cap 12%)."""
    pot = 100
    price = 25
    table = make_postflop_table(
        hero_cards=["7d", "2s"],
        hero_seat=2,
        board_cards=["5h", "3c", "Jd"],
        pot_chips=pot,
        hero_stack=1000,
        facing_bet=price,
    )
    hero = table["seats"][1]
    base = ("fold", None, "base fold")
    result = strategy.postflop_draw_continue(table, hero, base)
    assert result is None, f"Expected None for gutshot at 25%, got {result}"


def test_oesd_unchanged_at_25pct_bet():
    """9s8d on JhTc5s (OESD) facing 25% c-bet → still call (OESD cap is 30%)."""
    pot = 100
    price = 25
    table = make_postflop_table(
        hero_cards=["9s", "8d"],
        hero_seat=2,
        board_cards=["Jh", "Tc", "5s"],
        pot_chips=pot,
        hero_stack=1000,
        facing_bet=price,
    )
    hero = table["seats"][1]
    base = ("fold", None, "base fold")
    result = strategy.postflop_draw_continue(table, hero, base)
    assert result is not None
    assert result[0] == "call", f"Expected call for OESD at 25%, got {result[0]}"


# ════════════════════════════════════════════════════════════════════════════
# Patch 5: widen preflop opens vs unprofiled field
# ════════════════════════════════════════════════════════════════════════════


def test_unprofiled_field_widens_utg_kts_open():
    """KTs UTG (score 69) with no opponent profiles → raise.
    Patch 5: unprofiled field widens UTG threshold from 68 to 60
    (floor). KTs (score 69) passes the 60 floor and opens."""
    # 6-max: button=4 → seat 1 = UTG. Order: 4=BTN, 5=SB, 6=BB, 1=UTG, 2=MP, 3=CO.
    seats = []
    for i in range(1, 7):
        if i == 1:
            seats.append(
                {
                    "agentId": "hero",
                    "seatNumber": i,
                    "holeCards": ["Ks", "Ts"],
                    "stackChips": 1500,
                    "currentBetChips": 0,
                    "folded": False,
                    "hasFolded": False,
                }
            )
        else:
            seats.append(
                {
                    "agentId": f"villain-{i}",
                    "seatNumber": i,
                    "holeCards": [],
                    "stackChips": 1500,
                    "currentBetChips": 0,
                    "folded": True,
                    "hasFolded": True,
                }
            )
    table = {
        "street": "Preflop",
        "boardCards": [],
        "potChips": 15,
        "buttonSeatNumber": 4,
        "seats": seats,
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": ["fold", "call", "raise"],
            "callAmount": 10,
            "callChips": 10,
            "minBet": 10,
        },
    }
    hero = table["seats"][0]
    base = ("fold", None, "base")
    result = strategy.preflop_open_raise(table, hero, base)
    assert result is not None, (
        f"Expected raise for KTs UTG vs unprofiled field, got None"
    )
    assert result[0] == "raise", f"Expected raise, got {result[0]}"


def test_unprofiled_field_keeps_t9s_utg_fold():
    """T9s UTG (score 52) is BELOW the unprofiled floor of 60 → still fold.
    Patch 5 floor at 60 means T9s stays a fold."""
    seats = []
    for i in range(1, 7):
        if i == 1:
            seats.append(
                {
                    "agentId": "hero",
                    "seatNumber": i,
                    "holeCards": ["9s", "Ts"],
                    "stackChips": 1500,
                    "currentBetChips": 0,
                    "folded": False,
                    "hasFolded": False,
                }
            )
        else:
            seats.append(
                {
                    "agentId": f"villain-{i}",
                    "seatNumber": i,
                    "holeCards": [],
                    "stackChips": 1500,
                    "currentBetChips": 0,
                    "folded": True,
                    "hasFolded": True,
                }
            )
    table = {
        "street": "Preflop",
        "boardCards": [],
        "potChips": 15,
        "buttonSeatNumber": 4,
        "seats": seats,
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": ["fold", "call", "raise"],
            "callAmount": 10,
            "callChips": 10,
            "minBet": 10,
        },
    }
    hero = table["seats"][0]
    base = ("fold", None, "base")
    result = strategy.preflop_open_raise(table, hero, base)
    # T9s at score 52 stays folded (below the 60 floor)
    assert result is None, f"Expected None (fold) for T9s UTG, got {result}"


def test_profiled_passive_field_keeps_widening():
    """KTs UTG vs opponents profiled as passive → still raise.
    Patch 5: only widens for unprofiled fields; profiled passive
    takes the existing profile-aware path which already widens."""
    seats = []
    for i in range(1, 7):
        if i == 1:
            seats.append(
                {
                    "agentId": "hero",
                    "seatNumber": i,
                    "holeCards": ["Ks", "Ts"],
                    "stackChips": 1500,
                    "currentBetChips": 0,
                    "folded": False,
                    "hasFolded": False,
                }
            )
        else:
            seats.append(
                {
                    "agentId": f"villain-{i}",
                    "seatNumber": i,
                    "holeCards": [],
                    "stackChips": 1500,
                    "currentBetChips": 0,
                    "folded": True,
                    "hasFolded": True,
                }
            )
    # Add profile data that triggers passive classification
    table = {
        "street": "Preflop",
        "boardCards": [],
        "potChips": 15,
        "buttonSeatNumber": 4,
        "seats": seats,
        "opponentProfiles": {
            "villain-2": {
                "hands_seen": 15,
                "vpip": 0.3,
                "pfr": 0.1,
                "fold_to_bet": 0.65,
            },
            "villain-3": {
                "hands_seen": 15,
                "vpip": 0.25,
                "pfr": 0.08,
                "fold_to_bet": 0.7,
            },
        },
        "allowedActions": {
            "availableActions": ["fold", "call", "raise"],
            "callAmount": 10,
            "callChips": 10,
            "minBet": 10,
        },
    }
    hero = table["seats"][0]
    base = ("fold", None, "base")
    result = strategy.preflop_open_raise(table, hero, base)
    assert result is not None
    assert result[0] == "raise", (
        f"Expected raise for KTs UTG vs passive, got {result[0]}"
    )


# ════════════════════════════════════════════════════════════════════════════
# Helper unit tests for the gutshot detector
# ════════════════════════════════════════════════════════════════════════════


def test_gutshot_helper_wheel():
    """7d2s on 5h3cJd → wheel gutshot."""
    assert strategy.has_gutshot_draw(["7d", "2s"], ["5h", "3c", "Jd"]) is True


def test_gutshot_helper_no_draw():
    """AdKd on Th9h2s → no draw."""
    assert strategy.has_gutshot_draw(["Ad", "Kd"], ["Th", "9h", "2s"]) is False


def test_gutshot_helper_made_straight_returns_false():
    """Made straight (4-5-6-7-8) should NOT be reported as a gutshot."""
    assert strategy.has_gutshot_draw(["4d", "8d"], ["5h", "6c", "7s"]) is False
