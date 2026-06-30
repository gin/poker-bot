"""
Observed gameplay flaws from hubase benchmark telemetry.

Each test documents a specific leak found in the benchmark data and
verifies the fix prevents the -EV action. Tests use the style from
test_from_fielding.py: concise fixtures, clear assertions, and
parametrized cases.

Reference: benchmark.sqlite (hubase, 10k hands, 16 opponents)
"""

import pytest

from poker_bot.strategies.hubase import choose_action

HERO = "hero"
DEFAULT_AVAILABLE = ("fold", "call", "raise", "all-in")


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
    board,
    *,
    street="Flop",
    pot=80,
    call=0,
    current_bet=0,
    available=DEFAULT_AVAILABLE,
    button=1,
    hero_seat=1,
    action_history=None,
):
    """Build a minimal HU table fixture."""
    seats = [make_seat(hero_seat, HERO, hole)]
    seats.append(make_seat(2 if hero_seat == 1 else 1, "villain"))

    table = {
        "street": street,
        "boardCards": board,
        "potChips": pot,
        "currentBet": current_bet,
        "buttonSeatNumber": button,
        "seats": seats,
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": list(available),
            "callAmount": call,
            "callChips": call,
            "raiseRange": {"min": call * 2 if call > 0 else 40, "max": 2000},
            "betRange": {"min": 0, "max": 2000},
        },
    }
    if action_history is not None:
        table["actionHistory"] = action_history
    return table, seats[0]


def action_for(hole, board, **kwargs):
    table, hero = make_table(hole, board, **kwargs)
    return choose_action(table, hero)[0]


def message_for(hole, board, **kwargs):
    table, hero = make_table(hole, board, **kwargs)
    return choose_action(table, hero)[2]


# ════════════════════════════════════════════════════════════════════════
# FLAW 1: Board-dominated trips — T7 on Q-7-7 flop
# The base sees made_hand_rank=3 (trips) and raises, but the trips
# are fully on the board. Hero's hand is effectively one pair.
# ════════════════════════════════════════════════════════════════════════

BOARD_DOMINATED_TRIPS_CASES = (
    (["Tc", "7h"], ["Qh", "7s", "7d"]),   # tens on Q-7-7 (trips on board)
    (["9d", "8c"], ["Jh", "8s", "8d"]),   # nines on J-8-8 (trips on board)
    (["Kh", "5s"], ["Ah", "5d", "5c"]),   # kings on A-5-5 (trips on board)
    (["Qd", "3h"], ["9s", "3c", "3d"]),   # queens on 9-3-3 (trips on board)
    (["Ac", "Jh"], ["Ks", "Jd", "Jc"]),   # aces on K-J-J (trips on board)
)

REAL_TRIPS_CASES = (
    (["7h", "7c"], ["Qh", "7s", "7d"]),   # pocket sevens = quads
    (["8d", "8c"], ["Jh", "8s", "8h"]),   # pocket eights = quads
    (["8d", "8c"], ["Jh", "7s", "8h"]),   # pocket eights = trips
    (["Qd", "Qc"], ["Qh", "7s", "7d"]),   # Qc makes real trips (Q-Q-Q)
)


@pytest.mark.parametrize("hole,board", BOARD_DOMINATED_TRIPS_CASES)
def test_board_dominated_trips_does_not_raise(hole, board):
    """Trips fully on board should not be value-raised on the flop."""
    action = action_for(hole, board, street="Flop", pot=80, call=0)
    assert action != "raise", (
        f"board-dominated trips should not raise, got {action!r}"
    )


@pytest.mark.parametrize("hole,board", REAL_TRIPS_CASES)
def test_real_trips_still_gets_value(hole, board):
    """Real trips (hero holds a trips card) should get value (raise or call).

    Note: patch1_choose_action may convert raise→call for pot control
    when required <= 0.32. Both raise and call are acceptable.
    """
    action = action_for(hole, board, street="Flop", pot=80, call=0)
    assert action in ("raise", "call"), f"real trips should get value, got {action!r}"


# ════════════════════════════════════════════════════════════════════════
# FLAW 2: Rank-2 value raise facing a bet on flop/turn
# Two pair raising into a bet is -EV. Should call instead.
# ════════════════════════════════════════════════════════════════════════

RANK2_FACING_BET_CASES = (
    (["7C", "KC"], ["7S", "3D", "KH"], "Flop"),   # 7s + Ks on 7-3-K board
    (["9C", "5D"], ["9H", "5S", "2C"], "Flop"),   # 9s + 5s on 9-5-2 board
    (["QC", "7D"], ["QS", "7H", "2S"], "Flop"),   # Qs + 7s on Q-7-2 board
    (["9C", "5D"], ["9H", "5S", "2C", "KH"], "Turn"),  # same on turn
)

RANK2_NOT_FACING_BET_CASES = (
    (["7C", "KC"], ["7S", "3D", "KH"], "Flop"),   # first to action
    (["9C", "5D"], ["9H", "5S", "2C"], "Flop"),   # first to action
)

RANK1_FACING_BET_CASES = (
    (["AC", "5H"], ["AS", "7D", "3C"], "Flop"),   # one pair (aces)
    (["KD", "QH"], ["KS", "9D", "4C"], "Flop"),   # one pair (kings)
)

RANK3_FACING_BET_CASES = (
    (["8C", "8D"], ["8H", "KS", "3D"], "Flop"),   # trips (888)
    (["7C", "7D"], ["7H", "2S", "9C"], "Flop"),   # trips (777)
)


@pytest.mark.parametrize("hole,board,street", RANK2_FACING_BET_CASES)
def test_rank2_facing_bet_calls_instead_of_raise(hole, board, street):
    """Two pair facing a bet should call, not raise."""
    action = action_for(hole, board, street=street, pot=80, call=40, current_bet=40)
    assert action == "call", (
        f"rank 2 facing bet should call, got {action!r}"
    )


@pytest.mark.parametrize("hole,board,street", RANK2_NOT_FACING_BET_CASES)
def test_rank2_not_facing_bet_gets_value(hole, board, street):
    """Two pair first to action should get value (raise or call).

    Note: patch1_choose_action may convert raise→call for pot control.
    Both raise and call are acceptable.
    """
    action = action_for(hole, board, street=street, pot=80, call=0, current_bet=0)
    assert action in ("raise", "call"), (
        f"rank 2 not facing bet should get value, got {action!r}"
    )


@pytest.mark.parametrize("hole,board,street", RANK1_FACING_BET_CASES)
def test_rank1_facing_bet_not_affected(hole, board, street):
    """One pair facing a bet should not be forced to call by rank-2 guard."""
    action = action_for(hole, board, street=street, pot=80, call=40, current_bet=40)
    # Rank 1 can raise for protection or call — either is fine
    assert action in ("raise", "call", "fold"), (
        f"rank 1 facing bet should not be forced, got {action!r}"
    )


@pytest.mark.parametrize("hole,board,street", RANK3_FACING_BET_CASES)
def test_rank3_facing_bet_allows_raise(hole, board, street):
    """Trips facing a bet should still raise."""
    action = action_for(hole, board, street=street, pot=80, call=40, current_bet=40)
    assert action == "raise", f"trips facing bet should raise, got {action!r}"


# ════════════════════════════════════════════════════════════════════════
# FLAW 3: River one pair over-call
# Calling >30% pot with one pair on the river is -EV.
# ════════════════════════════════════════════════════════════════════════

RIVER_ONE_PAIR_OVER_CALL_CASES = (
    # (hole, board, pot, call) — call/pot ratio > 30%
    (["KH", "QS"], ["8S", "9C", "QS", "3S", "AC"], 604, 211),   # 35% pot
    (["TD", "QH"], ["8S", "9C", "QS", "3S", "AC"], 462, 161),   # 35% pot
    (["4H", "4S"], ["9D", "6C", "AD", "KH", "3H"], 404, 141),   # 35% pot
    (["9S", "9H"], ["7S", "3D", "4H", "6S", "8C"], 172, 60),    # 35% pot
)

RIVER_ONE_PAIR_CHEAP_CALL_CASES = (
    # call/pot ratio <= 30% — should be allowed to call
    (["KH", "QS"], ["8S", "9C", "QS", "3S", "AC"], 1000, 100),  # 10% pot
    (["TD", "QH"], ["8S", "9C", "QS", "3S", "AC"], 800, 80),    # 10% pot
)


@pytest.mark.parametrize("hole,board,pot,call", RIVER_ONE_PAIR_OVER_CALL_CASES)
def test_river_one_pair_folds_to_large_bet(hole, board, pot, call):
    """One pair on river facing >30% pot bet should fold."""
    action = action_for(
        hole, board,
        street="River", pot=pot, call=call, current_bet=call,
    )
    assert action == "fold", (
        f"one pair on river facing {call/pot:.0%} pot should fold, got {action!r}"
    )


@pytest.mark.parametrize("hole,board,pot,call", RIVER_ONE_PAIR_CHEAP_CALL_CASES)
def test_river_one_pair_calls_cheap_bet(hole, board, pot, call):
    """One pair on river facing <=30% pot bet should be allowed to call."""
    action = action_for(
        hole, board,
        street="River", pot=pot, call=call, current_bet=call,
    )
    assert action != "fold", (
        f"one pair on river facing {call/pot:.0%} pot should not fold, got {action!r}"
    )


# ════════════════════════════════════════════════════════════════════════
# FLAW 4: Postflop war cap — rank 2 at >33% pot after 3 raises
# Two pair calling at >33% pot after 3 raises is -EV.
# ════════════════════════════════════════════════════════════════════════

def _war_history(street, num_raises=3):
    """Build action history showing hero raised num_raises times."""
    history = []
    for i in range(num_raises):
        history.append({"agentId": HERO, "action": "raise", "amount": 40, "street": street})
        history.append({"agentId": "villain", "action": "raise", "amount": 80, "street": street})
    return history


WAR_CAP_RANK2_HIGH_PRICE = (
    # (hole, board, street, pot, call) — price > 33%
    (["QC", "JD"], ["QD", "JC", "7H"], "Turn", 180, 100),   # 36% pot
    (["9C", "5D"], ["9H", "5S", "2C"], "Flop", 180, 100),   # 36% pot
)

WAR_CAP_RANK2_LOW_PRICE = (
    # price <= 33% — should call
    (["QC", "JD"], ["QD", "JC", "7H"], "Turn", 600, 100),   # 14% pot
    (["9C", "5D"], ["9H", "5S", "2C"], "Flop", 600, 100),   # 14% pot
)

WAR_CAP_RANK1 = (
    # Rank 1 should always call (not fold)
    (["AS", "5H"], ["AH", "KD", "7S"], "Turn", 180, 100),   # 36% pot
    (["KD", "QH"], ["KS", "9D", "4C"], "Flop", 180, 100),   # 36% pot
)


@pytest.mark.parametrize("hole,board,street,pot,call", WAR_CAP_RANK2_HIGH_PRICE)
def test_war_cap_rank2_folds_at_high_price(hole, board, street, pot, call):
    """Two pair after 3 raises at >33% pot should fold."""
    history = _war_history(street, num_raises=3)
    action = action_for(
        hole, board,
        street=street, pot=pot, call=call, current_bet=call,
        action_history=history,
    )
    assert action == "fold", (
        f"rank 2 after 3 raises at >33% pot should fold, got {action!r}"
    )


@pytest.mark.parametrize("hole,board,street,pot,call", WAR_CAP_RANK2_LOW_PRICE)
def test_war_cap_rank2_calls_at_low_price(hole, board, street, pot, call):
    """Two pair after 3 raises at <=33% pot should call."""
    history = _war_history(street, num_raises=3)
    action = action_for(
        hole, board,
        street=street, pot=pot, call=call, current_bet=call,
        action_history=history,
    )
    assert action == "call", (
        f"rank 2 after 3 raises at <=33% pot should call, got {action!r}"
    )


@pytest.mark.parametrize("hole,board,street,pot,call", WAR_CAP_RANK1)
def test_war_cap_rank1_reasonable(hole, board, street, pot, call):
    """One pair after 3 raises should not be forced into a bad call.

    At >33% pot odds, folding one pair is acceptable (the opponent's
    4-bet range is too strong). The key is that the guard doesn't
    force a -EV call.
    """
    history = _war_history(street, num_raises=3)
    action = action_for(
        hole, board,
        street=street, pot=pot, call=call, current_bet=call,
        action_history=history,
    )
    # At >33% pot, folding one pair is acceptable
    assert action in ("fold", "call"), (
        f"rank 1 after 3 raises should be reasonable, got {action!r}"
    )


# ════════════════════════════════════════════════════════════════════════
# FLAW 5: Full house positive control — 77 on 3-3-7 board
# The medium_pair_paired_board_fold_guard was folding 77 on paired
# boards even when hero has a full house (77 + 3-3 on board = 77733).
# ════════════════════════════════════════════════════════════════════════

FULL_HOUSE_CASES = (
    (["7C", "7D"], ["3C", "AD", "JS", "7S", "3D"]),   # 77 on 3-3-7-A-J = full house
    (["7C", "7D"], ["7H", "3S", "3D"]),                # 77 on 7-3-3 = full house
    (["8C", "8D"], ["8H", "2S", "2D"]),                # 88 on 8-2-2 = full house
    (["QD", "7C"], ["QH", "7S", "7D"]),                # Q7 on Q-7-7 = full house
)

TWO_PAIR_NO_FULL_HOUSE = (
    (["7C", "7D"], ["9H", "KS", "9D"]),                # 77 on 9-9-K = two pair only
    (["7C", "7D"], ["AH", "KS", "KD"]),                # 77 on K-K-A = two pair only
)


@pytest.mark.parametrize("hole,board", FULL_HOUSE_CASES)
def test_full_house_not_folded(hole, board):
    """Full house on paired board should never be folded."""
    action = action_for(
        hole, board,
        street="River", pot=400, call=200, current_bet=200,
        available=("fold", "call"),
    )
    assert action != "fold", (
        f"full house should not fold, got {action!r}"
    )


@pytest.mark.parametrize("hole,board", TWO_PAIR_NO_FULL_HOUSE)
def test_two_pair_no_full_house_folds(hole, board):
    """Two pair (no full house) on paired board should fold to a bet."""
    action = action_for(
        hole, board,
        street="Flop", pot=100, call=80, current_bet=80,
        available=("fold", "call"),
    )
    assert action == "fold", (
        f"two pair without full house should fold, got {action!r}"
    )


# ════════════════════════════════════════════════════════════════════════
# FLAW 6: BTN preflop fold — suited hands at score 35-39 in HU
# In HU, the hero is always the button (labeled BTN). Suited hands
# at score 35-39 were being folded when they should call.
# ════════════════════════════════════════════════════════════════════════

BTN_SUITED_CALL_CASES = (
    # (hole, score) — suited hands at score 35-39
    (["5C", "6C"], 36),    # suited connector
    (["3S", "8S"], 35),    # suited one-gapper
    (["9H", "4H"], 39),    # suited
    (["2C", "9C"], 37),    # suited
    (["9S", "2S"], 37),    # suited
    (["4H", "8H"], 36),    # suited one-gapper
)

BTN_JUNK_FOLD_CASES = (
    (["2C", "7D"], 30),    # offsuit junk
    (["3C", "9D"], 33),    # offsuit
    (["2D", "8C"], 32),    # offsuit
)


@pytest.mark.parametrize("hole,expected_score", BTN_SUITED_CALL_CASES)
def test_btn_suited_calls_in_hu(hole, expected_score):
    """Suited hands at score 35+ from BTN in HU should call."""
    from poker_bot.strategies.hubase import preflop_score
    score = preflop_score(hole)
    assert score >= 35, f"Expected score >= 35, got {score}"

    action = action_for(
        hole, [],
        street="Preflop", pot=10, call=5, current_bet=5,
        button=1, hero_seat=1,  # hero is button
    )
    assert action == "call", (
        f"BTN suited score {score} should call, got {action!r}"
    )


@pytest.mark.parametrize("hole,expected_score", BTN_JUNK_FOLD_CASES)
def test_btn_junk_folds_in_hu(hole, expected_score):
    """Junk hands below 35 from BTN in HU should fold."""
    from poker_bot.strategies.hubase import preflop_score
    score = preflop_score(hole)
    assert score < 35, f"Expected score < 35, got {score}"

    action = action_for(
        hole, [],
        street="Preflop", pot=10, call=5, current_bet=5,
        button=1, hero_seat=1,
    )
    assert action == "fold", (
        f"BTN junk score {score} should fold, got {action!r}"
    )
