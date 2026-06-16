"""Scenario tests for flattened_v5 defense guards.

These lock in the v008 preflop defense behavior: playable blind hands should be
defended at capped prices, and effective_pot() should recover the true pot in the
selfplay simulator when posted/current bets live on seats instead of potChips.

The same file also covers the postflop SPR commitment guard: with two pair or
better, a low effective SPR after calling should override a base fold. SPR tests
cover both the selfplay live-bet representation and the completed-pot
representation.
"""

from poker_bot.strategies import flattened_v5 as strategy


def make_seats(bets, stacks, hero_seat, hero_cards):
    seats = []
    for index in range(6):
        seat_number = index + 1
        seats.append(
            {
                "agentId": f"villain-{seat_number}",
                "seatNumber": seat_number,
                "holeCards": [],
                "stackChips": stacks[index],
                "currentBetChips": bets[index] or 0,
                "folded": bets[index] is None and seat_number != hero_seat,
                "hasFolded": False,
            }
        )
    hero = seats[hero_seat - 1]
    hero["agentId"] = "hero"
    hero["holeCards"] = hero_cards
    hero["folded"] = False
    return seats, hero


def make_table(hero_cards, hero_seat=3, button=1, bets=None, stacks=None):
    # button=1 -> six-max order [BTN=1, SB=2, BB=3, UTG=4, MP=5, CO=6]
    bets = bets if bets is not None else [150, 25, 50, None, None, None]
    stacks = stacks or [2000] * 6
    seats, hero = make_seats(bets, stacks, hero_seat, hero_cards)
    max_bet = max(b or 0 for b in bets)
    call_amount = max_bet - (bets[hero_seat - 1] or 0)
    return {
        "street": "Preflop",
        "boardCards": [],
        "potChips": 0,  # simulator reports 0 preflop; blinds live on seats
        "buttonSeatNumber": button,
        "seats": seats,
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": ["fold", "call", "raise"],
            "callAmount": call_amount,
            "callChips": call_amount,
        },
    }, hero


def test_effective_pot_includes_posted_blinds():
    table, _hero = make_table(["AS", "QH"])
    # 150 + 25 + 50 = 225 of live bets, potChips=0
    assert strategy.effective_pot(table) == 225


def test_bb_defends_aqo_against_open():
    table, hero = make_table(["AS", "QH"])  # AQo, score 66
    base = ("fold", None, "base over-folds")
    decision = strategy.preflop_blind_defense(table, hero, base)
    assert decision is not None
    assert decision[0] == "call"


def test_bb_defends_ato_cheap():
    table, hero = make_table(["AH", "TC"], bets=[60, 25, 50, None, None, None])
    base = ("fold", None, "base over-folds")
    decision = strategy.preflop_blind_defense(table, hero, base)
    assert decision is not None
    assert decision[0] == "call"


def test_bb_folds_junk():
    table, hero = make_table(["7H", "2C"])  # 72o, well below min score
    base = ("fold", None, "base folds")
    assert strategy.preflop_blind_defense(table, hero, base) is None


def test_early_position_not_widened():
    # Hero UTG (seat 4): not a blind, must keep the base's tight fold.
    table, hero = make_table(
        ["AS", "QH"], hero_seat=4, bets=[None, 25, 50, 150, None, None]
    )
    base = ("fold", None, "base folds")
    assert strategy.preflop_blind_defense(table, hero, base) is None


def test_does_not_override_non_fold_base():
    table, hero = make_table(["AS", "QH"])
    base = ("raise", 300, "base raises")
    assert strategy.preflop_blind_defense(table, hero, base) is None


def test_price_cap_blocks_expensive_defend():
    # BB facing a huge 3-bet jam: price exceeds the BB cap, so no defend.
    table, hero = make_table(["AS", "QH"], bets=[1200, 25, 50, None, None, None])
    base = ("fold", None, "base folds")
    assert strategy.preflop_blind_defense(table, hero, base) is None


def test_spr_commitment_lock_uses_post_call_stack_and_pot():
    table, hero = make_table(["AS", "KH"], bets=[None] * 6)
    table["street"] = "Flop"
    table["boardCards"] = ["AS", "KH", "2H"]  # two pair; strong enough for SPR lock
    table["potChips"] = 0  # current bet is live on the bettor's seat

    # Only villain-1 is live and has bet 20. Hero is all-in calling 20, so the
    # genuine post-call SPR is 0 / 40, not the old effective_pot / potChips.
    for seat in table["seats"]:
        seat["currentBetChips"] = 0
        if seat["seatNumber"] not in {1, 3}:
            seat["folded"] = True
    table["seats"][0]["currentBetChips"] = 20
    hero["stackChips"] = 20
    table["allowedActions"] = {
        "availableActions": ["fold", "call"],
        "callAmount": 20,
        "callChips": 20,
    }

    decision = strategy.spr_commitment_lock(table, hero, ("fold", None, "base folds"))
    assert decision is not None
    assert decision[0] == "call"
    assert "spr 0.00 < 3.0" in decision[2]



def make_postflop_table(
    *,
    hero_cards,
    board_cards,
    pot=100,
    call=50,
    hero_stack=200,
    live_bet=0,
    opponent_stacks=None,
    available=("fold", "call"),
):
    opponent_stacks = opponent_stacks or [200]
    seats = [
        {
            "agentId": "hero",
            "seatNumber": 1,
            "holeCards": hero_cards,
            "stackChips": hero_stack,
            "currentBetChips": 0,
            "folded": False,
            "hasFolded": False,
        }
    ]
    for index, stack in enumerate(opponent_stacks, start=2):
        seats.append(
            {
                "agentId": f"v{index}",
                "seatNumber": index,
                "holeCards": [],
                "stackChips": stack,
                "currentBetChips": live_bet if index == 2 else 0,
                "folded": False,
                "hasFolded": False,
            }
        )
    return {
        "street": "River",
        "boardCards": board_cards,
        "potChips": pot,
        "buttonSeatNumber": 1,
        "seats": seats,
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": list(available),
            "callAmount": call,
            "callChips": call,
        },
    }, seats[0]



def test_spr_commitment_lock_completes_pot_representation():
    # pot=100, call=50, hero and villain both have 200.
    # After the call: effective stack = 150, pot = 150, SPR = 1.00.
    table, hero = make_postflop_table(
        hero_cards=["AS", "AD"],
        board_cards=["AS", "2H", "3D", "4C", "7S"],  # set
        pot=100,
        call=50,
        hero_stack=200,
        opponent_stacks=[200],
    )

    decision = strategy.spr_commitment_lock(table, hero, ("fold", None, "base folds"))

    assert decision is not None
    assert decision[0] == "call"
    assert "spr 1.00 < 3.0" in decision[2]



def test_spr_commitment_lock_ignores_high_spr():
    # pot=100, call=10, hero and villain both have 1000.
    # After the call: effective stack = 990, pot = 110, SPR = 9.00.
    table, hero = make_postflop_table(
        hero_cards=["AS", "AD"],
        board_cards=["AS", "2H", "3D", "4C", "7S"],  # set
        pot=100,
        call=10,
        hero_stack=1000,
        opponent_stacks=[1000],
    )

    decision = strategy.spr_commitment_lock(table, hero, ("fold", None, "base folds"))

    assert decision is None



def test_spr_commitment_lock_uses_multiway_threshold():
    # Four live opponents lower the SPR threshold to 1.5.
    table, hero = make_postflop_table(
        hero_cards=["AS", "AD"],
        board_cards=["AS", "2H", "3D", "4C", "7S"],  # set
        pot=100,
        call=50,
        hero_stack=100,
        opponent_stacks=[100, 100, 100, 100],
    )

    decision = strategy.spr_commitment_lock(table, hero, ("fold", None, "base folds"))

    assert decision is not None
    assert decision[0] == "call"
    assert "spr 0.33 < 1.5" in decision[2]



def test_spr_commitment_lock_requires_two_pair_or_better():
    # Same low SPR as the completed-pot test, but only top pair should not trigger.
    table, hero = make_postflop_table(
        hero_cards=["AS", "KH"],
        board_cards=["AS", "2H", "3D", "4C", "5S"],  # top pair, weak kicker
        pot=100,
        call=50,
        hero_stack=200,
        opponent_stacks=[200],
    )

    decision = strategy.spr_commitment_lock(table, hero, ("fold", None, "base folds"))

    assert decision is None



def test_spr_commitment_lock_only_rescues_base_folds():
    table, hero = make_postflop_table(
        hero_cards=["AS", "AD"],
        board_cards=["AS", "2H", "3D", "4C", "7S"],  # set
        pot=100,
        call=50,
        hero_stack=200,
        opponent_stacks=[200],
    )

    decision = strategy.spr_commitment_lock(table, hero, ("raise", 150, "base raises"))

    assert decision is None

