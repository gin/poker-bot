"""Scenario tests for flattened_v5 defense guards.

These lock in the v008 preflop defense behavior: playable blind hands should be
defended at capped prices, and effective_pot() should recover the true pot in the
selfplay simulator when posted/current bets live on seats instead of potChips.

The same file also covers the postflop SPR commitment guard: with two pair or
better, a low effective SPR after calling should override a base fold.
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
