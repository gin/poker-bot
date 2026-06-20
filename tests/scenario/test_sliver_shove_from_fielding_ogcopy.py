# """
# Min-raise-war / sliver-shove behavioral spec.
# Some bots decide a call by comparing HAND STRENGTH (or equity vs the opponent's
# perceived value range) to a threshold — WITHOUT re-pricing the bet against the
# pot. Attackers farm this: min-raise back and forth to inflate the pot cheaply
# (each min-raise is a tiny *additional* call, so naive pot-odds logic keeps
# calling), then shove a SLIVER at the end — e.g. 100 into a 9,000 pot.
# That sliver lays ~90:1: you need ~1.1% equity to call, which ANY two live cards
# clear. A bot that folds "to a raise / all-in" unless its hand is strong FOLDS and
# ships the whole pot for a 100-chip bet. Folding is -EV at *every* equity here.
# This file is a behavioral contract, NOT an implementation. Each test states how
# the strategy SHOULD act; how you satisfy it is up to you.
# Setup:
#   - Adjust the import below to wherever your choose_action(table, my_seat) lives.
#   - Tweak the `table()` helper if your table/seat schema differs.
#   - Run:  pytest test_sliver_shove_spec.py -v
# The "control" tests at the bottom must stay green: vs a NORMAL-sized bet, folding
# air is still correct. The fix is "never fold a priced sliver," NOT "call
# everything" — a change that turns the bot into a calling station is over-correcting.
# """
# import pytest

# # >>> POINT THIS AT YOUR STRATEGY <<<
# from poker_bot.strategies.profiled import choose_action


# HERO = "hero"


# def table(hole, board, *, street="River", pot=9000, call=100,
#           available=("fold", "call"), villains=2, profiles=None):
#     """Minimal table/seat the strategy can act on: facing an all-in on the river."""
#     seats = [{"seatNumber": 1, "agentId": HERO, "holeCards": hole, "folded": False}]
#     for i in range(villains):
#         seats.append({"seatNumber": 2 + i, "agentId": f"v{i}",
#                       "holeCards": [], "folded": False})
#     t = {
#         "street": street,
#         "boardCards": board,
#         "potChips": pot,
#         "seats": seats,
#         "opponentProfiles": profiles or {},
#         "allowedActions": {
#             "availableActions": list(available),
#             "callAmount": call,
#             "raiseRange": {"min": call * 2, "max": 2000},
#             "betRange": {"min": 0, "max": 2000},
#         },
#     }
#     return t, seats[0]


# def act(hole, board, **kw):
#     action, _amount, _reason = choose_action(*table(hole, board, **kw))
#     return action


# # ── the pot is huge, the shove is a sliver -> the price says CALL, never fold ──
# def test_air_does_not_fold_sliver_shove_90to1():
#     # 100 into 9000 = ~90:1; ~1% equity needed. 7-high clears it. Folding = farmed.
#     assert act(["7h", "2d"], ["Ks", "Qd", "9c", "4s", "3h"], pot=9000, call=100) != "fold"


# def test_top_pair_does_not_fold_sliver_shove():
#     # a made hand makes it stark: folding "to a raise" ships a 9000 pot for 100
#     assert act(["Jh", "9c"], ["Js", "6d", "3c", "8h", "2s"], pot=9000, call=100) != "fold"


# def test_air_does_not_fold_sliver_30to1():
#     assert act(["7h", "2d"], ["Ks", "Qd", "9c", "4s", "3h"], pot=9000, call=300) != "fold"


# def test_air_does_not_fold_sliver_multiway():
#     # the real spot: a min-raise war among several bots, then a sliver shove
#     assert act(["Th", "4c"], ["Ks", "Qd", "9c", "4s", "3h"],
#                pot=6000, call=150, villains=3) != "fold"


# # ── controls: vs NORMAL sizing, folding air is still correct (not a station) ───
# def test_air_folds_to_half_pot_bet():
#     assert act(["7h", "2d"], ["Ks", "Qd", "9c", "4s", "3h"], pot=600, call=300) == "fold"


# def test_air_folds_to_pot_bet():
#     assert act(["7h", "2d"], ["Ks", "Qd", "9c", "4s", "3h"], pot=600, call=600) == "fold"
