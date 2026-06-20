"""Telemetry-driven scenario tests for s2v009 poker gameplay flaws.

Reference hands from telemetry-luigi-tournament.sqlite (strategy='s2v009'):
  - cmqi8w7prprmabezqlfim1ele: Qd Tc on Jd Kc 3d Js Kd (fragile two pair on
    paired board) called against tight opponent
  - cmqjqdfklzvb5bezqsz6iucz1: 6d 6s on Qs Kh Js 4h 4d (fragile two pair on
    paired board) called against loose opponent
"""
from poker_bot.strategies.s2base import choose_action


HERO = "hero"


def table(hole, board, *, street="River", pot=400, call=100, available=("fold", "call"), profiles=None):
    """Minimal table/seat the strategy can act on."""
    seats = [
        {"seatNumber": 1, "agentId": HERO, "holeCards": hole, "folded": False, "stackChips": 5000, "currentBetChips": 0},
        {"seatNumber": 2, "agentId": "villain", "holeCards": [], "folded": False, "stackChips": 5000, "currentBetChips": call},
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


def act(hole, board, **kw):
    action, _amount, _reason = choose_action(*table(hole, board, **kw))
    return action


# ── Fragile two pair on paired board: opponent stats matter ───────────────────

def test_fragile_two_pair_on_paired_board_folds_vs_tight_opponent():
    """Qd Tc on Jd Kc 3d Js Kd vs tight opponent → fold.

    Reference: cmqi8w7prprmabezqlfim1ele from telemetry.
    Hero has two pair (Jacks + Kings) but the board is double-paired (J, K).
    Against a tight opponent's river bet, the range is condensed toward full
    houses. Folding is correct.
    """
    profiles = {"villain": {"hands_seen": 50, "vpip": 6, "pfr": 4}}
    assert act(["Qd", "Tc"], ["Jd", "Kc", "3d", "Js", "Kd"], pot=403, call=134, profiles=profiles) == "fold"


def test_fragile_two_pair_on_paired_board_continues_vs_loose_opponent():
    """6d 6s on Qs Kh Js 4h 4d vs loose opponent → continue.

    Reference: cmqjqdfklzvb5bezqsz6iucz1 from telemetry.
    Hero has two pair (6s + 4s) on a paired board. Against a loose opponent,
    the betting range is wider with more bluffs and thinner value, so continuing
    with a call is reasonable.
    """
    profiles = {"villain": {"hands_seen": 50, "vpip": 23, "pfr": 15}}
    assert act(["6d", "6s"], ["Qs", "Kh", "Js", "4h", "4d"], pot=416, call=145, profiles=profiles) != "fold"
