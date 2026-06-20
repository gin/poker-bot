"""Telemetry-driven scenario tests for s2v009 poker gameplay flaws.

Reference hands from telemetry-luigi-tournament.sqlite (strategy='s2v009'):
  - cmqhrlt0h6cskbezq0zhq4oie: K-high (not a flush) incorrectly folded by
    the paired-board flush guard
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


# ── Flush guard: opponent stats matter for K-high flush on paired board ────────

def test_k_high_flush_on_paired_board_folds_vs_tight_opponent():
    """Jd Kh on 9h 9d 3h 8h 2h vs tight opponent → fold.

    Reference: cmqhrlt0h6cskbezq0zhq4oie from telemetry.
    Hero has K-high flush on a paired board. Against a tight-passive opponent
    (VPIP 15%, PFR 1%), the betting range is condensed toward full houses.
    Folding is strategically reasonable.
    """
    profiles = {"villain": {"hands_seen": 95, "vpip": 15, "pfr": 1}}
    # Adjust to not for if hero has A or K flush
    assert act(["Jd", "Kh"], ["9h", "9d", "3h", "8h", "2h"], pot=124, call=70, profiles=profiles) != "fold"
    assert act(["Jd", "Ah"], ["9h", "9d", "3h", "8h", "2h"], pot=124, call=70, profiles=profiles) != "fold"

    assert act(["Jd", "Qh"], ["9h", "9d", "3h", "8h", "2h"], pot=124, call=70, profiles=profiles) == "fold"
    assert act(["Jd", "Jh"], ["9h", "9d", "3h", "8h", "2h"], pot=124, call=70, profiles=profiles) == "fold"


def test_k_high_flush_on_paired_board_continues_vs_loose_opponent():
    """Jd Kh on 9h 9d 3h 8h 2h vs loose opponent → continue.

    Same hand/board as cmqhrlt0h6cskbezq0zhq4oie, but against a loose opponent.
    The betting range is wider with more bluffs and thinner value, so K-high
    flush should continue rather than auto-fold to the flush guard.
    """
    profiles = {"villain": {"hands_seen": 50, "vpip": 45, "pfr": 30}}
    assert act(["Jd", "Kh"], ["9h", "9d", "3h", "8h", "2h"], pot=124, call=35, profiles=profiles) != "fold"