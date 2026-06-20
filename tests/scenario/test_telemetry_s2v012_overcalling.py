"""Telemetry-driven scenario tests for s2v012 over-calling flaws.

Reference hand from telemetry-luigi-tournament.sqlite (strategy='s2v012'):
  - cmql7dvor5s9vf6jtank1kk27: 5c Ks on As 2s 5s Kc, facing Turn jam
    from tight-aggressive opponent (VPIP 16%, PFR 14%) → should fold.

The bot called 2490 into 3122 pot (79.7% pot odds, 220% of remaining stack)
with the message "anti-bully continue rank 2". Against a TAG opponent with
0 all-ins and 0 large bets in 127 hands, this is a massive over-call.
"""

from poker_bot.strategies.s2base import choose_action


def make_turn_table(opponent_vpip=16, opponent_pfr=14):
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


def act(hero_cards, board_cards, pot, call, profiles=None):
    """Call s2v012.choose_action with a minimal table setup."""
    table = make_turn_table()
    table["boardCards"] = board_cards
    table["potChips"] = pot
    table["seats"][0]["holeCards"] = hero_cards
    table["seats"][0]["stackChips"] = 1131
    table["seats"][1]["currentBetChips"] = call
    table["allowedActions"]["callAmount"] = call
    if profiles is not None:
        table["opponentProfiles"] = profiles
    action, _amount, _message = choose_action(table, table["seats"][0])
    return action


# ── Over-calling large bets with medium-strength hands ────────────────────────

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
    assert act(["5c", "Ks"], ["As", "2s", "5s", "Kc"], pot=3122, call=2490, profiles=profiles) == "fold"


def test_two_pair_on_paired_board_continues_vs_loose_aggressive_turn_jam():
    """5c Ks on As 2s 5s Kc vs loose opponent Turn jam → continue.

    Same hand/board as cmql7dvor5s9vf6jtank1kk27, but against a loose-aggressive
    opponent (VPIP 43%, PFR 23%). The jamming range is wider with more bluffs
    and thinner value, so two pair should continue rather than auto-fold.
    """
    profiles = {"villain": {"hands_seen": 127, "vpip": 43, "pfr": 23}}
    assert act(["5c", "Ks"], ["As", "2s", "5s", "Kc"], pot=3122, call=2490, profiles=profiles) != "fold"
