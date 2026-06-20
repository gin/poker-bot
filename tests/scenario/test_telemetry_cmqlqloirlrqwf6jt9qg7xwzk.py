"""Telemetry-driven scenario test for cmqlqloirlrqwf6jt9qg7xwzk.

Reference hand from telemetry-luigi-tournament.sqlite (strategy='s2v013'):
  - cmqlqloirlrqwf6jt9qg7xwzk: 6d Jd on 4s Tc Th 2c (Turn)
  - Hero has middle pair (tens) with J kicker
  - Board is paired (tens), making trips possible
  - Against a tight opponent, middle pair with weak kicker should check the Turn

The telemetry shows hero betting 33 chips on the Turn with the message
"survival value pressure: Thin value against simple". This test locks in the
strategically correct play: checking with middle pair on a paired board instead
of thin-value betting a dominated medium-strength hand.
"""

from poker_bot.strategies.s2base import choose_action


def make_turn_table(opponent_vpip=38, opponent_pfr=23):
    """Build the 6d Jd on 4s Tc Th 2c scenario from telemetry."""
    return {
        "street": "Turn",
        "boardCards": ["4s", "Tc", "Th", "2c"],
        "potChips": 157,
        "buttonSeatNumber": 5,
        "opponentProfiles": {
            "villain": {
                "hands_seen": 189,
                "vpip": opponent_vpip,
                "pfr": opponent_pfr,
            }
        },
        "seats": [
            {
                "seatNumber": 1,
                "agentId": "hero",
                "holeCards": ["6d", "Jd"],
                "folded": False,
                "stackChips": 157,
                "currentBetChips": 0,
            },
            {
                "seatNumber": 2,
                "agentId": "villain",
                "holeCards": [],
                "folded": False,
                "stackChips": 157,
                "currentBetChips": 0,
            },
        ],
        "allowedActions": {
            "availableActions": ["fold", "check", "bet", "all-in"],
            "callAmount": 0,
            "minBet": 10,
            "minRaiseTo": 20,
        },
    }


def act(hero_cards, board_cards, pot, profiles=None):
    """Call s2base.choose_action with a minimal table setup."""
    table = make_turn_table()
    table["boardCards"] = board_cards
    table["potChips"] = pot
    table["seats"][0]["holeCards"] = hero_cards
    if profiles is not None:
        table["opponentProfiles"] = profiles
    action, _amount, _message = choose_action(table, table["seats"][0])
    return action


def test_middle_pair_on_paired_board_checks_vs_tight_turn():
    """6d Jd on 4s Tc Th 2c vs tight opponent Turn → check.

    Reference: cmqlqloirlrqwf6jt9qg7xwzk from telemetry.
    Hero has middle pair (tens) with J kicker on a paired board. Against a
    tight opponent (VPIP 38%, PFR 23% in telemetry, labeled "tight"), the Turn
    bet is thin value that's likely dominated. Checking is strategically
    correct — medium-strength hands with weak kickers should not thin-value bet
    on paired boards against tight opponents.
    """
    profiles = {
        "villain": {
            "hands_seen": 189,
            "vpip": 30,  # 15.9% — telemetry labels this opponent "tight"
            "pfr": 44,  # 23.3%
            "calls": 30,
            "bets": 30,
            "raises": 30,
            "folds": 99,
        }
    }
    assert act(["6d", "Jd"], ["4s", "Tc", "Th", "2c"], pot=157, profiles=profiles) == "check"
