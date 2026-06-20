"""Regression test for s2v009 non-nut flush River bluff-catch.

This captures the telemetry hand where hero held Jd Kh on a paired four-heart
board and the old vulnerable_flush_guard folded a cheap River bluff-catcher.
"""

from poker_bot.strategies import s2v009 as strategy


def test_s2v009_jd_kh_paired_board_river_facing_bet_bluff_catches():
    """Hero has JdKh on 9h9d3h8h2h facing 35 into 124 on the River.

    The old guard saw a non-nut flush on a paired board and folded the intended
    raise. Facing a bet, this is a bluff-catch spot, not a value-raise spot.
    """
    table = {
        "street": "River",
        "boardCards": ["9h", "9d", "3h", "8h", "2h"],
        "potChips": 124,
        "facing_bet": 1,
        "buttonSeatNumber": 4,
        "seats": [
            {
                "agentId": "cmq4d4i7n1g1sfahhe3gl5j3c",
                "seatNumber": 3,
                "holeCards": ["Jd", "Kh"],
                "stackChips": 957,
                "currentBetChips": 0,
                "folded": False,
                "hasFolded": False,
            },
            {
                "agentId": "villain",
                "seatNumber": 6,
                "holeCards": [],
                "stackChips": 3484,
                "currentBetChips": 35,
                "folded": False,
                "hasFolded": False,
            },
        ],
        "opponentProfiles": {},
        "allowedActions": {
            "availableActions": ["fold", "call", "raise", "all-in"],
            "callAmount": 35,
            "callChips": 35,
            "minBet": 2,
            "minRaiseTo": 70,
        },
    }
    hero = table["seats"][0]

    result = strategy.choose_action(table, hero)

    assert result is not None
    assert result[0] == "call"
    assert result[1] == 35
    assert result[2] == "non-nut flush on paired board: bluff catch"
