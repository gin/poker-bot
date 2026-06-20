"""Scenario tests for s2base top-pair continuation flaws.

Regression tests for telemetry hand 456144ac3a2b465498a111a4886b956b:
Kd Qc on 8s 9h Qs, facing 50% pot bet on the flop.

The bot folded with "Rank 1 below price, folding" even though top pair with
K kicker on a dry Q-high board is a clear call at 2:1 pot odds.
"""

import pytest

from poker_bot.strategies import s2base as strategy


def make_top_pair_table(opponent_vpip=0.5, opponent_pfr=0.3):
    """Build the Kd Qc on 8s 9h Qs scenario from telemetry."""
    return {
        "street": "Flop",
        "potChips": 890,
        "currentBet": 445,
        "buttonSeatNumber": 5,
        "opponentProfiles": {
            "opp4": {
                "hands_seen": 50,
                "vpip": int(round(opponent_vpip * 50)),
                "pfr": int(round(opponent_pfr * 50)),
            }
        },
        "seats": [
            {"seatNumber": 1, "agentId": "opp1", "stackChips": 5000, "holeCards": []},
            {"seatNumber": 2, "agentId": "opp2", "stackChips": 5000, "holeCards": []},
            {
                "seatNumber": 3,
                "agentId": "hero",
                "stackChips": 888,
                "holeCards": ["Kd", "Qc"],
            },
            {"seatNumber": 4, "agentId": "opp4", "stackChips": 5000, "holeCards": []},
            {"seatNumber": 5, "agentId": "btn", "stackChips": 5000, "holeCards": []},
            {"seatNumber": 6, "agentId": "opp6", "stackChips": 5000, "holeCards": []},
        ],
        "boardCards": ["8s", "9h", "Qs"],
        "allowedActions": {
            "availableActions": ["fold", "call", "raise", "all-in"],
            "callAmount": 445,
            "minBet": 40,
            "minRaiseTo": 890,
        },
    }


def test_top_pair_kicker_continues_at_50pct_pot():
    """Kd Qc on 8s 9h Qs vs 50% pot bet → call.

    Regression test for telemetry hand 456144ac3a2b465498a111a4886b956b.
    Hero has top pair with K kicker on a dry Q-high board, facing a 50% pot
    bet. The bot should call — it's getting 2:1 and needs only 33% equity.

    FIXME: Not part of the targeted set-mining patch — separate fix needed.
    """
    # pytest.xfail("top-pair continuation: separate fix needed")
    table = make_top_pair_table()
    hero = table["seats"][2]

    result = strategy.choose_action(table, hero)

    assert result is not None
    assert result[0] == "call", (
        f"Expected call for Kd Qc top pair vs 50% pot bet, got {result[0]}: {result[2]}"
    )


def test_top_pair_kicker_should_not_fold_to_loose_player_at_bad_price():
    """Kd Qc on 8s 9h Qs vs 80% pot bet from loose player → continue.

    Against a loose player, the bet range is wider with more bluffs and thinner
    value. Hero has top pair with K kicker on a dry Q-high board, so continuing
    with a call is reasonable even at a bad price.
    """
    table = make_top_pair_table(opponent_vpip=0.45, opponent_pfr=0.30)
    table["potChips"] = 500
    table["currentBet"] = 400
    table["allowedActions"]["callAmount"] = 400
    hero = table["seats"][2]

    result = strategy.choose_action(table, hero)

    assert result is not None
    assert result[0] != "fold", (
        f"Expected call/raise for Kd Qc top pair vs loose 80% pot bet, got {result[0]}: {result[2]}"
    )


def test_top_pair_kicker_folds_to_tight_player_at_bad_price():
    """Kd Qc on 8s 9h Qs vs 80% pot bet from tight player → fold.

    Against a tight player, the bet range is condensed toward strong value.
    Hero has top pair with K kicker, but the 80% pot bet price is too expensive
    to continue against a tight range.
    """
    table = make_top_pair_table(opponent_vpip=0.12, opponent_pfr=0.08)
    table["potChips"] = 500
    table["currentBet"] = 400
    table["allowedActions"]["callAmount"] = 400
    hero = table["seats"][2]

    result = strategy.choose_action(table, hero)

    assert result is not None
    assert result[0] == "fold", (
        f"Expected fold for Kd Qc top pair vs tight 80% pot bet, got {result[0]}: {result[2]}"
    )


def test_two_pair_still_raises_for_value():
    """TT on Ah Qd Ts (top set) → raise.

    Positive control: strong hands should still raise for value.
    """
    table = make_top_pair_table()
    table["boardCards"] = ["Ah", "Qd", "Ts"]
    table["seats"][2]["holeCards"] = ["Th", "Td"]
    hero = table["seats"][2]

    result = strategy.choose_action(table, hero)

    assert result is not None
    assert result[0] == "raise", (
        f"Expected raise for top set, got {result[0]}: {result[2]}"
    )
