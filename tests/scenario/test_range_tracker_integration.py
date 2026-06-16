from poker_bot.range_model import BayesianRangeTracker
from poker_bot.strategies import flattened_v5


def test_tracker_pressure_summary_is_available_to_flattened_v5(tmp_path):
    flattened_v5._RANGE_TRACKER = BayesianRangeTracker(tmp_path)
    table = {
        "street": "Flop",
        "buttonSeatNumber": 1,
        "potChips": 100,
        "currentBet": 10,
        "boardCards": ["AS", "KH", "7D"],
        "allowedActions": {
            "availableActions": ["fold", "call", "raise"],
            "callAmount": 10,
            "minBet": 2,
        },
        "seats": [
            {
                "agentId": "villain",
                "seatNumber": 1,
                "currentBetChips": 10,
                "stackChips": 1000,
            },
            {
                "agentId": "hero",
                "seatNumber": 2,
                "holeCards": ["QD", "JD"],
                "currentBetChips": 0,
                "stackChips": 1000,
            },
        ],
    }
    hero = table["seats"][1]

    summary = flattened_v5.bayesian_pressure_summary(table, hero)

    assert summary["tracker_samples"] == 1
    assert 0.0 <= summary["tracker_strength"] <= 1.0
    assert 0.0 <= summary["tracker_capped_probability"] <= 1.0
    assert tmp_path.exists()
