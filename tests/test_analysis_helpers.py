from collections import deque

from poker_bot.analysis.opponent_tendencies import summarize_tendencies
from poker_bot.analysis.table_context import active_opponents, table_agent_stats
from poker_bot.opponents import OpponentProfile


def test_summarize_tendencies_uses_local_profiles():
    profile = OpponentProfile(
        agent_id="villain",
        hands_seen=40,
        vpip=24,
        calls=20,
        folds=4,
        recent_actions=deque(maxlen=20),
    )

    tendencies = summarize_tendencies([profile])

    assert tendencies.has_calling_station
    assert tendencies.avg_vpip == 0.6
    assert tendencies.confidence == 1.0


def test_summarize_tendencies_uses_api_playing_style():
    stats = {
        "agentId": "villain",
        "sampleSize": 191,
        "vpip": 0.115,
        "pfr": 0.042,
        "af": 1.4,
        "playingStyle": {
            "label": "tight-measured",
            "tightness": "tight",
            "aggression": "measured",
            "archetype": "nit",
        },
    }

    tendencies = summarize_tendencies(agent_stats=[stats])

    assert tendencies.has_patient
    assert tendencies.all_patient
    assert tendencies.avg_pfr == 0.042
    assert tendencies.confidence == 1.0


def test_table_context_extracts_active_opponents_and_agent_stats():
    hero = {"agentId": "hero", "seatNumber": 1}
    table = {
        "seats": [
            hero,
            {"agentId": "active", "seatNumber": 2},
            {"agentId": "folded", "seatNumber": 3, "folded": True},
        ],
        "opponentAgentStats": {
            "active": {"agentId": "active", "sampleSize": 20},
            "hero": {"agentId": "hero", "sampleSize": 20},
        },
    }

    assert active_opponents(table, hero) == 1
    assert table_agent_stats(table, hero) == [{"agentId": "active", "sampleSize": 20}]
