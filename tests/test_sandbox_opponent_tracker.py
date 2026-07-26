import poker_bot.sandbox_opponent_tracker as tracker


def setup_function():
    tracker._PROFILES.clear()
    tracker._SEEN_HAND_KEYS.clear()
    tracker._SEEN_EVENT_KEYS.clear()


def _table(hand_id, events):
    return {
        "handId": hand_id,
        "selfSeatNumber": 1,
        "street": "River",
        "seats": [
            {"agentId": "hero", "seatNumber": 1},
            {"agentId": "villain", "seatNumber": 2},
        ],
        "events": events,
    }


def test_snapshot_replay_counts_one_preflop_vpip_and_pfr_per_hand():
    events = [
        {"id": "1", "agentId": "villain", "street": "Preflop", "action": "raise"},
        {"id": "2", "agentId": "villain", "street": "Preflop", "action": "raise"},
        {"id": "3", "agentId": "villain", "street": "Flop", "action": "bet"},
        {"id": "4", "agentId": "villain", "street": "Turn", "action": "call"},
    ]
    tracker.enrich_table(_table("h1", events))
    tracker.enrich_table(_table("h1", events))
    profile = tracker._PROFILES["villain"]

    assert (profile.preflop_hands_seen, profile.vpip, profile.pfr) == (1, 1, 1)
    assert profile.legacy_vpip_action_count == 2
    assert profile.legacy_pfr_raise_count == 2

    tracker.enrich_table(
        _table(
            "h2",
            [{"id": "5", "agentId": "villain", "street": "Preflop", "action": "call"}],
        )
    )
    assert (profile.preflop_hands_seen, profile.vpip, profile.pfr) == (2, 2, 1)


def test_live_all_in_call_counts_vpip_without_pfr():
    table = _table(
        "all-in-call",
        [
            {
                "id": "all-in-call",
                "agentId": "villain",
                "street": "Preflop",
                "action": "all-in",
                "toAmount": 100,
            }
        ],
    )
    table["currentBet"] = 100

    tracker.enrich_table(table)
    profile = tracker._PROFILES["villain"]

    assert (profile.preflop_hands_seen, profile.vpip, profile.pfr) == (1, 1, 0)
    assert profile.legacy_pfr_raise_count == 0


def test_live_all_in_raise_counts_vpip_and_pfr():
    table = _table(
        "all-in-raise",
        [
            {
                "id": "all-in-raise",
                "agentId": "villain",
                "street": "Preflop",
                "action": "all-in",
                "toAmount": 200,
            }
        ],
    )
    table["currentBet"] = 100

    tracker.enrich_table(table)
    profile = tracker._PROFILES["villain"]

    assert (profile.preflop_hands_seen, profile.vpip, profile.pfr) == (1, 1, 1)
    assert profile.legacy_pfr_raise_count == 1


def test_ambiguous_live_all_in_fails_closed_for_pfr():
    table = _table(
        "ambiguous-all-in",
        [
            {
                "id": "ambiguous-all-in",
                "agentId": "villain",
                "street": "Preflop",
                "action": "all-in",
            }
        ],
    )
    table["currentBet"] = 100

    tracker.enrich_table(table)
    profile = tracker._PROFILES["villain"]

    assert (profile.preflop_hands_seen, profile.vpip, profile.pfr) == (1, 1, 0)
    assert profile.legacy_pfr_raise_count == 0
