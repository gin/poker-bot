from poker_bot.opponents import OpponentProfile, record_action, summarize_profiles


def test_profile_labels_loose_aggressive_player():
    profile = OpponentProfile(agent_id="villain", hands_seen=20, vpip=12, pfr=7)
    profile.bets = 8
    profile.raises = 6
    profile.calls = 3
    profile.folds = 4

    assert profile.label() == "loose_aggressive"


def test_record_action_updates_fold_to_bet_and_summary():
    profile = OpponentProfile(agent_id="villain", hands_seen=10)

    record_action(profile, "fold", street="Turn", facing_bet=True)

    summary = summarize_profiles({"villain": profile})
    assert profile.folds == 1
    assert profile.fold_to_bet == 1
    assert profile.opportunities_to_fold_to_bet == 1
    assert summary["villain"]["fold_to_bet"] == 1.0
