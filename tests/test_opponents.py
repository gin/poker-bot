from poker_bot.opponents import (
    _MAX_PREFLOP_HAND_FLAGS,
    OpponentProfile,
    profile_from_mapping,
    record_action,
    record_hand_seen,
    summarize_profiles,
)


def test_profile_labels_loose_aggressive_player():
    profile = OpponentProfile(
        agent_id="villain", hands_seen=20, preflop_hands_seen=20, vpip=12, pfr=7
    )
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


def test_hand_level_preflop_flags_ignore_repeated_and_postflop_actions():
    profile = OpponentProfile(agent_id="villain")
    record_hand_seen(profile, "h1")
    record_action(
        profile,
        "raise",
        hand_id="h1",
        street="Preflop",
        voluntary=True,
        is_preflop_raise=True,
    )
    record_action(
        profile,
        "raise",
        hand_id="h1",
        street="Preflop",
        voluntary=True,
        is_preflop_raise=True,
    )
    record_action(
        profile, "bet", hand_id="h1", street="Flop", voluntary=False
    )
    record_action(
        profile, "call", hand_id="h1", street="Turn", voluntary=False
    )

    assert (profile.preflop_hands_seen, profile.vpip, profile.pfr) == (1, 1, 1)
    assert profile.legacy_vpip_action_count == 2
    assert profile.legacy_pfr_raise_count == 2

    record_hand_seen(profile, "h2")
    record_action(
        profile, "call", hand_id="h2", street="Preflop", voluntary=True
    )
    assert (profile.preflop_hands_seen, profile.vpip, profile.pfr) == (2, 2, 1)



def test_preflop_hand_flag_cache_is_bounded_and_tracks_the_next_hand():
    profile = OpponentProfile(agent_id="villain")
    hand_count = _MAX_PREFLOP_HAND_FLAGS * 2
    for hand_number in range(hand_count):
        hand_id = f"h{hand_number}"
        record_hand_seen(profile, hand_id)
        record_action(
            profile, "call", hand_id=hand_id, street="Preflop", voluntary=True
        )

    assert len(profile._preflop_hand_flags) == _MAX_PREFLOP_HAND_FLAGS
    assert (profile.preflop_hands_seen, profile.vpip, profile.pfr) == (
        hand_count,
        hand_count,
        0,
    )

    record_hand_seen(profile, "next")
    record_action(
        profile,
        "raise",
        hand_id="next",
        street="Preflop",
        voluntary=True,
        is_preflop_raise=True,
    )
    record_action(
        profile,
        "raise",
        hand_id="next",
        street="Preflop",
        voluntary=True,
        is_preflop_raise=True,
    )

    assert len(profile._preflop_hand_flags) == _MAX_PREFLOP_HAND_FLAGS
    assert (profile.preflop_hands_seen, profile.vpip, profile.pfr) == (
        hand_count + 1,
        hand_count + 1,
        1,
    )



def test_legacy_mapping_never_infers_a_canonical_preflop_denominator():
    profile = profile_from_mapping(
        "villain", {"hands_seen": 20, "vpip": 10, "pfr": 4}
    )

    assert profile.profile_stats_provenance == "legacy_untrusted"
    assert profile.preflop_hands_seen == 0
    assert profile.vpip_frequency == 0.0
    assert profile.pfr_frequency == 0.0


def test_direct_canonical_profile_uses_explicit_preflop_denominator():
    profile = OpponentProfile(
        agent_id="villain",
        hands_seen=20,
        preflop_hands_seen=20,
        vpip=10,
        pfr=4,
    )

    assert profile.vpip_frequency == 0.5
    assert profile.pfr_frequency == 0.2


def test_missing_hand_identity_is_untrusted():
    profile = OpponentProfile(agent_id="villain")
    record_hand_seen(profile, None)
    record_action(profile, "raise", street="Preflop", voluntary=True)

    assert profile.profile_stats_provenance == "legacy_untrusted"
    assert profile.vpip_frequency == 0.0
