from poker_bot.opponent_store import (
    connect,
    increment_hand_seen,
    load_profile,
    record_observed_action,
)


def test_opponent_store_records_and_loads_profile(tmp_path):
    db_path = tmp_path / "opponents.sqlite"
    conn = connect(db_path)

    increment_hand_seen(conn, "arena", "villain-1", handle="LooseOne")
    record_observed_action(
        conn,
        platform="arena",
        agent_id="villain-1",
        hand_id="h1",
        street="Preflop",
        action="raise",
        amount=150,
        pot=75,
        message="Strong hand, raising",
        facing_bet=True,
        stack_chips=2200,
        hero_stack_chips=1800,
        voluntary=True,
    )

    profile = load_profile(conn, "arena", "villain-1")

    assert profile is not None
    assert profile.agent_id == "villain-1"
    assert profile.name == "looseone"
    assert profile.hands_seen == 1
    assert profile.vpip == 1
    assert profile.pfr == 1
    assert profile.raises == 1
    assert profile.opportunities_to_fold_to_bet == 1

    action_row = conn.execute("select * from opponent_actions").fetchone()
    assert action_row["message"] == "Strong hand, raising"
