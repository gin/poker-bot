from pathlib import Path

from poker_bot.opponent_store import (
    connect,
    default_db_path,
    increment_hand_seen,
    load_profile,
    record_external_agent_stats,
    record_observed_action,
)


def test_default_db_path_is_repo_local(monkeypatch):
    monkeypatch.delenv("POKER_BOT_OPPONENT_DB", raising=False)

    db_path = default_db_path()

    assert db_path == Path(__file__).resolve().parents[1] / "gameplay.sqlite"


def test_default_db_path_can_be_overridden(monkeypatch, tmp_path):
    override = tmp_path / "custom.sqlite"
    monkeypatch.setenv("POKER_BOT_OPPONENT_DB", str(override))

    assert default_db_path() == override


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


def test_record_external_agent_stats_keeps_arena_stats_separate(tmp_path):
    db_path = tmp_path / "opponents.sqlite"
    conn = connect(db_path)

    record_external_agent_stats(
        conn,
        platform="arena",
        agent_id="villain-2",
        handle="KnownPro",
        competition_id="cmp-test",
        stats={"hands": 200, "vpip": 44, "pfr": 18},
    )

    opponent = conn.execute(
        """
        select o.handle, s.hands_seen, s.vpip, s.pfr, e.stats_json
        from opponents o
        join opponent_stats s on s.opponent_id = o.id
        join opponent_external_stats e on e.opponent_id = o.id
        where o.agent_id = 'villain-2'
        """
    ).fetchone()
    assert opponent["handle"] == "knownpro"
    assert opponent["hands_seen"] == 0
    assert opponent["vpip"] == 0
    assert opponent["pfr"] == 0
    assert '"hands": 200' in opponent["stats_json"]
