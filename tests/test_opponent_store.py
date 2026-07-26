import sqlite3
from pathlib import Path

import pytest

import poker_bot.opponent_store as opponent_store
from poker_bot.opponent_store import (
    connect,
    default_db_path,
    default_telemetry_db_path,
    increment_hand_seen,
    init_db,
    load_profile,
    merge_worker_db,
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


def test_telemetry_db_path_prefers_telemetry_env(monkeypatch, tmp_path):
    opponent_db = tmp_path / "opponents.sqlite"
    telemetry_db = tmp_path / "telemetry.sqlite"
    monkeypatch.setenv("POKER_BOT_OPPONENT_DB", str(opponent_db))
    monkeypatch.setenv("POKER_BOT_TELEMETRY_DB", str(telemetry_db))

    assert default_db_path() == opponent_db
    assert default_telemetry_db_path() == telemetry_db
    with connect(telemetry=True) as conn:
        assert conn.execute("select name from sqlite_master").fetchone()


def test_telemetry_db_path_falls_back_to_opponent_env(monkeypatch, tmp_path):
    opponent_db = tmp_path / "opponents.sqlite"
    monkeypatch.delenv("POKER_BOT_TELEMETRY_DB", raising=False)
    monkeypatch.setenv("POKER_BOT_OPPONENT_DB", str(opponent_db))

    assert default_telemetry_db_path() == opponent_db


def test_connect_enables_wal_and_busy_timeout_for_shared_db(tmp_path):
    db_path = tmp_path / "shared-opponents.sqlite"

    with connect(db_path) as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]

    assert journal_mode == "wal"
    assert busy_timeout == 5000
    assert foreign_keys == 1


def test_opponent_store_records_and_loads_profile(tmp_path):
    db_path = tmp_path / "opponents.sqlite"
    conn = connect(db_path)

    increment_hand_seen(
        conn, "arena", "villain-1", handle="LooseOne", hand_id="h1"
    )
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


def test_store_counts_preflop_flags_once_and_preserves_legacy_action_counts(tmp_path):
    conn = connect(tmp_path / "stats.sqlite")
    increment_hand_seen(conn, "arena", "villain", hand_id="h1")
    for action, street in (
        ("raise", "Preflop"),
        ("raise", "Preflop"),
        ("bet", "Flop"),
        ("call", "Turn"),
    ):
        record_observed_action(
            conn,
            platform="arena",
            agent_id="villain",
            hand_id="h1",
            street=street,
            action=action,
            voluntary=street == "Preflop",
            commit=False,
        )
    conn.commit()
    profile = load_profile(conn, "arena", "villain")

    assert profile is not None
    assert (profile.preflop_hands_seen, profile.vpip, profile.pfr) == (1, 1, 1)
    assert profile.profile_stats_provenance == "canonical"
    assert profile.legacy_vpip_action_count == 2
    assert profile.legacy_pfr_raise_count == 2



def test_canonical_preflop_aggregates_do_not_decrement_after_replayed_actions(tmp_path):
    conn = connect(tmp_path / "canonical-preflop.sqlite")
    actions_by_hand = {
        "raise-call": (("raise", True), ("call", True)),
        "raise-fold": (("raise", True), ("fold", False)),
        "call-fold": (("call", True), ("fold", False)),
        "duplicate-delivery": (("raise", True), ("raise", True)),
    }

    for hand_id, actions in actions_by_hand.items():
        for action, voluntary in actions:
            record_observed_action(
                conn,
                platform="arena",
                agent_id="villain",
                hand_id=hand_id,
                street="Preflop",
                action=action,
                voluntary=voluntary,
                commit=False,
            )
    conn.commit()

    aggregates = conn.execute(
        """
        select s.preflop_hands_seen, s.vpip, s.pfr,
               count(h.hand_id) as fact_hands,
               coalesce(sum(h.vpip), 0) as fact_vpip,
               coalesce(sum(h.pfr), 0) as fact_pfr
        from opponents o
        join opponent_stats s on s.opponent_id = o.id
        left join opponent_preflop_hands h on h.opponent_id = o.id
        where o.platform = 'arena' and o.agent_id = 'villain'
        group by s.opponent_id
        """
    ).fetchone()

    assert aggregates is not None
    assert (
        aggregates["preflop_hands_seen"],
        aggregates["vpip"],
        aggregates["pfr"],
    ) == (4, 4, 3)
    assert aggregates["vpip"] == aggregates["fact_vpip"]
    assert aggregates["pfr"] == aggregates["fact_pfr"]
    assert aggregates["preflop_hands_seen"] == aggregates["fact_hands"]
    assert (
        0
        <= aggregates["pfr"]
        <= aggregates["vpip"]
        <= aggregates["preflop_hands_seen"]
    )


def test_forced_all_in_is_not_vpip_or_pfr(tmp_path):
    conn = connect(tmp_path / "forced-all-in.sqlite")
    increment_hand_seen(conn, "arena", "villain", hand_id="h1")
    record_observed_action(
        conn,
        platform="arena",
        agent_id="villain",
        hand_id="h1",
        street="Preflop",
        action="all-in",
        voluntary=False,
    )
    increment_hand_seen(conn, "arena", "villain", hand_id="h2")
    record_observed_action(
        conn,
        platform="arena",
        agent_id="villain",
        hand_id="h2",
        street="Preflop",
        action="all-in",
        voluntary=True,
    )
    profile = load_profile(conn, "arena", "villain")

    assert profile is not None
    assert (profile.preflop_hands_seen, profile.vpip, profile.pfr) == (2, 1, 1)


def test_migration_backfills_only_complete_unique_hand_history(tmp_path):
    conn = connect(tmp_path / "migrate.sqlite")
    opponent_id = conn.execute(
        "insert into opponents(platform, agent_id) values ('arena', 'villain')"
    ).lastrowid
    conn.execute("insert into opponent_stats(opponent_id) values (?)", (opponent_id,))
    conn.execute(
        """
        update opponent_stats
        set hands_seen = 2, vpip = 8, pfr = 4, profile_stats_schema_version = 0
        where opponent_id = ?
        """,
        (opponent_id,),
    )
    conn.executemany(
        """
        insert into opponent_actions(
            opponent_id, hand_id, street, action, voluntary, is_preflop_raise
        )
        values (?, ?, 'Preflop', ?, ?, ?)
        """,
        [(opponent_id, "h1", "raise", 1, 1), (opponent_id, "h2", "fold", 0, 0)],
    )
    init_db(conn)
    profile = load_profile(conn, "arena", "villain")

    assert profile is not None
    assert (profile.preflop_hands_seen, profile.vpip, profile.pfr) == (2, 1, 1)
    assert profile.legacy_vpip_action_count == 8
    assert profile.legacy_pfr_raise_count == 4
    assert profile.profile_stats_provenance == "canonical"


def test_merge_worker_db_recomputes_canonical_preflop_facts_idempotently(tmp_path):
    main_path = tmp_path / "main.sqlite"
    worker_a_path = tmp_path / "worker-a.sqlite"
    worker_b_path = tmp_path / "worker-b.sqlite"
    worker_a = connect(worker_a_path)
    worker_b = connect(worker_b_path)

    for conn, hand_id, action in (
        (worker_a, "worker-a:h1", "raise"),
        (worker_a, "shared:h1", "call"),
        (worker_b, "worker-b:h1", "call"),
        (worker_b, "shared:h1", "raise"),
    ):
        increment_hand_seen(
            conn, "selfplay", "villain", hand_id=hand_id, commit=False
        )
        record_observed_action(
            conn,
            platform="selfplay",
            agent_id="villain",
            hand_id=hand_id,
            street="Preflop",
            action=action,
            voluntary=True,
            commit=False,
        )
        conn.commit()
    worker_a.close()
    worker_b.close()

    merge_worker_db(main_path, worker_a_path)
    merge_worker_db(main_path, worker_b_path)

    main = connect(main_path)
    facts = main.execute(
        """
        select hand_id, vpip, pfr
        from opponent_preflop_hands
        order by hand_id
        """
    ).fetchall()
    stats = main.execute(
        """
        select preflop_hands_seen, vpip, pfr, profile_stats_provenance
        from opponent_stats
        """
    ).fetchone()

    assert [tuple(row) for row in facts] == [
        ("shared:h1", 1, 1),
        ("worker-a:h1", 1, 1),
        ("worker-b:h1", 1, 0),
    ]
    assert stats is not None
    assert (
        stats["preflop_hands_seen"],
        stats["vpip"],
        stats["pfr"],
        stats["profile_stats_provenance"],
    ) == (3, 3, 2, "canonical")

    main.close()

    merge_worker_db(main_path, worker_a_path)
    main = connect(main_path)
    repeated = main.execute(
        """
        select preflop_hands_seen, vpip, pfr
        from opponent_stats
        """
    ).fetchone()
    fact_totals = main.execute(
        """
        select count(*) as hands, sum(vpip) as vpip, sum(pfr) as pfr
        from opponent_preflop_hands
        """
    ).fetchone()

    assert repeated is not None
    assert fact_totals is not None
    assert tuple(repeated) == (3, 3, 2)
    assert tuple(fact_totals) == tuple(repeated)
    assert 0 <= repeated["pfr"] <= repeated["vpip"] <= repeated["preflop_hands_seen"]
    main.close()


def test_merge_worker_db_marks_missing_hand_identity_untrusted(tmp_path):
    main_path = tmp_path / "main.sqlite"
    worker_path = tmp_path / "worker.sqlite"
    worker = connect(worker_path)
    increment_hand_seen(worker, "selfplay", "villain")
    worker.close()

    merge_worker_db(main_path, worker_path)

    main = connect(main_path)
    profile = load_profile(main, "selfplay", "villain")

    assert profile is not None
    assert profile.profile_stats_provenance == "legacy_untrusted"
    main.close()


def _write_completed_worker_snapshot(path, hand_id, action):
    worker = connect(path)
    increment_hand_seen(
        worker, "selfplay", "villain", hand_id=hand_id, commit=False
    )
    record_observed_action(
        worker,
        platform="selfplay",
        agent_id="villain",
        hand_id=hand_id,
        street="Preflop",
        action=action,
        voluntary=True,
        commit=False,
    )
    worker.commit()
    worker.close()


def _merged_worker_snapshot(conn):
    stats = conn.execute(
        """
        select hands_seen, preflop_hands_seen, vpip, pfr,
               legacy_vpip_action_count, legacy_pfr_raise_count,
               calls, bets, raises, folds, fold_to_bet,
               opportunities_to_fold_to_bet, showdowns, won_showdown,
               all_ins, large_bets, pressure_when_covering,
               profile_stats_schema_version, profile_stats_provenance
        from opponent_stats
        """
    ).fetchall()
    actions = conn.execute(
        """
        select opponent_id, hand_id, street, action, amount, pot, message,
               facing_bet, voluntary, is_preflop_raise, stack_chips,
               hero_stack_chips, created_at
        from opponent_actions
        order by id
        """
    ).fetchall()
    facts = conn.execute(
        """
        select opponent_id, hand_id, vpip, pfr
        from opponent_preflop_hands
        order by opponent_id, hand_id
        """
    ).fetchall()
    return (
        tuple(map(tuple, stats)),
        tuple(map(tuple, actions)),
        tuple(map(tuple, facts)),
    )


def test_merge_worker_db_receipt_skips_replays_and_allows_distinct_sources(
    tmp_path, monkeypatch
):
    main_path = tmp_path / "main.sqlite"
    worker_a_path = tmp_path / "worker-a.sqlite"
    worker_b_path = tmp_path / "worker-b.sqlite"
    _write_completed_worker_snapshot(worker_a_path, "worker-a:h1", "raise")
    _write_completed_worker_snapshot(worker_b_path, "worker-b:h1", "call")
    monkeypatch.setattr(
        opponent_store, "_worker_snapshot_digest", lambda _path: "forced-collision"
    )

    merge_worker_db(main_path, worker_a_path)
    main = connect(main_path)
    first = _merged_worker_snapshot(main)
    main.close()

    merge_worker_db(main_path, worker_a_path)
    main = connect(main_path)
    assert _merged_worker_snapshot(main) == first
    main.close()

    merge_worker_db(main_path, worker_b_path)
    main = connect(main_path)
    stats, actions, facts = _merged_worker_snapshot(main)

    assert stats[0][:5] == (2, 2, 2, 1, 0)
    assert len(actions) == 2
    assert len(facts) == 2
    assert facts == (
        (1, "worker-a:h1", 1, 1),
        (1, "worker-b:h1", 1, 0),
    )
    assert main.execute("select count(*) from worker_merge_receipts").fetchone()[0] == 2
    main.close()


def test_merge_worker_db_receipt_allows_reused_path_for_new_snapshot(tmp_path):
    main_path = tmp_path / "main.sqlite"
    worker_path = tmp_path / "worker.sqlite"
    _write_completed_worker_snapshot(worker_path, "first:h1", "raise")
    merge_worker_db(main_path, worker_path)

    for suffix in ("", "-wal", "-shm"):
        Path(f"{worker_path}{suffix}").unlink(missing_ok=True)
    _write_completed_worker_snapshot(worker_path, "second:h1", "call")
    merge_worker_db(main_path, worker_path)

    main = connect(main_path)
    stats, actions, facts = _merged_worker_snapshot(main)

    assert stats[0][:5] == (2, 2, 2, 1, 0)
    assert len(actions) == 2
    assert len(facts) == 2
    assert main.execute("select count(*) from worker_merge_receipts").fetchone()[0] == 2
    main.close()


def test_merge_worker_db_rolls_back_its_receipt_on_failure(tmp_path):
    main_path = tmp_path / "main.sqlite"
    worker_path = tmp_path / "worker.sqlite"
    _write_completed_worker_snapshot(worker_path, "worker:h1", "raise")
    worker = connect(worker_path)
    worker.execute("drop table guard_overrides")
    worker.commit()
    worker.close()

    with pytest.raises(sqlite3.OperationalError, match="guard_overrides"):
        merge_worker_db(main_path, worker_path)

    main = connect(main_path)
    assert main.execute("select count(*) from worker_merge_receipts").fetchone()[0] == 0
    main.close()
