"""Regression tests for init_db migration paths.

Covers the case where ``init_db`` is called on a database that was
created by an earlier version of the schema (no ``table_id`` column
and no ``idx_decisions_table_id`` index). The migration must add the
column and the index without raising.
"""

import importlib
import sqlite3
import sys

from poker_bot.opponent_store import connect


def _build_legacy_schema(db_path):
    """Recreate a pre-``table_id`` schema identical to what the bot
    shipped before the column was introduced."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        create table if not exists opponents (
            id integer primary key,
            platform text not null,
            agent_id text not null,
            handle text,
            first_seen_at text not null default current_timestamp,
            last_seen_at text not null default current_timestamp,
            unique(platform, agent_id)
        );
        create table if not exists opponent_stats (
            opponent_id integer primary key references opponents(id),
            hands_seen integer not null default 0,
            vpip integer not null default 0,
            pfr integer not null default 0,
            calls integer not null default 0,
            bets integer not null default 0,
            raises integer not null default 0,
            folds integer not null default 0,
            fold_to_bet integer not null default 0,
            opportunities_to_fold_to_bet integer not null default 0,
            showdowns integer not null default 0,
            won_showdown integer not null default 0,
            all_ins integer not null default 0,
            large_bets integer not null default 0,
            pressure_when_covering integer not null default 0,
            updated_at text not null default current_timestamp
        );
        create table if not exists opponent_actions (
            id integer primary key,
            opponent_id integer not null references opponents(id),
            hand_id text,
            street text,
            action text not null,
            amount integer,
            pot integer,
            facing_bet integer not null default 0,
            stack_chips integer,
            hero_stack_chips integer,
            created_at text not null default current_timestamp
        );
        create table if not exists opponent_external_stats (
            id integer primary key,
            opponent_id integer not null references opponents(id),
            competition_id text not null,
            source text not null,
            stats_json text not null,
            fetched_at text not null default current_timestamp,
            unique(opponent_id, competition_id, source)
        );
        create table if not exists telemetry_runs (
            run_id text primary key,
            strategy text not null,
            opponent text,
            players integer,
            seed integer,
            platform text not null default 'selfplay',
            metadata_json text,
            started_at text not null default current_timestamp
        );
        create table if not exists decision_telemetry (
            id integer primary key,
            run_id text not null references telemetry_runs(run_id),
            hand_id text not null,
            decision_index integer not null,
            strategy text not null,
            street text,
            hero_agent_id text,
            hero_seat_number integer,
            button_seat_number integer,
            hero_position text,
            hero_position_offset integer,
            seated_players integer,
            active_players integer,
            table_style text,
            pot_chips integer,
            current_bet integer,
            call_amount integer,
            min_bet integer,
            min_raise_to integer,
            hero_stack integer,
            hero_current_bet integer,
            max_opponent_stack integer,
            covered_by_larger_stack integer,
            hole_cards text,
            board_cards text,
            preflop_score integer,
            made_hand_rank integer,
            hand_bucket text,
            board_wet integer,
            board_paired integer,
            board_high integer,
            top_pair_or_better integer,
            available_actions text,
            chosen_action text,
            chosen_amount integer,
            amount_ratio_pot real,
            amount_ratio_stack real,
            facing_bet integer not null default 0,
            voluntary integer not null default 0,
            strategy_message text,
            hero_net_chips integer,
            won_hand integer,
            final_pot integer,
            created_at text not null default current_timestamp
        );
        create index if not exists idx_decisions_run
            on decision_telemetry(run_id);
        create index if not exists idx_decisions_hand
            on decision_telemetry(hand_id);
        create index if not exists idx_decisions_strategy
            on decision_telemetry(strategy);
        create index if not exists idx_decisions_bucket
            on decision_telemetry(street, hand_bucket, table_style);
        create index if not exists idx_decisions_action
            on decision_telemetry(chosen_action, street);
        create index if not exists idx_external_stats_competition
            on opponent_external_stats(competition_id, source);
        """
    )
    conn.execute(
        "insert into telemetry_runs(run_id, strategy) values ('legacy-run', 's')"
    )
    conn.execute(
        "insert into decision_telemetry(run_id, hand_id, decision_index, strategy) "
        "values ('legacy-run', 'tbl-xyz:2026-06-18T10:00:00', 0, 's')"
    )
    conn.execute(
        "insert into decision_telemetry(run_id, hand_id, decision_index, strategy) "
        "values ('legacy-run', 'tbl-xyz:2026-06-18T10:05:00', 1, 's')"
    )
    conn.execute(
        "insert into decision_telemetry(run_id, hand_id, decision_index, strategy) "
        "values ('legacy-run', 'plain-hand-id', 2, 's')"
    )
    conn.commit()
    conn.close()


def test_init_db_adds_table_id_column_on_legacy_schema(tmp_path):
    """Regression: init_db used to fail on legacy DBs because
    CREATE INDEX ... ON decision_telemetry(table_id) ran before the
    column was added. Now _ensure_columns adds the column first, then
    the index is created."""
    db = tmp_path / "legacy.sqlite"
    _build_legacy_schema(str(db))

    # Reload the module to flush any cached state.
    if "poker_bot.opponent_store" in sys.modules:
        importlib.reload(sys.modules["poker_bot.opponent_store"])

    conn = connect(str(db))
    try:
        cols = [
            r["name"] for r in conn.execute("pragma table_info(decision_telemetry)")
        ]
        assert "table_id" in cols, "table_id column should be added by _ensure_columns"

        indexes = [
            r["name"] for r in conn.execute("pragma index_list(decision_telemetry)")
        ]
        assert "idx_decisions_table_id" in indexes, (
            "idx_decisions_table_id should be created after _ensure_columns"
        )

        # The one-shot migration should have populated table_id for legacy
        # rows that look like ``tbl-xyz:2026-06-18T10:00:00``.
        rows = conn.execute(
            "select hand_id, table_id from decision_telemetry "
            "where run_id = 'legacy-run' order by decision_index"
        ).fetchall()
        by_hand = {r["hand_id"]: r["table_id"] for r in rows}
        assert by_hand["tbl-xyz:2026-06-18T10:00:00"] == "tbl-xyz"
        assert by_hand["tbl-xyz:2026-06-18T10:05:00"] == "tbl-xyz"
        # Legacy hand_ids without a colon stay NULL — the replay backfill
        # simply skips them.
        assert by_hand["plain-hand-id"] is None
    finally:
        conn.close()


def test_init_db_idempotent_on_already_migrated_db(tmp_path):
    """Running init_db a second time on an already-migrated DB is a no-op."""
    db = tmp_path / "twice.sqlite"
    _build_legacy_schema(str(db))
    if "poker_bot.opponent_store" in sys.modules:
        importlib.reload(sys.modules["poker_bot.opponent_store"])

    # First connect() migrates.
    c1 = connect(str(db))
    c1.close()

    # Second connect() must not raise.
    c2 = connect(str(db))
    try:
        cols_before = sorted(
            r["name"] for r in c2.execute("pragma table_info(decision_telemetry)")
        )
        rows_before = c2.execute(
            "select count(*) as n from decision_telemetry"
        ).fetchone()["n"]
    finally:
        c2.close()

    c3 = connect(str(db))
    try:
        cols_after = sorted(
            r["name"] for r in c3.execute("pragma table_info(decision_telemetry)")
        )
        rows_after = c3.execute(
            "select count(*) as n from decision_telemetry"
        ).fetchone()["n"]
    finally:
        c3.close()

    assert cols_before == cols_after
    assert rows_before == rows_after


def test_init_db_on_fresh_db_still_works(tmp_path):
    """A brand-new DB (no legacy state) must still initialise correctly."""
    db = tmp_path / "fresh.sqlite"
    if "poker_bot.opponent_store" in sys.modules:
        importlib.reload(sys.modules["poker_bot.opponent_store"])
    conn = connect(str(db))
    try:
        cols = [
            r["name"] for r in conn.execute("pragma table_info(decision_telemetry)")
        ]
        assert "table_id" in cols
        indexes = [
            r["name"] for r in conn.execute("pragma index_list(decision_telemetry)")
        ]
        assert "idx_decisions_table_id" in indexes
    finally:
        conn.close()
