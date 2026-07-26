"""SQLite-backed opponent profile storage."""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import os
import sqlite3
import uuid
from pathlib import Path

from poker_bot.opponents import OpponentProfile, profile_from_mapping
from poker_bot.strategies.adaptive import (
    board_texture,
    has_top_pair_or_better,
    made_hand_rank,
    preflop_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "gameplay.sqlite"
OPPONENT_DB_ENV = "POKER_BOT_OPPONENT_DB"
TELEMETRY_DB_ENV = "POKER_BOT_TELEMETRY_DB"

# Local-vs-API merge thresholds (see PLAN_OPPONENT_STATS.md).
# LOCAL_MIN_HANDS: when we have seen this many hands against an
# opponent ourselves, trust the local read over the API summary.
# API_STALE_DAYS: ignore API data older than this — the opponent
# may have changed play style, so stale data is worse than no data.
LOCAL_MIN_HANDS = 20
API_STALE_DAYS = 2


def _worker_snapshot_digest(source_path):
    """Hash the SQLite database and its WAL for a completed worker snapshot."""
    digest = hashlib.sha256()
    for suffix in ("", "-wal"):
        component = Path(f"{source_path}{suffix}")
        digest.update(suffix.encode())
        if component.exists():
            with component.open("rb") as snapshot:
                for chunk in iter(lambda: snapshot.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _worker_snapshot_receipt(worker_path, worker_conn):
    """Return a receipt key for an immutable, completed worker snapshot."""
    source_path = Path(worker_path).expanduser().resolve()
    run_identity = ",".join(
        str(row[0])
        for row in worker_conn.execute(
            "select run_id from telemetry_runs order by run_id"
        )
    )
    return str(source_path), run_identity, _worker_snapshot_digest(source_path)


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value and value.strip():
        return Path(value).expanduser()
    return default


def default_db_path():
    return _env_path(OPPONENT_DB_ENV, DEFAULT_DB_PATH)


def default_telemetry_db_path():
    return _env_path(TELEMETRY_DB_ENV, default_db_path())


def connect(path=None, *, telemetry: bool = False, optimize_writes: bool = False):
    if telemetry and path is None:
        db_path = default_telemetry_db_path()
    else:
        db_path = Path(path).expanduser() if path is not None else default_db_path()
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    if str(db_path) != ":memory:":
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        if optimize_writes:
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-64000")
    init_db(conn)
    return conn


def init_db(conn):
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
            preflop_hands_seen integer not null default 0,
            profile_stats_schema_version integer not null default 2,
            profile_stats_provenance text not null default 'canonical',
            legacy_vpip_action_count integer not null default 0,
            legacy_pfr_raise_count integer not null default 0,
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
            message text,
            facing_bet integer not null default 0,
            voluntary integer,
            is_preflop_raise integer,
            stack_chips integer,
            hero_stack_chips integer,
            created_at text not null default current_timestamp
        );
        create table if not exists opponent_preflop_hands (
            opponent_id integer not null references opponents(id),
            hand_id text not null,
            vpip integer not null default 0,
            pfr integer not null default 0,
            primary key(opponent_id, hand_id)
        );
        create index if not exists idx_opponent_preflop_hands_opponent
            on opponent_preflop_hands(opponent_id);

        create table if not exists worker_merge_receipts (
            source_path text not null,
            run_identity text not null,
            content_sha256 text not null,
            merged_at text not null default current_timestamp,
            primary key(source_path, run_identity, content_sha256)
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
            table_id text,
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
            opp_vpip real not null default 0.5,
            opp_pfr real not null default 0.3,
            opp_fold_to_bet real not null default 0.5,
            opp_aggro real not null default 0.5,
            opp_showdown real not null default 0.3,
            opp_hands_seen real not null default 0.0,
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
    _ensure_columns(
        conn,
        "opponent_stats",
        {
            "preflop_hands_seen": "integer not null default 0",
            "profile_stats_schema_version": "integer not null default 0",
            "profile_stats_provenance": "text not null default 'legacy_untrusted'",
            "legacy_vpip_action_count": "integer not null default 0",
            "legacy_pfr_raise_count": "integer not null default 0",
        },
    )
    _ensure_columns(
        conn,
        "opponent_actions",
        {
            "message": "text",
            "voluntary": "integer",
            "is_preflop_raise": "integer",
        },
    )
    _ensure_columns(
        conn,
        "decision_telemetry",
        {
            "hero_position": "text",
            "hero_position_offset": "integer",
            "seated_players": "integer",
            "table_id": "text",
            "opp_vpip": "real not null default 0.5",
            "opp_pfr": "real not null default 0.3",
            "opp_fold_to_bet": "real not null default 0.5",
            "opp_aggro": "real not null default 0.5",
            "opp_showdown": "real not null default 0.3",
            "opp_hands_seen": "real not null default 0.0",
        },
    )
    # Indexes on columns added by _ensure_columns must be created AFTER
    # the column exists. We can't put them in the executescript above
    # because pre-existing DBs that predate the column would fail with
    # "no such column" before _ensure_columns runs. Wrapping each CREATE
    # INDEX in a try/except keeps init_db idempotent and tolerant of
    # legacy schemas.
    for index_sql in (
        "create index if not exists idx_decisions_table_id "
        "on decision_telemetry(table_id)",
    ):
        with contextlib.suppress(Exception):
            conn.execute(index_sql)
    # One-shot migration: derive table_id from hand_id for rows written
    # before the table_id column existed. hand_id format is
    # ``f"{tableId}:{boundary_iso}"``; the tableId is everything before
    # the first colon. Rows where hand_id does not contain ':' (legacy
    # hand_id == tableId rows) are left alone — table_id stays NULL and
    # the replay backfill simply skips them.
    if conn.execute(
        "select 1 from decision_telemetry "
        "where table_id is null and hand_id like '%:%' limit 1"
    ).fetchone():
        conn.execute(
            "update decision_telemetry "
            "set table_id = substr(hand_id, 1, instr(hand_id, ':') - 1) "
            "where table_id is null and hand_id like '%:%'"
        )
    ensure_guard_overrides_table(conn)
    _migrate_legacy_preflop_stats(conn)
    conn.commit()

def _migrate_legacy_preflop_stats(conn):
    """Convert only provably complete action histories to v2 hand facts."""
    legacy_rows = conn.execute(
        """
        select opponent_id, hands_seen, vpip, pfr
        from opponent_stats
        where profile_stats_schema_version < 2
        """
    ).fetchall()
    for row in legacy_rows:
        opponent_id = row["opponent_id"]
        hands_seen = int(row["hands_seen"])
        coverage = conn.execute(
            """
            select count(distinct hand_id) as hand_count,
                   sum(case when hand_id is null then 1 else 0 end) as null_count,
                   sum(case when voluntary is null
                                 or is_preflop_raise is null
                            then 1 else 0 end) as unclassified_count
            from opponent_actions
            where opponent_id = ?
            """,
            (opponent_id,),
        ).fetchone()
        hand_count = int(coverage["hand_count"] or 0)
        complete = (
            hands_seen > 0
            and hand_count == hands_seen
            and not coverage["null_count"]
            and not coverage["unclassified_count"]
        )
        conn.execute(
            """
            update opponent_stats
            set legacy_vpip_action_count = vpip,
                legacy_pfr_raise_count = pfr,
                vpip = 0,
                pfr = 0,
                preflop_hands_seen = 0,
                profile_stats_schema_version = 2,
                profile_stats_provenance = ?
            where opponent_id = ?
            """,
            ("canonical" if complete else "legacy_untrusted", opponent_id),
        )
        if not complete:
            continue
        rows = conn.execute(
            """
            select hand_id,
                   max(case when street = 'Preflop' and voluntary = 1
                            then 1 else 0 end) as vpip,
                   max(case when street = 'Preflop'
                              and voluntary = 1
                              and is_preflop_raise = 1
                            then 1 else 0 end) as pfr
            from opponent_actions
            where opponent_id = ?
            group by hand_id
            """,
            (opponent_id,),
        ).fetchall()
        conn.executemany(
            """
            insert into opponent_preflop_hands(opponent_id, hand_id, vpip, pfr)
            values (?, ?, ?, ?)
            on conflict(opponent_id, hand_id) do update set
                vpip = max(opponent_preflop_hands.vpip, excluded.vpip),
                pfr = max(opponent_preflop_hands.pfr, excluded.pfr)
            """,
            [
                (opponent_id, item["hand_id"], item["vpip"], item["pfr"])
                for item in rows
            ],
        )
        facts = conn.execute(
            """
            select count(*) as hand_count, coalesce(sum(vpip), 0) as vpip,
                   coalesce(sum(pfr), 0) as pfr
            from opponent_preflop_hands
            where opponent_id = ?
            """,
            (opponent_id,),
        ).fetchone()
        conn.execute(
            """
            update opponent_stats
            set preflop_hands_seen = ?,
                vpip = ?,
                pfr = ?
            where opponent_id = ?
            """,
            (facts["hand_count"], facts["vpip"], facts["pfr"], opponent_id),
        )



def ensure_guard_overrides_table(conn) -> None:
    """Create the guard_overrides table if it does not exist."""
    conn.executescript(
        """
        create table if not exists guard_overrides (
            id integer primary key,
            run_id text not null references telemetry_runs(run_id),
            hand_id text not null,
            decision_index integer not null,
            guard_id text not null,
            pre_decision integer not null default 0,
            original_action text not null,
            final_action text not null,
            reason text not null,
            street text,
            pot_chips integer,
            call_amount integer,
            available_actions text,
            shadow integer not null default 0,
            applied integer not null default 1,
            original_amount integer,
            final_amount integer,
            phase text,
            precedence integer,
            created_at text not null default current_timestamp
        );
        create index if not exists idx_guard_overrides_run
            on guard_overrides(run_id);
        create index if not exists idx_guard_overrides_guard
            on guard_overrides(guard_id);
        """
    )
    # Legacy DBs created before the shadow-mode columns existed.
    _ensure_columns(
        conn,
        "guard_overrides",
        {
            "shadow": "integer not null default 0",
            "applied": "integer not null default 1",
            "original_amount": "integer",
            "final_amount": "integer",
            "phase": "text",
            "precedence": "integer",
        },
    )
    conn.commit()


def record_guard_event(
    conn,
    *,
    run_id,
    hand_id,
    decision_index,
    event,
    commit=False,
):
    """Persist one GuardEvent (poker_bot.guards.telemetry) for a hero decision.

    Keyed by (run_id, hand_id, decision_index) so rows join straight to
    decision_telemetry for hand context and outcome (hero_net_chips, won_hand).
    """
    conn.execute(
        """
        insert into guard_overrides(
            run_id, hand_id, decision_index, guard_id, pre_decision,
            original_action, final_action, reason,
            street, pot_chips, call_amount, available_actions,
            shadow, applied, original_amount, final_amount, phase, precedence
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            hand_id,
            decision_index,
            event.guard_id,
            1 if event.phase == "pre" else 0,
            event.original_action,
            event.final_action,
            event.reason,
            event.street,
            event.pot,
            event.call_price,
            event.available_actions,
            1 if event.shadow else 0,
            1 if event.applied else 0,
            event.original_amount,
            event.final_amount,
            event.phase,
            event.precedence,
        ),
    )
    if commit:
        conn.commit()


def summarize_guard_overrides(conn, run_id=None, min_fires=1):
    """Per-guard summary: fire counts and outcome of the decisions touched.

    avg_net_chips comes from joining decision_telemetry on
    (run_id, hand_id, decision_index) — i.e. the outcome of hands where the
    guard fired, which is the number to compare against the run average when
    judging whether a guard is taxing good spots.
    """
    where = "where g.run_id = ?" if run_id is not None else ""
    params: list = [run_id] if run_id is not None else []
    params.append(min_fires)
    return conn.execute(
        f"""
        select g.guard_id,
               g.phase,
               count(*) as fires,
               sum(g.applied) as applied,
               sum(g.shadow) as shadow,
               g.original_action || ' -> ' || g.final_action as transition,
               avg(d.hero_net_chips) as avg_net_chips,
               avg(d.won_hand) as win_rate,
               count(d.hero_net_chips) as decided
        from guard_overrides g
        left join decision_telemetry d
          on d.run_id = g.run_id
         and d.hand_id = g.hand_id
         and d.decision_index = g.decision_index
        {where}
        group by g.guard_id, g.phase, transition
        having fires >= ?
        order by fires desc
        """,
        params,
    ).fetchall()


def _ensure_columns(conn, table, columns):
    existing = {
        row["name"] for row in conn.execute(f"pragma table_info({table})").fetchall()
    }
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"alter table {table} add column {name} {definition}")


def normalize_handle(handle):
    if handle is None:
        return None
    cleaned = str(handle).strip().lower()
    return cleaned or None


def upsert_opponent(conn, platform, agent_id, handle=None, *, commit=True):
    if not agent_id:
        raise ValueError("agent_id is required")
    normalized = normalize_handle(handle)
    conn.execute(
        """
        insert into opponents(platform, agent_id, handle)
        values (?, ?, ?)
        on conflict(platform, agent_id) do update set
            handle = coalesce(excluded.handle, opponents.handle),
            last_seen_at = current_timestamp
        """,
        (platform, agent_id, normalized),
    )
    row = conn.execute(
        "select id from opponents where platform = ? and agent_id = ?",
        (platform, agent_id),
    ).fetchone()
    opponent_id = int(row["id"])
    conn.execute(
        """
        insert into opponent_stats(opponent_id)
        values (?)
        on conflict(opponent_id) do nothing
        """,
        (opponent_id,),
    )
    if commit:
        conn.commit()
    return opponent_id


def increment_hand_seen(
    conn, platform, agent_id, handle=None, *, hand_id=None, commit=True
):
    """Record one dealt hand, with a zeroed preflop fact when identifiable."""
    opponent_id = upsert_opponent(conn, platform, agent_id, handle, commit=False)
    if hand_id:
        cursor = conn.execute(
            """
            insert or ignore into opponent_preflop_hands(
                opponent_id, hand_id, vpip, pfr
            )
            values (?, ?, 0, 0)
            """,
            (opponent_id, hand_id),
        )
        if cursor.rowcount:
            conn.execute(
                """
                update opponent_stats
                set hands_seen = hands_seen + 1,
                    preflop_hands_seen = preflop_hands_seen + 1
                where opponent_id = ?
                """,
                (opponent_id,),
            )
    else:
        conn.execute(
            """
            update opponent_stats
            set hands_seen = hands_seen + 1,
                profile_stats_provenance = 'legacy_untrusted'
            where opponent_id = ?
            """,
            (opponent_id,),
        )
    if commit:
        conn.commit()
    return opponent_id

def _record_canonical_preflop_action(
    conn, opponent_id, hand_id, street, action, voluntary, is_preflop_raise
):
    """Idempotently update one opponent's hand-level preflop facts."""
    if not hand_id:
        conn.execute(
            """
            update opponent_stats
            set profile_stats_provenance = 'legacy_untrusted'
            where opponent_id = ?
            """,
            (opponent_id,),
        )
        return
    if street != "Preflop":
        return
    vpip = int(voluntary)
    pfr = int(voluntary and is_preflop_raise)
    existing = conn.execute(
        """
        select vpip, pfr from opponent_preflop_hands
        where opponent_id = ? and hand_id = ?
        """,
        (opponent_id, hand_id),
    ).fetchone()
    if existing is None:
        conn.execute(
            """
            insert into opponent_preflop_hands(opponent_id, hand_id, vpip, pfr)
            values (?, ?, ?, ?)
            """,
            (opponent_id, hand_id, vpip, pfr),
        )
        conn.execute(
            """
            update opponent_stats
            set hands_seen = hands_seen + 1,
                preflop_hands_seen = preflop_hands_seen + 1,
                vpip = vpip + ?,
                pfr = pfr + ?,
                profile_stats_schema_version = 2
            where opponent_id = ?
            """,
            (vpip, pfr, opponent_id),
        )
        return
    old_vpip = int(existing["vpip"])
    old_pfr = int(existing["pfr"])
    vpip_delta = max(old_vpip, vpip) - old_vpip
    pfr_delta = max(old_pfr, pfr) - old_pfr
    if vpip_delta or pfr_delta:
        conn.execute(
            """
            update opponent_preflop_hands
            set vpip = max(vpip, ?), pfr = max(pfr, ?)
            where opponent_id = ? and hand_id = ?
            """,
            (vpip, pfr, opponent_id, hand_id),
        )
        conn.execute(
            """
            update opponent_stats
            set vpip = vpip + ?, pfr = pfr + ?
            where opponent_id = ?
            """,
            (vpip_delta, pfr_delta, opponent_id),
        )


def record_external_agent_stats(
    conn,
    *,
    platform,
    agent_id,
    competition_id,
    stats,
    handle=None,
    source="arena_agent_stats",
):
    opponent_id = upsert_opponent(conn, platform, agent_id, handle)
    payload = stats if isinstance(stats, dict) else {"value": stats}
    stats_json = json.dumps(payload, sort_keys=True)
    conn.execute(
        """
        insert into opponent_external_stats(
            opponent_id, competition_id, source, stats_json
        )
        values (?, ?, ?, ?)
        on conflict(opponent_id, competition_id, source) do update set
            stats_json = excluded.stats_json,
            fetched_at = current_timestamp
        """,
        (opponent_id, competition_id, source, stats_json),
    )
    conn.commit()
    return opponent_id


def _parse_fetched_at(value):
    """Parse a stored timestamp string into a tz-aware UTC datetime.

    Accepts both SQLite's default ``YYYY-MM-DD HH:MM:SS`` (UTC)
    format and ISO 8601 with the ``T`` separator and ``Z`` suffix.
    Returns None for missing or unparseable values.
    """
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=_dt.UTC)
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        # ISO 8601 with 'Z' suffix → fromisoformat in Py3.11+ handles
        # "Z" but we normalize for safety.
        normalized = text.replace("Z", "+00:00")
        parsed = _dt.datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_dt.UTC)
        return parsed
    except (ValueError, TypeError):
        return None


def apply_external_stats_merge(profile, *, today=None):
    """Mutate ``profile`` in place: replace local counters with API
    values when local sample is too small to make a judgment.

    Per the user spec (PLAN_OPPONENT_STATS.md §2):

    - If ``profile.hands_seen >= LOCAL_MIN_HANDS`` (20), keep the
      local read — even if it disagrees with the API. The local
      read is fresher and tracks possible mid-competition
      strategy changes.
    - Else, if the API data is missing, stale (> API_STALE_DAYS),
      or has too small a sample, fall back to local (whatever we
      have, even if sparse).
    - Else, replace ``profile.vpip``, ``profile.pfr``,
      ``profile.hands_seen`` with API-derived values so the
      strategy's existing frequency reads work without any code
      changes. The API does not provide per-action breakdowns
      (calls / bets / raises / folds / fold_to_bet), so those
      local counters are preserved.

    The ``api_source_used`` flag is set on the profile so future
    strategy changes can opt out per-call.
    """
    today = today or _dt.datetime.now(_dt.UTC)
    api = profile.api_stats
    if api is None:
        profile.api_source_used = False
        return

    # Local has enough: keep local observations.
    if profile.hands_seen >= LOCAL_MIN_HANDS:
        profile.api_source_used = False
        return

    # Stale API: ignore.
    fetched_at = _parse_fetched_at(getattr(profile, "api_fetched_at", None))
    if fetched_at is None:
        profile.api_source_used = False
        return
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=_dt.UTC)
    if (today - fetched_at) > _dt.timedelta(days=API_STALE_DAYS):
        profile.api_source_used = False
        return

    # API sample size — prefer explicit sampleSize, fall back to hands.
    api_sample = int(api.get("sampleSize") or api.get("hands") or 0)
    if api_sample < LOCAL_MIN_HANDS:
        profile.api_source_used = False
        return

    def _to_fraction(value):
        """Auto-detect scale: > 1.5 means percent, else fraction."""
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return v / 100.0 if v > 1.5 else v

    api_vpip = _to_fraction(api.get("vpip"))
    api_pfr = _to_fraction(api.get("pfr"))

    # AF (aggression factor) is always 0-1 per arena. Convert to
    # an "aggression frequency" comparable to local — clamp to
    # [0, 0.7] so a 1.0 AF doesn't yield an unrealistically
    # high local-style frequency.
    api_aggr_freq = None
    api_af = api.get("af")
    if api_af is not None:
        try:
            api_aggr_freq = min(0.70, max(0.0, float(api_af)))
        except (TypeError, ValueError):
            api_aggr_freq = None
    if api_aggr_freq is None:
        style = api.get("playingStyle") or {}
        aggr_label = str(style.get("aggression") or "").strip().lower()
        api_aggr_freq = {
            "passive": 0.10,
            "measured": 0.28,
            "aggressive": 0.46,
        }.get(aggr_label)

    # API values are already hand-level rates. Keep their denominator explicit
    # so profile consumers never divide canonical numerators by action counts.
    if api_vpip is not None:
        profile.vpip = int(round(api_vpip * api_sample))
    if api_pfr is not None:
        profile.pfr = int(round(api_pfr * api_sample))

    # Record the API aggression frequency for any future reader.
    if api_aggr_freq is not None:
        profile.api_aggr_freq = api_aggr_freq

    profile.hands_seen = api_sample
    profile.preflop_hands_seen = api_sample
    profile.profile_stats_schema_version = 2
    profile.profile_stats_provenance = "canonical"
    profile.api_source_used = True
    profile.api_sample_size = api_sample


def record_observed_action(
    conn,
    *,
    platform,
    agent_id,
    handle=None,
    hand_id=None,
    street=None,
    action,
    amount=None,
    pot=None,
    message=None,
    facing_bet=False,
    stack_chips=None,
    hero_stack_chips=None,
    voluntary=False,
    is_preflop_raise=None,
    commit=True,
):
    opponent_id = upsert_opponent(conn, platform, agent_id, handle, commit=False)
    all_in = int(
        amount is not None and stack_chips is not None and amount >= stack_chips
    )
    large_bet = int(
        amount is not None and pot is not None and pot > 0 and amount >= pot
    )
    pressure_when_covering = int(
        action in {"bet", "raise"}
        and stack_chips is not None
        and hero_stack_chips is not None
        and stack_chips > hero_stack_chips
    )
    if is_preflop_raise is None:
        is_preflop_raise = (
            voluntary and street == "Preflop" and action in {"bet", "raise", "all-in"}
        )

    conn.execute(
        """
        insert into opponent_actions(
            opponent_id, hand_id, street, action, amount, pot, message, facing_bet,
            voluntary, is_preflop_raise, stack_chips, hero_stack_chips
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            opponent_id,
            hand_id,
            street,
            action,
            amount,
            pot,
            message,
            int(facing_bet),
            int(voluntary),
            int(is_preflop_raise),
            stack_chips,
            hero_stack_chips,
        ),
    )
    _record_canonical_preflop_action(
        conn,
        opponent_id,
        hand_id,
        street,
        action,
        voluntary,
        is_preflop_raise,
    )
    conn.execute(
        """
        update opponent_stats
        set legacy_vpip_action_count = legacy_vpip_action_count + ?,
            legacy_pfr_raise_count = legacy_pfr_raise_count + ?,
            calls = calls + ?,
            bets = bets + ?,
            raises = raises + ?,
            folds = folds + ?,
            fold_to_bet = fold_to_bet + ?,
            opportunities_to_fold_to_bet = opportunities_to_fold_to_bet + ?,
            all_ins = all_ins + ?,
            large_bets = large_bets + ?,
            pressure_when_covering = pressure_when_covering + ?,
            updated_at = current_timestamp
        where opponent_id = ?
        """,
        (
            int(voluntary),
            int(is_preflop_raise),
            int(action == "call"),
            int(action == "bet"),
            int(action in {"raise", "all-in"}),
            int(action == "fold"),
            int(action == "fold" and facing_bet),
            int(facing_bet),
            all_in,
            large_bet,
            pressure_when_covering,
            opponent_id,
        ),
    )
    if commit:
        conn.commit()
    return opponent_id


def load_profile(conn, platform, agent_id):
    row = conn.execute(
        """
        select o.agent_id, o.handle as name, s.*, e.stats_json,
               e.fetched_at as api_fetched_at
        from opponents o
        join opponent_stats s on s.opponent_id = o.id
        left join (
            select opponent_id, stats_json, fetched_at
            from opponent_external_stats
            where source = 'arena_agent_stats'
            order by fetched_at desc
            limit 1
        ) e on e.opponent_id = o.id
        where o.platform = ? and o.agent_id = ?
        """,
        (platform, agent_id),
    ).fetchone()
    if row is None:
        return None
    return profile_from_mapping(row["agent_id"], dict(row))


def load_profiles_for_agents(conn, platform, agent_ids):
    profiles = {}
    for agent_id in agent_ids:
        profile = load_profile(conn, platform, agent_id)
        if profile is not None:
            apply_external_stats_merge(profile)
            profiles[agent_id] = profile
    return profiles


def profile_to_mapping(profile: OpponentProfile):
    mapping = {
        "name": profile.name,
        "hands_seen": profile.hands_seen,
        "vpip": profile.vpip,
        "pfr": profile.pfr,
        "preflop_hands_seen": profile.preflop_hands_seen,
        "profile_stats_schema_version": profile.profile_stats_schema_version,
        "profile_stats_provenance": profile.profile_stats_provenance,
        "legacy_vpip_action_count": profile.legacy_vpip_action_count,
        "legacy_pfr_raise_count": profile.legacy_pfr_raise_count,
        "calls": profile.calls,
        "bets": profile.bets,
        "raises": profile.raises,
        "folds": profile.folds,
        "fold_to_bet": profile.fold_to_bet,
        "opportunities_to_fold_to_bet": profile.opportunities_to_fold_to_bet,
        "showdowns": profile.showdowns,
        "weak_aggressive_showdowns": profile.weak_aggressive_showdowns,
    }
    if profile.api_stats is not None:
        mapping["api_stats"] = profile.api_stats
    return mapping


def create_telemetry_run(
    conn,
    *,
    strategy,
    opponent=None,
    players=None,
    seed=None,
    platform="selfplay",
    metadata_json=None,
    run_id=None,
    commit=True,
):
    run_id = run_id or uuid.uuid4().hex
    conn.execute(
        """
        insert or ignore into telemetry_runs(
            run_id, strategy, opponent, players, seed, platform, metadata_json
        )
        values (?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, strategy, opponent, players, seed, platform, metadata_json),
    )
    if commit:
        conn.commit()
    return run_id


def _join_cards(cards):
    return " ".join(cards or [])


def _join_actions(actions):
    return ",".join(actions or [])


def _safe_ratio(numerator, denominator):
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def _safe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _active_players(table):
    return sum(
        1
        for seat in table.get("seats", [])
        if not seat.get("folded", False) and not seat.get("hasFolded", False)
    )


def _seated_numbers(table):
    numbers = []
    for seat in table.get("seats", []):
        if not seat.get("agentId") and seat.get("seatNumber") is None:
            continue
        seat_number = _safe_int(seat.get("seatNumber"))
        if seat_number is not None:
            numbers.append(seat_number)
    return sorted(set(numbers))


def _button_seat_number(table):
    value = table.get("buttonSeatNumber")
    if value is not None:
        return _safe_int(value)
    for key in (
        "dealerButtonSeatNumber",
        "dealerSeatNumber",
        "buttonSeat",
        "dealerSeat",
        "button",
    ):
        value = table.get(key)
        if isinstance(value, dict):
            value = value.get("seatNumber")
        seat_number = _safe_int(value)
        if seat_number is not None:
            return seat_number

    # Arena fallback — no explicit dealer field. Derive the button from blind
    # posters: the seat whose currentBetChips matches smallBlindChips is the SB,
    # and the button is the seat immediately before the SB in circular order.
    # This works reliably at preflop where the blinds are live in currentBetChips.
    sb_chips = _safe_int(table.get("smallBlindChips"))
    bb_chips = _safe_int(table.get("bigBlindChips"))
    if sb_chips is not None and bb_chips is not None and sb_chips > 0 and bb_chips > 0:
        seats = table.get("seats", [])
        for s in seats:
            bet = _safe_int(s.get("currentBetChips"))
            sb_seat = _safe_int(s.get("seatNumber"))
            if (
                bet is not None
                and sb_seat is not None
                and bet == sb_chips
                and bet < bb_chips
            ):
                active_seats = sorted(
                    sn
                    for seat in seats
                    if (sn := _safe_int(seat.get("seatNumber"))) is not None
                )
                if sb_seat in active_seats and len(active_seats) >= 2:
                    idx = active_seats.index(sb_seat)
                    return active_seats[(idx - 1) % len(active_seats)]
    return None


def _position_label(player_count, offset):
    labels_by_count = {
        2: ["BTN/SB", "BB"],
        3: ["BTN", "SB", "BB"],
        4: ["BTN", "SB", "BB", "CO"],
        5: ["BTN", "SB", "BB", "UTG", "CO"],
        6: ["BTN", "SB", "BB", "UTG", "HJ", "CO"],
    }
    labels = labels_by_count.get(player_count)
    if labels is None:
        if offset == 0:
            return "BTN"
        if offset == 1:
            return "SB"
        if offset == 2:
            return "BB"
        return f"POS{offset}"
    if offset < len(labels):
        return labels[offset]
    return None


def _hero_position(table, seat):
    seats = _seated_numbers(table)
    button = _button_seat_number(table)
    hero_seat = _safe_int(seat.get("seatNumber"))
    if not seats or button is None or hero_seat is None:
        return None, None, len(seats)
    if button not in seats or hero_seat not in seats:
        return None, None, len(seats)

    button_index = seats.index(button)
    ordered = seats[button_index:] + seats[:button_index]
    offset = ordered.index(hero_seat)
    return _position_label(len(ordered), offset), offset, len(ordered)


def _table_style(table, my_seat):
    try:
        from poker_bot.strategies import survival_lookup

        return survival_lookup.table_style(table, my_seat)
    except Exception:
        return "unknown"


def _hand_bucket(hole_cards, board_cards):
    if not board_cards:
        score = preflop_score(hole_cards)
        if score >= 80:
            return "strong"
        if score >= 50:
            return "medium"
        return "air"
    rank = made_hand_rank(hole_cards, board_cards)
    if rank >= 2:
        return "strong"
    if rank == 1 or has_top_pair_or_better(hole_cards, board_cards):
        return "medium"
    return "air"


def record_decision_telemetry(
    conn,
    *,
    run_id,
    hand_id,
    decision_index,
    strategy,
    table,
    seat,
    action,
    amount=None,
    message=None,
    facing_bet=False,
    voluntary=False,
    commit=True,
):
    allowed = table.get("allowedActions", {})
    hole_cards = seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = int(table.get("potChips") or 0)
    stack = int(seat.get("stackChips") or 0)
    opponent_stacks = [
        int(other.get("stackChips") or 0)
        for other in table.get("seats", [])
        if other.get("agentId") != seat.get("agentId")
        and not other.get("folded", False)
        and not other.get("hasFolded", False)
    ]
    max_opponent_stack = max(opponent_stacks, default=0)
    texture = board_texture(board_cards) if board_cards else {}
    rank = made_hand_rank(hole_cards, board_cards) if board_cards else 0
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    amount_value = int(amount) if amount is not None else None
    button_seat = _button_seat_number(table)
    hero_position, hero_position_offset, seated_players = _hero_position(table, seat)

    # Extract opponent profiling features for the active opponent
    opp_features = _extract_telemetry_opp_features(table, seat)

    conn.execute(
        """
        insert into decision_telemetry(
            run_id, hand_id, table_id, decision_index, strategy, street,
            hero_agent_id, hero_seat_number, button_seat_number, hero_position,
            hero_position_offset, seated_players, active_players,
            table_style, pot_chips, current_bet, call_amount, min_bet,
            min_raise_to, hero_stack, hero_current_bet, max_opponent_stack,
            covered_by_larger_stack, hole_cards, board_cards, preflop_score,
            made_hand_rank, hand_bucket, board_wet, board_paired, board_high,
            top_pair_or_better, available_actions, chosen_action, chosen_amount,
            amount_ratio_pot, amount_ratio_stack, facing_bet, voluntary,
            strategy_message,
            opp_vpip, opp_pfr, opp_fold_to_bet, opp_aggro, opp_showdown,
            opp_hands_seen
        )
        values (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?
        )
        """,
        (
            run_id,
            hand_id,
            str(table.get("tableId") or table.get("id") or "") or None,
            decision_index,
            strategy,
            table.get("street"),
            seat.get("agentId"),
            seat.get("seatNumber"),
            button_seat,
            hero_position,
            hero_position_offset,
            seated_players,
            _active_players(table),
            _table_style(table, seat),
            pot,
            table.get("currentBet"),
            allowed.get("callAmount") or allowed.get("callChips") or 0,
            allowed.get("minBet"),
            allowed.get("minRaiseTo"),
            stack,
            seat.get("currentBetChips"),
            max_opponent_stack,
            int(max_opponent_stack > stack),
            _join_cards(hole_cards),
            _join_cards(board_cards),
            preflop_score(hole_cards),
            rank,
            _hand_bucket(hole_cards, board_cards),
            int(texture.get("wet", False)),
            int(texture.get("paired", False)),
            int(texture.get("high", False)),
            int(top_pair),
            _join_actions(allowed.get("availableActions", [])),
            action,
            amount_value,
            _safe_ratio(amount_value, pot),
            _safe_ratio(amount_value, stack),
            int(facing_bet),
            int(voluntary),
            message,
            *opp_features,
        ),
    )
    if commit:
        conn.commit()


def _extract_telemetry_opp_features(table, seat):
    """Extract normalized opponent profile features for telemetry storage.

    Returns a list of 6 float values:
    [vpip, pfr, fold_to_bet, aggro, showdown_rate, hands_seen_norm]
    """
    profiles = table.get("opponentProfiles")
    if not profiles:
        return [0.5, 0.3, 0.5, 0.5, 0.3, 0.0]

    hero_id = seat.get("agentId")
    # Find the primary opponent (first non-folded, non-hero)
    opp_profile = None
    for other_seat in table.get("seats", []):
        if other_seat.get("agentId") == hero_id:
            continue
        if other_seat.get("folded"):
            continue
        opp_id = other_seat.get("agentId")
        if opp_id in profiles:
            opp_profile = profiles[opp_id]
            break

    if opp_profile is None:
        return [0.5, 0.3, 0.5, 0.5, 0.3, 0.0]

    hands_seen = max(getattr(opp_profile, "hands_seen", 0), 1)
    vpip_count = getattr(opp_profile, "vpip", 0)
    pfr_count = getattr(opp_profile, "pfr", 0)
    calls = getattr(opp_profile, "calls", 0)
    bets = getattr(opp_profile, "bets", 0)
    raises = getattr(opp_profile, "raises", 0)
    fold_to_bet = getattr(opp_profile, "fold_to_bet", 0)
    fold_opp = getattr(opp_profile, "opportunities_to_fold_to_bet", 0)
    showdowns = getattr(opp_profile, "showdowns", 0)

    # Normalize: VPIP can exceed hands_seen (multiple voluntary actions/hand)
    vpip = min(vpip_count / (hands_seen * 2.5), 1.0)
    pfr = min(pfr_count / (hands_seen * 2.5), 1.0)
    fold_to_bet_rate = min(fold_to_bet / fold_opp, 1.0) if fold_opp > 0 else 0.5
    bet_raise = bets + raises
    total_actions = calls + bets + raises + getattr(opp_profile, "folds", 0)
    aggro = min(bet_raise / total_actions, 1.0) if total_actions > 0 else 0.5
    showdown_rate = min(showdowns / hands_seen, 1.0)
    hands_norm = min(hands_seen / 500.0, 1.0)

    return [vpip, pfr, fold_to_bet_rate, aggro, showdown_rate, hands_norm]


def update_hand_telemetry_outcome(
    conn,
    *,
    run_id,
    hand_id,
    hero_net_chips,
    won_hand,
    final_pot=None,
    commit=True,
):
    """Overwrite outcome columns for every decision row of a hand.

    Canonical source: the arena's Showdown snapshot (payoutChips -
    totalCommittedChips plus winners[]). Use this whenever the Showdown
    payload is observed, because it is more accurate than any estimate
    derived from stack deltas.
    """
    conn.execute(
        """
        update decision_telemetry
        set hero_net_chips = ?,
            won_hand = ?,
            final_pot = ?
        where run_id = ? and hand_id = ?
        """,
        (hero_net_chips, int(won_hand), final_pot, run_id, hand_id),
    )
    if commit:
        conn.commit()


def fill_hand_telemetry_outcome_from_delta(
    conn,
    *,
    run_id,
    hand_id,
    hero_net_chips,
    won_hand,
    commit=True,
):
    """Fill outcome columns only where hero_net_chips is currently NULL.

    Fallback path used when the Showdown snapshot was not observed (the
    common case — hands transition through Showdown faster than the bot
    polls). Estimates hero_net_chips from the stack delta between the last
    seen snapshot of the previous hand and the first snapshot of this one.
    Never overwrites an existing outcome so the more accurate Showdown
    value wins whenever both fire.
    """
    conn.execute(
        """
        update decision_telemetry
        set hero_net_chips = ?,
            won_hand = ?
        where run_id = ? and hand_id = ? and hero_net_chips is null
        """,
        (hero_net_chips, int(won_hand), run_id, hand_id),
    )
    if commit:
        conn.commit()


def fill_hand_telemetry_outcome_from_replay(
    conn,
    *,
    run_id,
    table_id,
    chip_delta,
    won_hand,
    commit=True,
):
    """Fill outcome columns from a /agent/{id}/replays entry.

    The arena's `/replays` endpoint reports `chipDelta` (payoutChips -
    totalCommittedChips) for the queried agent per settled hand. Joins on
    ``table_id`` because the arena's `handId` field is documented as
    *"currently the table cuid"* — i.e. effectively the table id.

    Like the delta-fill, this never overwrites an existing outcome, so
    the Showdown and stack-delta paths keep precedence when they fire
    first.

    `won_hand` should be derived by the caller from the sign of
    ``chip_delta`` (push = loss for our purposes). Set ``won_hand=None``
    to leave the column NULL; we coerce to 0 for non-null non-truthy
    values.
    """
    if not table_id:
        return
    won_hand_int = int(bool(won_hand)) if won_hand is not None else None
    set_parts = ["hero_net_chips = ?"]
    params: list = [int(chip_delta)]
    if won_hand_int is not None:
        set_parts.append("won_hand = ?")
        params.append(won_hand_int)
    params.extend([run_id, table_id])
    set_clause = ", ".join(set_parts)
    conn.execute(
        f"""
        update decision_telemetry
        set {set_clause}
        where run_id = ? and table_id = ? and hero_net_chips is null
        """,
        params,
    )
    if commit:
        conn.commit()


def summarize_losing_buckets(conn, run_id, min_spots=20, limit=20):
    return conn.execute(
        """
        select street, hand_bucket, table_style, chosen_action,
               count(*) as spots,
               avg(hero_net_chips) as avg_delta
        from decision_telemetry
        where run_id = ? and hero_net_chips is not null
        group by street, hand_bucket, table_style, chosen_action
        having spots >= ?
        order by avg_delta asc
        limit ?
        """,
        (run_id, min_spots, limit),
    ).fetchall()


def merge_worker_db(main_path, worker_path):
    """Merge a per-worker opponent DB into the main DB.

    Used by parallel benchmark runs. Callers must pass completed, immutable
    worker snapshots; changed snapshots are distinct merges, not incremental
    deltas. This function copies event counters into ``main_path`` while
    de-duplicating canonical preflop hand facts.
    """
    main_conn = connect(main_path)
    worker_conn = sqlite3.connect(worker_path)
    worker_conn.row_factory = sqlite3.Row
    attached = False
    try:
        receipt = _worker_snapshot_receipt(worker_path, worker_conn)
        main_conn.execute("ATTACH DATABASE ? AS worker", (str(worker_path),))
        attached = True
        receipt_cursor = main_conn.execute(
            """
            insert or ignore into worker_merge_receipts(
                source_path, run_identity, content_sha256
            )
            values (?, ?, ?)
            """,
            receipt,
        )
        if not receipt_cursor.rowcount:
            main_conn.commit()
            return
        main_conn.execute(
            """
            INSERT OR IGNORE INTO opponents(
                platform, agent_id, handle, first_seen_at, last_seen_at
            )
            SELECT platform, agent_id, handle, first_seen_at, last_seen_at
            FROM worker.opponents
            """
        )
        main_conn.execute(
            """
            CREATE TEMP TABLE worker_opponent_map AS
            SELECT worker_opponents.id AS worker_id,
                   main_opponents.id AS main_id
            FROM worker.opponents AS worker_opponents
            JOIN opponents AS main_opponents
              ON main_opponents.platform = worker_opponents.platform
             AND main_opponents.agent_id = worker_opponents.agent_id
            """
        )
        main_conn.execute(
            "INSERT OR IGNORE INTO telemetry_runs SELECT * FROM worker.telemetry_runs"
        )
        main_conn.execute(
            """
            INSERT OR IGNORE INTO opponent_stats(opponent_id)
            SELECT main_id FROM worker_opponent_map
            """
        )
        main_conn.execute(
            """
            UPDATE opponent_stats
            SET hands_seen = hands_seen + coalesce((
                    SELECT sum(worker_stats.hands_seen)
                    FROM worker.opponent_stats AS worker_stats
                    JOIN worker_opponent_map AS map
                      ON map.worker_id = worker_stats.opponent_id
                    WHERE map.main_id = opponent_stats.opponent_id
                ), 0),
                calls = calls + coalesce((
                    SELECT sum(worker_stats.calls)
                    FROM worker.opponent_stats AS worker_stats
                    JOIN worker_opponent_map AS map
                      ON map.worker_id = worker_stats.opponent_id
                    WHERE map.main_id = opponent_stats.opponent_id
                ), 0),
                bets = bets + coalesce((
                    SELECT sum(worker_stats.bets)
                    FROM worker.opponent_stats AS worker_stats
                    JOIN worker_opponent_map AS map
                      ON map.worker_id = worker_stats.opponent_id
                    WHERE map.main_id = opponent_stats.opponent_id
                ), 0),
                raises = raises + coalesce((
                    SELECT sum(worker_stats.raises)
                    FROM worker.opponent_stats AS worker_stats
                    JOIN worker_opponent_map AS map
                      ON map.worker_id = worker_stats.opponent_id
                    WHERE map.main_id = opponent_stats.opponent_id
                ), 0),
                folds = folds + coalesce((
                    SELECT sum(worker_stats.folds)
                    FROM worker.opponent_stats AS worker_stats
                    JOIN worker_opponent_map AS map
                      ON map.worker_id = worker_stats.opponent_id
                    WHERE map.main_id = opponent_stats.opponent_id
                ), 0),
                fold_to_bet = fold_to_bet + coalesce((
                    SELECT sum(worker_stats.fold_to_bet)
                    FROM worker.opponent_stats AS worker_stats
                    JOIN worker_opponent_map AS map
                      ON map.worker_id = worker_stats.opponent_id
                    WHERE map.main_id = opponent_stats.opponent_id
                ), 0),
                opportunities_to_fold_to_bet =
                    opportunities_to_fold_to_bet + coalesce((
                    SELECT sum(worker_stats.opportunities_to_fold_to_bet)
                    FROM worker.opponent_stats AS worker_stats
                    JOIN worker_opponent_map AS map
                      ON map.worker_id = worker_stats.opponent_id
                    WHERE map.main_id = opponent_stats.opponent_id
                ), 0),
                showdowns = showdowns + coalesce((
                    SELECT sum(worker_stats.showdowns)
                    FROM worker.opponent_stats AS worker_stats
                    JOIN worker_opponent_map AS map
                      ON map.worker_id = worker_stats.opponent_id
                    WHERE map.main_id = opponent_stats.opponent_id
                ), 0),
                won_showdown = won_showdown + coalesce((
                    SELECT sum(worker_stats.won_showdown)
                    FROM worker.opponent_stats AS worker_stats
                    JOIN worker_opponent_map AS map
                      ON map.worker_id = worker_stats.opponent_id
                    WHERE map.main_id = opponent_stats.opponent_id
                ), 0),
                all_ins = all_ins + coalesce((
                    SELECT sum(worker_stats.all_ins)
                    FROM worker.opponent_stats AS worker_stats
                    JOIN worker_opponent_map AS map
                      ON map.worker_id = worker_stats.opponent_id
                    WHERE map.main_id = opponent_stats.opponent_id
                ), 0),
                large_bets = large_bets + coalesce((
                    SELECT sum(worker_stats.large_bets)
                    FROM worker.opponent_stats AS worker_stats
                    JOIN worker_opponent_map AS map
                      ON map.worker_id = worker_stats.opponent_id
                    WHERE map.main_id = opponent_stats.opponent_id
                ), 0),
                pressure_when_covering = pressure_when_covering + coalesce((
                    SELECT sum(worker_stats.pressure_when_covering)
                    FROM worker.opponent_stats AS worker_stats
                    JOIN worker_opponent_map AS map
                      ON map.worker_id = worker_stats.opponent_id
                    WHERE map.main_id = opponent_stats.opponent_id
                ), 0),
                updated_at = current_timestamp
            WHERE opponent_id IN (SELECT main_id FROM worker_opponent_map)
            """
        )
        main_conn.execute(
            """
            INSERT INTO opponent_preflop_hands(opponent_id, hand_id, vpip, pfr)
            SELECT map.main_id, hands.hand_id, hands.vpip, hands.pfr
            FROM worker.opponent_preflop_hands AS hands
            JOIN worker_opponent_map AS map ON map.worker_id = hands.opponent_id
            WHERE 1
            ON CONFLICT(opponent_id, hand_id) DO UPDATE SET
                vpip = max(opponent_preflop_hands.vpip, excluded.vpip),
                pfr = max(opponent_preflop_hands.pfr, excluded.pfr)
            """
        )
        main_conn.execute(
            """
            UPDATE opponent_stats
            SET preflop_hands_seen = coalesce((
                    SELECT count(*)
                    FROM opponent_preflop_hands AS hands
                    WHERE hands.opponent_id = opponent_stats.opponent_id
                ), 0),
                vpip = coalesce((
                    SELECT sum(hands.vpip)
                    FROM opponent_preflop_hands AS hands
                    WHERE hands.opponent_id = opponent_stats.opponent_id
                ), 0),
                pfr = coalesce((
                    SELECT sum(hands.pfr)
                    FROM opponent_preflop_hands AS hands
                    WHERE hands.opponent_id = opponent_stats.opponent_id
                ), 0),
                profile_stats_schema_version = 2,
                profile_stats_provenance = CASE
                    WHEN profile_stats_provenance != 'canonical'
                      OR EXISTS(
                          SELECT 1
                          FROM worker.opponent_stats AS worker_stats
                          JOIN worker_opponent_map AS map
                            ON map.worker_id = worker_stats.opponent_id
                          WHERE map.main_id = opponent_stats.opponent_id
                            AND (
                                worker_stats.profile_stats_schema_version < 2
                                OR worker_stats.profile_stats_provenance != 'canonical'
                                OR worker_stats.preflop_hands_seen != (
                                    SELECT count(*)
                                    FROM worker.opponent_preflop_hands AS hands
                                    WHERE hands.opponent_id = worker_stats.opponent_id
                                )
                            )
                      )
                    THEN 'legacy_untrusted'
                    ELSE 'canonical'
                END
            WHERE opponent_id IN (SELECT main_id FROM worker_opponent_map)
            """
        )
        main_conn.execute(
            """
            INSERT OR IGNORE INTO opponent_external_stats (
                opponent_id, competition_id, source, stats_json, fetched_at
            )
            SELECT map.main_id, stats.competition_id, stats.source,
                   stats.stats_json, stats.fetched_at
            FROM worker.opponent_external_stats AS stats
            JOIN worker_opponent_map AS map
              ON map.worker_id = stats.opponent_id
            """
        )
        main_conn.execute(
            """
            INSERT INTO opponent_actions (
                opponent_id, hand_id, street, action, amount, pot, message,
                facing_bet, voluntary, is_preflop_raise, stack_chips,
                hero_stack_chips, created_at
            )
            SELECT map.main_id, actions.hand_id, actions.street, actions.action,
                   actions.amount, actions.pot, actions.message,
                   actions.facing_bet, actions.voluntary,
                   actions.is_preflop_raise, actions.stack_chips,
                   actions.hero_stack_chips, actions.created_at
            FROM worker.opponent_actions AS actions
            JOIN worker_opponent_map AS map
              ON map.worker_id = actions.opponent_id
            """
        )
        main_conn.execute(
            """
            INSERT INTO decision_telemetry (
                run_id, hand_id, table_id, decision_index, strategy, street,
                hero_agent_id, hero_seat_number, button_seat_number,
                hero_position, hero_position_offset, seated_players,
                active_players, table_style, pot_chips, current_bet,
                call_amount, min_bet, min_raise_to, hero_stack,
                hero_current_bet, max_opponent_stack, covered_by_larger_stack,
                hole_cards, board_cards, preflop_score, made_hand_rank,
                hand_bucket, board_wet, board_paired, board_high,
                top_pair_or_better, available_actions, chosen_action,
                chosen_amount, amount_ratio_pot, amount_ratio_stack,
                facing_bet, voluntary, strategy_message, hero_net_chips,
                won_hand, final_pot, created_at
            )
            SELECT run_id, hand_id, table_id, decision_index, strategy, street,
                   hero_agent_id, hero_seat_number, button_seat_number,
                   hero_position, hero_position_offset, seated_players,
                   active_players, table_style, pot_chips, current_bet,
                   call_amount, min_bet, min_raise_to, hero_stack,
                   hero_current_bet, max_opponent_stack, covered_by_larger_stack,
                   hole_cards, board_cards, preflop_score, made_hand_rank,
                   hand_bucket, board_wet, board_paired, board_high,
                   top_pair_or_better, available_actions, chosen_action,
                   chosen_amount, amount_ratio_pot, amount_ratio_stack,
                   facing_bet, voluntary, strategy_message, hero_net_chips,
                   won_hand, final_pot, created_at
            FROM worker.decision_telemetry
            """
        )
        main_conn.execute(
            """
            INSERT INTO guard_overrides (
                run_id, hand_id, decision_index, guard_id, pre_decision,
                original_action, final_action, reason, street, pot_chips,
                call_amount, available_actions, shadow, applied,
                original_amount, final_amount, phase, precedence, created_at
            )
            SELECT run_id, hand_id, decision_index, guard_id, pre_decision,
                   original_action, final_action, reason, street, pot_chips,
                   call_amount, available_actions, shadow, applied,
                   original_amount, final_amount, phase, precedence, created_at
            FROM worker.guard_overrides
            """
        )
        main_conn.commit()
    except Exception:
        main_conn.rollback()
        raise
    finally:
        if attached:
            main_conn.execute("DROP TABLE IF EXISTS worker_opponent_map")
            main_conn.execute("DETACH DATABASE worker")
        main_conn.close()
        worker_conn.close()
