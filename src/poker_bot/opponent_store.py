"""SQLite-backed opponent profile storage."""

from __future__ import annotations

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


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value and value.strip():
        return Path(value).expanduser()
    return default


def default_db_path():
    return _env_path(OPPONENT_DB_ENV, DEFAULT_DB_PATH)


def default_telemetry_db_path():
    return _env_path(TELEMETRY_DB_ENV, default_db_path())


def connect(path=None, *, telemetry: bool = False):
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
    _ensure_columns(
        conn,
        "opponent_actions",
        {
            "message": "text",
        },
    )
    _ensure_columns(
        conn,
        "decision_telemetry",
        {
            "hero_position": "text",
            "hero_position_offset": "integer",
            "seated_players": "integer",
        },
    )
    conn.commit()


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


def upsert_opponent(conn, platform, agent_id, handle=None):
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
    conn.commit()
    return opponent_id


def increment_hand_seen(conn, platform, agent_id, handle=None):
    opponent_id = upsert_opponent(conn, platform, agent_id, handle)
    conn.execute(
        """
        update opponent_stats
        set hands_seen = hands_seen + 1,
            updated_at = current_timestamp
        where opponent_id = ?
        """,
        (opponent_id,),
    )
    conn.commit()
    return opponent_id


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
):
    opponent_id = upsert_opponent(conn, platform, agent_id, handle)
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

    conn.execute(
        """
        insert into opponent_actions(
            opponent_id, hand_id, street, action, amount, pot, message, facing_bet,
            stack_chips, hero_stack_chips
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            stack_chips,
            hero_stack_chips,
        ),
    )
    conn.execute(
        """
        update opponent_stats
        set vpip = vpip + ?,
            pfr = pfr + ?,
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
            int(action == "raise" and street == "Preflop"),
            int(action == "call"),
            int(action == "bet"),
            int(action == "raise"),
            int(action == "fold"),
            int(action == "fold" and facing_bet),
            int(facing_bet),
            all_in,
            large_bet,
            pressure_when_covering,
            opponent_id,
        ),
    )
    conn.commit()
    return opponent_id


def load_profile(conn, platform, agent_id):
    row = conn.execute(
        """
        select o.agent_id, o.handle as name, s.*, e.stats_json
        from opponents o
        join opponent_stats s on s.opponent_id = o.id
        left join (
            select opponent_id, stats_json
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
            profiles[agent_id] = profile
    return profiles


def profile_to_mapping(profile: OpponentProfile):
    mapping = {
        "name": profile.name,
        "hands_seen": profile.hands_seen,
        "vpip": profile.vpip,
        "pfr": profile.pfr,
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

    conn.execute(
        """
        insert into decision_telemetry(
            run_id, hand_id, decision_index, strategy, street,
            hero_agent_id, hero_seat_number, button_seat_number, hero_position,
            hero_position_offset, seated_players, active_players,
            table_style, pot_chips, current_bet, call_amount, min_bet,
            min_raise_to, hero_stack, hero_current_bet, max_opponent_stack,
            covered_by_larger_stack, hole_cards, board_cards, preflop_score,
            made_hand_rank, hand_bucket, board_wet, board_paired, board_high,
            top_pair_or_better, available_actions, chosen_action, chosen_amount,
            amount_ratio_pot, amount_ratio_stack, facing_bet, voluntary,
            strategy_message
        )
        values (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            run_id,
            hand_id,
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
        ),
    )
    conn.commit()


def update_hand_telemetry_outcome(
    conn,
    *,
    run_id,
    hand_id,
    hero_net_chips,
    won_hand,
    final_pot=None,
):
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

    Used by parallel benchmark runs. Each worker writes to a private
    SQLite file; this function copies the data into ``main_path``,
    summing ``opponent_stats`` counters and de-duplicating unique
    tables.
    """
    main_conn = connect(main_path)
    worker_conn = sqlite3.connect(worker_path)
    worker_conn.row_factory = sqlite3.Row
    try:
        main_conn.execute("ATTACH DATABASE ? AS worker", (str(worker_path),))
        main_conn.execute(
            "INSERT OR IGNORE INTO opponents SELECT * FROM worker.opponents"
        )
        main_conn.execute(
            "INSERT OR IGNORE INTO opponent_external_stats "
            "SELECT * FROM worker.opponent_external_stats"
        )
        main_conn.execute(
            "INSERT OR IGNORE INTO telemetry_runs SELECT * FROM worker.telemetry_runs"
        )
        worker_stats = worker_conn.execute("SELECT * FROM opponent_stats").fetchall()
        for row in worker_stats:
            main_row = main_conn.execute(
                "SELECT * FROM opponent_stats WHERE opponent_id = ?",
                (row["opponent_id"],),
            ).fetchone()
            if main_row is None:
                main_conn.execute(
                    """
                    INSERT INTO opponent_stats (
                        opponent_id, hands_seen, vpip, pfr, calls, bets, raises,
                        folds, fold_to_bet, opportunities_to_fold_to_bet,
                        showdowns, won_showdown, all_ins, large_bets,
                        pressure_when_covering
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["opponent_id"],
                        row["hands_seen"],
                        row["vpip"],
                        row["pfr"],
                        row["calls"],
                        row["bets"],
                        row["raises"],
                        row["folds"],
                        row["fold_to_bet"],
                        row["opportunities_to_fold_to_bet"],
                        row["showdowns"],
                        row["won_showdown"],
                        row["all_ins"],
                        row["large_bets"],
                        row["pressure_when_covering"],
                    ),
                )
            else:
                main_conn.execute(
                    """
                    UPDATE opponent_stats
                    SET hands_seen = hands_seen + ?,
                        vpip = vpip + ?,
                        pfr = pfr + ?,
                        calls = calls + ?,
                        bets = bets + ?,
                        raises = raises + ?,
                        folds = folds + ?,
                        fold_to_bet = fold_to_bet + ?,
                        opportunities_to_fold_to_bet =
                            opportunities_to_fold_to_bet + ?,
                        showdowns = showdowns + ?,
                        won_showdown = won_showdown + ?,
                        all_ins = all_ins + ?,
                        large_bets = large_bets + ?,
                        pressure_when_covering = pressure_when_covering + ?,
                        updated_at = current_timestamp
                    WHERE opponent_id = ?
                    """,
                    (
                        row["hands_seen"],
                        row["vpip"],
                        row["pfr"],
                        row["calls"],
                        row["bets"],
                        row["raises"],
                        row["folds"],
                        row["fold_to_bet"],
                        row["opportunities_to_fold_to_bet"],
                        row["showdowns"],
                        row["won_showdown"],
                        row["all_ins"],
                        row["large_bets"],
                        row["pressure_when_covering"],
                        row["opponent_id"],
                    ),
                )
        main_conn.execute(
            """
            INSERT INTO opponent_actions (
                opponent_id, hand_id, street, action, amount, pot, message,
                facing_bet, stack_chips, hero_stack_chips, created_at
            )
            SELECT opponent_id, hand_id, street, action, amount, pot, message,
                   facing_bet, stack_chips, hero_stack_chips, created_at
            FROM worker.opponent_actions
            """
        )
        main_conn.execute(
            """
            INSERT INTO decision_telemetry (
                run_id, hand_id, decision_index, strategy, street, hero_agent_id,
                hero_seat_number, button_seat_number, hero_position,
                hero_position_offset, seated_players, active_players,
                table_style, pot_chips, current_bet, call_amount, min_bet,
                min_raise_to, hero_stack, hero_current_bet, max_opponent_stack,
                covered_by_larger_stack, hole_cards, board_cards, preflop_score,
                made_hand_rank, hand_bucket, board_wet, board_paired,
                board_high, top_pair_or_better, available_actions, chosen_action,
                chosen_amount, amount_ratio_pot, amount_ratio_stack, facing_bet,
                voluntary, strategy_message, hero_net_chips, won_hand, final_pot,
                created_at
            )
            SELECT run_id, hand_id, decision_index, strategy, street,
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
        main_conn.commit()
    finally:
        main_conn.execute("DETACH DATABASE worker")
        main_conn.close()
        worker_conn.close()

