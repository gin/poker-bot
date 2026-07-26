"""Tests for scripts/sync_arena_tables.py — the arena recent-tables mirror.

Mocks the paged fetch (no network) and asserts the schema contract, seat
parsing (space-joined hole cards), the incremental stop-on-known-page logic,
and re-run idempotence. Style mirrors tests/test_sandbox_agent_stats.py
(sys.path.insert for the scripts/ import).
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sync_arena_tables as sync  # noqa: E402

HERO_ID = "cmpzvsdsavulpc7zaxq9t2j6c"


def _table(table_id, villain_id, villain_handle="villain", board=None,
           hole=("Kd", "9h")):
    return {
        "id": table_id,
        "tableNumber": 1000,
        "status": "Completed",
        "startedAt": "2026-07-09T18:02:13.648Z",
        "endedAt": "2026-07-09T18:02:17.011Z",
        "playerCount": 2,
        "handCount": 1,
        "boardCards": board or [],
        "winners": [{"seatNumber": 2, "agentId": villain_id, "amount": 15}],
        "seats": [
            {
                "seatNumber": 1,
                "agentId": HERO_ID,
                "agentName": "Luigi v4",
                "agentHandle": "cosmometrics",
                "holeCards": list(hole),
                "payoutChips": 0,
                "stackChips": 995,
            },
            {
                "seatNumber": 2,
                "agentId": villain_id,
                "agentName": "Villain",
                "agentHandle": villain_handle,
                "holeCards": ["As", "Ah"],
                "payoutChips": 15,
                "stackChips": 1005,
            },
        ],
    }


def _pager(pages):
    """Build a fetch stub serving `pages` (list of {'hasMore','data'}) by offset."""
    by_offset = {}
    offset = 0
    for page in pages:
        by_offset[offset] = page
        offset += len(page["data"])

    def fetch(competition_id, api_key, offset, limit=100, timeout=None):
        return by_offset.get(offset, {"hasMore": False, "data": []})

    return fetch


def test_schema_created(tmp_path):
    conn = sync.open_db(tmp_path / "arena_tables.sqlite")
    tables = {
        row[0]
        for row in conn.execute("select name from sqlite_master where type='table'")
    }
    assert {"arena_tables", "arena_seats", "sync_meta"} <= tables
    indexes = {
        row[0]
        for row in conn.execute("select name from sqlite_master where type='index'")
    }
    assert "idx_arena_seats_agent" in indexes
    conn.close()


def test_sync_inserts_rows_and_parses_seats(tmp_path):
    conn = sync.open_db(tmp_path / "arena_tables.sqlite")
    fetch = _pager(
        [
            {"hasMore": True, "data": [_table("t1", "v1", board=["Td", "9c", "Qc"])]},
            {"hasMore": False, "data": [_table("t2", "v2")]},
        ]
    )
    new, total, pages = sync.sync_competition(conn, "comp-1", "k", 200, fetch=fetch)
    assert (new, total) == (2, 2)
    assert pages == 2

    row = conn.execute(
        "select competition_id, board_cards, hand_count from arena_tables "
        "where id='t1'"
    ).fetchone()
    assert row == ("comp-1", "Td 9c Qc", 1)

    # Hole cards stored space-joined.
    hole = conn.execute(
        "select hole_cards, handle from arena_seats "
        "where table_id='t1' and seat_number=2"
    ).fetchone()
    assert hole == ("As Ah", "villain")

    # winners_json + raw_json retained.
    winners, raw = conn.execute(
        "select winners_json, raw_json from arena_tables where id='t1'"
    ).fetchone()
    assert "\"seatNumber\": 2" in winners
    assert "\"boardCards\"" in raw

    # Per-competition last-sync recorded.
    meta = dict(conn.execute("select key, value from sync_meta"))
    assert "last_sync:comp-1" in meta
    conn.close()


def test_first_sync_backfills_past_known_pages(tmp_path):
    """Before a completed backfill, an all-known page must NOT stop paging:
    page 0 is always the newest tables, so '0 new' cannot distinguish
    'caught up' from 'never fetched deeper history'."""
    calls = []

    def fetch(competition_id, api_key, offset, limit=100, timeout=None):
        calls.append(offset)
        pages = {
            0: {"hasMore": True, "data": [_table("t1", "v1")]},
            1: {"hasMore": True, "data": [_table("t1", "v1")]},  # all-known
            2: {"hasMore": False, "data": [_table("t99", "v99")]},  # deeper!
        }
        return pages.get(offset, {"hasMore": False, "data": []})

    conn = sync.open_db(tmp_path / "arena_tables.sqlite")
    new, total, pages = sync.sync_competition(conn, "comp-1", "k", 200, fetch=fetch)
    assert new == 2  # t1 AND the deeper t99
    assert total == 2
    assert calls == [0, 1, 2]
    # Reaching hasMore=False records the completed backfill.
    done = conn.execute(
        "select value from sync_meta where key = 'backfill_done:comp-1'"
    ).fetchone()
    assert done is not None
    conn.close()


def test_incremental_stops_on_all_known_page_after_backfill(tmp_path):
    """Once a backfill completed, a fully-known page halts paging."""
    calls = []

    def fetch(competition_id, api_key, offset, limit=100, timeout=None):
        calls.append(offset)
        pages = {
            0: {"hasMore": True, "data": [_table("t1", "v1")]},
            1: {"hasMore": True, "data": [_table("t1", "v1")]},  # all-known
            2: {"hasMore": True, "data": [_table("t99", "v99")]},
        }
        return pages.get(offset, {"hasMore": False, "data": []})

    conn = sync.open_db(tmp_path / "arena_tables.sqlite")
    conn.execute(
        "insert into sync_meta(key, value) values('backfill_done:comp-1', 'x')"
    )
    new, total, pages = sync.sync_competition(conn, "comp-1", "k", 200, fetch=fetch)
    assert new == 1
    assert total == 1
    # Fetched offset 0 (new) then offset 1 (all-known -> stop); never offset 2.
    assert calls == [0, 1]
    # --backfill overrides the early stop even when backfill_done is set
    # (pages through to the empty page past the last data).
    calls.clear()
    new, total, pages = sync.sync_competition(
        conn, "comp-1", "k", 200, fetch=fetch, backfill=True
    )
    assert calls == [0, 1, 2, 3]
    assert total == 2
    conn.close()


def test_limit_pages_cap(tmp_path):
    def fetch(competition_id, api_key, offset, limit=100, timeout=None):
        # Always a full new page with hasMore True -> only the cap stops it.
        return {"hasMore": True, "data": [_table(f"t{offset}", f"v{offset}")]}

    conn = sync.open_db(tmp_path / "arena_tables.sqlite")
    new, total, pages = sync.sync_competition(conn, "comp-1", "k", 3, fetch=fetch)
    assert pages == 3
    assert new == 3
    conn.close()


def test_rerun_inserts_nothing_new(tmp_path):
    db = tmp_path / "arena_tables.sqlite"
    pages = [
        {"hasMore": True, "data": [_table("t1", "v1")]},
        {"hasMore": False, "data": [_table("t2", "v2")]},
    ]
    conn = sync.open_db(db)
    sync.sync_competition(conn, "comp-1", "k", 200, fetch=_pager(pages))
    conn.close()

    conn = sync.open_db(db)
    new, total, npages = sync.sync_competition(
        conn, "comp-1", "k", 200, fetch=_pager(pages)
    )
    assert new == 0
    assert total == 2
    # First page is all-known, so it stops after one fetch.
    assert npages == 1
    conn.close()


def test_verify_against_telemetry(tmp_path):
    db = tmp_path / "arena_tables.sqlite"
    conn = sync.open_db(db)
    # v1 exact-matches; shifted variant of v2 shares the 14-char suffix.
    v2 = "cmr1v3z8q2dlphdvhmbf2t96p"
    v2_seat = "cmr2v3z8q2dlphdvhmbf2t96p"  # one segment shifted, same suffix
    fetch = _pager(
        [
            {"hasMore": False, "data": [
                _table("t1", "cmq6anuh00b4xf9kbh76xqi85"),
                _table("t2", v2_seat),
                _table("t3", "cmZZZunknownunknownunkno"),
            ]},
        ]
    )
    sync.sync_competition(conn, "comp-1", "k", 200, fetch=fetch)

    tele = tmp_path / "telemetry.sqlite"
    tconn = sqlite3.connect(tele)
    tconn.execute(
        "create table opponents (agent_id text, handle text)"
    )
    tconn.executemany(
        "insert into opponents(agent_id, handle) values (?, ?)",
        [("cmq6anuh00b4xf9kbh76xqi85", "a"), (v2, "b")],
    )
    tconn.commit()
    tconn.close()

    result = sync.verify_against_telemetry(conn, tele, HERO_ID)
    # 3 non-hero villains total (hero seat excluded).
    assert result["non_hero_seat_agents"] == 3
    assert result["exact_match"] == 1  # only v1 exact
    assert result["suffix_match"] == 2  # v1 exact + v2 by suffix
    conn.close()


def test_expand_competitions():
    assert sync._expand_competitions(["a,b", "c", "b"]) == ["a", "b", "c"]
    assert sync._expand_competitions(["  x , y "]) == ["x", "y"]
