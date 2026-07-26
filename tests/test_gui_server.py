"""Tests for the replay GUI arena-overlay join (gui/server.py).

Focus: /api/hand attaches fully-revealed hole cards + winners from the
arena mirror when configured and the table matches, and degrades quietly
when the arena DB is absent or the table is unknown.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from poker_bot.opponent_store import init_db

REPO_ROOT = Path(__file__).resolve().parents[1]

HERO = "hero_agent_id_0001"
VILLAIN = "villain_agent_id_002"
RUN_ID = "run-1"
HAND_ID = "tbl_match:2026-07-07T08:51:50+00:00"
TABLE_ID = "tbl_match"


def _load_server():
    spec = importlib.util.spec_from_file_location(
        "gui_server", REPO_ROOT / "gui" / "server.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_telemetry(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    conn.execute(
        "insert into telemetry_runs(run_id, strategy) values (?, ?)",
        (RUN_ID, "s4"),
    )
    conn.execute(
        """
        insert into decision_telemetry(
            run_id, hand_id, table_id, decision_index, strategy, street,
            hero_agent_id, hero_position, hole_cards, board_cards,
            chosen_action, chosen_amount, pot_chips, hero_stack,
            hero_net_chips, won_hand, created_at
        ) values (?, ?, ?, 0, 's4', 'Flop', ?, 'BTN', '5c 5d',
                  'Td 9c Qc', 'bet', 20, 40, 950, -50, 0,
                  '2026-07-07T08:51:50')
        """,
        (RUN_ID, HAND_ID, TABLE_ID, HERO),
    )
    # Opponent profile rows so _profile_row returns something.
    cur = conn.execute(
        "insert into opponents(platform, agent_id, handle) values "
        "('playground', ?, 'villain')",
        (VILLAIN,),
    )
    opp_id = cur.lastrowid
    conn.execute(
        "insert into opponent_stats(opponent_id, hands_seen, vpip, pfr, "
        "calls, bets, raises, folds) values (?, 10, 5, 3, 4, 2, 1, 3)",
        (opp_id,),
    )
    conn.execute(
        "insert into opponent_actions(opponent_id, hand_id, street, action, "
        "amount, pot, created_at) values (?, ?, 'Flop', 'call', 20, 40, "
        "'2026-07-07T08:51:51')",
        (opp_id, HAND_ID),
    )
    conn.commit()
    conn.close()


def _build_arena(path: Path, table_id: str = TABLE_ID) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        create table arena_tables (
            id text primary key, competition_id text not null,
            table_number integer, started_at text, ended_at text,
            player_count integer, hand_count integer, board_cards text,
            winners_json text not null, raw_json text not null,
            fetched_at text not null
        );
        create table arena_seats (
            table_id text not null references arena_tables(id),
            seat_number integer not null, agent_id text, handle text,
            hole_cards text, payout_chips integer, stack_chips integer,
            primary key (table_id, seat_number)
        );
        create table sync_meta (key text primary key, value text);
        """
    )
    winners = json.dumps(
        [
            {
                "seatNumber": 2,
                "agentId": VILLAIN,
                "agentName": "Villain",
                "amount": 100,
                "handName": "Two Pair",
                "message": "rivered it",
            }
        ]
    )
    conn.execute(
        "insert into arena_tables(id, competition_id, board_cards, "
        "winners_json, raw_json, fetched_at) values (?, 'comp', "
        "'Td 9c Qc Js 2h', ?, '{}', '2026-07-07')",
        (table_id, winners),
    )
    conn.executemany(
        "insert into arena_seats(table_id, seat_number, agent_id, handle, "
        "hole_cards, payout_chips, stack_chips) values (?, ?, ?, ?, ?, ?, ?)",
        [
            (table_id, 1, HERO, "cosmometrics", "5c 5d", -50, 950),
            (table_id, 2, VILLAIN, "villain", "Kd 9h", 100, 1050),
        ],
    )
    conn.commit()
    conn.close()


@pytest.fixture
def server(tmp_path):
    module = _load_server()
    tele = tmp_path / "telemetry.sqlite"
    _build_telemetry(tele)
    module.DB_PATH = tele
    module.OPP_DB_PATH = None
    module.ARENA_DB_PATH = None
    return module


def _hand(module):
    with TestClient(module.app) as client:
        resp = client.get(
            "/api/hand", params={"run_id": RUN_ID, "hand_id": HAND_ID}
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_attaches_revealed_and_winners_when_arena_present(server, tmp_path):
    arena = tmp_path / "arena.sqlite"
    _build_arena(arena)
    server.ARENA_DB_PATH = arena

    data = _hand(server)

    assert data["final_board"] == "Td 9c Qc Js 2h"
    assert data["winners"] == [
        {
            "agentName": "Villain",
            "amount": 100,
            "handName": "Two Pair",
            "message": "rivered it",
        }
    ]
    assert len(data["opponents"]) == 1
    opp = data["opponents"][0]
    assert opp["agent_id"] == VILLAIN
    assert opp["revealed_hole_cards"] == "Kd 9h"
    assert opp["payout_chips"] == 100


def test_omits_arena_fields_when_arena_absent(server):
    server.ARENA_DB_PATH = None

    data = _hand(server)

    assert "winners" not in data
    assert "final_board" not in data
    assert "revealed_hole_cards" not in data["opponents"][0]


def test_graceful_when_table_missing_from_arena(server, tmp_path):
    arena = tmp_path / "arena.sqlite"
    _build_arena(arena, table_id="some_other_table")  # no row for TABLE_ID
    server.ARENA_DB_PATH = arena

    data = _hand(server)

    assert "winners" not in data
    assert "revealed_hole_cards" not in data["opponents"][0]


def test_arena_hand_helper_returns_none_without_table_id(server, tmp_path):
    arena = tmp_path / "arena.sqlite"
    _build_arena(arena)
    server.ARENA_DB_PATH = arena
    # Old sim data: no table_id -> None, no lookup attempted.
    assert server._arena_hand(None) is None
    assert server._arena_hand("") is None
    found = server._arena_hand(TABLE_ID)
    assert found is not None
    assert found["seats_by_agent"][VILLAIN]["revealed_hole_cards"] == "Kd 9h"
