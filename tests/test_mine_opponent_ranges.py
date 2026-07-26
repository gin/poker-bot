"""Tests for scripts/mine_opponent_ranges.py (revealed-card range miner).

Builds tiny fixture arena + telemetry SQLite DBs with known cards / boards /
actions, then asserts classification, street-board slicing, big-bet air%,
and playbook emission thresholds + no-overwrite behavior. Style mirrors
tests/test_sandbox_agent_stats.py (sys.path.insert for scripts/).
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mine_opponent_ranges as miner  # noqa: E402

HERO = "hero_agent_000000000"
VILLAIN = "villain_agent_11111"


def _make_arena_db(path, tables):
    """tables: list of (table_id, board, winners, seats[(agent,handle,hole)])."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        create table arena_tables (id text primary key, competition_id text,
          table_number integer, started_at text, ended_at text,
          player_count integer, hand_count integer, board_cards text,
          winners_json text not null, raw_json text not null,
          fetched_at text not null);
        create table arena_seats (table_id text, seat_number integer,
          agent_id text, handle text, hole_cards text, payout_chips integer,
          stack_chips integer, primary key (table_id, seat_number));
        """
    )
    for table_id, board, winners, seats in tables:
        conn.execute(
            "insert into arena_tables values (?,?,?,?,?,?,?,?,?,?,?)",
            (
                table_id,
                "comp",
                1,
                None,
                None,
                len(seats),
                1,
                board,
                json.dumps(winners),
                "{}",
                "now",
            ),
        )
        for i, (agent, handle, hole) in enumerate(seats, start=1):
            conn.execute(
                "insert into arena_seats values (?,?,?,?,?,?,?)",
                (table_id, i, agent, handle, hole, 0, 1000),
            )
    conn.commit()
    conn.close()


def _make_telemetry_db(path, actions):
    """actions: list of (agent, handle, hand_id, street, action, amount, pot,
    stack). Inserted in order (id ordering preserved)."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        create table opponents (id integer primary key, platform text,
          agent_id text not null, handle text);
        create table opponent_actions (id integer primary key,
          opponent_id integer, hand_id text, street text, action text,
          amount integer, pot integer, stack_chips integer);
        """
    )
    ids = {}
    for agent, handle, *_ in actions:
        if agent not in ids:
            ids[agent] = len(ids) + 1
            conn.execute(
                "insert into opponents values (?,?,?,?)",
                (ids[agent], "arena", agent, handle),
            )
    for agent, _handle, hand_id, street, action, amount, pot, stack in actions:
        conn.execute(
            "insert into opponent_actions(opponent_id, hand_id, street, "
            "action, amount, pot, stack_chips) values (?,?,?,?,?,?,?)",
            (ids[agent], hand_id, street, action, amount, pot, stack),
        )
    conn.commit()
    conn.close()


# ── Pure-helper unit tests ──────────────────────────────────────────────────


def test_card_normalization_and_parse():
    assert miner.normalize_card("ad") == "Ad"
    assert miner.normalize_card("KD") == "Kd"
    assert miner.parse_cards("ad kh") == ["Ad", "Kh"]
    assert miner.parse_cards("") == []
    assert miner.parse_cards(None) == []


def test_street_board_slicing():
    board = ["Ts", "9c", "Qc", "Js", "2h"]
    assert miner.street_board(board, "preflop") == []
    assert miner.street_board(board, "flop") == ["Ts", "9c", "Qc"]
    assert miner.street_board(board, "turn") == ["Ts", "9c", "Qc", "Js"]
    assert miner.street_board(board, "river") == board


def test_classify_air_on_known_river():
    # 2c 7d on Ts 9c Qc Js 2h -> a pair of deuces (weak pair) = marginal;
    # 3d 4h -> no pair, no draw on completed board = air.
    board = ["Ts", "9c", "Qc", "Js", "2h"]
    assert miner.classify_strength(["3d", "4h"], "river", board) == "air"
    assert miner.classify_strength(["2c", "7d"], "river", board) == "marginal"
    # Top pair (Queen) = value.
    assert miner.classify_strength(["Qd", "4h"], "river", board) == "value"


def test_classify_preflop_by_score():
    assert miner.classify_strength(["Ah", "Kh"], "preflop", []) == "value"  # AKs 80
    assert miner.classify_strength(["Kd", "Qc"], "preflop", []) == "marginal"
    assert miner.classify_strength(["7d", "2c"], "preflop", []) == "air"


def test_sizing_bucket():
    assert miner.sizing_bucket(10, 100) == "small"
    assert miner.sizing_bucket(80, 100) == "medium"
    assert miner.sizing_bucket(150, 100) == "overbet"
    assert miner.sizing_bucket(10, 0) is None
    assert miner.sizing_bucket(None, 100) is None


# ── Integration tests over fixture DBs ──────────────────────────────────────


def test_river_air_big_bet_counted(tmp_path):
    arena = tmp_path / "arena.sqlite"
    tel = tmp_path / "tel.sqlite"
    board = "Ts 9c Qc Js 2h"
    _make_arena_db(
        arena,
        [
            (
                "t1",
                board,
                [{"agentId": HERO, "amount": 100}],
                [(HERO, "hero", "Ac Kc"), (VILLAIN, "villain", "3d 4h")],
            )
        ],
    )
    # Villain bets 80 into 100 (0.8x -> big bet) on the river with air.
    _make_telemetry_db(
        tel,
        [(VILLAIN, "villain", "t1:2026-01-01T00:00:00", "River", "bet", 80, 100, 900)],
    )
    reports = miner.mine(arena, [tel], HERO, min_hands=1, opponent=None)
    assert len(reports) == 1
    rep = reports[0]
    assert rep["handle"] == "villain"
    assert rep["per_street"]["river"] == {"value": 0, "marginal": 0, "air": 1}
    assert rep["big_bet_total"] == 1
    assert rep["big_bet_air"] == 1
    assert rep["big_bet_air_pct"] == 1.0
    assert rep["sizing"]["medium"]["air"] == 1


def test_preflop_raise_and_allin_ranges(tmp_path):
    arena = tmp_path / "arena.sqlite"
    tel = tmp_path / "tel.sqlite"
    _make_arena_db(
        arena,
        [
            ("t1", "", [], [(VILLAIN, "villain", "Ad Kh")]),
            ("t2", "", [{"agentId": VILLAIN, "amount": 50}],
             [(VILLAIN, "villain", "Qs Qd")]),
        ],
    )
    _make_telemetry_db(
        tel,
        [
            (VILLAIN, "villain", "t1:x", "Preflop", "raise", 6, 3, 900),
            # amount >= stack -> all-in shove.
            (VILLAIN, "villain", "t2:y", "Preflop", "raise", 900, 3, 900),
        ],
    )
    reports = miner.mine(arena, [tel], HERO, min_hands=1, opponent=None)
    rep = reports[0]
    assert rep["preflop_raise_range"] == ["Ad Kh", "Qs Qd"]
    assert rep["all_in_range"] == ["Qs Qd"]
    # Villain won t2 -> a showdown win.
    assert rep["showdown_wins"] == 1


def test_calldown_quality_and_hero_excluded(tmp_path):
    arena = tmp_path / "arena.sqlite"
    tel = tmp_path / "tel.sqlite"
    board = "Ts 9c Qc Js 2h"
    _make_arena_db(
        arena,
        [("t1", board, [], [(HERO, "hero", "Ac Kc"),
                            (VILLAIN, "villain", "3d 4h")])],
    )
    _make_telemetry_db(
        tel,
        [
            (HERO, "hero", "t1:x", "River", "bet", 50, 100, 900),  # excluded
            (VILLAIN, "villain", "t1:x", "River", "call", 50, 100, 900),  # air call
        ],
    )
    reports = miner.mine(arena, [tel], HERO, min_hands=1, opponent=None)
    assert len(reports) == 1  # hero excluded
    rep = reports[0]
    assert rep["calldown"] == {"value": 0, "marginal": 0, "air": 1}
    assert rep["calldown_loose_pct"] == 1.0


def test_min_hands_filter(tmp_path):
    arena = tmp_path / "arena.sqlite"
    tel = tmp_path / "tel.sqlite"
    _make_arena_db(arena, [("t1", "", [], [(VILLAIN, "villain", "Ad Kh")])])
    _make_telemetry_db(
        tel, [(VILLAIN, "villain", "t1:x", "Preflop", "raise", 6, 3, 900)]
    )
    assert miner.mine(arena, [tel], HERO, min_hands=2, opponent=None) == []
    assert len(miner.mine(arena, [tel], HERO, min_hands=1, opponent=None)) == 1


# ── Playbook emission ───────────────────────────────────────────────────────


def test_derive_knobs_thresholds():
    bluffy = {"big_bet_total": 20, "big_bet_air_pct": 0.6,
              "calldown_total": 0, "calldown_loose_pct": 0.0}
    assert miner.derive_knobs(bluffy) == {"bluffcatch_max_price": 0.5}

    honest = {"big_bet_total": 20, "big_bet_air_pct": 0.1,
              "calldown_total": 0, "calldown_loose_pct": 0.0}
    assert miner.derive_knobs(honest) == {"no_bluffcatch": True}

    station = {"big_bet_total": 3, "big_bet_air_pct": 0.9,
               "calldown_total": 20, "calldown_loose_pct": 0.7}
    assert miner.derive_knobs(station) == {"station_bet_fraction": 0.85}

    thin = {"big_bet_total": 5, "big_bet_air_pct": 0.9,
            "calldown_total": 5, "calldown_loose_pct": 0.9}
    assert miner.derive_knobs(thin) == {}  # n<15 gates everything out


def test_playbook_no_overwrite_without_force():
    existing = {"villain": {"bluffcatch_max_price": 0.9, "custom": "keep"}}
    proposals = {"villain": {"bluffcatch_max_price": 0.5, "no_bluffcatch": True}}

    merged, _ = miner.merge_playbook(existing, proposals, force=False)
    assert merged["villain"]["bluffcatch_max_price"] == 0.9  # existing kept
    assert merged["villain"]["no_bluffcatch"] is True  # new key added
    assert merged["villain"]["custom"] == "keep"

    forced, _ = miner.merge_playbook(existing, proposals, force=True)
    assert forced["villain"]["bluffcatch_max_price"] == 0.5  # overwritten


def test_emit_playbook_writes_file(tmp_path):
    path = tmp_path / "opponent_playbook.json"
    reports = [{"handle": "villain", "big_bet_total": 20, "big_bet_air_pct": 0.6,
                "calldown_total": 0, "calldown_loose_pct": 0.0}]
    miner.emit_playbook(reports, path, force=False)
    data = json.loads(path.read_text())
    assert data["villain"]["bluffcatch_max_price"] == 0.5


def test_wake_up_stat_cross_street_and_check_raise():
    """Passive earlier (flop call, or same-street check) then bet/raise =
    a wake-up; revealed cards classify it. The trap hand's line: check-call
    flop, check-raise turn with a made flush."""
    board = ["Ks", "Qh", "4s", "Qs", "Ac"]
    hands = {
        "h1": {  # trap: called flop, then raised turn with the made flush
            "hole": ["2s", "3s"], "board": board,
            "reached_showdown": True, "won": True,
            "actions": [("Flop", "call", 36, 80, 5000),
                        ("Turn", "raise", 401, 731, 5000)],
        },
        "h2": {  # same-street check-raise, strong
            "hole": ["Qd", "Qc"], "board": board,
            "reached_showdown": False, "won": False,
            "actions": [("Turn", "check", None, 188, 5000),
                        ("Turn", "raise", 400, 731, 5000)],
        },
        "h3": {  # plain barrel, no passivity first: NOT a wake-up
            "hole": ["7h", "2d"], "board": board,
            "reached_showdown": False, "won": False,
            "actions": [("Flop", "bet", 50, 80, 5000)],
        },
    }
    report = miner.build_report("villain", "agent-x", hands)
    assert report["wake_up_total"] == 2
    assert report["wake_up_strong"] == 2
    assert report["wake_up_strong_pct"] == 1.0


def test_trap_prone_knob_gated_on_n():
    hot = {"big_bet_total": 0, "big_bet_air_pct": 0, "calldown_total": 0,
           "calldown_loose_pct": 0, "wake_up_total": 8, "wake_up_strong_pct": 0.75}
    assert miner.derive_knobs(hot) == {"trap_prone": True}
    thin = dict(hot, wake_up_total=5)
    assert miner.derive_knobs(thin) == {}
    honest = dict(hot, wake_up_strong_pct=0.5)
    assert miner.derive_knobs(honest) == {}
