# """Tests for hand-outcome telemetry backfill paths in main.py.

# Covers the two complementary backfill strategies:

# * Stack-delta inference: when a new hand appears at a table where we
#   observed the previous hand, hero_net_chips is estimated from the
#   stack delta between the last seen snapshot of the previous hand and
#   the first seen snapshot of the new one.
# * Showdown overwrite: when the arena returns a Showdown snapshot with
#   ``winners``, the canonical payout-based outcome overwrites any
#   stack-delta estimate (because the Showdown value is more accurate).

# Both paths rely on ``_derive_hand_id`` for a stable per-hand identifier,
# since the arena payload does not expose a handId field.
# """

# import importlib

# from poker_bot.opponent_store import (
#     connect,
#     create_telemetry_run,
#     fill_hand_telemetry_outcome_from_delta,
#     fill_hand_telemetry_outcome_from_replay,
#     record_decision_telemetry,
#     update_hand_telemetry_outcome,
# )


# HERO_ID = "hero-agent"


# def _make_seat(stack, committed=0, payout=0):
#     return {
#         "agentId": HERO_ID,
#         "seatNumber": 1,
#         "holeCards": ["AS", "KS"],
#         "stackChips": stack,
#         "currentBetChips": 0,
#         "totalCommittedChips": committed,
#         "payoutChips": payout,
#     }


# def _make_table(
#     seats, *, street="Preflop", table_id="T1", winners=None, hero_acting=True
# ):
#     table = {
#         "tableId": table_id,
#         "street": street,
#         "boardCards": [],
#         "potChips": 75,
#         "currentBet": 50,
#         "buttonSeatNumber": 1,
#         "actingAgentId": HERO_ID if hero_acting else None,
#         "actingSeatNumber": 1 if hero_acting else None,
#         "allowedActions": {
#             "availableActions": ["fold", "call", "raise"],
#             "callAmount": 50,
#             "minBet": 50,
#             "minRaiseTo": 150,
#         },
#         "seats": seats,
#     }
#     if winners is not None:
#         table["winners"] = winners
#     return table


# def _load_main():
#     """Import main.py lazily so test collection doesn't require its CLI deps."""
#     import sys

#     src_path = "src"
#     if src_path not in sys.path:
#         sys.path.insert(0, src_path)
#     import main as main_module

#     return importlib.reload(main_module)


# def test_process_hand_identity_initializes_new_hand():
#     main = _load_main()
#     state: dict = {}
#     table = _make_table([_make_seat(stack=2000)], table_id="T-A")

#     hand_id, is_new, prev_id, prev_stack, prev_committed = main._process_hand_identity(
#         table, state
#     )

#     assert is_new is True
#     assert hand_id.startswith("T-A:")
#     assert prev_id is None
#     assert prev_stack is None
#     assert prev_committed is None
#     # State was populated
#     assert state["live_hand_state"]["T-A"]["hand_id"] == hand_id


# def test_process_hand_identity_stable_across_snapshots_in_same_hand():
#     main = _load_main()
#     state: dict = {}

#     table_preflop = _make_table(
#         [_make_seat(stack=2000)], street="Preflop", table_id="T-B"
#     )
#     table_flop = _make_table([_make_seat(stack=1990)], street="Flop", table_id="T-B")

#     hand_id_1, is_new_1, *_ = main._process_hand_identity(table_preflop, state)
#     hand_id_2, is_new_2, *_ = main._process_hand_identity(table_flop, state)

#     assert is_new_1 is True
#     assert is_new_2 is False
#     assert hand_id_1 == hand_id_2


# def test_process_hand_identity_detects_preflop_reappearance_as_new_hand():
#     main = _load_main()
#     state: dict = {}

#     table_preflop_1 = _make_table(
#         [_make_seat(stack=2000)], street="Preflop", table_id="T-C"
#     )
#     table_flop_1 = _make_table([_make_seat(stack=1990)], street="Flop", table_id="T-C")
#     table_preflop_2 = _make_table(
#         [_make_seat(stack=1985)], street="Preflop", table_id="T-C"
#     )

#     h1, is_new_1, _, _, _ = main._process_hand_identity(table_preflop_1, state)
#     main._record_hand_stack_state(table_preflop_1, state, HERO_ID)
#     h2, is_new_2, _, _, _ = main._process_hand_identity(table_flop_1, state)
#     main._record_hand_stack_state(table_flop_1, state, HERO_ID)
#     h3, is_new_3, prev_3, prev_stack_3, _ = main._process_hand_identity(
#         table_preflop_2, state
#     )

#     # hand 1 boundary
#     assert is_new_1 is True
#     assert h1.startswith("T-C:")
#     # mid-hand: same hand_id, not new
#     assert is_new_2 is False
#     assert h2 == h1
#     # new hand: Preflop returned after Flop
#     assert is_new_3 is True
#     assert h3.startswith("T-C:")
#     assert h3 != h1
#     assert prev_3 == h1
#     assert prev_stack_3 == 1990  # last recorded stack of previous hand (at Flop)


# def test_process_hand_identity_distinct_tables_independent():
#     main = _load_main()
#     state: dict = {}

#     t_x = _make_table([_make_seat(stack=2000)], table_id="T-X")
#     t_y = _make_table([_make_seat(stack=3000)], table_id="T-Y")

#     h_x, _, _, _, _ = main._process_hand_identity(t_x, state)
#     h_y, _, _, _, _ = main._process_hand_identity(t_y, state)

#     assert h_x.startswith("T-X:")
#     assert h_y.startswith("T-Y:")
#     assert h_x != h_y
#     assert "T-X" in state["live_hand_state"]
#     assert "T-Y" in state["live_hand_state"]


# def test_record_hand_stack_state_writes_under_current_hand():
#     main = _load_main()
#     state: dict = {}
#     table = _make_table([_make_seat(stack=1850, committed=150)], table_id="T-Z")
#     main._process_hand_identity(table, state)

#     main._record_hand_stack_state(table, state, HERO_ID)

#     hand_info = state["live_hand_state"]["T-Z"]
#     assert hand_info["last_hero_stack"] == 1850
#     assert hand_info["last_hero_committed"] == 150


# def test_fill_hand_telemetry_outcome_from_delta_only_fills_null(tmp_path):
#     conn = connect(tmp_path / "telemetry.sqlite")
#     run_id = create_telemetry_run(conn, strategy="s", opponent="o")
#     seat = _make_seat(stack=2000)
#     table = _make_table([seat])

#     # Two decisions for the same hand.
#     record_decision_telemetry(
#         conn,
#         run_id=run_id,
#         hand_id="h1",
#         decision_index=0,
#         strategy="s",
#         table=table,
#         seat=seat,
#         action="call",
#         message="first",
#     )
#     record_decision_telemetry(
#         conn,
#         run_id=run_id,
#         hand_id="h1",
#         decision_index=1,
#         strategy="s",
#         table=table,
#         seat=seat,
#         action="call",
#         message="second",
#     )

#     # Pre-fill both rows with a Showdown-canonical value, then manually
#     # reset one back to NULL to exercise the "fill NULL only" path on it.
#     update_hand_telemetry_outcome(
#         conn,
#         run_id=run_id,
#         hand_id="h1",
#         hero_net_chips=999,  # sentinel — must NOT be overwritten
#         won_hand=True,
#     )
#     conn.execute(
#         "update decision_telemetry "
#         "set hero_net_chips = null, won_hand = null "
#         "where run_id = ? and hand_id = ? and decision_index = 0",
#         (run_id, "h1"),
#     )

#     # Now try to fill from a delta estimate: should land on decision_index=0 only
#     fill_hand_telemetry_outcome_from_delta(
#         conn,
#         run_id=run_id,
#         hand_id="h1",
#         hero_net_chips=-75,
#         won_hand=False,
#     )

#     rows = conn.execute(
#         "select decision_index, hero_net_chips, won_hand "
#         "from decision_telemetry where run_id = ? and hand_id = ? "
#         "order by decision_index",
#         (run_id, "h1"),
#     ).fetchall()

#     assert rows[0]["hero_net_chips"] == -75  # filled from delta (was NULL)
#     assert rows[0]["won_hand"] == 0
#     assert rows[1]["hero_net_chips"] == 999  # protected — Showdown wins
#     assert rows[1]["won_hand"] == 1


# def test_stack_delta_backfill_noop_when_inputs_missing():
#     main = _load_main()

#     # No-op when prev_hand_id is None
#     assert (
#         main.backfill_hand_outcome_from_stack_delta(
#             telemetry_conn=None,
#             run_id="run",
#             prev_hand_id=None,
#             prev_stack=1000,
#             cur_stack=900,
#             cur_committed=0,
#         )
#         is False
#     )

#     # No-op when prev_stack is None
#     assert (
#         main.backfill_hand_outcome_from_stack_delta(
#             telemetry_conn=None,
#             run_id="run",
#             prev_hand_id="h1",
#             prev_stack=None,
#             cur_stack=900,
#             cur_committed=0,
#         )
#         is False
#     )

#     # No-op when telemetry_conn is None
#     assert (
#         main.backfill_hand_outcome_from_stack_delta(
#             telemetry_conn=None,
#             run_id="run",
#             prev_hand_id="h1",
#             prev_stack=1000,
#             cur_stack=900,
#             cur_committed=0,
#         )
#         is False
#     )


# def test_stack_delta_backfill_detects_loss(tmp_path):
#     main = _load_main()
#     conn = connect(tmp_path / "telemetry.sqlite")
#     run_id = create_telemetry_run(conn, strategy="s", opponent="o")
#     seat = _make_seat(stack=2000)
#     table = _make_table([seat])

#     # Seed a decision for the previous hand
#     record_decision_telemetry(
#         conn,
#         run_id=run_id,
#         hand_id="h_prev",
#         decision_index=0,
#         strategy="s",
#         table=table,
#         seat=seat,
#         action="call",
#         message="hand 1 call",
#     )

#     # Hero loses 100 chips between hands
#     wrote = main.backfill_hand_outcome_from_stack_delta(
#         conn,
#         run_id=run_id,
#         prev_hand_id="h_prev",
#         prev_stack=2000,
#         cur_stack=1880,
#         cur_committed=20,
#     )

#     assert wrote is True
#     row = conn.execute(
#         "select hero_net_chips, won_hand "
#         "from decision_telemetry where run_id = ? and hand_id = ?",
#         (run_id, "h_prev"),
#     ).fetchone()
#     # net = (1880 + 20) - 2000 = -100
#     assert row["hero_net_chips"] == -100
#     assert row["won_hand"] == 0


# def test_stack_delta_backfill_detects_win(tmp_path):
#     main = _load_main()
#     conn = connect(tmp_path / "telemetry.sqlite")
#     run_id = create_telemetry_run(conn, strategy="s", opponent="o")
#     seat = _make_seat(stack=2000)
#     table = _make_table([seat])

#     record_decision_telemetry(
#         conn,
#         run_id=run_id,
#         hand_id="h_win",
#         decision_index=0,
#         strategy="s",
#         table=table,
#         seat=seat,
#         action="raise",
#         message="hand 1 raise",
#     )

#     wrote = main.backfill_hand_outcome_from_stack_delta(
#         conn,
#         run_id=run_id,
#         prev_hand_id="h_win",
#         prev_stack=2000,
#         cur_stack=2350,
#         cur_committed=10,
#     )

#     assert wrote is True
#     row = conn.execute(
#         "select hero_net_chips, won_hand "
#         "from decision_telemetry where run_id = ? and hand_id = ?",
#         (run_id, "h_win"),
#     ).fetchone()
#     # net = (2350 + 10) - 2000 = +360
#     assert row["hero_net_chips"] == 360
#     assert row["won_hand"] == 1


# def test_stack_delta_backfill_detects_fold_with_zero_investment(tmp_path):
#     main = _load_main()
#     conn = connect(tmp_path / "telemetry.sqlite")
#     run_id = create_telemetry_run(conn, strategy="s", opponent="o")
#     seat = _make_seat(stack=2000)
#     table = _make_table([seat])

#     record_decision_telemetry(
#         conn,
#         run_id=run_id,
#         hand_id="h_fold",
#         decision_index=0,
#         strategy="s",
#         table=table,
#         seat=seat,
#         action="fold",
#         message="hand 1 fold",
#     )

#     wrote = main.backfill_hand_outcome_from_stack_delta(
#         conn,
#         run_id=run_id,
#         prev_hand_id="h_fold",
#         prev_stack=2000,
#         cur_stack=2000,
#         cur_committed=0,
#     )

#     assert wrote is True
#     row = conn.execute(
#         "select hero_net_chips, won_hand "
#         "from decision_telemetry where run_id = ? and hand_id = ?",
#         (run_id, "h_fold"),
#     ).fetchone()
#     assert row["hero_net_chips"] == 0
#     assert row["won_hand"] == 0


# def test_process_single_table_records_decisions_under_derived_hand_id(
#     tmp_path, monkeypatch
# ):
#     """End-to-end: process_single_table writes decision rows with the derived
#     hand_id (tableId:boundary) — not just tableId — so two hands at the same
#     table get distinct rows in decision_telemetry.
#     """
#     main = _load_main()
#     db_path = tmp_path / "telemetry.sqlite"
#     conn = connect(db_path)
#     run_id = create_telemetry_run(conn, strategy="s", opponent="o")
#     # Force the telemetry_run_id the bot will use
#     state: dict = {"telemetry_run_id": run_id}

#     class _StubStats:
#         def close(self):
#             pass

#     from dataclasses import dataclass, field

#     @dataclass
#     class Ctx:
#         api_key: str = "k"
#         competition_id: str = "c"
#         agent_id: str = HERO_ID
#         strategy_name: str = "s"
#         choose_action: object = lambda table, seat: ("fold", None, "stub fold")
#         api_fn: object = lambda *a, **kw: {}  # empty dict = success (no error key)
#         state: dict = field(default_factory=dict)
#         telemetry_conn: object = None
#         telemetry_run_id: str = ""
#         stats_fetcher: object = None

#     ctx = Ctx()
#     ctx.state = state
#     ctx.telemetry_conn = conn
#     ctx.telemetry_run_id = run_id
#     ctx.stats_fetcher = _StubStats()

#     # First hand at table T1: Preflop only (hero folds).
#     t1_preflop = _make_table([_make_seat(stack=2000)], street="Preflop", table_id="T-1")
#     main.process_single_table(t1_preflop, ctx)

#     # Snapshot mid-Flop for the same hand (so the next Preflop triggers a boundary).
#     t1_flop = _make_table([_make_seat(stack=2000)], street="Flop", table_id="T-1")
#     main.process_single_table(t1_flop, ctx)

#     # Second hand at the same table: Preflop again — should be a new hand_id.
#     t1_preflop_2 = _make_table(
#         [_make_seat(stack=1990)], street="Preflop", table_id="T-1"
#     )
#     main.process_single_table(t1_preflop_2, ctx)

#     hand_ids = [
#         row["hand_id"]
#         for row in conn.execute(
#             "select distinct hand_id from decision_telemetry order by hand_id"
#         ).fetchall()
#     ]

#     # At least two distinct hand_ids, all starting with "T-1:" (not just "T-1")
#     assert len(hand_ids) >= 2
#     for hid in hand_ids:
#         assert hid.startswith("T-1:")
#         assert hid != "T-1"  # the old buggy form would be exactly "T-1"


# def test_stack_delta_and_showdown_compose_with_showdown_winning(tmp_path):
#     """The stack-delta fill runs first; if a Showdown snapshot is later
#     recorded for the same hand via ``update_hand_telemetry_outcome`` (which
#     overwrites), the canonical Showdown value wins over the delta estimate.
#     """
#     conn = connect(tmp_path / "telemetry.sqlite")
#     run_id = create_telemetry_run(conn, strategy="s", opponent="o")
#     seat = _make_seat(stack=2000)
#     table = _make_table([seat])

#     # Seed a decision row for hand h1.
#     record_decision_telemetry(
#         conn,
#         run_id=run_id,
#         hand_id="h1",
#         decision_index=0,
#         strategy="s",
#         table=table,
#         seat=seat,
#         action="call",
#         message="hand 1",
#     )

#     # First: stack-delta fill (estimate). This lands because hero_net_chips
#     # is currently NULL.
#     fill_hand_telemetry_outcome_from_delta(
#         conn,
#         run_id=run_id,
#         hand_id="h1",
#         hero_net_chips=-50,
#         won_hand=False,
#     )

#     row = conn.execute(
#         "select hero_net_chips, won_hand from decision_telemetry "
#         "where run_id = ? and hand_id = ?",
#         (run_id, "h1"),
#     ).fetchone()
#     assert row["hero_net_chips"] == -50
#     assert row["won_hand"] == 0

#     # Later: Showdown snapshot arrives. update_hand_telemetry_outcome
#     # overwrites — this is the canonical value the bot would record if it
#     # happened to poll the table during Showdown.
#     update_hand_telemetry_outcome(
#         conn,
#         run_id=run_id,
#         hand_id="h1",
#         hero_net_chips=42,
#         won_hand=True,
#         final_pot=84,
#     )

#     row = conn.execute(
#         "select hero_net_chips, won_hand, final_pot "
#         "from decision_telemetry where run_id = ? and hand_id = ?",
#         (run_id, "h1"),
#     ).fetchone()
#     # Showdown wins.
#     assert row["hero_net_chips"] == 42
#     assert row["won_hand"] == 1
#     assert row["final_pot"] == 84

#     # After Showdown has won, the stack-delta fill no longer has anything
#     # to land on (hero_net_chips is no longer NULL).
#     fill_hand_telemetry_outcome_from_delta(
#         conn,
#         run_id=run_id,
#         hand_id="h1",
#         hero_net_chips=-999,  # would-be bad estimate
#         won_hand=False,
#     )
#     row = conn.execute(
#         "select hero_net_chips, won_hand from decision_telemetry "
#         "where run_id = ? and hand_id = ?",
#         (run_id, "h1"),
#     ).fetchone()
#     assert row["hero_net_chips"] == 42  # Showdown value preserved
#     assert row["won_hand"] == 1


# # ---------------------------------------------------------------------------
# # /agent/{id}/replays backfill
# # ---------------------------------------------------------------------------


# def _seed_replay_decision(conn, run_id, *, hand_id, table_id, action="call"):
#     """Insert a decision row tagged with table_id so the replay fill can find it."""
#     seat = _make_seat(stack=2000)
#     table = _make_table([seat], table_id=table_id)
#     record_decision_telemetry(
#         conn,
#         run_id=run_id,
#         hand_id=hand_id,
#         decision_index=0,
#         strategy="s",
#         table=table,
#         seat=seat,
#         action=action,
#         message="seed",
#     )
#     return seat, table


# def test_fill_hand_telemetry_outcome_from_replay_only_fills_null(tmp_path):
#     conn = connect(tmp_path / "telemetry.sqlite")
#     run_id = create_telemetry_run(conn, strategy="s", opponent="o")

#     # Two decisions at the same table_id, two distinct hand_ids (the case
#     # the bot produces after the table_id column was added).
#     _seed_replay_decision(
#         conn, run_id, hand_id="t1:2026-06-18T10:00:00+00:00", table_id="t1"
#     )
#     _seed_replay_decision(
#         conn, run_id, hand_id="t1:2026-06-18T10:05:00+00:00", table_id="t1"
#     )

#     # Pre-fill ONE row to a sentinel value via raw SQL targeting one
#     # specific hand_id; leave the other NULL so the replay fill has
#     # something to land on while the protected row is preserved.
#     conn.execute(
#         "update decision_telemetry set hero_net_chips = 999, won_hand = 1 "
#         "where run_id = ? and hand_id = ?",
#         (run_id, "t1:2026-06-18T10:00:00+00:00"),
#     )

#     fill_hand_telemetry_outcome_from_replay(
#         conn,
#         run_id=run_id,
#         table_id="t1",
#         chip_delta=-50,
#         won_hand=False,
#     )

#     rows = conn.execute(
#         "select hand_id, hero_net_chips, won_hand "
#         "from decision_telemetry where run_id = ? "
#         "and table_id = ? order by hand_id",
#         (run_id, "t1"),
#     ).fetchall()

#     by_hand = {r["hand_id"]: r for r in rows}
#     # Protected row stays at 999 — replay fill never overwrites.
#     assert by_hand["t1:2026-06-18T10:00:00+00:00"]["hero_net_chips"] == 999
#     assert by_hand["t1:2026-06-18T10:00:00+00:00"]["won_hand"] == 1
#     # NULL row got filled by the replay path.
#     assert by_hand["t1:2026-06-18T10:05:00+00:00"]["hero_net_chips"] == -50
#     assert by_hand["t1:2026-06-18T10:05:00+00:00"]["won_hand"] == 0


# def test_fill_hand_telemetry_outcome_from_replay_matches_by_table_id(tmp_path):
#     """Two rows with the same table_id but different hand_ids both get filled
#     when both are NULL (single replay entry covers a whole table per the
#     arena's handId == tableCuid convention).
#     """
#     conn = connect(tmp_path / "telemetry.sqlite")
#     run_id = create_telemetry_run(conn, strategy="s", opponent="o")
#     _seed_replay_decision(
#         conn, run_id, hand_id="t1:2026-06-18T10:00:00+00:00", table_id="t1"
#     )
#     _seed_replay_decision(
#         conn, run_id, hand_id="t1:2026-06-18T10:05:00+00:00", table_id="t1"
#     )

#     fill_hand_telemetry_outcome_from_replay(
#         conn,
#         run_id=run_id,
#         table_id="t1",
#         chip_delta=200,
#         won_hand=True,
#     )

#     rows = conn.execute(
#         "select hero_net_chips, won_hand from decision_telemetry "
#         "where run_id = ? and table_id = ?",
#         (run_id, "t1"),
#     ).fetchall()
#     assert len(rows) == 2
#     for r in rows:
#         assert r["hero_net_chips"] == 200
#         assert r["won_hand"] == 1


# def test_fill_hand_telemetry_outcome_from_replay_skips_other_run(tmp_path):
#     """The fill is scoped to run_id so a different run's rows aren't touched."""
#     conn = connect(tmp_path / "telemetry.sqlite")
#     run_a = create_telemetry_run(conn, strategy="a", opponent="o")
#     run_b = create_telemetry_run(conn, strategy="b", opponent="o")
#     _seed_replay_decision(conn, run_a, hand_id="t1:A", table_id="t1")
#     _seed_replay_decision(conn, run_b, hand_id="t1:B", table_id="t1")

#     fill_hand_telemetry_outcome_from_replay(
#         conn,
#         run_id=run_a,
#         table_id="t1",
#         chip_delta=42,
#         won_hand=True,
#     )

#     assert (
#         conn.execute(
#             "select hero_net_chips from decision_telemetry "
#             "where run_id = ? and table_id = ?",
#             (run_a, "t1"),
#         ).fetchone()["hero_net_chips"]
#         == 42
#     )
#     assert (
#         conn.execute(
#             "select hero_net_chips from decision_telemetry "
#             "where run_id = ? and table_id = ?",
#             (run_b, "t1"),
#         ).fetchone()["hero_net_chips"]
#         is None
#     )


# def test_fill_hand_telemetry_outcome_from_replay_noop_when_table_id_empty(tmp_path):
#     conn = connect(tmp_path / "telemetry.sqlite")
#     run_id = create_telemetry_run(conn, strategy="s", opponent="o")
#     _seed_replay_decision(conn, run_id, hand_id="t1:x", table_id="t1")

#     # Missing/empty table_id should be a no-op (defensive — the bot should
#     # never pass None but the function tolerates it).
#     fill_hand_telemetry_outcome_from_replay(
#         conn,
#         run_id=run_id,
#         table_id="",
#         chip_delta=10,
#         won_hand=True,
#     )
#     assert (
#         conn.execute(
#             "select hero_net_chips from decision_telemetry where run_id = ?",
#             (run_id,),
#         ).fetchone()["hero_net_chips"]
#         is None
#     )


# def test_backfill_outcomes_from_replays_handles_api_failure():
#     """A failing api_fn must not propagate — the bot must keep playing."""
#     main = _load_main()

#     def boom(*args, **kwargs):
#         raise RuntimeError("simulated network failure")

#     seen = main.backfill_outcomes_from_replays(
#         telemetry_conn=None,  # skip DB write
#         run_id="r",
#         agent_id="agent-x",
#         competition_id="c",
#         api_fn=boom,
#     )
#     assert seen == 0


# def test_backfill_outcomes_from_replays_handles_error_dict():
#     main = _load_main()

#     def err_dict(*args, **kwargs):
#         return {"error": "rate_limited", "message": "slow down"}

#     seen = main.backfill_outcomes_from_replays(
#         telemetry_conn=None,
#         run_id="r",
#         agent_id="agent-x",
#         competition_id="c",
#         api_fn=err_dict,
#     )
#     assert seen == 0


# def test_backfill_outcomes_from_replays_noop_when_inputs_missing():
#     main = _load_main()

#     calls = []

#     def spy(*args, **kwargs):
#         calls.append((args, kwargs))
#         return []

#     # Missing telemetry_conn → no API call
#     assert (
#         main.backfill_outcomes_from_replays(
#             telemetry_conn=None,
#             run_id="r",
#             agent_id="a",
#             competition_id="c",
#             api_fn=spy,
#         )
#         == 0
#     )
#     # Missing run_id → no API call
#     assert (
#         main.backfill_outcomes_from_replays(
#             telemetry_conn=object(),
#             run_id=None,
#             agent_id="a",
#             competition_id="c",
#             api_fn=spy,
#         )
#         == 0
#     )
#     # Missing agent_id → no API call
#     assert (
#         main.backfill_outcomes_from_replays(
#             telemetry_conn=object(),
#             run_id="r",
#             agent_id="",
#             competition_id="c",
#             api_fn=spy,
#         )
#         == 0
#     )
#     assert calls == []


# def test_backfill_outcomes_from_replays_fills_canonical_data(tmp_path):
#     """End-to-end: a stubbed /replays response fills our decision rows."""
#     main = _load_main()
#     conn = connect(tmp_path / "telemetry.sqlite")
#     run_id = create_telemetry_run(conn, strategy="s", opponent="o")

#     # Seed three decision rows at three different tables.
#     _seed_replay_decision(conn, run_id, hand_id="t1:01", table_id="t1")
#     _seed_replay_decision(conn, run_id, hand_id="t2:01", table_id="t2")
#     _seed_replay_decision(conn, run_id, hand_id="t3:01", table_id="t3")

#     api_calls = []

#     def fake_api(method, path, **kwargs):
#         api_calls.append((method, path))
#         # /replays returns an array. Mirror the spec shape.
#         return [
#             {
#                 "handId": "t1",
#                 "tableId": "t1",
#                 "chipDelta": 250,  # hero won
#                 "winnerHandle": "hero",
#                 "settledAt": 1_700_000_000_000,
#                 "replayUrl": "https://...",
#             },
#             {
#                 "handId": "t2",
#                 "tableId": "t2",
#                 "chipDelta": -75,  # hero lost
#                 "winnerHandle": "opponent",
#                 "settledAt": 1_700_000_001_000,
#                 "replayUrl": "https://...",
#             },
#             {
#                 # chipDelta=0 (push). won_hand should be 0 (loss).
#                 "handId": "t3",
#                 "tableId": "t3",
#                 "chipDelta": 0,
#                 "winnerHandle": "opponent",
#                 "settledAt": 1_700_000_002_000,
#                 "replayUrl": "https://...",
#             },
#         ]

#     seen = main.backfill_outcomes_from_replays(
#         telemetry_conn=conn,
#         run_id=run_id,
#         agent_id="agent-x",
#         competition_id="c-1",
#         api_fn=fake_api,
#     )

#     assert seen == 3

#     # Verify exactly one API call with the expected path
#     assert len(api_calls) == 1
#     method, path = api_calls[0]
#     assert method == "GET"
#     assert path == "/agent/agent-x/replays?competitionId=c-1&limit=50"

#     # Verify the rows
#     rows = {
#         r["table_id"]: r
#         for r in conn.execute(
#             "select table_id, hero_net_chips, won_hand "
#             "from decision_telemetry where run_id = ?",
#             (run_id,),
#         ).fetchall()
#     }
#     assert rows["t1"]["hero_net_chips"] == 250
#     assert rows["t1"]["won_hand"] == 1
#     assert rows["t2"]["hero_net_chips"] == -75
#     assert rows["t2"]["won_hand"] == 0
#     assert rows["t3"]["hero_net_chips"] == 0
#     assert rows["t3"]["won_hand"] == 0


# def test_backfill_outcomes_from_replays_falls_back_to_hand_id_key(tmp_path):
#     """If an entry lacks tableId, the function uses handId as the table id
#     (per the arena spec note 'currently the table cuid')."""
#     main = _load_main()
#     conn = connect(tmp_path / "telemetry.sqlite")
#     run_id = create_telemetry_run(conn, strategy="s", opponent="o")
#     _seed_replay_decision(conn, run_id, hand_id="t1:x", table_id="t1")

#     def fake_api(*args, **kwargs):
#         return [
#             {
#                 # No tableId; only handId present.
#                 "handId": "t1",
#                 "chipDelta": 99,
#             },
#         ]

#     main.backfill_outcomes_from_replays(
#         telemetry_conn=conn,
#         run_id=run_id,
#         agent_id="a",
#         competition_id=None,
#         api_fn=fake_api,
#     )
#     row = conn.execute(
#         "select hero_net_chips, won_hand from decision_telemetry "
#         "where run_id = ? and table_id = ?",
#         (run_id, "t1"),
#     ).fetchone()
#     assert row["hero_net_chips"] == 99
#     assert row["won_hand"] == 1


# def test_backfill_outcomes_from_replays_skips_malformed_entries(tmp_path):
#     """Entries missing table_id/handId OR chipDelta are silently skipped."""
#     main = _load_main()
#     conn = connect(tmp_path / "telemetry.sqlite")
#     run_id = create_telemetry_run(conn, strategy="s", opponent="o")
#     _seed_replay_decision(conn, run_id, hand_id="t1:x", table_id="t1")

#     def fake_api(*args, **kwargs):
#         return [
#             {"chipDelta": 10},  # no tableId/handId — skip
#             {"tableId": "t1"},  # no chipDelta — skip
#             {"tableId": "t1", "chipDelta": "not-a-number"},  # bad int — skip
#             {"tableId": "t1", "chipDelta": 7},  # valid
#             "not a dict",
#             None,
#         ]

#     seen = main.backfill_outcomes_from_replays(
#         telemetry_conn=conn,
#         run_id=run_id,
#         agent_id="a",
#         competition_id="c",
#         api_fn=fake_api,
#     )
#     assert seen == 1
#     row = conn.execute(
#         "select hero_net_chips, won_hand from decision_telemetry "
#         "where run_id = ? and table_id = ?",
#         (run_id, "t1"),
#     ).fetchone()
#     assert row["hero_net_chips"] == 7


# def test_handle_idle_state_triggers_replay_backfill(tmp_path, monkeypatch):
#     """The replay backfill fires from handle_idle_state, throttled, only when
#     idle (no active tables)."""
#     main = _load_main()
#     db_path = tmp_path / "telemetry.sqlite"
#     conn = connect(db_path)
#     run_id = create_telemetry_run(conn, strategy="s", opponent="o")

#     class _StubStats:
#         def close(self):
#             pass

#     from dataclasses import dataclass, field

#     @dataclass
#     class Ctx:
#         api_key: str = "k"
#         competition_id: str = "c"
#         agent_id: str = "a"
#         strategy_name: str = "s"
#         choose_action: object = lambda *a, **kw: (None, None, None)
#         api_fn: object = lambda *a, **kw: {}
#         state: dict = field(default_factory=dict)
#         telemetry_conn: object = None
#         telemetry_run_id: str = ""
#         stats_fetcher: object = None

#     calls = []

#     def spy(*args, **kwargs):
#         calls.append((args, kwargs))
#         return []

#     ctx = Ctx()
#     ctx.state = {}
#     ctx.telemetry_conn = conn
#     ctx.telemetry_run_id = run_id
#     ctx.api_fn = spy
#     ctx.stats_fetcher = _StubStats()

#     # Bot in queue: pending has lobby set so should_attempt_join returns False
#     # and we go straight to the replay-backfill path. last_join_attempt_epoch
#     # is also pre-set so the join-throttling path doesn't fire either.
#     import time as _time

#     pending = {
#         "participant": {"chipState": "available"},
#         "runner": {},
#         "lobby": {"position": 1, "total": 3},
#     }
#     main.handle_idle_state(ctx, pending, consecutive_empty=1)

#     assert len(calls) == 1, "backfill should fire on first idle call"
#     assert calls[0][0][0] == "GET"
#     assert calls[0][0][1].startswith("/agent/a/replays")

#     # Second idle call: throttled, no second API call.
#     main.handle_idle_state(ctx, pending, consecutive_empty=2)
#     assert len(calls) == 1, "backfill should be throttled"

#     # Manually advance the throttle: rewind the stored epoch.
#     ctx.state["last_replay_backfill_epoch"] = 0.0
#     main.handle_idle_state(ctx, pending, consecutive_empty=3)
#     assert len(calls) == 2, "backfill should fire again after throttle window"


# def test_handle_idle_state_does_not_trigger_when_telemetry_disabled(tmp_path):
#     main = _load_main()

#     from dataclasses import dataclass, field

#     @dataclass
#     class Ctx:
#         api_key: str = "k"
#         competition_id: str = "c"
#         agent_id: str = "a"
#         strategy_name: str = "s"
#         choose_action: object = lambda *a, **kw: (None, None, None)
#         api_fn: object = lambda *a, **kw: {"error": "no"}
#         state: dict = field(default_factory=dict)
#         telemetry_conn: object = None
#         telemetry_run_id: str = ""
#         stats_fetcher: object = None

#     calls = []

#     def spy(*args, **kwargs):
#         calls.append((args, kwargs))
#         return []

#     ctx = Ctx()
#     ctx.state = {}
#     ctx.telemetry_conn = None  # telemetry disabled
#     ctx.telemetry_run_id = None
#     ctx.api_fn = spy

#     pending = {
#         "participant": {"chipState": "available"},
#         "runner": {},
#         "lobby": {"position": 1, "total": 3},
#     }
#     main.handle_idle_state(ctx, pending, consecutive_empty=1)

#     assert calls == [], "backfill must not run without a telemetry conn"
