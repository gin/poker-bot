"""Tests for guard telemetry: event buffer, shadow mode, and DB attribution."""

from eval.selfplay import run_selfplay
from poker_bot.guards.context import GuardContext
from poker_bot.guards.registry import GuardRail
from poker_bot.guards.telemetry import (
    ERROR_GUARD_ID,
    clear_events,
    drain_events,
    record_guard_error,
)
from poker_bot.opponent_store import (
    connect,
    create_telemetry_run,
    merge_worker_db,
    record_decision_telemetry,
    record_guard_event,
    summarize_guard_overrides,
    update_hand_telemetry_outcome,
)

HERO = "hero"


def make_ctx(street="River", call=100, pot=400):
    seat = {
        "seatNumber": 1,
        "agentId": HERO,
        "holeCards": ["2S", "7D"],
        "stackChips": 2000,
        "currentBetChips": 0,
    }
    table = {
        "street": street,
        "boardCards": ["3H", "8C", "JD", "QS", "KH"] if street == "River" else [],
        "potChips": pot,
        "currentBet": call,
        "buttonSeatNumber": 1,
        "allowedActions": {
            "availableActions": ["fold", "call", "raise"],
            "callAmount": call,
            "minRaiseTo": call * 2,
        },
        "seats": [
            seat,
            {
                "seatNumber": 2,
                "agentId": "opp",
                "holeCards": [],
                "stackChips": 2000,
                "currentBetChips": call,
            },
        ],
    }
    return GuardContext.build(table, seat)


def make_rail(guards):
    """Build a GuardRail from (guard_id, phase, precedence, shadow, func) specs."""
    rail = GuardRail()
    for guard_id, phase, precedence, shadow, func in guards:
        rail.register(guard_id, phase, precedence, ["all"], shadow=shadow)(func)
    return rail


def fire_call(ctx, proposed=None):
    return ("call", ctx.call_price, "test fire")


def no_fire(ctx, proposed=None):
    return None


def setup_function(_func):
    clear_events()


# ── registry-level ───────────────────────────────────────────────────────────


def test_active_post_guard_overrides_and_records_applied_event():
    rail = make_rail([("g1", "post", 10, False, fire_call)])
    ctx = make_ctx()
    decision, guard_id = rail.run_post(ctx, ("fold", None, "core fold"))
    assert guard_id == "g1"
    assert decision[0] == "call"
    events = drain_events()
    assert len(events) == 1
    assert events[0].guard_id == "g1"
    assert events[0].applied is True
    assert events[0].shadow is False
    assert events[0].original_action == "fold"
    assert events[0].final_action == "call"


def test_shadow_guard_records_event_but_does_not_override():
    rail = make_rail([("g1", "post", 10, True, fire_call)])
    ctx = make_ctx()
    decision, guard_id = rail.run_post(ctx, ("fold", None, "core fold"))
    assert guard_id == "approved"
    assert decision == ("fold", None, "core fold")
    events = drain_events()
    assert len(events) == 1
    assert events[0].applied is False
    assert events[0].shadow is True


def test_first_active_guard_wins_second_logged_unapplied():
    def fire_fold(ctx, proposed=None):
        return ("fold", None, "second guard")

    rail = make_rail(
        [
            ("first", "post", 10, False, fire_call),
            ("second", "post", 20, False, fire_fold),
        ]
    )
    ctx = make_ctx()
    decision, guard_id = rail.run_post(ctx, ("raise", 200, "core raise"))
    assert guard_id == "first"
    assert decision[0] == "call"
    events = {e.guard_id: e for e in drain_events()}
    assert events["first"].applied is True
    assert events["second"].applied is False
    assert events["second"].shadow is False


def test_pre_guard_events_use_pending_original_action():
    rail = make_rail([("g1", "pre", 10, False, fire_call)])
    ctx = make_ctx()
    result = rail.run_pre(ctx)
    assert result is not None
    (action, _amount, _msg), guard_id = result
    assert action == "call"
    assert guard_id == "g1"
    events = drain_events()
    assert events[0].original_action == "__pending__"
    assert events[0].phase == "pre"


def test_env_shadow_override_forces_shadow(monkeypatch):
    monkeypatch.setenv("POKER_GUARD_SHADOW", "g1")
    rail = make_rail([("g1", "post", 10, False, fire_call)])
    ctx = make_ctx()
    decision, guard_id = rail.run_post(ctx, ("fold", None, "core fold"))
    assert guard_id == "approved"
    assert decision[0] == "fold"
    events = drain_events()
    assert events[0].shadow is True
    assert events[0].applied is False


def test_env_disable_override_skips_guard(monkeypatch):
    monkeypatch.setenv("POKER_GUARD_DISABLE", "g1")
    rail = make_rail([("g1", "post", 10, False, fire_call)])
    ctx = make_ctx()
    decision, guard_id = rail.run_post(ctx, ("fold", None, "core fold"))
    assert guard_id == "approved"
    assert drain_events() == []


def test_raising_post_guard_records_error_event_and_cascade_continues():
    def broken(ctx, proposed=None):
        raise RuntimeError("boom -- must never leak into telemetry")

    rail = make_rail(
        [
            ("broken", "post", 10, False, broken),
            ("g1", "post", 20, False, fire_call),
        ]
    )
    ctx = make_ctx()
    decision, guard_id = rail.run_post(ctx, ("fold", None, "core fold"))
    # A later healthy guard still runs and applies despite the earlier raise.
    assert guard_id == "g1"
    assert decision[0] == "call"

    events = {e.guard_id: e for e in drain_events()}
    assert events["g1"].applied is True
    error_event = events[ERROR_GUARD_ID]
    assert error_event.phase == "post"
    assert error_event.applied is False
    assert error_event.shadow is True
    assert "RuntimeError" in error_event.reason
    # Generic sanitized reason: the raw exception message never leaks.
    assert "boom" not in error_event.reason


def test_raising_pre_guard_records_error_event_and_cascade_continues():
    def broken(ctx):
        raise ValueError("hole cards AsKs -- must never leak into telemetry")

    rail = make_rail(
        [
            ("broken", "pre", 10, False, broken),
            ("g1", "pre", 20, False, fire_call),
        ]
    )
    ctx = make_ctx()
    result = rail.run_pre(ctx)
    assert result is not None
    (action, _amount, _msg), guard_id = result
    # A later healthy guard still runs and applies despite the earlier raise.
    assert guard_id == "g1"
    assert action == "call"

    events = {e.guard_id: e for e in drain_events()}
    assert events["g1"].applied is True
    error_event = events[ERROR_GUARD_ID]
    assert error_event.phase == "pre"
    assert error_event.applied is False
    assert error_event.shadow is True
    assert "ValueError" in error_event.reason
    # Generic sanitized reason: the raw exception message never leaks.
    assert "AsKs" not in error_event.reason
    assert "leak" not in error_event.reason


def test_non_firing_guards_record_no_events():
    rail = make_rail([("g1", "post", 10, False, no_fire)])
    ctx = make_ctx()
    decision, guard_id = rail.run_post(ctx, ("fold", None, "core fold"))
    assert guard_id == "approved"
    assert drain_events() == []


# ── pipeline error events ───────────────────────────────────────────────────


def test_record_guard_error_reports_class_and_phase_without_leaking_message():
    record_guard_error("pre", ValueError("hole cards AsKs leaked here"))
    events = drain_events()
    assert len(events) == 1
    event = events[0]
    assert event.guard_id == ERROR_GUARD_ID
    assert event.phase == "pre"
    assert event.applied is False
    assert event.shadow is True
    assert "ValueError" in event.reason
    assert "pre" in event.reason
    # The exception's own message must never be persisted -- it can carry
    # arbitrary state (hole cards, table contents, etc.).
    assert "AsKs" not in event.reason
    assert "leaked" not in event.reason


def test_record_guard_error_never_calls_str_on_the_exception():
    class Radioactive(Exception):
        """An exception whose __str__ would leak state if ever invoked."""

        def __str__(self):
            raise AssertionError("record_guard_error must never call str(exc)")

    # Must not raise (fail-open) AND must still record the event -- if the
    # implementation ever called str(exc)/f"{exc}", __str__ would raise,
    # get swallowed by record_guard_error's own guard, and silently drop
    # the event, which the length assertion below would catch.
    record_guard_error("post", Radioactive())
    events = drain_events()
    assert len(events) == 1
    assert "Radioactive" in events[0].reason


# ── DB-level ─────────────────────────────────────────────────────────────────


def _one_event(**overrides):
    from poker_bot.guards.telemetry import GuardEvent

    defaults = dict(
        guard_id="g1",
        phase="post",
        precedence=10,
        shadow=False,
        applied=True,
        original_action="fold",
        original_amount=None,
        final_action="call",
        final_amount=100,
        reason="test",
        street="River",
        pot=400,
        call_price=100,
        available_actions="fold,call,raise",
    )
    defaults.update(overrides)
    return GuardEvent(**defaults)


def _seat_and_table():
    seat = {
        "agentId": "player-agent",
        "seatNumber": 1,
        "holeCards": ["AS", "KS"],
        "stackChips": 1800,
        "currentBetChips": 50,
    }
    table = {
        "street": "Preflop",
        "boardCards": [],
        "potChips": 75,
        "currentBet": 50,
        "buttonSeatNumber": 1,
        "allowedActions": {
            "availableActions": ["fold", "call", "raise"],
            "callAmount": 50,
            "minRaiseTo": 150,
        },
        "seats": [
            seat,
            {
                "agentId": "bot-agent-1",
                "seatNumber": 2,
                "stackChips": 2200,
                "currentBetChips": 25,
                "holeCards": [],
            },
        ],
    }
    return seat, table


def test_record_guard_event_joins_decision_telemetry_outcome(tmp_path):
    conn = connect(tmp_path / "telemetry.sqlite")
    run_id = create_telemetry_run(
        conn, strategy="multi_core", opponent="simple", players=2, seed=1
    )
    seat, table = _seat_and_table()
    record_decision_telemetry(
        conn,
        run_id=run_id,
        hand_id="h1",
        decision_index=0,
        strategy="multi_core",
        table=table,
        seat=seat,
        action="call",
        amount=50,
        message="guarded call [guard:g1]",
        facing_bet=True,
        voluntary=True,
    )
    record_guard_event(
        conn, run_id=run_id, hand_id="h1", decision_index=0, event=_one_event()
    )
    record_guard_event(
        conn,
        run_id=run_id,
        hand_id="h1",
        decision_index=0,
        event=_one_event(guard_id="g2", shadow=True, applied=False),
    )
    update_hand_telemetry_outcome(
        conn,
        run_id=run_id,
        hand_id="h1",
        hero_net_chips=125,
        won_hand=True,
        final_pot=300,
    )
    conn.commit()

    rows = {row["guard_id"]: row for row in summarize_guard_overrides(conn, run_id)}
    assert rows["g1"]["fires"] == 1
    assert rows["g1"]["applied"] == 1
    assert rows["g1"]["shadow"] == 0
    assert rows["g1"]["avg_net_chips"] == 125
    assert rows["g1"]["transition"] == "fold -> call"
    assert rows["g2"]["shadow"] == 1
    assert rows["g2"]["applied"] == 0


def test_merge_worker_db_copies_guard_overrides(tmp_path):
    main_path = tmp_path / "main.sqlite"
    worker_path = tmp_path / "worker.sqlite"
    connect(main_path).close()
    worker = connect(worker_path)
    run_id = create_telemetry_run(
        worker, strategy="multi_core", opponent="simple", players=2, seed=1
    )
    record_guard_event(
        worker, run_id=run_id, hand_id="h1", decision_index=0, event=_one_event()
    )
    worker.commit()
    worker.close()

    merge_worker_db(main_path, worker_path)
    main = connect(main_path)
    rows = main.execute("select * from guard_overrides").fetchall()
    assert len(rows) == 1
    assert rows[0]["guard_id"] == "g1"
    assert rows[0]["run_id"] == run_id
    main.close()


# ── integration ──────────────────────────────────────────────────────────────


def test_selfplay_persists_guard_events_with_correct_attribution(tmp_path):
    db_path = tmp_path / "telemetry.sqlite"
    run_selfplay(
        "multi_core",
        hands=300,
        seed=7,
        opponent_name="multi_core",
        players=2,
        opponent_db=db_path,
        telemetry=True,
    )
    conn = connect(db_path)
    fires = conn.execute("select count(*) from guard_overrides").fetchone()[0]
    assert fires > 0, "expected multi_core guards to fire at least once in 300 hands"

    # Attribution check: an applied pre-guard short-circuits the core, so the
    # hero's recorded chosen_action must equal the guard's final_action. In a
    # mirror match (multi_core vs multi_core) misattributed opponent events
    # would break this.
    mismatches = conn.execute(
        """
        select count(*)
        from guard_overrides g
        join decision_telemetry d
          on d.run_id = g.run_id
         and d.hand_id = g.hand_id
         and d.decision_index = g.decision_index
        where g.applied = 1 and g.phase = 'pre'
          and d.chosen_action != g.final_action
        """
    ).fetchone()[0]
    assert mismatches == 0

    # Every guard row must join to a hero decision row.
    orphans = conn.execute(
        """
        select count(*)
        from guard_overrides g
        left join decision_telemetry d
          on d.run_id = g.run_id
         and d.hand_id = g.hand_id
         and d.decision_index = g.decision_index
        where d.id is null
        """
    ).fetchone()[0]
    assert orphans == 0
    conn.close()
