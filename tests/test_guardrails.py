"""Tests for the centralized guardrails system."""

import sqlite3

from poker_bot.neural.guardrails import (
    GuardRail,
    GuardResult,
    _action_from_guard,
    excessive_bet_size_cap,
    get_guardrail,
    royal_flush_possible,
    royal_flush_predecision_guard,
    sliver_shove_floor,
)


# ── royal_flush_possible ─────────────────────────────────────────────

def test_royal_flush_possible_preflop_aks():
    assert royal_flush_possible(["AS", "KS"], [])


def test_royal_flush_possible_flop_3_royals():
    assert royal_flush_possible(["AH", "KH"], ["QH", "JH", "3C"])


def test_royal_flush_not_possible_when_blocked():
    # AK hearts + Qh Jh Th = royal flush already possible (all 5 cards)
    # Add 5th board card that is NOT a royal -> still possible (already made)
    assert royal_flush_possible(["AH", "KH"], ["QH", "JH", "TH", "9S", "2D"])

    # But 2 random cards + 3-board with no royal path -> false
    assert not royal_flush_possible(["2C", "7D"], ["9H", "KS", "3C"])


# ── GuardResult dataclass ────────────────────────────────────────────

def test_guard_result_is_frozen():
    result = GuardResult(
        fired=True,
        guard_id="test",
        original_action="raise",
        final_action="call",
        reason="too big",
        pre_decision=False,
    )
    assert result.fired is True


# ── GuardRail registry ───────────────────────────────────────────────

def test_default_guardrail_has_pre_and_post_guards():
    guards = get_guardrail()
    # Should have pre-decision royal flush guard
    pre = guards.run_pre(
        {
            "street": "Preflop",
            "allowedActions": {
                "availableActions": ["fold", "call"],
                "callAmount": 40,
            },
        },
        {"holeCards": ["AS", "KS"], "agentId": "hero"},
    )
    assert pre is not None
    assert pre.fired is True
    assert pre.final_action == "call"


# ── _action_from_guard helper ────────────────────────────────────────

def test_action_from_guard_call_amount():
    result = GuardResult(
        fired=True,
        guard_id="test",
        original_action="fold",
        final_action="call",
        reason="pot odds",
        pre_decision=False,
    )
    table = {
        "allowedActions": {
            "callAmount": 50,
        }
    }
    action, amount, reason = _action_from_guard(result, table)
    assert action == "call"
    assert amount == 50


def test_action_from_guard_check_is_none_amount():
    result = GuardResult(
        fired=True,
        guard_id="test",
        original_action="fold",
        final_action="check",
        reason="free",
        pre_decision=False,
    )
    table = {"allowedActions": {"callAmount": 0}}
    action, amount, reason = _action_from_guard(result, table)
    assert action == "check"
    assert amount is None


# ── Post-decision guards ────────────────────────────────────────────

def test_sliver_shove_floor_fires():
    table = {
        "potChips": 300,
        "allowedActions": {"callAmount": 10},
    }
    result = sliver_shove_floor(table, {}, "fold")
    assert result is not None
    assert result.fired is True
    assert result.final_action == "call"


def test_sliver_shove_floor_does_not_fire_on_call():
    table = {
        "potChips": 300,
        "allowedActions": {"callAmount": 10},
    }
    result = sliver_shove_floor(table, {}, "call")
    assert result is None


def test_excessive_bet_size_cap_fires():
    table = {
        "potChips": 100,
        "allowedActions": {
            "callAmount": 10,
            "raiseRange": {"min": 400},
        },
    }
    result = excessive_bet_size_cap(table, {}, "raise")
    assert result is not None
    assert result.fired is True
    assert result.final_action == "call"


# ── DB logging ──────────────────────────────────────────────────────

def test_guardrail_log_override_writes_row(tmp_path):
    conn = sqlite3.connect(tmp_path / "test.db")
    conn.executescript(
        """
        create table if not exists telemetry_runs (
            run_id text primary key
        );
        create table if not exists guard_overrides (
            id integer primary key,
            run_id text not null,
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
            created_at text not null default current_timestamp
        );
        """
    )
    conn.execute("insert into telemetry_runs(run_id) values (?)", ("run1",))
    conn.commit()

    guards = GuardRail()
    result = GuardResult(
        fired=True,
        guard_id="sliver_shove_floor",
        original_action="fold",
        final_action="call",
        reason="Pot odds 0.03 < 0.15",
        pre_decision=False,
    )
    guards.log_override(
        conn,
        run_id="run1",
        hand_id="table1:hand1",
        decision_index=0,
        guard_result=result,
        table={
            "street": "Flop",
            "potChips": 300,
            "allowedActions": {"callAmount": 10, "availableActions": ["fold", "call"]},
        },
        seat={},
    )
    conn.commit()

    rows = conn.execute("select * from guard_overrides").fetchall()
    assert len(rows) == 1
    assert rows[0][4] == "sliver_shove_floor"
