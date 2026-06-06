import json
import os
import tempfile
import unittest

from main import (
    DEFAULT_STRATEGY_NAME,
    PUBLIC_MESSAGE_ALPHABET,
    STRATEGY_ENV_VAR,
    action_request_body,
    enrich_table_with_opponent_profiles,
    init_live_telemetry,
    load_configured_strategy,
    load_credentials,
    load_strategy_name,
    make_api_client,
    normalize_live_table_metadata,
    public_action_message,
    record_live_decision,
    record_live_observed_actions,
    record_live_opponents_seen,
)
from poker_bot.opponent_store import (
    connect,
    increment_hand_seen,
    record_observed_action,
)


class DummyResult:
    def __init__(self, stdout):
        self.stdout = stdout


def fake_runner(cmd, capture_output, text, timeout):
    return DummyResult('{"success": true}')


class PokerBotTests(unittest.TestCase):
    def test_load_credentials_json(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            json.dump(
                {
                    "apiKey": "arena_sk_abc",
                    "competitionId": "cmpy_1",
                    "agentId": "cmpy_agent",
                },
                tmp,
            )
            tmp_path = tmp.name

        api_key, competition_id, agent_id = load_credentials(tmp_path)
        self.assertEqual(api_key, "arena_sk_abc")
        self.assertEqual(competition_id, "cmpy_1")
        self.assertEqual(agent_id, "cmpy_agent")
        os.unlink(tmp_path)

    def test_load_credentials_env_style(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write("apiKey=arena_sk_xyz\n")
            tmp.write("competitionId=cmpy_2\n")
            tmp.write("agentId=cmpy_agent2\n")
            tmp_path = tmp.name

        api_key, competition_id, agent_id = load_credentials(tmp_path)
        self.assertEqual(api_key, "arena_sk_xyz")
        self.assertEqual(competition_id, "cmpy_2")
        self.assertEqual(agent_id, "cmpy_agent2")
        os.unlink(tmp_path)

    def test_load_strategy_name_defaults_when_no_config_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = os.path.join(tmpdir, "missing-strategy")

            strategy_name = load_strategy_name(missing_path, environ={})

        self.assertEqual(strategy_name, DEFAULT_STRATEGY_NAME)

    def test_load_strategy_name_prefers_environment(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write("simple\n")
            tmp_path = tmp.name

        try:
            strategy_name = load_strategy_name(
                tmp_path,
                environ={STRATEGY_ENV_VAR: "adaptive"},
            )
        finally:
            os.unlink(tmp_path)

        self.assertEqual(strategy_name, "adaptive")

    def test_load_strategy_name_reads_plain_config_file(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write("# active strategy\n")
            tmp.write("simple\n")
            tmp_path = tmp.name

        try:
            strategy_name = load_strategy_name(tmp_path, environ={})
        finally:
            os.unlink(tmp_path)

        self.assertEqual(strategy_name, "simple")

    def test_load_strategy_name_reads_json_config_file(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            json.dump({"strategy": "counter_adaptive"}, tmp)
            tmp_path = tmp.name

        try:
            strategy_name = load_strategy_name(tmp_path, environ={})
        finally:
            os.unlink(tmp_path)

        self.assertEqual(strategy_name, "counter_adaptive")

    def test_load_configured_strategy_returns_callable(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write("simple\n")
            tmp_path = tmp.name

        try:
            strategy_name, strategy = load_configured_strategy(tmp_path, environ={})
        finally:
            os.unlink(tmp_path)

        self.assertEqual(strategy_name, "simple")
        self.assertTrue(callable(strategy))

    def test_api_runner_injection(self):
        api_fn = make_api_client("arena_sk_test", runner=fake_runner)
        result = api_fn("GET", "/test")
        self.assertEqual(result, {"success": True})

    def test_action_request_body_uses_public_random_message(self):
        body = action_request_body("cmp-test", "table-1", "raise", amount=150)

        self.assertEqual(body["competitionId"], "cmp-test")
        self.assertEqual(body["tableId"], "table-1")
        self.assertEqual(body["action"], "raise")
        self.assertEqual(body["amount"], 150)
        self.assertEqual(len(body["message"]), 16)
        self.assertTrue(set(body["message"]).issubset(set(PUBLIC_MESSAGE_ALPHABET)))

    def test_public_action_message_has_requested_length(self):
        message = public_action_message()

        self.assertEqual(len(message), 16)
        self.assertTrue(set(message).issubset(set(PUBLIC_MESSAGE_ALPHABET)))

    def test_live_telemetry_records_decision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "telemetry.sqlite")
            old_db = os.environ.get("POKER_BOT_OPPONENT_DB")
            old_enabled = os.environ.get("POKER_BOT_TELEMETRY")
            os.environ["POKER_BOT_OPPONENT_DB"] = db_path
            os.environ["POKER_BOT_TELEMETRY"] = "1"
            try:
                state = {}
                conn, run_id = init_live_telemetry(state, "cmp-test")
                table = {
                    "tableId": "table-1",
                    "street": "Preflop",
                    "boardCards": [],
                    "potChips": 75,
                    "currentBet": 50,
                    "buttonSeatNumber": 1,
                    "allowedActions": {
                        "availableActions": ["fold", "call", "raise"],
                        "callAmount": 50,
                        "callChips": 50,
                        "minBet": 50,
                        "minRaiseTo": 150,
                    },
                    "seats": [
                        {
                            "agentId": "hero",
                            "seatNumber": 1,
                            "holeCards": ["AS", "KS"],
                            "stackChips": 1800,
                            "currentBetChips": 25,
                        }
                    ],
                }
                my_seat = table["seats"][0]

                record_live_decision(
                    conn,
                    run_id,
                    state,
                    table,
                    my_seat,
                    "raise",
                    150,
                    "test live raise",
                )

                row = connect(db_path).execute(
                    "select * from decision_telemetry where run_id = ?",
                    (run_id,),
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row["hand_id"], "table-1")
                self.assertEqual(row["chosen_action"], "raise")
                self.assertEqual(state["telemetry_decision_indexes"]["table-1"], 1)
            finally:
                if old_db is None:
                    os.environ.pop("POKER_BOT_OPPONENT_DB", None)
                else:
                    os.environ["POKER_BOT_OPPONENT_DB"] = old_db
                if old_enabled is None:
                    os.environ.pop("POKER_BOT_TELEMETRY", None)
                else:
                    os.environ["POKER_BOT_TELEMETRY"] = old_enabled

    def test_live_telemetry_resets_run_when_strategy_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "telemetry.sqlite")
            old_db = os.environ.get("POKER_BOT_OPPONENT_DB")
            old_enabled = os.environ.get("POKER_BOT_TELEMETRY")
            os.environ["POKER_BOT_OPPONENT_DB"] = db_path
            os.environ["POKER_BOT_TELEMETRY"] = "1"
            try:
                state = {}
                conn, first_run = init_live_telemetry(
                    state,
                    "cmp-test",
                    strategy_name="simple",
                )
                state["telemetry_decision_indexes"]["table-1"] = 3

                _conn, second_run = init_live_telemetry(
                    state,
                    "cmp-test",
                    strategy_name="adaptive",
                )

                self.assertNotEqual(first_run, second_run)
                self.assertEqual(state["telemetry_strategy"], "adaptive")
                self.assertEqual(state["telemetry_decision_indexes"], {})
                rows = conn.execute(
                    "select strategy from telemetry_runs order by started_at, run_id"
                ).fetchall()
                self.assertEqual(
                    {row["strategy"] for row in rows},
                    {"simple", "adaptive"},
                )
            finally:
                if old_db is None:
                    os.environ.pop("POKER_BOT_OPPONENT_DB", None)
                else:
                    os.environ["POKER_BOT_OPPONENT_DB"] = old_db
                if old_enabled is None:
                    os.environ.pop("POKER_BOT_TELEMETRY", None)
                else:
                    os.environ["POKER_BOT_TELEMETRY"] = old_enabled

    def test_live_telemetry_records_opponents_seen_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "telemetry.sqlite")
            old_db = os.environ.get("POKER_BOT_OPPONENT_DB")
            os.environ["POKER_BOT_OPPONENT_DB"] = db_path
            try:
                state = {}
                conn, _run_id = init_live_telemetry(state, "cmp-test")
                table = {
                    "tableId": "table-2",
                    "seats": [
                        {"agentId": "hero", "seatNumber": 1},
                        {"agentId": "villain-1", "seatNumber": 2, "name": "V1"},
                        {"agentId": "villain-2", "seatNumber": 3, "handle": "V2"},
                    ],
                }

                record_live_opponents_seen(conn, state, table, "hero")
                record_live_opponents_seen(conn, state, table, "hero")

                db = connect(db_path)
                opponents = db.execute(
                    "select count(*) as count from opponents"
                ).fetchone()["count"]
                hands_seen = db.execute(
                    """
                    select sum(hands_seen) as total
                    from opponent_stats
                    """
                ).fetchone()["total"]
                self.assertEqual(opponents, 2)
                self.assertEqual(hands_seen, 2)
            finally:
                if old_db is None:
                    os.environ.pop("POKER_BOT_OPPONENT_DB", None)
                else:
                    os.environ["POKER_BOT_OPPONENT_DB"] = old_db

    def test_enrich_table_with_opponent_profiles_loads_db_profiles(self):
        conn = connect(":memory:")
        increment_hand_seen(conn, "arena", "villain-1", handle="Caller")
        record_observed_action(
            conn,
            platform="arena",
            agent_id="villain-1",
            action="call",
            street="Preflop",
            voluntary=True,
        )
        table = {
            "seats": [
                {"agentId": "hero", "seatNumber": 1},
                {"agentId": "villain-1", "seatNumber": 2},
            ]
        }

        enriched = enrich_table_with_opponent_profiles(conn, table, "hero")

        self.assertIn("opponentProfiles", enriched)
        self.assertIn("villain-1", enriched["opponentProfiles"])
        profile = enriched["opponentProfiles"]["villain-1"]
        self.assertEqual(profile.hands_seen, 1)
        self.assertEqual(profile.calls, 1)

    def test_live_observed_actions_records_history_once(self):
        conn = connect(":memory:")
        state = {}
        table = {
            "tableId": "hand-1",
            "street": "Preflop",
            "potChips": 75,
            "actionHistory": [
                {
                    "id": "event-1",
                    "agentId": "villain-1",
                    "action": "raise",
                    "amount": "150",
                    "street": "Preflop",
                    "facingBet": True,
                    "message": "Strong hand, raising",
                }
            ],
            "seats": [
                {"agentId": "hero", "seatNumber": 1, "stackChips": 1800},
                {
                    "agentId": "villain-1",
                    "seatNumber": 2,
                    "name": "Raiser",
                    "stackChips": 2200,
                },
            ],
        }

        first_count = record_live_observed_actions(conn, state, table, "hero")
        second_count = record_live_observed_actions(conn, state, table, "hero")

        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 0)
        row = conn.execute(
            """
            select s.raises, s.pfr, s.vpip, s.opportunities_to_fold_to_bet,
                   count(a.id) as action_rows,
                   max(a.message) as message
            from opponents o
            join opponent_stats s on s.opponent_id = o.id
            join opponent_actions a on a.opponent_id = o.id
            where o.agent_id = 'villain-1'
            group by s.opponent_id
            """
        ).fetchone()
        self.assertEqual(row["raises"], 1)
        self.assertEqual(row["pfr"], 1)
        self.assertEqual(row["vpip"], 1)
        self.assertEqual(row["opportunities_to_fold_to_bet"], 1)
        self.assertEqual(row["action_rows"], 1)
        self.assertEqual(row["message"], "Strong hand, raising")

    def test_live_observed_actions_records_action_taken_summary(self):
        conn = connect(":memory:")
        state = {}
        table = {
            "tableId": "hand-action-taken",
            "street": "Preflop",
            "potChips": 10,
            "events": [
                {
                    "id": "cmq4fdpwj1w85s57w0shoqyme",
                    "sequence": 13,
                    "type": "ActionTaken",
                    "street": "Preflop",
                    "occurredAt": 1780875646861,
                    "summary": {
                        "action": "call",
                        "amount": 2,
                        "toAmount": None,
                        "reasoning": "priced in",
                        "cards": None,
                        "boardCards": None,
                        "seatNumber": 6,
                        "agentName": "zctp 397",
                    },
                }
            ],
            "seats": [
                {"agentId": "hero", "seatNumber": 1, "stackChips": 1800},
                {"agentId": "villain-6", "seatNumber": 6, "stackChips": 2200},
            ],
        }

        count = record_live_observed_actions(conn, state, table, "hero")

        self.assertEqual(count, 1)
        row = conn.execute(
            """
            select o.agent_id, o.handle, a.street, a.action, a.amount,
                   a.message, s.calls, s.vpip
            from opponents o
            join opponent_stats s on s.opponent_id = o.id
            join opponent_actions a on a.opponent_id = o.id
            where o.agent_id = 'villain-6'
            """
        ).fetchone()
        self.assertEqual(row["agent_id"], "villain-6")
        self.assertEqual(row["handle"], "zctp 397")
        self.assertEqual(row["street"], "Preflop")
        self.assertEqual(row["action"], "call")
        self.assertEqual(row["amount"], 2)
        self.assertEqual(row["message"], "priced in")
        self.assertEqual(row["calls"], 1)
        self.assertEqual(row["vpip"], 1)
        self.assertIn(
            "hand-action-taken:cmq4fdpwj1w85s57w0shoqyme",
            state["observed_action_event_keys"],
        )

    def test_live_observed_actions_prefers_raise_to_amount_from_summary(self):
        conn = connect(":memory:")
        state = {}
        table = {
            "tableId": "hand-raise-taken",
            "street": "Turn",
            "potChips": 100,
            "events": [
                {
                    "sequence": 2,
                    "type": "ActionTaken",
                    "street": "Turn",
                    "occurredAt": 1780875647000,
                    "summary": {
                        "action": "raise",
                        "amount": 30,
                        "toAmount": 90,
                        "reasoning": None,
                        "seatNumber": 3,
                        "agentName": "raiser",
                    },
                }
            ],
            "seats": [
                {"agentId": "hero", "seatNumber": 1, "stackChips": 1800},
                {"agentId": "villain-3", "seatNumber": 3, "stackChips": 2200},
            ],
        }

        count = record_live_observed_actions(conn, state, table, "hero")

        self.assertEqual(count, 1)
        row = conn.execute(
            """
            select a.action, a.amount, s.raises
            from opponents o
            join opponent_stats s on s.opponent_id = o.id
            join opponent_actions a on a.opponent_id = o.id
            where o.agent_id = 'villain-3'
            """
        ).fetchone()
        self.assertEqual(row["action"], "raise")
        self.assertEqual(row["amount"], 90)
        self.assertEqual(row["raises"], 1)
        self.assertIn(
            "hand-raise-taken:seq:2:villain-3:Turn:raise:90",
            state["observed_action_event_keys"],
        )

    def test_normalize_live_table_metadata_uses_button_alias(self):
        table = {"dealerSeatNumber": "4"}

        normalize_live_table_metadata(table)

        self.assertEqual(table["buttonSeatNumber"], "4")


if __name__ == "__main__":
    unittest.main()
