import json

from poker_bot.arena_stats import (
    ArenaStatsFetcher,
    fetch_and_record_agent_stats,
    schedule_table_opponent_stats,
    table_opponent_agents,
)
from poker_bot.opponent_store import connect


def test_table_opponent_agents_skips_hero_and_empty_seats():
    table = {
        "seats": [
            {"agentId": "hero", "agentHandle": "us"},
            {"agentId": "villain-1", "agentHandle": "fast"},
            {"seatNumber": 3},
            {"agentId": "villain-2", "agentName": "Patient"},
        ]
    }

    assert table_opponent_agents(table, "hero") == [
        ("villain-1", "fast"),
        ("villain-2", "Patient"),
    ]


def test_schedule_table_opponent_stats_queues_once_per_table_opponent():
    calls = []

    class FakeFetcher:
        def enqueue(self, table_id, agent_id, handle=None):
            key = (table_id, agent_id, handle)
            if key in calls:
                return False
            calls.append(key)
            return True

    table = {
        "tableId": "table-1",
        "seats": [
            {"agentId": "hero"},
            {"agentId": "villain-1", "agentHandle": "one"},
            {"agentId": "villain-2", "agentName": "Two"},
        ],
    }

    assert schedule_table_opponent_stats(FakeFetcher(), table, "hero") == 2
    assert calls == [
        ("table-1", "villain-1", "one"),
        ("table-1", "villain-2", "Two"),
    ]


def test_arena_stats_fetcher_dedupes_same_table_but_refetches_new_table():
    fetcher = ArenaStatsFetcher(lambda *_args: {}, "cmp-test", start=False)

    assert fetcher.enqueue("table-1", "villain-1", handle="one")
    assert not fetcher.enqueue("table-1", "villain-1", handle="one")
    assert fetcher.enqueue("table-2", "villain-1", handle="one")


def test_fetch_and_record_agent_stats_uses_agent_stats_endpoint(tmp_path):
    calls = []

    def api_fn(method, path, data=None):
        calls.append((method, path, data))
        return {"hands": 123, "vpip": 37}

    db_path = tmp_path / "gameplay.sqlite"
    ok = fetch_and_record_agent_stats(
        api_fn,
        "cmp test",
        "agent/with space",
        handle="Handle",
        db_path=db_path,
    )

    assert ok
    assert calls == [
        (
            "GET",
            "/agent/agent%2Fwith%20space/stats?competitionId=cmp%20test",
            None,
        )
    ]
    conn = connect(db_path)
    row = conn.execute(
        """
        select e.stats_json
        from opponents o
        join opponent_external_stats e on e.opponent_id = o.id
        where o.agent_id = 'agent/with space'
        """
    ).fetchone()
    assert json.loads(row["stats_json"]) == {"hands": 123, "vpip": 37}


def test_fetch_and_record_agent_stats_ignores_error_response(tmp_path):
    def api_fn(method, path, data=None):
        return {"error": "not_found"}

    db_path = tmp_path / "gameplay.sqlite"

    assert not fetch_and_record_agent_stats(
        api_fn,
        "cmp-test",
        "villain",
        db_path=db_path,
    )
