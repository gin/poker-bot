"""Background enrichment for Arena-provided opponent stats."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from poker_bot.opponent_store import connect, record_external_agent_stats


@dataclass(frozen=True)
class AgentStatsTask:
    table_id: str
    agent_id: str
    handle: str | None = None


def seat_handle(seat):
    return (
        seat.get("agentHandle")
        or seat.get("agentName")
        or seat.get("handle")
        or seat.get("name")
    )


def table_opponent_agents(table, hero_agent_id):
    opponents = []
    for seat in table.get("seats", []):
        agent_id = seat.get("agentId")
        if not agent_id or agent_id == hero_agent_id:
            continue
        opponents.append((str(agent_id), seat_handle(seat)))
    return opponents


def fetch_and_record_agent_stats(
    api_fn,
    competition_id,
    agent_id,
    *,
    handle=None,
    db_path: str | Path | None = None,
    platform="arena",
):
    path = (
        f"/agent/{quote(str(agent_id), safe='')}/stats"
        f"?competitionId={quote(str(competition_id), safe='')}"
    )
    response = api_fn("GET", path)
    if not isinstance(response, dict) or "error" in response:
        return False

    conn = connect(db_path)
    try:
        record_external_agent_stats(
            conn,
            platform=platform,
            agent_id=agent_id,
            handle=handle,
            competition_id=competition_id,
            stats=response,
        )
    finally:
        conn.close()
    return True


class ArenaStatsFetcher:
    """Fetch Arena stats off the action path, once per table/opponent sighting."""

    def __init__(
        self,
        api_fn,
        competition_id,
        *,
        db_path: str | Path | None = None,
        platform="arena",
        start=True,
    ):
        self.api_fn = api_fn
        self.competition_id = competition_id
        self.db_path = db_path
        self.platform = platform
        self._tasks: queue.Queue[AgentStatsTask | None] = queue.Queue()
        self._queued_keys: set[tuple[str, str]] = set()
        self._lock = threading.Lock()
        self._thread = None
        if start:
            self.start()

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="arena-agent-stats",
            daemon=True,
        )
        self._thread.start()

    def close(self, timeout=2):
        if self._thread is None:
            return
        self._tasks.put(None)
        self._thread.join(timeout=timeout)

    def enqueue(self, table_id, agent_id, handle=None):
        if not agent_id:
            return False
        key = (str(table_id or "unknown-table"), str(agent_id))
        with self._lock:
            if key in self._queued_keys:
                return False
            self._queued_keys.add(key)
        self._tasks.put(AgentStatsTask(key[0], key[1], handle))
        return True

    def _run(self):
        while True:
            task = self._tasks.get()
            if task is None:
                return
            try:
                fetch_and_record_agent_stats(
                    self.api_fn,
                    self.competition_id,
                    task.agent_id,
                    handle=task.handle,
                    db_path=self.db_path,
                    platform=self.platform,
                )
            except Exception as exc:
                print(
                    f"  Arena stats fetch failed for {task.agent_id}: {exc}",
                    flush=True,
                )


def schedule_table_opponent_stats(fetcher, table, hero_agent_id):
    if fetcher is None:
        return 0
    table_id = str(table.get("tableId") or table.get("id") or "unknown-table")
    queued = 0
    for agent_id, handle in table_opponent_agents(table, hero_agent_id):
        queued += int(fetcher.enqueue(table_id, agent_id, handle=handle))
    return queued
