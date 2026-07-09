"""Gameplay replay & analysis GUI — FastAPI over the telemetry SQLite.

No ETL: every request queries the DB read-only (WAL-safe), so the app works
live against a DB the bot is currently writing.

Launch:
    uv run python gui/server.py --db ~/playground-luigi-multi-core-s5.sqlite
    # or rely on POKER_BOT_TELEMETRY_DB / default gameplay.sqlite
Then open http://127.0.0.1:8400
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

GUI_DIR = Path(__file__).resolve().parent
REPO_ROOT = GUI_DIR.parent

app = FastAPI(title="poker-bot replay")
DB_PATH: Path | None = None
OPP_DB_PATH: Path | None = None  # live runs keep profiles in a separate file

TAG_RE = re.compile(r"\[(guard|exploit):([a-z0-9_-]+)\]")


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def tags_of(message: str | None) -> list[str]:
    return [f"{kind}:{name}" for kind, name in TAG_RE.findall(message or "")]


@app.get("/")
def index():
    return FileResponse(GUI_DIR / "index.html")


@app.get("/app.js")
def app_js():
    return FileResponse(GUI_DIR / "app.js")


@app.get("/style.css")
def style_css():
    return FileResponse(GUI_DIR / "style.css")


@app.get("/api/runs")
def runs():
    conn = db()
    try:
        return [
            dict(r)
            for r in conn.execute(
                """
                select r.run_id, r.strategy, r.opponent, r.players,
                       r.platform, r.started_at,
                       count(distinct d.hand_id) as hands,
                       (select coalesce(sum(net), 0) from (
                            select min(hero_net_chips) as net
                            from decision_telemetry
                            where run_id = r.run_id and hero_net_chips is not null
                            group by hand_id)) as net_chips
                from telemetry_runs r
                left join decision_telemetry d on d.run_id = r.run_id
                group by r.run_id
                order by r.started_at desc
                """
            )
        ]
    finally:
        conn.close()


@app.get("/api/hands")
def hands(
    run_id: str,
    order: str = "recent",
    tag: str = "",
    opponent: str = "",
    limit: int = 200,
):
    order_sql = {
        "recent": "last_seen desc",
        "biggest_loss": "net asc",
        "biggest_win": "net desc",
    }.get(order, "last_seen desc")
    conn = db()
    try:
        rows = conn.execute(
            f"""
            select hand_id,
                   min(hero_net_chips) as net,
                   max(won_hand) as won,
                   min(hole_cards) as hole_cards,
                   max(created_at) as last_seen,
                   count(*) as decisions,
                   max(case when street='River' then 4
                            when street='Turn' then 3
                            when street='Flop' then 2 else 1 end) as street_depth,
                   group_concat(strategy_message, ' || ') as messages,
                   max(active_players) as players,
                   max(hero_position) as position
            from decision_telemetry
            where run_id = ?
            group by hand_id
            order by {order_sql}
            limit ?
            """,
            (run_id, max(1, min(limit, 1000))),
        ).fetchall()
        out = []
        for r in rows:
            all_tags = sorted(set(tags_of(r["messages"])))
            if tag and not any(tag in t for t in all_tags):
                continue
            out.append(
                {
                    "hand_id": r["hand_id"],
                    "net": r["net"],
                    "won": r["won"],
                    "hole_cards": r["hole_cards"],
                    "decisions": r["decisions"],
                    "street_depth": r["street_depth"],
                    "players": r["players"],
                    "position": r["position"],
                    "tags": all_tags,
                    "last_seen": r["last_seen"],
                }
            )
        return out
    finally:
        conn.close()


def _profile_row(conn, agent_id):
    """Merge profiles across the telemetry and opponent DBs.

    Live runs split the stores: observed counters land in one file, API
    external stats in the other. Take observed fields from whichever side
    has seen more hands, and api_stats from whichever side has them.
    """
    candidates = [p for p in [_profile_row_from(conn, agent_id)] if p]
    if OPP_DB_PATH is not None:
        opp_conn = sqlite3.connect(f"file:{OPP_DB_PATH}?mode=ro", uri=True)
        opp_conn.row_factory = sqlite3.Row
        try:
            extra = _profile_row_from(opp_conn, agent_id)
            if extra:
                candidates.append(extra)
        finally:
            opp_conn.close()
    if not candidates:
        return None
    best = max(candidates, key=lambda p: p["hands_seen"] or 0)
    for p in candidates:
        if p is not best and p.get("api_stats") and not best.get("api_stats"):
            best["api_stats"] = p["api_stats"]
            best["api_fetched_at"] = p.get("api_fetched_at")
    return best


def _profile_row_from(conn, agent_id):
    row = conn.execute(
        """
        select o.agent_id, o.handle, s.*
        from opponents o join opponent_stats s on s.opponent_id = o.id
        where o.agent_id = ?
        """,
        (agent_id,),
    ).fetchone()
    if row is None:
        return None
    actions = row["calls"] + row["bets"] + row["raises"] + row["folds"]
    hands_seen = max(row["hands_seen"], 1)
    profile = {
        "agent_id": row["agent_id"],
        "handle": row["handle"],
        "hands_seen": row["hands_seen"],
        "vpip_pct": round(100 * row["vpip"] / hands_seen, 1),
        "pfr_pct": round(100 * row["pfr"] / hands_seen, 1),
        "aggression_pct": round(
            100 * (row["bets"] + row["raises"]) / actions, 1
        )
        if actions
        else None,
        "call_pct": round(100 * row["calls"] / actions, 1) if actions else None,
        "fold_to_bet_pct": round(
            100 * row["fold_to_bet"] / row["opportunities_to_fold_to_bet"], 1
        )
        if row["opportunities_to_fold_to_bet"]
        else None,
        "showdowns": row["showdowns"],
        "won_showdown": row["won_showdown"],
    }
    ext = conn.execute(
        """
        select e.stats_json, e.fetched_at from opponent_external_stats e
        join opponents o on o.id = e.opponent_id where o.agent_id = ?
        order by e.fetched_at desc limit 1
        """,
        (agent_id,),
    ).fetchone()
    if ext:
        try:
            profile["api_stats"] = json.loads(ext["stats_json"])
            profile["api_fetched_at"] = ext["fetched_at"]
        except ValueError:
            pass
    return profile


@app.get("/api/hand")
def hand(run_id: str, hand_id: str):
    conn = db()
    try:
        decisions = [
            dict(r)
            for r in conn.execute(
                """
                select * from decision_telemetry
                where run_id = ? and hand_id = ?
                order by decision_index
                """,
                (run_id, hand_id),
            )
        ]
        if not decisions:
            raise HTTPException(404, "hand not found")
        hero_id = decisions[0].get("hero_agent_id")
        for d in decisions:
            d["tags"] = tags_of(d.get("strategy_message"))

        observed = [
            dict(r)
            for r in conn.execute(
                """
                select o.agent_id, o.handle, a.street, a.action, a.amount,
                       a.pot, a.message, a.facing_bet, a.created_at, a.id
                from opponent_actions a join opponents o on o.id = a.opponent_id
                where a.hand_id = ?
                order by a.id
                """,
                (hand_id,),
            )
        ]

        # Timeline: group by street, hero decisions interleaved with observed
        # opponent actions by timestamp (hero rows come from decision_telemetry
        # which carries the richer context; skip hero dupes in observed).
        streets = ["Preflop", "Flop", "Turn", "River", "Showdown"]
        timeline = []
        for street in streets:
            events = []
            for d in decisions:
                if d["street"] == street:
                    events.append(
                        {
                            "kind": "hero",
                            "at": d["created_at"],
                            "action": d["chosen_action"],
                            "amount": d["chosen_amount"],
                            "pot": d["pot_chips"],
                            "message": d["strategy_message"],
                            "tags": d["tags"],
                            "hole_cards": d["hole_cards"],
                            "board": d["board_cards"],
                            "facing_bet": d["facing_bet"],
                            "hero_stack": d["hero_stack"],
                        }
                    )
            for a in observed:
                if a["street"] == street and a["agent_id"] != hero_id:
                    events.append(
                        {
                            "kind": "opponent",
                            "at": a["created_at"],
                            "who": a["handle"] or a["agent_id"][:10],
                            "agent_id": a["agent_id"],
                            "action": a["action"],
                            "amount": a["amount"],
                            "pot": a["pot"],
                        }
                    )
            events.sort(key=lambda e: e["at"] or "")
            board = next(
                (d["board_cards"] for d in decisions if d["street"] == street), None
            )
            if events or board:
                timeline.append({"street": street, "board": board, "events": events})

        opponents = {}
        for a in observed:
            if a["agent_id"] != hero_id and a["agent_id"] not in opponents:
                profile = _profile_row(conn, a["agent_id"])
                if profile:
                    opponents[a["agent_id"]] = profile

        return {
            "hand_id": hand_id,
            "hero": {
                "agent_id": hero_id,
                "hole_cards": decisions[0]["hole_cards"],
                "position": decisions[0]["hero_position"],
                "net": min(
                    (d["hero_net_chips"] for d in decisions
                     if d["hero_net_chips"] is not None),
                    default=None,
                ),
                "won": decisions[0]["won_hand"],
            },
            "timeline": timeline,
            "decisions": decisions,
            "opponents": list(opponents.values()),
        }
    finally:
        conn.close()


@app.get("/api/tags")
def tag_summary(run_id: str):
    conn = db()
    try:
        rows = conn.execute(
            """
            select strategy_message, hero_net_chips from decision_telemetry
            where run_id = ? and (strategy_message like '%[guard:%'
                                  or strategy_message like '%[exploit:%')
            """,
            (run_id,),
        ).fetchall()
        agg: dict[str, dict] = {}
        for r in rows:
            for tag in tags_of(r["strategy_message"]):
                entry = agg.setdefault(tag, {"fires": 0, "net": 0, "known": 0})
                entry["fires"] += 1
                if r["hero_net_chips"] is not None:
                    entry["net"] += r["hero_net_chips"]
                    entry["known"] += 1
        return [
            {
                "tag": tag,
                "fires": e["fires"],
                "avg_net": round(e["net"] / e["known"], 1) if e["known"] else None,
            }
            for tag, e in sorted(agg.items(), key=lambda kv: -kv[1]["fires"])
        ]
    finally:
        conn.close()


def main():
    global DB_PATH
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db",
        default=os.environ.get("POKER_BOT_TELEMETRY_DB")
        or str(REPO_ROOT / "gameplay.sqlite"),
    )
    parser.add_argument(
        "--opponent-db",
        default=os.environ.get("POKER_BOT_OPPONENT_DB"),
        help="separate opponent-profile DB (live runs split the stores)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8400)
    args = parser.parse_args()
    DB_PATH = Path(args.db).expanduser()
    if not DB_PATH.is_file():
        raise SystemExit(f"no such db: {DB_PATH}")
    global OPP_DB_PATH
    if args.opponent_db:
        candidate = Path(args.opponent_db).expanduser()
        if candidate.is_file():
            OPP_DB_PATH = candidate
    import uvicorn

    print(f"replay GUI: http://{args.host}:{args.port}  (db: {DB_PATH})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
