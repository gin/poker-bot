"""Per-guard attribution report over guard_overrides telemetry.

Answers, per guard: how often did it fire, was it applied or shadow, what
action transition did it make, and what was the outcome of the hands it
touched compared to the run average. This judges a guard on exactly the
decisions it touches instead of whole-run bb/100 noise.

Usage:
    guard-report                          # latest telemetry run, default DB
    guard-report --run-id RUN [--db PATH] [--min-fires N] [--by-street]
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from poker_bot.opponent_store import (  # noqa: E402
    connect,
    default_telemetry_db_path,
    summarize_guard_overrides,
)


def _latest_run_id(conn):
    row = conn.execute(
        "select run_id from telemetry_runs order by started_at desc limit 1"
    ).fetchone()
    return row["run_id"] if row else None


def _run_baseline(conn, run_id):
    """Run-wide averages per decision, for comparison against fired spots."""
    where = "where run_id = ?" if run_id else ""
    params = (run_id,) if run_id else ()
    row = conn.execute(
        f"""
        select count(*) as decisions,
               avg(hero_net_chips) as avg_net_chips,
               avg(won_hand) as win_rate
        from decision_telemetry {where}
        """,
        params,
    ).fetchone()
    return row


def _street_breakdown(conn, run_id, guard_id):
    where = "where g.guard_id = ?"
    params: list = [guard_id]
    if run_id:
        where += " and g.run_id = ?"
        params.append(run_id)
    return conn.execute(
        f"""
        select g.street,
               count(*) as fires,
               sum(g.applied) as applied,
               avg(d.hero_net_chips) as avg_net_chips
        from guard_overrides g
        left join decision_telemetry d
          on d.run_id = g.run_id
         and d.hand_id = g.hand_id
         and d.decision_index = g.decision_index
        {where}
        group by g.street
        order by fires desc
        """,
        params,
    ).fetchall()


def _fmt(value, spec="{:.1f}", empty="-"):
    return spec.format(value) if value is not None else empty


def print_report(conn, run_id, *, min_fires=1, by_street=False, out=sys.stdout):
    baseline = _run_baseline(conn, run_id)
    decisions = baseline["decisions"] or 0
    print(f"run: {run_id or '(all runs)'}", file=out)
    print(
        f"hero decisions: {decisions}"
        f" | run avg net chips/decision: {_fmt(baseline['avg_net_chips'])}"
        f" | run win rate: {_fmt(baseline['win_rate'], '{:.1%}')}",
        file=out,
    )

    rows = summarize_guard_overrides(conn, run_id, min_fires=min_fires)
    if not rows:
        print("no guard fires recorded", file=out)
        return

    header = (
        f"{'guard':<38} {'phase':<5} {'fires':>5} {'appl':>5} {'shdw':>5} "
        f"{'fire%':>6} {'avg_chips':>10} {'win%':>6}  transition"
    )
    print(header, file=out)
    print("-" * len(header), file=out)
    seen_guards = []
    for row in rows:
        fire_rate = row["fires"] / decisions if decisions else 0.0
        print(
            f"{row['guard_id']:<38} {row['phase'] or '-':<5} "
            f"{row['fires']:>5} {row['applied']:>5} {row['shadow']:>5} "
            f"{fire_rate:>6.2%} {_fmt(row['avg_net_chips']):>10} "
            f"{_fmt(row['win_rate'], '{:.1%}'):>6}  {row['transition']}",
            file=out,
        )
        if row["guard_id"] not in seen_guards:
            seen_guards.append(row["guard_id"])

    if by_street:
        for guard_id in seen_guards:
            print(f"\n{guard_id} by street:", file=out)
            for row in _street_breakdown(conn, run_id, guard_id):
                print(
                    f"  {row['street'] or '-':<8} fires={row['fires']:<5} "
                    f"applied={row['applied']:<5} "
                    f"avg_chips={_fmt(row['avg_net_chips'])}",
                    file=out,
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db",
        default=None,
        help=f"telemetry SQLite path (default: {default_telemetry_db_path()})",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="telemetry run to report on (default: most recent run)",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="aggregate across every run in the DB",
    )
    parser.add_argument("--min-fires", type=int, default=1)
    parser.add_argument("--by-street", action="store_true")
    args = parser.parse_args(argv)

    conn = connect(args.db, telemetry=args.db is None)
    try:
        run_id = args.run_id
        if run_id is None and not args.all_runs:
            run_id = _latest_run_id(conn)
            if run_id is None:
                print("no telemetry runs found", file=sys.stderr)
                return 1
        print_report(
            conn, run_id, min_fires=args.min_fires, by_street=args.by_street
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
