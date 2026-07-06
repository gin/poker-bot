"""Run benchmark matrices for poker strategies."""

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

# Use 'spawn' to avoid fork() deadlock issues with threaded parents
multiprocessing.set_start_method("spawn", force=True)

from eval.profiler import (
    PlayProfiler,
    StrategyProfile,
    aggregate_profiles,
    format_profile,
)
from eval.selfplay import BIG_BLIND, SelfPlayResult, run_selfplay
from poker_bot import opponent_store

DEFAULT_HANDS = 10000
DEFAULT_OPPONENTS = ("simple", "adaptive", "royal_adaptive")
DEFAULT_PLAYERS = (2, 6)
DEFAULT_SEEDS = (1, 2, 3, 4, 5, 6)
DEFAULT_WORKER_CAP = 16


@dataclass(frozen=True)
class BenchmarkCase:
    strat: str
    opponent: str
    players: int
    seed: int
    hands: int


@dataclass(frozen=True)
class BenchmarkAggregate:
    strat: str
    opponent: str
    players: int
    seeds: tuple[int, ...]
    hands: int
    wins: int
    losses: int
    pushes: int
    net_chips: int
    elapsed: float
    profile: StrategyProfile | None = None

    @property
    def bb_per_100(self):
        if self.hands == 0:
            return 0.0
        return self.net_chips / BIG_BLIND / self.hands * 100

    @property
    def chips_per_hand(self):
        if self.hands == 0:
            return 0.0
        return self.net_chips / self.hands


@dataclass(frozen=True)
class BenchmarkComparison:
    opponent: str
    players: int
    candidate_bb_per_100: float
    baseline_bb_per_100: float
    delta_bb_per_100: float
    min_delta_bb_per_100: float

    @property
    def passed(self):
        return self.delta_bb_per_100 >= self.min_delta_bb_per_100


@dataclass(frozen=True)
class BenchmarkReport:
    strat: str
    cases: tuple[BenchmarkCase, ...]
    results: tuple[SelfPlayResult, ...]
    aggregates: tuple[BenchmarkAggregate, ...]
    elapsed: float
    profile_enabled: bool = False
    pass_threshold: float | None = None
    baseline_strat: str | None = None
    baseline_results: tuple[SelfPlayResult, ...] = ()
    baseline_aggregates: tuple[BenchmarkAggregate, ...] = ()
    comparisons: tuple[BenchmarkComparison, ...] = ()
    min_delta_bb_per_100: float = 0.0
    workers: int = 1
    h2h: bool = False

    @property
    def passed(self):
        checks = []
        if self.pass_threshold is not None:
            checks.append(
                all(row.bb_per_100 >= self.pass_threshold for row in self.aggregates)
            )
        if self.baseline_strat is not None:
            checks.append(all(row.passed for row in self.comparisons))
        if not checks:
            return None
        return all(checks)


def _split_csv(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(value)


def parse_csv_strings(value):
    parsed = _split_csv(value)
    if parsed is None:
        return None
    if not parsed:
        raise ValueError("expected at least one value")
    return tuple(parsed)


def parse_csv_ints(value):
    parsed = _split_csv(value)
    if parsed is None:
        return None
    if not parsed:
        raise ValueError("expected at least one integer")
    try:
        return tuple(int(part) for part in parsed)
    except ValueError as exc:
        raise ValueError(f"invalid integer list: {value}") from exc


def load_config(path):
    if path is None:
        return {}
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("benchmark config must be a JSON object")
    return data


def resolve_options(args):
    config = load_config(args.config)
    opponents = (
        parse_csv_strings(args.opponents)
        or parse_csv_strings(config.get("opponents"))
        or DEFAULT_OPPONENTS
    )
    players = (
        parse_csv_ints(args.players)
        or parse_csv_ints(config.get("players"))
        or DEFAULT_PLAYERS
    )
    h2h = bool(args.h2h or config.get("h2h", False))
    if h2h:
        players = (
            parse_csv_ints(args.players)
            or parse_csv_ints(config.get("players"))
            or (2,)
        )
        if players != (2,):
            raise ValueError("--h2h only supports heads-up --players 2")
    seeds = (
        parse_csv_ints(args.seeds)
        or parse_csv_ints(config.get("seeds"))
        or DEFAULT_SEEDS
    )
    hands = args.hands if args.hands is not None else config.get("hands", DEFAULT_HANDS)
    track_opponents = bool(
        h2h or args.track_opponents or config.get("track_opponents", False)
    )
    baseline = args.baseline if args.baseline is not None else config.get("baseline")
    min_delta_bb_per_100 = (
        args.min_delta_bb_per_100
        if args.min_delta_bb_per_100 is not None
        else config.get("min_delta_bb_per_100", 0.0)
    )
    fail_under_bb100 = (
        args.fail_under_bb100
        if args.fail_under_bb100 is not None
        else config.get("fail_under_bb100")
    )
    use_profile = bool(args.profile or config.get("profile", False))
    workers = (
        int(args.workers) if args.workers is not None else int(config.get("workers", 0))
    )
    return {
        "opponents": opponents,
        "players": players,
        "seeds": seeds,
        "hands": int(hands),
        "track_opponents": track_opponents,
        "baseline": baseline,
        "min_delta_bb_per_100": float(min_delta_bb_per_100),
        "fail_under_bb100": fail_under_bb100
        if fail_under_bb100 is None
        else float(fail_under_bb100),
        "profile": use_profile,
        "workers": workers,
        "h2h": h2h,
    }


def build_cases(strat, opponents, players, seeds, hands):
    if hands <= 0:
        raise ValueError("--hands must be positive")
    cases = []
    for opponent in opponents:
        for player_count in players:
            if player_count < 2 or player_count > 6:
                raise ValueError("--players values must be between 2 and 6")
            for seed in seeds:
                cases.append(
                    BenchmarkCase(
                        strat=strat,
                        opponent=opponent,
                        players=player_count,
                        seed=seed,
                        hands=hands,
                    )
                )
    return tuple(cases)


def aggregate_results(cases, results):
    grouped = {}
    for case, result in zip(cases, results, strict=True):
        key = (result.strat, result.opponent, result.players)
        grouped.setdefault(key, []).append((case, result))

    rows = []
    for (strat, opponent, players), group in grouped.items():
        profiles = []
        for _case, r in group:
            if r.profile is not None:
                profiles.append(r.profile)
        rows.append(
            BenchmarkAggregate(
                strat=strat,
                opponent=opponent,
                players=players,
                seeds=tuple(case.seed for case, _result in group),
                hands=sum(result.hands for _case, result in group),
                wins=sum(result.wins for _case, result in group),
                losses=sum(result.losses for _case, result in group),
                pushes=sum(result.pushes for _case, result in group),
                net_chips=sum(result.net_chips for _case, result in group),
                elapsed=sum(result.elapsed for _case, result in group),
                profile=aggregate_profiles(profiles) if profiles else None,
            )
        )
    return tuple(rows)


def compare_aggregates(candidate_rows, baseline_rows, min_delta_bb_per_100):
    baseline_by_key = {(row.opponent, row.players): row for row in baseline_rows}
    comparisons = []
    for candidate in candidate_rows:
        baseline = baseline_by_key.get((candidate.opponent, candidate.players))
        if baseline is None:
            continue
        delta = candidate.bb_per_100 - baseline.bb_per_100
        comparisons.append(
            BenchmarkComparison(
                opponent=candidate.opponent,
                players=candidate.players,
                candidate_bb_per_100=candidate.bb_per_100,
                baseline_bb_per_100=baseline.bb_per_100,
                delta_bb_per_100=delta,
                min_delta_bb_per_100=min_delta_bb_per_100,
            )
        )
    return tuple(comparisons)


def _resolve_workers(workers: int) -> int:
    """Resolve the actual worker count from the requested value.

    - 0 or negative: default to half the CPU count (min 1).
    - > DEFAULT_WORKER_CAP: require POKER_BENCHMARK_ALLOW_HIGH_WORKERS=1.
    """
    if workers <= 0:
        cpu = os.cpu_count() or 1
        workers = max(1, cpu // 2)
    if (
        workers > DEFAULT_WORKER_CAP
        and os.environ.get("POKER_BENCHMARK_ALLOW_HIGH_WORKERS") != "1"
    ):
        raise ValueError(
            f"workers={workers} exceeds safety cap {DEFAULT_WORKER_CAP}; "
            "set POKER_BENCHMARK_ALLOW_HIGH_WORKERS=1 to override"
        )
    return workers


def _h2h_platform(case: BenchmarkCase) -> str:
    return (
        f"benchmark-h2h:{case.strat}:vs:{case.opponent}:p{case.players}:seed{case.seed}"
    )


def _run_case_worker(args):
    """Run a single benchmark case in a worker process.

    ``args`` is a tuple ``(case, runner, track_opponents, profile,
    worker_db_path, h2h, telemetry, telemetry_run_id)``. The worker uses
    ``worker_db_path`` as the opponent DB (a per-worker private file) so
    that parallel writes do not collide; the orchestrator merges those
    files afterwards.
    """
    (
        case,
        runner,
        track_opponents,
        profile,
        worker_db_path,
        h2h,
        telemetry,
        telemetry_run_id,
    ) = args
    profiler = PlayProfiler() if profile else None
    runner_kwargs = {}
    if h2h:
        runner_kwargs["platform"] = _h2h_platform(case)
    if telemetry_run_id is not None:
        runner_kwargs["telemetry_run_id"] = telemetry_run_id
    result = runner(
        case.strat,
        hands=case.hands,
        seed=case.seed,
        opponent_name=case.opponent,
        players=case.players,
        track_opponents=track_opponents,
        opponent_db=worker_db_path,
        profiler=profiler,
        telemetry=telemetry,
        **runner_kwargs,
    )
    if profile and profiler is not None:
        sp = profiler.compute_profile()
        result = replace(result, profile=sp)
    return result


def _run_cases_parallel(
    cases,
    *,
    runner,
    track_opponents,
    profile,
    worker_db_paths,
    h2h,
    telemetry,
    telemetry_run_id,
    label: str,
) -> list[SelfPlayResult]:
    actual_workers = len(worker_db_paths) if worker_db_paths else _resolve_workers(0)
    results: list[SelfPlayResult | None] = [None] * len(cases)
    total = len(cases)
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as pool:
        futures = {}
        for i, case in enumerate(cases):
            worker_db = worker_db_paths[i % actual_workers] if worker_db_paths else None
            future = pool.submit(
                _run_case_worker,
                (
                    case,
                    runner,
                    track_opponents,
                    profile,
                    worker_db,
                    h2h,
                    telemetry,
                    telemetry_run_id,
                ),
            )
            futures[future] = i
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
            except Exception:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            results[idx] = result
            completed += 1
            print(f"[benchmark] {label} completed {completed}/{total} cases")
    return results  # type: ignore[return-value]


def _run_cases_sequential(
    cases,
    *,
    runner,
    track_opponents,
    profile,
    opponent_db,
    h2h,
    telemetry,
    telemetry_run_id,
) -> list[SelfPlayResult]:
    return [
        _run_case_worker(
            (
                case,
                runner,
                track_opponents,
                profile,
                opponent_db,
                h2h,
                telemetry,
                telemetry_run_id,
            )
        )
        for case in cases
    ]


def run_benchmark(
    strat,
    *,
    opponents=DEFAULT_OPPONENTS,
    players=DEFAULT_PLAYERS,
    seeds=DEFAULT_SEEDS,
    hands=DEFAULT_HANDS,
    track_opponents=False,
    opponent_db=None,
    fail_under_bb100=None,
    baseline_strat=None,
    min_delta_bb_per_100=0.0,
    profile=False,
    runner=run_selfplay,
    workers=0,
    h2h=False,
    telemetry=False,
    telemetry_run_id=None,
):
    if h2h:
        if tuple(players) != (2,):
            raise ValueError("--h2h only supports heads-up --players 2")
        track_opponents = True
    # When telemetry is enabled but no opponent_db is given, fall back to
    # the default telemetry DB path so both opponent stats and gameplay
    # telemetry land in the same place. Mirrors the selfplay CLI's
    # default behavior so benchmark and selfplay stay symmetric.
    if telemetry and opponent_db is None:
        opponent_db = opponent_store.default_telemetry_db_path()
    cases = build_cases(strat, opponents, players, seeds, hands)
    started = time.perf_counter()
    actual_workers = _resolve_workers(workers)

    worker_db_paths: list[str] | None = None
    temp_dir_ctx: tempfile.TemporaryDirectory | None = None
    if opponent_db is not None and actual_workers > 1:
        temp_dir_ctx = tempfile.TemporaryDirectory()
        worker_db_paths = [
            os.path.join(temp_dir_ctx.name, f"worker_{i}.sqlite")
            for i in range(actual_workers)
        ]
        for path in worker_db_paths:
            opponent_store.connect(path)

    if actual_workers <= 1:
        results = _run_cases_sequential(
            cases,
            runner=runner,
            track_opponents=track_opponents,
            profile=profile,
            opponent_db=opponent_db,
            h2h=h2h,
            telemetry=telemetry,
            telemetry_run_id=telemetry_run_id,
        )
    else:
        results = _run_cases_parallel(
            cases,
            runner=runner,
            track_opponents=track_opponents,
            profile=profile,
            worker_db_paths=worker_db_paths,
            h2h=h2h,
            telemetry=telemetry,
            telemetry_run_id=telemetry_run_id,
            label="candidate",
        )

    baseline_results: list[SelfPlayResult] = []
    if baseline_strat is not None:
        baseline_cases = [replace(case, strat=baseline_strat) for case in cases]
        if actual_workers <= 1:
            baseline_results = _run_cases_sequential(
                baseline_cases,
                runner=runner,
                track_opponents=track_opponents,
                profile=profile,
                opponent_db=opponent_db,
                h2h=h2h,
                telemetry=telemetry,
                telemetry_run_id=telemetry_run_id,
            )
        else:
            baseline_results = _run_cases_parallel(
                baseline_cases,
                runner=runner,
                track_opponents=track_opponents,
                profile=profile,
                worker_db_paths=worker_db_paths,
                h2h=h2h,
                telemetry=telemetry,
                telemetry_run_id=telemetry_run_id,
                label="baseline",
            )

    if worker_db_paths is not None and temp_dir_ctx is not None:
        for path in worker_db_paths:
            opponent_store.merge_worker_db(opponent_db, path)
        temp_dir_ctx.cleanup()

    elapsed = time.perf_counter() - started
    aggregates = aggregate_results(cases, results)
    baseline_aggregates = ()
    comparisons = ()
    if baseline_strat is not None:
        baseline_aggregates = aggregate_results(cases, baseline_results)
        comparisons = compare_aggregates(
            aggregates,
            baseline_aggregates,
            min_delta_bb_per_100,
        )
    return BenchmarkReport(
        strat=strat,
        cases=cases,
        results=tuple(results),
        aggregates=aggregates,
        elapsed=elapsed,
        profile_enabled=profile,
        pass_threshold=fail_under_bb100,
        baseline_strat=baseline_strat,
        baseline_results=tuple(baseline_results),
        baseline_aggregates=baseline_aggregates,
        comparisons=comparisons,
        min_delta_bb_per_100=min_delta_bb_per_100,
        workers=actual_workers,
        h2h=h2h,
    )


def _signed_int(value):
    return f"{int(value):+d}"


def _signed_float(value):
    return f"{value:+.1f}"


def report_opponent_width(report):
    labels = [row.opponent for row in report.aggregates]
    labels.extend(row.opponent for row in report.comparisons)
    if not labels:
        return 24
    return max(24, max(len(label) for label in labels))


def format_report(report):
    opponent_width = report_opponent_width(report)
    lines = [
        f"benchmark   : {report.strat}",
        f"cases       : {len(report.cases)}",
        f"elapsed     : {report.elapsed:.1f}s",
        "",
        (
            f"{'opponent':<{opponent_width}} players seeds hands      net    "
            "bb/100 chips/hand  W/L/P"
        ),
        "-" * (54 + opponent_width),
    ]
    for row in report.aggregates:
        seeds = len(row.seeds)
        lines.append(
            f"{row.opponent:<{opponent_width}} "
            f"{row.players:<7} "
            f"{seeds:<5} "
            f"{row.hands:<8} "
            f"{_signed_int(row.net_chips):>8} "
            f"{_signed_float(row.bb_per_100):>7} "
            f"{_signed_float(row.chips_per_hand):>10}  "
            f"{row.wins}/{row.losses}/{row.pushes}"
        )
        if report.profile_enabled and row.profile is not None:
            profile_text = format_profile(row.profile)
            for line in profile_text.split("\n"):
                lines.append(f"  {line}")
    if report.baseline_strat is not None:
        lines.extend(
            [
                "",
                f"baseline    : {report.baseline_strat}",
                (
                    f"{'opponent':<{opponent_width}} players candidate "
                    "baseline delta gate"
                ),
                "-" * (48 + opponent_width),
            ]
        )
        for row in report.comparisons:
            status = "PASS" if row.passed else "FAIL"
            lines.append(
                f"{row.opponent:<{opponent_width}} "
                f"{row.players:<7} "
                f"{_signed_float(row.candidate_bb_per_100):>9} "
                f"{_signed_float(row.baseline_bb_per_100):>8} "
                f"{_signed_float(row.delta_bb_per_100):>6} "
                f"{status}"
            )
    if report.passed is not None:
        status = "PASS" if report.passed else "FAIL"
        gates = []
        if report.pass_threshold is not None:
            gates.append(f"bb/100 >= {report.pass_threshold:.1f}")
        if report.baseline_strat is not None:
            gates.append(
                f"delta vs {report.baseline_strat} >= {report.min_delta_bb_per_100:.1f}"
            )
        lines.extend(["", f"gate        : {status} {'; '.join(gates)}"])
    return "\n".join(lines)


def report_to_jsonable(report):
    return {
        "strat": report.strat,
        "elapsed": report.elapsed,
        "pass_threshold": report.pass_threshold,
        "passed": report.passed,
        "baseline_strat": report.baseline_strat,
        "min_delta_bb_per_100": report.min_delta_bb_per_100,
        "workers": report.workers,
        "h2h": report.h2h,
        "cases": [asdict(case) for case in report.cases],
        "results": [asdict(result) for result in report.results],
        "baseline_results": [asdict(result) for result in report.baseline_results],
        "aggregates": [
            {
                **asdict(row),
                "bb_per_100": row.bb_per_100,
                "chips_per_hand": row.chips_per_hand,
            }
            for row in report.aggregates
        ],
        "baseline_aggregates": [
            {
                **asdict(row),
                "bb_per_100": row.bb_per_100,
                "chips_per_hand": row.chips_per_hand,
            }
            for row in report.baseline_aggregates
        ],
        "comparisons": [
            {
                **asdict(row),
                "passed": row.passed,
            }
            for row in report.comparisons
        ],
    }


def write_json_report(report, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report_to_jsonable(report), f, indent=2)
        f.write("\n")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a self-play benchmark matrix for a poker strategy."
    )
    parser.add_argument(
        "--strat",
        required=True,
        help="Candidate strategy module under poker_bot.strategies.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON benchmark config, e.g. benchmarks/default.json.",
    )
    parser.add_argument(
        "--opponents",
        "--opponent",
        dest="opponents",
        default=None,
        help="Comma-separated opponent strategies. Overrides config.",
    )
    parser.add_argument(
        "--players",
        default=None,
        help="Comma-separated player counts, 2-6. Overrides config.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated integer seeds. Overrides config.",
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=None,
        help=f"Hands per case. Defaults to {DEFAULT_HANDS}.",
    )
    parser.add_argument(
        "--track-opponents",
        action="store_true",
        help="Maintain opponent profiles during self-play.",
    )
    parser.add_argument(
        "--h2h",
        action="store_true",
        help=(
            "Run heads-up matchups against each opponent with opponent tracking "
            "enabled for both strategies."
        ),
    )
    parser.add_argument(
        "--opponent-db",
        default=None,
        help="Optional SQLite database path for persistent benchmark profiles.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the benchmark report as JSON.",
    )
    parser.add_argument(
        "--fail-under-bb100",
        type=float,
        default=None,
        help="Exit non-zero unless every aggregate reaches this bb/100.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional baseline strategy that the candidate must beat.",
    )
    parser.add_argument(
        "--min-delta-bb100",
        type=float,
        default=None,
        dest="min_delta_bb_per_100",
        help=(
            "Minimum candidate bb/100 improvement over baseline for every "
            "aggregate. Defaults to 0.0 when --baseline is used."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable strategy profiling (VPIP, PFR, AF, 3-BET%, WTSD, W$SD, BLUFF).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        dest="workers",
        help=(
            "Number of worker processes for parallel case execution. "
            "0 (default) = half the CPU count. Capped at 16 unless "
            "POKER_BENCHMARK_ALLOW_HIGH_WORKERS=1 is set."
        ),
    )
    parser.add_argument(
        "--telemetry",
        action="store_true",
        help=(
            "Record hero decision telemetry to the SQLite database. "
            "Writes alongside opponent stats when --opponent-db is set; "
            "otherwise writes to the default telemetry DB path."
        ),
    )
    parser.add_argument(
        "--telemetry-run-id",
        default=None,
        dest="telemetry_run_id",
        help=(
            "Optional run id for decision telemetry. When omitted, a "
            "fresh UUID is generated per benchmark invocation so all "
            "cases share one run id (useful for cross-case queries)."
        ),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        options = resolve_options(args)
        report = run_benchmark(
            args.strat,
            opponents=options["opponents"],
            players=options["players"],
            seeds=options["seeds"],
            hands=options["hands"],
            track_opponents=options["track_opponents"],
            opponent_db=args.opponent_db,
            fail_under_bb100=options["fail_under_bb100"],
            baseline_strat=options["baseline"],
            min_delta_bb_per_100=options["min_delta_bb_per_100"],
            profile=options["profile"],
            workers=options["workers"],
            h2h=options["h2h"],
            telemetry=args.telemetry,
            telemetry_run_id=args.telemetry_run_id,
        )
    except ValueError as exc:
        print(f"benchmark: {exc}", file=sys.stderr)
        return 2

    print(format_report(report))
    if args.output_json:
        write_json_report(report, args.output_json)
    if report.passed is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
