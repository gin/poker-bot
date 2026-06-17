"""Promotion gate for candidate poker strategies."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from eval import benchmark, population_selfplay
from eval.selfplay import BIG_BLIND, run_selfplay

DEFAULT_CONFIG = Path("benchmarks/promotion_gate.json")
DEFAULT_CHAMPION_JSON = Path("benchmarks/champion.json")
DEFAULT_OUTPUT_DIR = Path("benchmark-runs")
DEFAULT_HISTORY_DIR = Path("promotions")
DEFAULT_HISTORY_INDEX = "index.jsonl"
DEFAULT_SCENARIO_TESTS = ("tests/scenario",)
DEFAULT_PROMOTION_OPPONENTS = (
    "simple",
    "adaptive",
    "counter_adaptive",
    "threshold_pressure",
    "anti_threshold",
    "profiled_counter_adaptive",
    "simple+adaptive+counter_adaptive+threshold_pressure+anti_threshold",
    (
        "adaptive+counter_adaptive+threshold_pressure+anti_threshold"
        "+profiled_counter_adaptive"
    ),
    (
        "counter_adaptive+threshold_pressure+anti_threshold"
        "+profiled_counter_adaptive+{champion}"
    ),
)
DEFAULT_COUNTER_STRATEGIES = (
    "adaptive",
    "counter_adaptive",
    "threshold_pressure",
    "anti_threshold",
    "profiled_counter_adaptive",
    "simple+adaptive+counter_adaptive+threshold_pressure+anti_threshold",
    (
        "adaptive+counter_adaptive+threshold_pressure+anti_threshold"
        "+profiled_counter_adaptive"
    ),
    (
        "counter_adaptive+threshold_pressure+anti_threshold"
        "+profiled_counter_adaptive+{champion}"
    ),
)
DEFAULT_SMALL_MARGIN_BB100 = -2.5
DEFAULT_CATASTROPHIC_FLOOR_BB100 = -10.0
DEFAULT_SIMPLE_MIN_BB100 = 0.0
DEFAULT_MIN_SEED_PASS_RATE = 0.8


@dataclass(frozen=True)
class ScenarioTestResult:
    command: tuple[str, ...]
    passed: bool
    exit_code: int
    elapsed: float
    stdout: str
    stderr: str


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SeedVariance:
    opponent: str
    players: int
    seeds: tuple[int, ...]
    candidate_bb_per_100: tuple[float, ...]
    baseline_bb_per_100: tuple[float, ...]
    delta_bb_per_100: tuple[float, ...]
    candidate_mean_bb_per_100: float
    candidate_stddev_bb_per_100: float
    candidate_stderr_bb_per_100: float
    candidate_ci95_low_bb_per_100: float
    candidate_ci95_high_bb_per_100: float
    delta_mean_bb_per_100: float
    delta_stddev_bb_per_100: float
    delta_stderr_bb_per_100: float
    delta_ci95_low_bb_per_100: float
    delta_ci95_high_bb_per_100: float
    seed_passes: int
    seed_count: int
    seed_pass_rate: float


@dataclass(frozen=True)
class PromotionGateConfig:
    hands: int
    opponents: tuple[str, ...]
    players: tuple[int, ...]
    seeds: tuple[int, ...]
    track_opponents: bool
    scenario_tests: tuple[str, ...]
    simple_min_bb100: float
    min_delta_bb_per_100: float
    catastrophic_floor_bb100: float
    counter_strategies: tuple[str, ...]
    min_seed_pass_rate: float
    population_config: str | None
    workers: int = 0


@dataclass(frozen=True)
class PromotionGateReport:
    candidate: str
    champion: str
    champion_metadata: dict
    config: PromotionGateConfig
    scenario_tests: ScenarioTestResult
    benchmark_report: benchmark.BenchmarkReport | None
    population_report: population_selfplay.PopulationReport | None
    seed_variance: tuple[SeedVariance, ...]
    checks: tuple[GateCheck, ...]
    reproducibility: dict
    elapsed: float
    generated_at: str
    attempt_id: str
    promoted: bool
    champion_updated: bool
    output_json: str | None = None
    output_markdown: str | None = None
    history_index: str | None = None

    @property
    def passed(self):
        return self.scenario_tests.passed and all(check.passed for check in self.checks)


def _csv_or_tuple(value, default):
    parsed = benchmark.parse_csv_strings(value)
    if parsed is None:
        return tuple(default)
    return parsed


def _csv_ints_or_tuple(value, default):
    parsed = benchmark.parse_csv_ints(value)
    if parsed is None:
        return tuple(default)
    return parsed


def load_champion_strategy(path=DEFAULT_CHAMPION_JSON):
    payload = load_champion_metadata(path)
    strategy = payload.get("strategy")
    if not isinstance(strategy, str) or not strategy:
        raise ValueError(f"{Path(path)} must define a non-empty strategy")
    return strategy


def load_champion_metadata(path=DEFAULT_CHAMPION_JSON):
    champion_path = Path(path)
    with champion_path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{champion_path} must contain a JSON object")
    return payload


def load_promotion_config(path=DEFAULT_CONFIG):
    config_path = Path(path)
    with config_path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("promotion gate config must be a JSON object")

    return PromotionGateConfig(
        hands=int(data.get("hands", benchmark.DEFAULT_HANDS)),
        opponents=_csv_or_tuple(data.get("opponents"), DEFAULT_PROMOTION_OPPONENTS),
        players=_csv_ints_or_tuple(data.get("players"), (6,)),
        seeds=_csv_ints_or_tuple(data.get("seeds"), benchmark.DEFAULT_SEEDS),
        track_opponents=bool(data.get("track_opponents", True)),
        scenario_tests=_csv_or_tuple(
            data.get("scenario_tests"),
            DEFAULT_SCENARIO_TESTS,
        ),
        simple_min_bb100=float(data.get("simple_min_bb100", DEFAULT_SIMPLE_MIN_BB100)),
        min_delta_bb_per_100=float(
            data.get("min_delta_bb_per_100", DEFAULT_SMALL_MARGIN_BB100)
        ),
        catastrophic_floor_bb100=float(
            data.get(
                "catastrophic_floor_bb100",
                DEFAULT_CATASTROPHIC_FLOOR_BB100,
            )
        ),
        counter_strategies=_csv_or_tuple(
            data.get("counter_strategies"),
            DEFAULT_COUNTER_STRATEGIES,
        ),
        min_seed_pass_rate=float(
            data.get("min_seed_pass_rate", DEFAULT_MIN_SEED_PASS_RATE)
        ),
        population_config=data.get("population_config"),
        workers=int(data.get("workers", 0)),
    )


def resolve_champion_placeholders(values, champion):
    return tuple(value.replace("{champion}", champion) for value in values)


def build_opponent_pool(config, champion):
    opponents = list(resolve_champion_placeholders(config.opponents, champion))
    if champion not in opponents:
        opponents.append(champion)
    return tuple(opponents)


def run_scenario_tests(paths, *, command_runner=subprocess.run):
    command = (sys.executable, "-m", "pytest", *paths)
    started = time.perf_counter()
    completed = command_runner(
        command,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - started
    return ScenarioTestResult(
        command=tuple(command),
        passed=completed.returncode == 0,
        exit_code=completed.returncode,
        elapsed=elapsed,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _aggregates_by_opponent(report):
    rows = {}
    for row in report.aggregates:
        rows.setdefault(row.opponent, []).append(row)
    return rows


def _mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def _sample_stddev(values):
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _series_stats(values):
    mean = _mean(values)
    stddev = _sample_stddev(values)
    stderr = stddev / math.sqrt(len(values)) if values else 0.0
    ci95 = 1.96 * stderr
    return mean, stddev, stderr, mean - ci95, mean + ci95


def calculate_seed_variance(report, min_delta_bb_per_100):
    baseline_by_case = {
        (result.opponent, result.players, case.seed): result
        for case, result in zip(
            report.cases,
            report.baseline_results,
            strict=True,
        )
    }
    grouped = {}
    for case, result in zip(report.cases, report.results, strict=True):
        baseline = baseline_by_case[(result.opponent, result.players, case.seed)]
        key = (result.opponent, result.players)
        grouped.setdefault(key, []).append((case.seed, result, baseline))

    rows = []
    for (opponent, players), group in sorted(grouped.items()):
        seeds = tuple(seed for seed, _result, _baseline in group)
        candidate_values = tuple(
            result.bb_per_100 for _seed, result, _baseline in group
        )
        baseline_values = tuple(
            baseline.bb_per_100 for _seed, _result, baseline in group
        )
        delta_values = tuple(
            result.bb_per_100 - baseline.bb_per_100 for _seed, result, baseline in group
        )
        (
            candidate_mean,
            candidate_stddev,
            candidate_stderr,
            candidate_low,
            candidate_high,
        ) = _series_stats(candidate_values)
        delta_mean, delta_stddev, delta_stderr, delta_low, delta_high = _series_stats(
            delta_values
        )
        seed_passes = sum(1 for value in delta_values if value >= min_delta_bb_per_100)
        seed_count = len(delta_values)
        rows.append(
            SeedVariance(
                opponent=opponent,
                players=players,
                seeds=seeds,
                candidate_bb_per_100=candidate_values,
                baseline_bb_per_100=baseline_values,
                delta_bb_per_100=delta_values,
                candidate_mean_bb_per_100=candidate_mean,
                candidate_stddev_bb_per_100=candidate_stddev,
                candidate_stderr_bb_per_100=candidate_stderr,
                candidate_ci95_low_bb_per_100=candidate_low,
                candidate_ci95_high_bb_per_100=candidate_high,
                delta_mean_bb_per_100=delta_mean,
                delta_stddev_bb_per_100=delta_stddev,
                delta_stderr_bb_per_100=delta_stderr,
                delta_ci95_low_bb_per_100=delta_low,
                delta_ci95_high_bb_per_100=delta_high,
                seed_passes=seed_passes,
                seed_count=seed_count,
                seed_pass_rate=seed_passes / seed_count if seed_count else 0.0,
            )
        )
    return tuple(rows)


def evaluate_gate(report, config, champion):
    rows_by_opponent = _aggregates_by_opponent(report)
    checks = []

    simple_rows = rows_by_opponent.get("simple", [])
    simple_passed = bool(simple_rows) and all(
        row.bb_per_100 > config.simple_min_bb100 for row in simple_rows
    )
    simple_detail = f"simple rows must be > {config.simple_min_bb100:.1f} bb/100"
    if simple_rows:
        observed = ", ".join(
            f"{row.players}p {row.bb_per_100:+.1f}" for row in simple_rows
        )
        simple_detail = f"{simple_detail}; observed {observed}"
    checks.append(GateCheck("positive bb/100 vs simple", simple_passed, simple_detail))

    delta_passed = all(row.passed for row in report.comparisons)
    worst_delta = min(
        (row.delta_bb_per_100 for row in report.comparisons),
        default=math.inf,
    )
    worst_delta_str = "n/a" if worst_delta == math.inf else f"{worst_delta:+.1f}"
    checks.append(
        GateCheck(
            "champion regression margin",
            delta_passed,
            (
                f"candidate must stay within {config.min_delta_bb_per_100:.1f} bb/100 "
                f"of {champion}; worst delta {worst_delta_str}"
            ),
        )
    )

    counter_names = set(
        resolve_champion_placeholders(config.counter_strategies, champion)
    )
    counter_rows = [row for row in report.aggregates if row.opponent in counter_names]
    baseline_by_key = {
        (row.opponent, row.players): row.bb_per_100
        for row in report.baseline_aggregates
    }
    deltas = []
    for row in counter_rows:
        baseline_bb = baseline_by_key.get((row.opponent, row.players))
        if baseline_bb is None:
            continue
        deltas.append(row.bb_per_100 - baseline_bb)
    catastrophic_passed = bool(deltas) and all(
        delta >= config.catastrophic_floor_bb100 for delta in deltas
    )
    worst_counter = min(deltas, default=0.0)
    checks.append(
        GateCheck(
            "no catastrophic counter loss",
            catastrophic_passed,
            (
                f"known counters must be >= "
                f"{config.catastrophic_floor_bb100:.1f} bb/100 vs {champion}; "
                f"worst delta {worst_counter:+.1f}"
            ),
        )
    )
    return tuple(checks)


def evaluate_seed_consistency(seed_variance, config):
    if not seed_variance:
        return GateCheck(
            "seed consistency",
            False,
            "no seed-level benchmark data was available",
        )

    weak_rows = [
        row
        for row in seed_variance
        if row.seed_pass_rate < config.min_seed_pass_rate
        or row.delta_ci95_low_bb_per_100 < config.min_delta_bb_per_100
    ]
    worst_rate = min((row.seed_pass_rate for row in seed_variance), default=0.0)
    worst_ci_low = min(
        (row.delta_ci95_low_bb_per_100 for row in seed_variance),
        default=0.0,
    )
    if weak_rows:
        examples = ", ".join(
            f"{row.opponent} {row.players}p {row.seed_passes}/{row.seed_count}"
            for row in weak_rows[:3]
        )
        detail = (
            f"requires at least {config.min_seed_pass_rate:.0%} of seeds per row "
            f"to clear delta {config.min_delta_bb_per_100:.1f} and "
            f"delta CI95 low >= {config.min_delta_bb_per_100:.1f}; "
            f"weak rows: {examples}; "
            f"worst rate {worst_rate:.0%}, worst delta CI95 low {worst_ci_low:+.1f}"
        )
    else:
        detail = (
            f"all rows clear {config.min_seed_pass_rate:.0%} seed pass rate "
            f"and delta CI95 low >= {config.min_delta_bb_per_100:.1f}; "
            f"worst rate {worst_rate:.0%}, worst delta CI95 low {worst_ci_low:+.1f}"
        )
    return GateCheck("seed consistency", not weak_rows, detail)


def strategy_snapshot(name):
    module_name = f"poker_bot.strategies.{name}"
    snapshot = {"module": module_name, "found": False, "path": None, "sha256": None}
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return snapshot
    path = Path(spec.origin)
    snapshot["found"] = path.exists()
    snapshot["path"] = str(path)
    if path.exists() and path.is_file():
        snapshot["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def run_git_command(args):
    try:
        completed = subprocess.run(
            ("git", *args),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return {"ok": False, "error": str(exc), "stdout": "", "stderr": ""}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def build_reproducibility_snapshot(candidate, champion):
    status = run_git_command(("status", "--short"))
    head = run_git_command(("rev-parse", "HEAD"))
    branch = run_git_command(("rev-parse", "--abbrev-ref", "HEAD"))
    status_lines = status["stdout"].splitlines() if status["stdout"] else []
    return {
        "strategies": {
            "candidate": strategy_snapshot(candidate),
            "champion": strategy_snapshot(champion),
        },
        "git": {
            "head": head["stdout"] if head["ok"] else None,
            "branch": branch["stdout"] if branch["ok"] else None,
            "dirty": bool(status_lines),
            "status_short": status_lines,
            "status_error": status["stderr"] if not status["ok"] else None,
        },
    }


def format_attempt_id(generated_at):
    return (
        generated_at.replace("+00:00", "Z")
        .replace(":", "")
        .replace("-", "")
        .replace(".", "")
    )


def _history_row(report):
    summary = _aggregate_summary(report)
    failed_gates = [check.name for check in report.checks if not check.passed]
    return {
        "attempt_id": report.attempt_id,
        "generated_at": report.generated_at,
        "candidate": report.candidate,
        "champion": report.champion,
        "passed": report.passed,
        "promoted": report.promoted,
        "champion_updated": report.champion_updated,
        "failed_gates": failed_gates,
        "output_json": report.output_json,
        "output_markdown": report.output_markdown,
        "hands": summary["hands"],
        "bb_per_100": summary["bb_per_100"],
        "worst_bb_per_100": summary["worst_bb_per_100"],
        "worst_delta_bb_per_100": summary["worst_delta_bb_per_100"],
        "worst_seed_pass_rate": summary["worst_seed_pass_rate"],
        "population_passed": summary["population_passed"],
        "population_worst_row_bb_per_100": summary["population_worst_row_bb_per_100"],
    }


def run_promotion_gate(
    candidate,
    *,
    config_path=DEFAULT_CONFIG,
    champion_json=DEFAULT_CHAMPION_JSON,
    benchmark_runner=run_selfplay,
    scenario_runner=subprocess.run,
    dry_run=False,
    output_json=None,
    output_markdown=None,
    history_index=None,
    generated_at=None,
):
    started = time.perf_counter()
    config = load_promotion_config(config_path)
    champion_metadata = load_champion_metadata(champion_json)
    champion = champion_metadata.get("strategy")
    if not isinstance(champion, str) or not champion:
        raise ValueError(f"{Path(champion_json)} must define a non-empty strategy")
    if generated_at is None:
        generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    attempt_id = format_attempt_id(generated_at)
    reproducibility = build_reproducibility_snapshot(candidate, champion)

    scenario_tests = run_scenario_tests(
        config.scenario_tests,
        command_runner=scenario_runner,
    )
    if not scenario_tests.passed:
        report = PromotionGateReport(
            candidate=candidate,
            champion=champion,
            champion_metadata=champion_metadata,
            config=config,
            scenario_tests=scenario_tests,
            benchmark_report=None,
            population_report=None,
            seed_variance=(),
            checks=(),
            reproducibility=reproducibility,
            elapsed=time.perf_counter() - started,
            generated_at=generated_at,
            attempt_id=attempt_id,
            promoted=False,
            champion_updated=False,
            output_json=str(output_json) if output_json else None,
            output_markdown=str(output_markdown) if output_markdown else None,
            history_index=str(history_index) if history_index else None,
        )
        write_outputs(report, output_json, output_markdown, history_index)
        return report

    opponents = build_opponent_pool(config, champion)
    benchmark_report = benchmark.run_benchmark(
        candidate,
        opponents=opponents,
        players=config.players,
        seeds=config.seeds,
        hands=config.hands,
        track_opponents=config.track_opponents,
        baseline_strat=champion,
        min_delta_bb_per_100=config.min_delta_bb_per_100,
        runner=benchmark_runner,
        workers=config.workers,
    )
    seed_variance = calculate_seed_variance(
        benchmark_report, config.min_delta_bb_per_100
    )
    population_report = None
    population_check = ()
    if config.population_config:
        population_report = population_selfplay.run_population_selfplay(
            candidate=candidate,
            config_path=config.population_config,
            champion_json=champion_json,
            runner=benchmark_runner,
        )
        population_gate = population_selfplay.population_gate_summary(
            population_report,
            candidate,
        )
        population_check = (
            GateCheck(
                "population score",
                population_gate["passed"] is True,
                population_gate["detail"],
            ),
        )
    checks = (
        *evaluate_gate(benchmark_report, config, champion),
        evaluate_seed_consistency(seed_variance, config),
        *population_check,
    )
    passed = all(check.passed for check in checks)

    report = PromotionGateReport(
        candidate=candidate,
        champion=champion,
        champion_metadata=champion_metadata,
        config=config,
        scenario_tests=scenario_tests,
        benchmark_report=benchmark_report,
        population_report=population_report,
        seed_variance=seed_variance,
        checks=checks,
        reproducibility=reproducibility,
        elapsed=time.perf_counter() - started,
        generated_at=generated_at,
        attempt_id=attempt_id,
        promoted=passed and not dry_run,
        champion_updated=False,
        output_json=str(output_json) if output_json else None,
        output_markdown=str(output_markdown) if output_markdown else None,
        history_index=str(history_index) if history_index else None,
    )
    if passed and not dry_run:
        write_json_report(report, output_json)
        write_markdown_report(report, output_markdown)
        report = replace(report, champion_updated=True)
        update_champion_json(
            champion_json,
            report,
            config_path=config_path,
            output_json=output_json,
        )
        write_outputs(report, output_json, output_markdown, history_index)
    else:
        write_outputs(report, output_json, output_markdown, history_index)
    return report


def _aggregate_summary(report):
    if report.benchmark_report is None:
        return {
            "hands": 0,
            "net_chips": 0,
            "bb_per_100": 0.0,
            "worst_bb_per_100": 0.0,
            "worst_delta_bb_per_100": 0.0,
            "worst_seed_pass_rate": 0.0,
            "worst_delta_ci95_low_bb_per_100": 0.0,
            "population_passed": None,
            "population_mean_row_bb_per_100": None,
            "population_worst_row_bb_per_100": None,
            "population_seed_pass_rate": None,
        }
    rows = report.benchmark_report.aggregates
    hands = sum(row.hands for row in rows)
    net_chips = sum(row.net_chips for row in rows)
    bb_per_100 = 0.0
    if hands:
        bb_per_100 = net_chips / BIG_BLIND / hands * 100
    population_score = (
        report.population_report.target_score
        if report.population_report is not None
        else None
    )
    return {
        "hands": hands,
        "net_chips": net_chips,
        "bb_per_100": bb_per_100,
        "worst_bb_per_100": min((row.bb_per_100 for row in rows), default=0.0),
        "worst_delta_bb_per_100": min(
            (row.delta_bb_per_100 for row in report.benchmark_report.comparisons),
            default=0.0,
        ),
        "worst_seed_pass_rate": min(
            (row.seed_pass_rate for row in report.seed_variance),
            default=0.0,
        ),
        "worst_delta_ci95_low_bb_per_100": min(
            (row.delta_ci95_low_bb_per_100 for row in report.seed_variance),
            default=0.0,
        ),
        "population_passed": (
            population_score.passed if population_score is not None else None
        ),
        "population_mean_row_bb_per_100": (
            population_score.mean_row_bb_per_100
            if population_score is not None
            else None
        ),
        "population_worst_row_bb_per_100": (
            population_score.worst_row_bb_per_100
            if population_score is not None
            else None
        ),
        "population_seed_pass_rate": (
            population_score.seed_pass_rate if population_score is not None else None
        ),
    }


def report_to_jsonable(report):
    return {
        "candidate": report.candidate,
        "champion": report.champion,
        "champion_metadata": report.champion_metadata,
        "generated_at": report.generated_at,
        "attempt_id": report.attempt_id,
        "elapsed": report.elapsed,
        "passed": report.passed,
        "promoted": report.promoted,
        "champion_updated": report.champion_updated,
        "output_json": report.output_json,
        "output_markdown": report.output_markdown,
        "history_index": report.history_index,
        "config": asdict(report.config),
        "scenario_tests": asdict(report.scenario_tests),
        "checks": [asdict(check) for check in report.checks],
        "seed_variance": [asdict(row) for row in report.seed_variance],
        "reproducibility": report.reproducibility,
        "summary": _aggregate_summary(report),
        "benchmark": (
            benchmark.report_to_jsonable(report.benchmark_report)
            if report.benchmark_report is not None
            else None
        ),
        "population": (
            population_selfplay.report_to_jsonable(report.population_report)
            if report.population_report is not None
            else None
        ),
    }


def write_json_report(report, path):
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report_to_jsonable(report), f, indent=2)
        f.write("\n")


def write_history_index(report, path):
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        json.dump(_history_row(report), f)
        f.write("\n")


def format_markdown_report(report):
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"# Promotion gate: {report.candidate}",
        "",
        f"Status: {status}.",
        f"Current champion: `{report.champion}`.",
        f"Generated at: {report.generated_at}.",
        f"Attempt ID: `{report.attempt_id}`.",
        "",
        "## Scenario Tests",
        "",
        f"- Command: `{' '.join(report.scenario_tests.command)}`",
        f"- Status: {'PASS' if report.scenario_tests.passed else 'FAIL'}",
        f"- Exit code: {report.scenario_tests.exit_code}",
        f"- Elapsed: {report.scenario_tests.elapsed:.1f}s",
    ]
    if not report.scenario_tests.passed:
        lines.extend(
            [
                "",
                "Benchmark skipped because scenario tests failed.",
            ],
        )
        stdout_tail = report.scenario_tests.stdout[-2000:]
        stderr_tail = report.scenario_tests.stderr[-2000:]
        if stdout_tail:
            lines.extend(
                [
                    "",
                    "**Scenario stdout (last 2000 chars):**",
                    "",
                    "```text",
                    stdout_tail,
                    "```",
                ],
            )
        if stderr_tail:
            lines.extend(
                [
                    "",
                    "**Scenario stderr (last 2000 chars):**",
                    "",
                    "```text",
                    stderr_tail,
                    "```",
                ],
            )
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "## Gates",
            "",
            "| Gate | Status | Detail |",
            "| --- | --- | --- |",
        ]
    )
    for check in report.checks:
        lines.append(
            f"| {check.name} | {'PASS' if check.passed else 'FAIL'} | {check.detail} |"
        )

    if report.seed_variance:
        lines.extend(
            [
                "",
                "## Seed Variance",
                "",
                (
                    "| Opponent | Players | Seeds Passed | Delta Mean | "
                    "Delta Stddev | Delta CI95 | Candidate CI95 |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in report.seed_variance:
            delta_ci = (
                f"{row.delta_ci95_low_bb_per_100:+.1f}.."
                f"{row.delta_ci95_high_bb_per_100:+.1f}"
            )
            candidate_ci = (
                f"{row.candidate_ci95_low_bb_per_100:+.1f}.."
                f"{row.candidate_ci95_high_bb_per_100:+.1f}"
            )
            lines.append(
                f"| {row.opponent} | {row.players} | "
                f"{row.seed_passes}/{row.seed_count} | "
                f"{row.delta_mean_bb_per_100:+.1f} | "
                f"{row.delta_stddev_bb_per_100:.1f} | "
                f"{delta_ci} | {candidate_ci} |"
            )

    candidate_snapshot = report.reproducibility.get("strategies", {}).get(
        "candidate",
        {},
    )
    champion_snapshot = report.reproducibility.get("strategies", {}).get(
        "champion",
        {},
    )
    git_snapshot = report.reproducibility.get("git", {})
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Candidate path: `{candidate_snapshot.get('path')}`",
            f"- Candidate SHA256: `{candidate_snapshot.get('sha256')}`",
            f"- Champion path: `{champion_snapshot.get('path')}`",
            f"- Champion SHA256: `{champion_snapshot.get('sha256')}`",
            f"- Git HEAD: `{git_snapshot.get('head')}`",
            f"- Git dirty: `{git_snapshot.get('dirty')}`",
        ]
    )

    if report.population_report is not None:
        population_gate = population_selfplay.population_gate_summary(
            report.population_report,
            report.candidate,
        )
        lines.extend(
            [
                "",
                "## Population Self-Play",
                "",
                (
                    f"Status: "
                    f"{'PASS' if population_gate['passed'] else 'FAIL'} "
                    f"{population_gate['detail']}"
                ),
                "",
                "```text",
                population_selfplay.format_report(report.population_report),
                "```",
            ]
        )

    if report.benchmark_report is not None:
        lines.extend(
            [
                "",
                "## Benchmark",
                "",
                "```text",
                benchmark.format_report(report.benchmark_report),
                "```",
            ]
        )
    return "\n".join(lines)


def write_markdown_report(report, path):
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_markdown_report(report))


def write_outputs(report, output_json, output_markdown, history_index=None):
    write_json_report(report, output_json)
    write_markdown_report(report, output_markdown)
    write_history_index(report, history_index)


def update_champion_json(path, report, *, config_path, output_json):
    summary = _aggregate_summary(report)
    payload = {
        "strategy": report.candidate,
        "previous_strategy": report.champion,
        "config": str(config_path),
        "report": str(output_json) if output_json else None,
        "promoted_at": report.generated_at,
        "summary": {
            "status": "promoted",
            "reason": "Passed scenario tests and promotion benchmark gate.",
            **summary,
            "min_delta_bb_per_100": report.config.min_delta_bb_per_100,
            "catastrophic_floor_bb_per_100": report.config.catastrophic_floor_bb100,
            "population_config": report.config.population_config,
        },
        "gates": [asdict(check) for check in report.checks],
    }
    champion_path = Path(path)
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = champion_path.with_suffix(".json.tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")
    tmp_path.replace(champion_path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run scenario tests and benchmark gates before promotion."
    )
    parser.add_argument(
        "--strat",
        required=True,
        help="Candidate strategy module under poker_bot.strategies.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Promotion gate JSON config.",
    )
    parser.add_argument(
        "--champion-json",
        default=str(DEFAULT_CHAMPION_JSON),
        help="Champion metadata JSON updated only after a passing gate.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for default JSON and Markdown promotion reports.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    parser.add_argument(
        "--history-index",
        default=None,
        help="Append one JSONL summary row for this attempt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all gates and write reports without updating champion.json.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    attempt_id = format_attempt_id(generated_at)
    history_dir = output_dir / DEFAULT_HISTORY_DIR
    output_json = (
        Path(args.output_json)
        if args.output_json is not None
        else history_dir / f"{args.strat}-{attempt_id}.json"
    )
    output_markdown = (
        Path(args.output_markdown)
        if args.output_markdown is not None
        else history_dir / f"{args.strat}-{attempt_id}.md"
    )
    history_index = (
        Path(args.history_index)
        if args.history_index is not None
        else history_dir / DEFAULT_HISTORY_INDEX
    )
    try:
        report = run_promotion_gate(
            args.strat,
            config_path=args.config,
            champion_json=args.champion_json,
            dry_run=args.dry_run,
            output_json=output_json,
            output_markdown=output_markdown,
            history_index=history_index,
            generated_at=generated_at,
        )
    except ValueError as exc:
        print(f"promotion-gate: {exc}", file=sys.stderr)
        return 2

    print(format_markdown_report(report))
    if not report.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
