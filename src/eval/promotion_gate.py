"""Promotion gate for candidate poker strategies."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from eval import benchmark, population_selfplay
from eval.selfplay import (
    BIG_BLIND,
    INITIAL_STACK,
    PROFILE_ROUTE_DIAGNOSTICS_SCHEMA_VERSION,
    PROFILE_STATE_PERSISTENT,
    PROFILE_STATE_SHARDED_RESEARCH,
    run_selfplay,
)

DEFAULT_CONFIG = Path("benchmarks/promotion_gate.json")
DEFAULT_SMOKE_CONFIG = Path("benchmarks/promotion_gate.smoke.json")
DEFAULT_CHAMPION_JSON = Path("benchmarks/champion.json")
DEFAULT_OUTPUT_DIR = Path("benchmark-runs")
DEFAULT_HISTORY_DIR = Path("promotions")
DEFAULT_HISTORY_INDEX = "index.jsonl"
DEFAULT_SCENARIO_TESTS = ("tests/scenario",)
DEFAULT_PROFILE = "production"
VALID_PROFILES = ("production", "smoke")
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
# Sequential paired evaluation is OFF by default (single run at `hands`);
# it activates only when a config sets `target_delta_ci95_half_width_bb100`.
# 5.0 bb/100 is the documented default half-width target for production:
# it is coarser than the -2.5 bb/100 non-inferiority floor (so precision
# never masks a real regression signal at that floor) while remaining
# achievable within a bounded number of additional hands given this
# simulator's typical per-seed bb/100 variance. Deployments with tighter
# variance or a larger hand budget can lower it per-config.
DEFAULT_DELTA_CI95_HALF_WIDTH_BB100 = 5.0


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
    candidate_stddev_bb_per_100: float | None
    candidate_stderr_bb_per_100: float | None
    candidate_ci95_low_bb_per_100: float | None
    candidate_ci95_high_bb_per_100: float | None
    delta_mean_bb_per_100: float
    delta_stddev_bb_per_100: float | None
    delta_stderr_bb_per_100: float | None
    delta_ci95_low_bb_per_100: float | None
    delta_ci95_high_bb_per_100: float | None
    seed_passes: int
    seed_count: int
    seed_pass_rate: float
    initial_stack: int = INITIAL_STACK


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
    initial_stacks: tuple[int, ...] = (INITIAL_STACK,)
    workers: int = 1
    profile: str = "production"
    regime_overrides: dict = field(default_factory=dict)
    max_hands: int | None = None
    batch_hands: int | None = None
    target_delta_ci95_half_width_bb100: float | None = None
    profile_state_mode: str = PROFILE_STATE_PERSISTENT


    def __post_init__(self):
        if self.profile_state_mode not in {
            PROFILE_STATE_PERSISTENT,
            PROFILE_STATE_SHARDED_RESEARCH,
        }:
            raise ValueError(
                "profile_state_mode must be 'persistent' or 'sharded_research'"
            )
        if self.profile == "production" and (
            self.profile_state_mode != PROFILE_STATE_PERSISTENT
        ):
            raise ValueError(
                "production promotion requires profile_state_mode='persistent'"
            )
        if self.target_delta_ci95_half_width_bb100 is not None:
            if self.target_delta_ci95_half_width_bb100 <= 0:
                raise ValueError(
                    "target_delta_ci95_half_width_bb100 must be positive"
                )
            if self.max_hands is None:
                raise ValueError(
                    "max_hands is required when "
                    "target_delta_ci95_half_width_bb100 is set, to bound "
                    "sequential sampling"
                )
            if self.max_hands < self.hands:
                raise ValueError("max_hands must be >= hands")
            if self.batch_hands is not None and self.batch_hands <= 0:
                raise ValueError("batch_hands must be positive")


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
    model_checkpoint: str | None = None
    aggregate_variance: AggregateVariance | None = None
    evaluated_hands: int = 0
    evaluation_stages: int = 0
    sequential_precision_achieved: bool | None = None

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

    profile = str(data.get("profile", DEFAULT_PROFILE))
    if profile not in VALID_PROFILES:
        raise ValueError(
            f"promotion gate config profile must be one of {VALID_PROFILES}, "
            f"got {profile!r}"
        )
    regime_overrides_raw = data.get("regime_overrides") or {}
    if not isinstance(regime_overrides_raw, dict):
        raise ValueError(
            "regime_overrides must be a JSON object keyed by player count"
        )
    regime_overrides = {
        int(players): dict(overrides)
        for players, overrides in regime_overrides_raw.items()
    }

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
        initial_stacks=_csv_ints_or_tuple(data.get("initial_stacks"), (INITIAL_STACK,)),
        workers=int(data.get("workers", 1)),
        profile=profile,
        profile_state_mode=str(
            data.get("profile_state_mode", PROFILE_STATE_PERSISTENT)
        ),
        regime_overrides=regime_overrides,
        max_hands=(
            int(data["max_hands"]) if data.get("max_hands") is not None else None
        ),
        batch_hands=(
            int(data["batch_hands"]) if data.get("batch_hands") is not None else None
        ),
        target_delta_ci95_half_width_bb100=(
            float(data["target_delta_ci95_half_width_bb100"])
            if data.get("target_delta_ci95_half_width_bb100") is not None
            else None
        ),
    )


def resolve_regime_floor(config, players, key):
    """Look up a per-player-count override for ``key`` (one of
    ``min_delta_bb_per_100`` or ``catastrophic_floor_bb100``), falling back
    to the config-wide default.

    Lets one production gate enforce a tighter non-inferiority /
    catastrophic floor for thin regimes (HU, 3-handed) than for 6-max, so
    a large 6-max win can never offset a collapse elsewhere.
    """
    override = config.regime_overrides.get(players)
    if override and key in override:
        return float(override[key])
    return getattr(config, key)


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


def _regularized_incomplete_beta(x, a, b):
    """Return I_x(a, b) using a continued fraction."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    fraction = d
    for iteration in range(1, 201):
        twice_iteration = 2 * iteration
        coefficient = iteration * (b - iteration) * x / (
            (qam + twice_iteration) * (a + twice_iteration)
        )
        d = 1.0 + coefficient * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + coefficient / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        fraction *= d * c
        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + twice_iteration) * (qap + twice_iteration))
        )
        d = 1.0 + coefficient * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + coefficient / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= 3e-14:
            break
    return front * fraction / a


def _student_t_cdf(value, degrees_of_freedom):
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * _regularized_incomplete_beta(
        x, degrees_of_freedom / 2.0, 0.5
    )
    return 1.0 - tail if value > 0.0 else tail


def _student_t_critical_95(degrees_of_freedom):
    """Return the two-sided 95% Student-t critical value for a positive df."""

    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive")
    lower, upper = 0.0, 1.0
    while _student_t_cdf(upper, degrees_of_freedom) < 0.975:
        upper *= 2.0
    for _ in range(60):
        midpoint = (lower + upper) / 2.0
        if _student_t_cdf(midpoint, degrees_of_freedom) < 0.975:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _series_stats(values):
    mean = _mean(values)
    if len(values) < 2:
        return mean, None, None, None, None
    stddev = _sample_stddev(values)
    stderr = stddev / math.sqrt(len(values))
    ci95 = _student_t_critical_95(len(values) - 1) * stderr
    return mean, stddev, stderr, mean - ci95, mean + ci95


def calculate_seed_variance(report, min_delta_fn):
    baseline_by_case = {
        (result.opponent, result.players, result.initial_stack, case.seed): result
        for case, result in zip(
            report.cases,
            report.baseline_results,
            strict=True,
        )
    }
    grouped = {}
    for case, result in zip(report.cases, report.results, strict=True):
        baseline = baseline_by_case[
            (result.opponent, result.players, result.initial_stack, case.seed)
        ]
        key = (result.opponent, result.players, result.initial_stack)
        grouped.setdefault(key, []).append((case.seed, result, baseline))

    rows = []
    for (opponent, players, initial_stack), group in sorted(grouped.items()):
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
        seed_passes = sum(1 for value in delta_values if value >= min_delta_fn(players))
        seed_count = len(delta_values)
        rows.append(
            SeedVariance(
                opponent=opponent,
                players=players,
                initial_stack=initial_stack,
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


@dataclass(frozen=True)
class AggregateVariance:
    seed_count: int
    delta_mean_bb_per_100: float
    delta_stddev_bb_per_100: float | None
    delta_stderr_bb_per_100: float | None
    delta_ci95_low_bb_per_100: float | None
    delta_ci95_high_bb_per_100: float | None


def calculate_aggregate_variance(seed_variance):
    """Pool every (opponent, players, seed) paired delta into one overall
    95% CI, alongside the per-row breakdown already in ``seed_variance``.
    This is the "aggregate" half of "paired deltas and 95% confidence
    intervals, both aggregate and per player count/opponent."
    """
    all_deltas = [delta for row in seed_variance for delta in row.delta_bb_per_100]
    mean, stddev, stderr, low, high = _series_stats(all_deltas)
    return AggregateVariance(
        seed_count=len(all_deltas),
        delta_mean_bb_per_100=mean,
        delta_stddev_bb_per_100=stddev,
        delta_stderr_bb_per_100=stderr,
        delta_ci95_low_bb_per_100=low,
        delta_ci95_high_bb_per_100=high,
    )


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
            f"{row.players}p@{row.initial_stack} {row.bb_per_100:+.1f}"
            for row in simple_rows
        )
        simple_detail = f"{simple_detail}; observed {observed}"
    checks.append(GateCheck("positive bb/100 vs simple", simple_passed, simple_detail))

    delta_margins = [
        (
            row,
            resolve_regime_floor(config, row.players, "min_delta_bb_per_100"),
            row.delta_bb_per_100,
            row.delta_bb_per_100
            - resolve_regime_floor(config, row.players, "min_delta_bb_per_100"),
        )
        for row in report.comparisons
    ]
    delta_passed = bool(delta_margins) and all(
        margin >= 0 for _row, _floor, _delta, margin in delta_margins
    )
    if delta_margins:
        worst_row, worst_floor, worst_delta, _worst_margin = min(
            delta_margins, key=lambda item: item[3]
        )
        delta_detail = (
            "candidate paired delta must clear each regime's non-inferiority "
            f"floor (default {config.min_delta_bb_per_100:.1f}); worst "
            f"{worst_row.players}p@{worst_row.initial_stack} "
            f"{worst_row.opponent} {worst_delta:+.1f} vs {worst_floor:+.1f}"
        )
    else:
        delta_detail = "no paired candidate/champion comparisons were available"
    checks.append(
        GateCheck("champion regression margin", delta_passed, delta_detail)
    )
    counter_names = set(
        resolve_champion_placeholders(config.counter_strategies, champion)
    )
    counter_rows = [row for row in report.aggregates if row.opponent in counter_names]
    baseline_by_key = {
        (row.opponent, row.players, row.initial_stack): row.bb_per_100
        for row in report.baseline_aggregates
    }
    counter_margins = []
    for row in counter_rows:
        baseline_bb = baseline_by_key.get(
            (row.opponent, row.players, row.initial_stack)
        )
        if baseline_bb is None:
            continue
        floor = resolve_regime_floor(config, row.players, "catastrophic_floor_bb100")
        delta = row.bb_per_100 - baseline_bb
        counter_margins.append((row, floor, delta, delta - floor))
    catastrophic_passed = bool(counter_margins) and all(
        margin >= 0 for _row, _floor, _delta, margin in counter_margins
    )
    worst_row, worst_floor, worst_delta, _worst_margin = min(
        counter_margins,
        key=lambda item: item[3],
        default=(None, config.catastrophic_floor_bb100, 0.0, 0.0),
    )
    worst_counter_str = (
        "n/a"
        if worst_row is None
        else (
            f"{worst_delta:+.1f} (floor {worst_floor:+.1f}, "
            f"{worst_row.players}p@{worst_row.initial_stack} {worst_row.opponent})"
        )
    )
    checks.append(
        GateCheck(
            "no catastrophic counter loss",
            catastrophic_passed,
            (
                f"known counters must clear each regime's configured floor "
                f"(default {config.catastrophic_floor_bb100:.1f} bb/100) vs "
                f"{champion}; worst delta {worst_counter_str}"
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
        or row.delta_ci95_low_bb_per_100 is None
        or row.delta_ci95_low_bb_per_100
        < resolve_regime_floor(config, row.players, "min_delta_bb_per_100")
    ]
    worst_rate = min((row.seed_pass_rate for row in seed_variance), default=0.0)
    ci_lows = [
        row.delta_ci95_low_bb_per_100
        for row in seed_variance
        if row.delta_ci95_low_bb_per_100 is not None
    ]
    worst_ci_low = min(ci_lows, default=None)
    if weak_rows:
        examples = ", ".join(
            f"{row.opponent} {row.players}p@{row.initial_stack} "
            f"{row.seed_passes}/{row.seed_count}"
            for row in weak_rows[:3]
        )
        detail = (
            f"requires at least {config.min_seed_pass_rate:.0%} of seeds per row "
            "and a finite Student-t paired-delta CI95 lower bound against each "
            f"regime's floor (default {config.min_delta_bb_per_100:.1f}); "
            f"weak rows: {examples}; worst rate {worst_rate:.0%}"
        )
        if worst_ci_low is None:
            detail += "; insufficient paired samples for CI95"
        else:
            detail += f", worst delta CI95 low {worst_ci_low:+.1f}"
    else:
        detail = (
            f"all rows clear {config.min_seed_pass_rate:.0%} seed pass rate "
            "and each regime's Student-t delta CI95 lower bound "
            f"(default {config.min_delta_bb_per_100:.1f}); "
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
    snapshot["found"] = True
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


@contextlib.contextmanager
def _env_override(key: str, value: str):
    """Temporarily override an os.environ key inside a with block."""
    prev = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def build_reproducibility_snapshot(
    candidate,
    champion,
    *,
    candidate_checkpoint=None,
    champion_checkpoint=None,
):
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
        "profile": report.config.profile,
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
    nn_mode=None,
    candidate_checkpoint=None,
    champion_checkpoint=None,
    workers=None,
):
    started = time.perf_counter()
    config = load_promotion_config(config_path)
    # Every logical (strategy, opponent, players, stack, seed) benchmark
    # case now runs against its own fresh, isolated opponent-profile DB
    # (see eval.benchmark._case_db_paths) regardless of worker count, so
    # profile-dependent production promotions no longer need to force
    # sequential (workers=1) execution to avoid cross-case contamination.
    actual_workers = workers if workers is not None else config.workers
    champion_metadata = load_champion_metadata(champion_json)
    champion = champion_metadata.get("strategy")
    if not isinstance(champion, str) or not champion:
        raise ValueError(f"{Path(champion_json)} must define a non-empty strategy")
    if generated_at is None:
        generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    attempt_id = format_attempt_id(generated_at)
    reproducibility = build_reproducibility_snapshot(
        candidate,
        champion,
        candidate_checkpoint=candidate_checkpoint,
        champion_checkpoint=champion_checkpoint,
    )

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
            model_checkpoint=candidate_checkpoint,
        )
        write_outputs(report, output_json, output_markdown, history_index)
        return report

    # For NN strategies, set NN_MODE so the arbiter activates the model.
    nn_mode_ctx = contextlib.nullcontext()
    if nn_mode:
        nn_mode_ctx = _env_override("NN_MODE", nn_mode)

    # For NN strategies, also set policy/value checkpoint paths for
    # candidate and baseline. Rebuilt fresh per stage below via
    # _candidate_ctxs()/_baseline_ctxs() -- see their docstring in
    # _run_stage for why (env-override context managers are one-shot).

    opponents = build_opponent_pool(config, champion)
    sequential_enabled = config.target_delta_ci95_half_width_bb100 is not None

    def _candidate_ctxs():
        policy_ctx = contextlib.nullcontext()
        value_ctx = contextlib.nullcontext()
        if candidate_checkpoint:
            policy_ctx = _env_override("NN_POLICY_PATH", str(candidate_checkpoint))
            value_candidate = str(
                Path(candidate_checkpoint).with_name(
                    Path(candidate_checkpoint).stem.replace("policy_", "value_")
                    + Path(candidate_checkpoint).suffix
                )
            )
            if Path(value_candidate).exists():
                value_ctx = _env_override("NN_VALUE_PATH", value_candidate)
        return policy_ctx, value_ctx

    def _baseline_ctxs():
        policy_ctx = contextlib.nullcontext()
        value_ctx = contextlib.nullcontext()
        if champion_checkpoint:
            policy_ctx = _env_override("NN_POLICY_PATH", str(champion_checkpoint))
            value_champion = str(
                Path(champion_checkpoint).with_name(
                    Path(champion_checkpoint).stem.replace("policy_", "value_")
                    + Path(champion_checkpoint).suffix
                )
            )
            if Path(value_champion).exists():
                value_ctx = _env_override("NN_VALUE_PATH", value_champion)
        return policy_ctx, value_ctx

    def _run_stage(hands_this_stage):
        """Run one paired candidate-vs-baseline benchmark stage at the
        given cumulative (not incremental) hand count per case.

        Rerunning from hand 0 each stage -- rather than incrementally
        appending new hands -- is deterministic and preserves exact
        paired deal streams: both strategies share the same per-(opponent,
        players, seed) RNG seed, and the deck consumes exactly one shuffle
        per hand regardless of policy (see the deal-stream pairing tests
        in tests/test_selfplay.py), so hands 0..N-1 of a run at hands=M>N
        are byte-identical to a run at hands=N. Context managers are
        rebuilt fresh each call: `_env_override` is a one-shot generator
        context manager and cannot be re-entered, and the arbiter cache is
        cleared before every run regardless of stage.
        """
        candidate_policy_ctx, candidate_value_ctx = _candidate_ctxs()
        with candidate_policy_ctx, candidate_value_ctx:
            from poker_bot.strategies.nnbase import _clear_arbiter_cache

            _clear_arbiter_cache()
            stage_candidate_report = benchmark.run_benchmark(
                candidate,
                opponents=opponents,
                players=config.players,
                seeds=config.seeds,
                hands=hands_this_stage,
                initial_stacks=config.initial_stacks,
                track_opponents=config.track_opponents,
                baseline_strat=None,  # no baseline for candidate-only run
                min_delta_bb_per_100=config.min_delta_bb_per_100,
                runner=benchmark_runner,
                workers=actual_workers,
            )

        baseline_policy_ctx, baseline_value_ctx = _baseline_ctxs()
        with baseline_policy_ctx, baseline_value_ctx:
            from poker_bot.strategies.nnbase import _clear_arbiter_cache

            _clear_arbiter_cache()
            stage_baseline_report = benchmark.run_benchmark(
                champion,
                opponents=opponents,
                players=config.players,
                seeds=config.seeds,
                hands=hands_this_stage,
                initial_stacks=config.initial_stacks,
                track_opponents=config.track_opponents,
                baseline_strat=None,
                min_delta_bb_per_100=config.min_delta_bb_per_100,
                runner=benchmark_runner,
                workers=actual_workers,
            )
        return stage_candidate_report, stage_baseline_report

    cumulative_hands = config.hands
    stage_count = 0
    precision_achieved = not sequential_enabled
    with nn_mode_ctx:
        while True:
            candidate_report, baseline_report = _run_stage(cumulative_hands)
            stage_count += 1
            seed_variance = calculate_seed_variance(
                replace(candidate_report, baseline_results=baseline_report.results),
                lambda players: resolve_regime_floor(
                    config, players, "min_delta_bb_per_100"
                ),
            )
            if not sequential_enabled:
                break
            half_widths = [
                (row.delta_ci95_high_bb_per_100 - row.delta_ci95_low_bb_per_100)
                / 2
                for row in seed_variance
                if row.delta_ci95_low_bb_per_100 is not None
                and row.delta_ci95_high_bb_per_100 is not None
            ]
            precision_achieved = (
                len(half_widths) == len(seed_variance)
                and bool(half_widths)
                and all(
                    width <= config.target_delta_ci95_half_width_bb100
                    for width in half_widths
                )
            )
            if precision_achieved or cumulative_hands >= config.max_hands:
                break
            batch = config.batch_hands or config.hands
            cumulative_hands = min(cumulative_hands + batch, config.max_hands)

    # Combine candidate and baseline reports for gate evaluation. Each
    # benchmark.run_benchmark() call above was run with baseline_strat=None
    # (candidate and baseline strategies are run as two separate,
    # single-strategy benchmark passes so NN checkpoint env vars can be
    # swapped between them), so each report's own `.comparisons` is empty.
    # The real per-(opponent, players) comparisons must be built here by
    # matching the two reports' aggregates -- concatenating the (empty)
    # per-report comparisons would make every gate that reads
    # `report.comparisons` vacuously pass.
    aggregate_variance = calculate_aggregate_variance(seed_variance)
    comparisons = benchmark.compare_aggregates(
        candidate_report.aggregates,
        baseline_report.aggregates,
        config.min_delta_bb_per_100,
    )
    benchmark_report = replace(
        candidate_report,
        baseline_results=baseline_report.results,
        aggregates=candidate_report.aggregates,
        baseline_aggregates=baseline_report.aggregates,
        comparisons=comparisons,
        elapsed=candidate_report.elapsed + baseline_report.elapsed,
        baseline_strat=champion,
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
    sequential_check = ()
    if sequential_enabled:
        sequential_check = (
            GateCheck(
                "sequential precision",
                precision_achieved,
                (
                    f"paired-delta CI95 half-width must be <= "
                    f"{config.target_delta_ci95_half_width_bb100:.2f} bb/100 on "
                    f"every row within max_hands={config.max_hands}; evaluated "
                    f"{cumulative_hands} hands/case across {stage_count} stage(s)"
                    + (
                        ""
                        if precision_achieved
                        else " -- inconclusive, never promote on unresolved noise"
                    )
                ),
            ),
        )
    checks = (
        *evaluate_gate(benchmark_report, config, champion),
        evaluate_seed_consistency(seed_variance, config),
        *sequential_check,
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
        aggregate_variance=aggregate_variance,
        checks=checks,
        reproducibility=reproducibility,
        elapsed=time.perf_counter() - started,
        generated_at=generated_at,
        attempt_id=attempt_id,
        promoted=passed and not dry_run and config.profile == "production",
        champion_updated=False,
        output_json=str(output_json) if output_json else None,
        output_markdown=str(output_markdown) if output_markdown else None,
        history_index=str(history_index) if history_index else None,
        evaluated_hands=cumulative_hands,
        evaluation_stages=stage_count,
        sequential_precision_achieved=(
            precision_achieved if sequential_enabled else None
        ),
    )
    if passed and not dry_run and config.profile == "production":
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
            (
                row.delta_ci95_low_bb_per_100
                for row in report.seed_variance
                if row.delta_ci95_low_bb_per_100 is not None
            ),
            default=None,
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


def _route_diagnostics_summary(report):
    """Expose evaluator-only route observations without changing gate checks."""

    if report.benchmark_report is None:
        return {
            "schema_version": PROFILE_ROUTE_DIAGNOSTICS_SCHEMA_VERSION,
            "candidate": [],
            "baseline": [],
        }

    def _rows(rows):
        return [
            {
                "opponent": row.opponent,
                "players": row.players,
                "initial_stack": row.initial_stack,
                **asdict(row.route_diagnostics),
                "activation_fraction": row.route_diagnostics.activation_fraction,
            }
            for row in rows
        ]

    diagnostics = _rows(report.benchmark_report.aggregates)
    baseline_diagnostics = _rows(report.benchmark_report.baseline_aggregates)
    return {
        "schema_version": max(
            (
                item["schema_version"]
                for item in [*diagnostics, *baseline_diagnostics]
            ),
            default=PROFILE_ROUTE_DIAGNOSTICS_SCHEMA_VERSION,
        ),
        "candidate": diagnostics,
        "baseline": baseline_diagnostics,
    }


def report_to_jsonable(report):
    return {
        "candidate": report.candidate,
        "champion": report.champion,
        "champion_metadata": report.champion_metadata,
        "profile": report.config.profile,
        "promotion_quality": report.config.profile == "production",
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
        "aggregate_variance": (
            asdict(report.aggregate_variance)
            if report.aggregate_variance is not None
            else None
        ),
        "evaluated_hands": report.evaluated_hands,
        "evaluation_stages": report.evaluation_stages,
        "sequential_precision_achieved": report.sequential_precision_achieved,
        "target_delta_ci95_half_width_bb100": (
            report.config.target_delta_ci95_half_width_bb100
        ),
        "reproducibility": report.reproducibility,
        "summary": _aggregate_summary(report),
        "route_diagnostics": _route_diagnostics_summary(report),
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


def _evaluated_hands_summary(report):
    base = (
        f"Evaluated hands: {report.evaluated_hands} per case across "
        f"{report.evaluation_stages} stage(s)"
    )
    if report.sequential_precision_achieved is None:
        return base + "."
    target = report.config.target_delta_ci95_half_width_bb100
    status = "achieved" if report.sequential_precision_achieved else "NOT achieved"
    suffix = "" if report.sequential_precision_achieved else " -- inconclusive"
    return (
        f"{base} (target paired-delta CI95 half-width: {target:.2f} bb/100, "
        f"{status}{suffix})."
    )


def format_markdown_report(report):
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"# Promotion gate: {report.candidate}",
        "",
        f"Status: {status}.",
        f"Current champion: `{report.champion}`.",
        f"Generated at: {report.generated_at}.",
        f"Attempt ID: `{report.attempt_id}`.",
        f"Profile: `{report.config.profile}`.",
        _evaluated_hands_summary(report),
    ]
    if report.config.profile != "production":
        lines.append(
            "\n> ⚠️ **SMOKE PROFILE** -- this run is NOT promotion-quality and "
            "never updates `champion.json`, regardless of PASS/FAIL. Use the "
            "production config for an authoritative promotion decision."
        )
    lines.append("")
    lines.extend(
        [
            "## Scenario Tests",
            "",
            f"- Command: `{' '.join(report.scenario_tests.command)}`",
            f"- Status: {'PASS' if report.scenario_tests.passed else 'FAIL'}",
            f"- Exit code: {report.scenario_tests.exit_code}",
            f"- Elapsed: {report.scenario_tests.elapsed:.1f}s",
        ]
    )
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
        seed_variance_lines = [
            "",
            "## Seed Variance",
            "",
        ]
        if report.aggregate_variance is not None:
            agg = report.aggregate_variance
            aggregate_ci = (
                f"[{agg.delta_ci95_low_bb_per_100:+.1f}, "
                f"{agg.delta_ci95_high_bb_per_100:+.1f}]"
                if agg.delta_ci95_low_bb_per_100 is not None
                and agg.delta_ci95_high_bb_per_100 is not None
                else "insufficient paired samples"
            )
            seed_variance_lines.append(
                f"**Aggregate (all {agg.seed_count} paired opponent/player/seed "
                f"deltas):** mean {agg.delta_mean_bb_per_100:+.1f} bb/100, "
                f"95% CI {aggregate_ci}"
            )
            seed_variance_lines.append("")
        seed_variance_lines.extend(
            [
                (
                    "| Opponent | Players | Stack | Seeds Passed | Delta Mean | "
                    "Delta Stddev | Delta CI95 | Candidate CI95 |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        lines.extend(seed_variance_lines)
        for row in report.seed_variance:
            delta_ci = (
                f"{row.delta_ci95_low_bb_per_100:+.1f}.."
                f"{row.delta_ci95_high_bb_per_100:+.1f}"
                if row.delta_ci95_low_bb_per_100 is not None
                and row.delta_ci95_high_bb_per_100 is not None
                else "insufficient"
            )
            candidate_ci = (
                f"{row.candidate_ci95_low_bb_per_100:+.1f}.."
                f"{row.candidate_ci95_high_bb_per_100:+.1f}"
                if row.candidate_ci95_low_bb_per_100 is not None
                and row.candidate_ci95_high_bb_per_100 is not None
                else "insufficient"
            )
            delta_stddev = (
                f"{row.delta_stddev_bb_per_100:.1f}"
                if row.delta_stddev_bb_per_100 is not None
                else "insufficient"
            )
            lines.append(
                f"| {row.opponent} | {row.players} | {row.initial_stack} | "
                f"{row.seed_passes}/{row.seed_count} | "
                f"{row.delta_mean_bb_per_100:+.1f} | {delta_stddev} | "
                f"{delta_ci} | {candidate_ci} |"
            )

    route_diagnostics = _route_diagnostics_summary(report)
    if route_diagnostics["candidate"] or route_diagnostics["baseline"]:
        lines.extend(
            [
                "",
                "## Route Diagnostics",
                "",
                (
                    f"Evaluator schema v{route_diagnostics['schema_version']}; "
                    "observational only and excluded from promotion checks."
                ),
                "",
                (
                    "| Role | Opponent | Players | Stack | Profile state | "
                    "Alternate Hands / Observed | Activation | "
                    "Decisions (alt/fallback/unknown) |"
                ),
                "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
            ]
        )
        for role in ("candidate", "baseline"):
            for row in route_diagnostics[role]:
                lines.append(
                    f"| {role} | {row['opponent']} | {row['players']} | "
                    f"{row['initial_stack']} | {row['profile_state_mode']} | "
                    f"{row['alternate_hands']}/{row['observed_hands']} | "
                    f"{row['activation_fraction']:.1%} | "
                    f"{row['alternate_decisions']}/{row['fallback_decisions']}/"
                    f"{row['unknown_decisions']} |"
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
        default=None,
        help=(
            "Promotion gate JSON config. Defaults to the production gate "
            f"({DEFAULT_CONFIG}), or the smoke config with --smoke."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use the fast smoke-check config "
            f"({DEFAULT_SMOKE_CONFIG}) instead of the production gate. "
            "Output is NOT promotion-quality: champion.json is never "
            "updated, regardless of PASS/FAIL."
        ),
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
    parser.add_argument(
        "--nn-mode",
        default=None,
        help="NN_MODE env var to set during benchmark (e.g. '6max_active'). "
        "Required when evaluating an nnbase-like strategy.",
    )
    parser.add_argument(
        "--model-checkpoint",
        default=None,
        help="Path to candidate .pt checkpoint. Stored in reproducibility snapshot.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of workers for self-play (0 = auto, default: %(default)s)",
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
    config_path = (
        Path(args.config)
        if args.config is not None
        else (DEFAULT_SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG)
    )
    try:
        report = run_promotion_gate(
            args.strat,
            config_path=config_path,
            champion_json=args.champion_json,
            dry_run=args.dry_run,
            output_json=output_json,
            output_markdown=output_markdown,
            history_index=history_index,
            generated_at=generated_at,
            nn_mode=args.nn_mode,
            candidate_checkpoint=args.model_checkpoint,
            workers=args.workers,
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
