"""Run population-based self-play matrices for poker strategies."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from eval import benchmark
from eval.selfplay import BIG_BLIND, SelfPlayResult, run_selfplay

DEFAULT_CONFIG = Path("benchmarks/population_selfplay.json")
DEFAULT_CHAMPION_JSON = Path("benchmarks/champion.json")
DEFAULT_OUTPUT_DIR = Path("benchmark-runs/population")
DEFAULT_HANDS = 2000
DEFAULT_PLAYERS = (6,)
DEFAULT_SEEDS = (1, 2, 3)
DEFAULT_STRATEGIES = (
    "{candidate}",
    "{champion}",
    "auto_research",
    "auto_research_v001",
    "auto_research_v002",
    "auto_research_v003",
)
DEFAULT_OPPONENTS = (
    "simple",
    "adaptive",
    "counter_adaptive",
    "threshold_pressure",
    "anti_threshold",
    "profiled_counter_adaptive",
    "{champion}",
    "simple+adaptive+counter_adaptive+threshold_pressure+anti_threshold",
    (
        "counter_adaptive+threshold_pressure+anti_threshold"
        "+profiled_counter_adaptive+{champion}"
    ),
)
DEFAULT_MIN_MEAN_BB100 = 0.0
DEFAULT_MIN_WORST_BB100 = -25.0
DEFAULT_MIN_SEED_PASS_RATE = 0.5


@dataclass(frozen=True)
class PopulationConfig:
    hands: int
    strategies: tuple[str, ...]
    opponents: tuple[str, ...]
    players: tuple[int, ...]
    seeds: tuple[int, ...]
    track_opponents: bool
    score_strategy: str | None
    min_mean_bb100: float
    min_worst_bb100: float
    min_seed_bb100: float
    min_seed_pass_rate: float


@dataclass(frozen=True)
class PopulationCase:
    strat: str
    opponent: str
    players: int
    seed: int
    hands: int


@dataclass(frozen=True)
class PopulationAggregate:
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
class PopulationScore:
    strat: str
    hands: int
    net_chips: int
    bb_per_100: float
    mean_row_bb_per_100: float
    worst_row_bb_per_100: float
    best_row_bb_per_100: float
    seed_passes: int
    seed_count: int
    seed_pass_rate: float
    counter_exposure_bb100: float
    min_mean_bb100: float
    min_worst_bb100: float
    min_seed_pass_rate: float

    @property
    def passed(self):
        return (
            self.mean_row_bb_per_100 >= self.min_mean_bb100
            and self.worst_row_bb_per_100 >= self.min_worst_bb100
            and self.seed_pass_rate >= self.min_seed_pass_rate
        )


@dataclass(frozen=True)
class PopulationReport:
    candidate: str | None
    champion: str
    config: PopulationConfig
    cases: tuple[PopulationCase, ...]
    results: tuple[SelfPlayResult, ...]
    aggregates: tuple[PopulationAggregate, ...]
    scores: tuple[PopulationScore, ...]
    elapsed: float
    output_json: str | None = None
    output_markdown: str | None = None

    @property
    def target_score(self):
        if self.config.score_strategy is None:
            return None
        for score in self.scores:
            if score.strat == self.config.score_strategy:
                return score
        return None

    @property
    def passed(self):
        score = self.target_score
        if score is None:
            return None
        return score.passed


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


def _dedupe(values):
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return tuple(output)


def load_champion_strategy(path=DEFAULT_CHAMPION_JSON):
    champion_path = Path(path)
    with champion_path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{champion_path} must contain a JSON object")
    strategy = payload.get("strategy")
    if not isinstance(strategy, str) or not strategy:
        raise ValueError(f"{champion_path} must define a non-empty strategy")
    return strategy


def resolve_placeholders(values, *, candidate, champion, field_name):
    resolved = []
    for value in values:
        if "{candidate}" in value:
            if candidate is None:
                raise ValueError(
                    f"{field_name} uses {{candidate}} but no candidate was provided"
                )
            value = value.replace("{candidate}", candidate)
        value = value.replace("{champion}", champion)
        resolved.append(value)
    return _dedupe(resolved)


def load_population_config(
    path=DEFAULT_CONFIG,
    *,
    candidate=None,
    champion_json=DEFAULT_CHAMPION_JSON,
    overrides=None,
):
    config_path = Path(path)
    with config_path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("population config must be a JSON object")
    if overrides:
        data = {
            **data,
            **{key: value for key, value in overrides.items() if value is not None},
        }

    champion = load_champion_strategy(champion_json)
    strategies = resolve_placeholders(
        _csv_or_tuple(data.get("strategies"), DEFAULT_STRATEGIES),
        candidate=candidate,
        champion=champion,
        field_name="strategies",
    )
    opponents = resolve_placeholders(
        _csv_or_tuple(data.get("opponents"), DEFAULT_OPPONENTS),
        candidate=candidate,
        champion=champion,
        field_name="opponents",
    )
    score_strategy = data.get("score_strategy", "{candidate}" if candidate else None)
    if isinstance(score_strategy, str):
        score_strategy = resolve_placeholders(
            (score_strategy,),
            candidate=candidate,
            champion=champion,
            field_name="score_strategy",
        )[0]

    min_worst_bb100 = float(data.get("min_worst_bb100", DEFAULT_MIN_WORST_BB100))
    return PopulationConfig(
        hands=int(data.get("hands", DEFAULT_HANDS)),
        strategies=strategies,
        opponents=opponents,
        players=_csv_ints_or_tuple(data.get("players"), DEFAULT_PLAYERS),
        seeds=_csv_ints_or_tuple(data.get("seeds"), DEFAULT_SEEDS),
        track_opponents=bool(data.get("track_opponents", True)),
        score_strategy=score_strategy,
        min_mean_bb100=float(data.get("min_mean_bb100", DEFAULT_MIN_MEAN_BB100)),
        min_worst_bb100=min_worst_bb100,
        min_seed_bb100=float(data.get("min_seed_bb100", min_worst_bb100)),
        min_seed_pass_rate=float(
            data.get("min_seed_pass_rate", DEFAULT_MIN_SEED_PASS_RATE)
        ),
    )


def build_cases(config):
    if config.hands < 0:
        raise ValueError("hands must be non-negative")
    cases = []
    for strat in config.strategies:
        for opponent in config.opponents:
            for player_count in config.players:
                if player_count < 2 or player_count > 6:
                    raise ValueError("players values must be between 2 and 6")
                for seed in config.seeds:
                    cases.append(
                        PopulationCase(
                            strat=strat,
                            opponent=opponent,
                            players=player_count,
                            seed=seed,
                            hands=config.hands,
                        )
                    )
    return tuple(cases)


def aggregate_results(cases, results):
    grouped = {}
    for case, result in zip(cases, results, strict=True):
        key = (result.strat, result.opponent, result.players)
        grouped.setdefault(key, []).append((case, result))

    rows = []
    for (strat, opponent, players), group in sorted(grouped.items()):
        rows.append(
            PopulationAggregate(
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
            )
        )
    return tuple(rows)


def _mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def score_population(config, aggregates, results):
    aggregates_by_strategy = {}
    results_by_strategy = {}
    for row in aggregates:
        aggregates_by_strategy.setdefault(row.strat, []).append(row)
    for result in results:
        results_by_strategy.setdefault(result.strat, []).append(result)

    scores = []
    for strat in config.strategies:
        rows = aggregates_by_strategy.get(strat, [])
        seed_results = results_by_strategy.get(strat, [])
        hands = sum(row.hands for row in rows)
        net_chips = sum(row.net_chips for row in rows)
        bb_per_100 = net_chips / BIG_BLIND / hands * 100 if hands else 0.0
        row_values = [row.bb_per_100 for row in rows]
        seed_passes = sum(
            1 for result in seed_results if result.bb_per_100 >= config.min_seed_bb100
        )
        seed_count = len(seed_results)
        counter_exposure = _mean([max(0.0, -value) for value in row_values])
        scores.append(
            PopulationScore(
                strat=strat,
                hands=hands,
                net_chips=net_chips,
                bb_per_100=bb_per_100,
                mean_row_bb_per_100=_mean(row_values),
                worst_row_bb_per_100=min(row_values, default=0.0),
                best_row_bb_per_100=max(row_values, default=0.0),
                seed_passes=seed_passes,
                seed_count=seed_count,
                seed_pass_rate=seed_passes / seed_count if seed_count else 0.0,
                counter_exposure_bb100=counter_exposure,
                min_mean_bb100=config.min_mean_bb100,
                min_worst_bb100=config.min_worst_bb100,
                min_seed_pass_rate=config.min_seed_pass_rate,
            )
        )
    return tuple(scores)


def run_population_selfplay(
    *,
    candidate=None,
    config_path=DEFAULT_CONFIG,
    champion_json=DEFAULT_CHAMPION_JSON,
    runner=run_selfplay,
    output_json=None,
    output_markdown=None,
    overrides=None,
):
    started = time.perf_counter()
    champion = load_champion_strategy(champion_json)
    config = load_population_config(
        config_path,
        candidate=candidate,
        champion_json=champion_json,
        overrides=overrides,
    )
    cases = build_cases(config)
    results = []
    for case in cases:
        results.append(
            runner(
                case.strat,
                hands=case.hands,
                seed=case.seed,
                opponent_name=case.opponent,
                players=case.players,
                track_opponents=config.track_opponents,
                opponent_db=None,
            )
        )

    aggregates = aggregate_results(cases, results)
    report = PopulationReport(
        candidate=candidate,
        champion=champion,
        config=config,
        cases=cases,
        results=tuple(results),
        aggregates=aggregates,
        scores=score_population(config, aggregates, results),
        elapsed=time.perf_counter() - started,
        output_json=str(output_json) if output_json else None,
        output_markdown=str(output_markdown) if output_markdown else None,
    )
    write_outputs(report, output_json, output_markdown)
    return report


def population_gate_summary(report, strategy=None):
    target = strategy or report.config.score_strategy
    if target is None:
        return {
            "strategy": None,
            "passed": None,
            "detail": "population report has no score strategy",
            "score": None,
        }
    score = next((row for row in report.scores if row.strat == target), None)
    if score is None:
        return {
            "strategy": target,
            "passed": False,
            "detail": f"{target} was not present in the population matrix",
            "score": None,
        }
    detail = (
        f"{target}: mean {score.mean_row_bb_per_100:+.1f} bb/100 "
        f">= {score.min_mean_bb100:.1f}, worst {score.worst_row_bb_per_100:+.1f} "
        f">= {score.min_worst_bb100:.1f}, seeds "
        f"{score.seed_passes}/{score.seed_count} "
        f">= {score.min_seed_pass_rate:.0%}"
    )
    return {
        "strategy": target,
        "passed": score.passed,
        "detail": detail,
        "score": score,
    }


def _signed_int(value):
    return f"{value:+d}"


def _signed_float(value):
    return f"{value:+.1f}"


def format_report(report):
    lines = [
        f"population  : {report.candidate or 'matrix'}",
        f"champion    : {report.champion}",
        f"cases       : {len(report.cases)}",
        f"elapsed     : {report.elapsed:.1f}s",
        "",
        "Scores",
        (
            "strategy                  hands       net    bb/100 mean row "
            "worst row seeds exposure gate"
        ),
        "-" * 94,
    ]
    for score in report.scores:
        status = "PASS" if score.passed else "FAIL"
        lines.append(
            f"{score.strat:<24} "
            f"{score.hands:<8} "
            f"{_signed_int(score.net_chips):>8} "
            f"{_signed_float(score.bb_per_100):>7} "
            f"{_signed_float(score.mean_row_bb_per_100):>8} "
            f"{_signed_float(score.worst_row_bb_per_100):>9} "
            f"{score.seed_passes}/{score.seed_count:<5} "
            f"{score.counter_exposure_bb100:>8.1f} "
            f"{status}"
        )

    opponent_width = max(
        24,
        max((len(row.opponent) for row in report.aggregates), default=24),
    )
    lines.extend(
        [
            "",
            "Payoff Matrix",
            (
                f"{'strategy':<24} {'opponent':<{opponent_width}} players "
                "seeds hands      net    bb/100  W/L/P"
            ),
            "-" * (73 + opponent_width),
        ]
    )
    for row in report.aggregates:
        lines.append(
            f"{row.strat:<24} "
            f"{row.opponent:<{opponent_width}} "
            f"{row.players:<7} "
            f"{len(row.seeds):<5} "
            f"{row.hands:<8} "
            f"{_signed_int(row.net_chips):>8} "
            f"{_signed_float(row.bb_per_100):>7}  "
            f"{row.wins}/{row.losses}/{row.pushes}"
        )
    gate = population_gate_summary(report)
    if gate["passed"] is not None:
        status = "PASS" if gate["passed"] else "FAIL"
        lines.extend(["", f"gate        : {status} {gate['detail']}"])
    return "\n".join(lines)


def format_markdown_report(report):
    gate = population_gate_summary(report)
    status = "PASS" if gate["passed"] else "FAIL" if gate["passed"] is False else "N/A"
    lines = [
        f"# Population self-play: {report.candidate or 'matrix'}",
        "",
        f"Status: {status}.",
        f"Current champion: `{report.champion}`.",
        "",
        "## Scores",
        "",
        (
            "| Strategy | Status | bb/100 | Mean Row | Worst Row | "
            "Seed Passes | Exposure |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for score in report.scores:
        lines.append(
            f"| {score.strat} | {'PASS' if score.passed else 'FAIL'} | "
            f"{score.bb_per_100:+.1f} | {score.mean_row_bb_per_100:+.1f} | "
            f"{score.worst_row_bb_per_100:+.1f} | "
            f"{score.seed_passes}/{score.seed_count} | "
            f"{score.counter_exposure_bb100:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Matrix",
            "",
            "```text",
            format_report(report),
            "```",
        ]
    )
    return "\n".join(lines)


def report_to_jsonable(report):
    gate = population_gate_summary(report)
    if gate["score"] is not None:
        gate = {
            **gate,
            "score": {
                **asdict(gate["score"]),
                "passed": gate["score"].passed,
            },
        }
    return {
        "candidate": report.candidate,
        "champion": report.champion,
        "elapsed": report.elapsed,
        "passed": report.passed,
        "output_json": report.output_json,
        "output_markdown": report.output_markdown,
        "config": asdict(report.config),
        "cases": [asdict(case) for case in report.cases],
        "results": [asdict(result) for result in report.results],
        "aggregates": [
            {
                **asdict(row),
                "bb_per_100": row.bb_per_100,
                "chips_per_hand": row.chips_per_hand,
            }
            for row in report.aggregates
        ],
        "scores": [
            {
                **asdict(score),
                "passed": score.passed,
            }
            for score in report.scores
        ],
        "gate": gate,
    }


def write_json_report(report, path):
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report_to_jsonable(report), f, indent=2)
        f.write("\n")


def write_markdown_report(report, path):
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_markdown_report(report))


def write_outputs(report, output_json, output_markdown):
    write_json_report(report, output_json)
    write_markdown_report(report, output_markdown)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a population self-play payoff matrix."
    )
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--champion-json", default=str(DEFAULT_CHAMPION_JSON))
    parser.add_argument("--strategies", default=None)
    parser.add_argument("--opponents", default=None)
    parser.add_argument("--players", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--hands", type=int, default=None)
    parser.add_argument("--score-strategy", default=None)
    parser.add_argument("--track-opponents", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    overrides = {
        "strategies": args.strategies,
        "opponents": args.opponents,
        "players": args.players,
        "seeds": args.seeds,
        "hands": args.hands,
        "score_strategy": args.score_strategy,
        "track_opponents": True if args.track_opponents else None,
    }
    try:
        report = run_population_selfplay(
            candidate=args.candidate,
            config_path=args.config,
            champion_json=args.champion_json,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
            overrides=overrides,
        )
    except ValueError as exc:
        print(f"population-selfplay: {exc}", file=sys.stderr)
        return 2

    print(format_report(report))
    if report.passed is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
