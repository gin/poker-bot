import json
import subprocess
from pathlib import Path

import pytest

from eval import promotion_gate
from eval.benchmark import (
    BenchmarkAggregate,
    BenchmarkComparison,
    BenchmarkReport,
    build_cases,
)
from eval.selfplay import BIG_BLIND, SelfPlayResult, resolve_opponent_lineup


def completed(returncode=0):
    return subprocess.CompletedProcess(
        args=("pytest", "tests/scenario"),
        returncode=returncode,
        stdout="scenario output",
        stderr="",
    )


def write_config(
    tmp_path,
    *,
    catastrophic_floor=-50.0,
    population_config=None,
    profile=None,
    regime_overrides=None,
):
    config_path = tmp_path / "promotion_gate.json"
    payload = {
        "hands": 100,
        "opponents": [
            "simple",
            "adaptive",
            "counter_adaptive",
            "threshold_pressure",
            "anti_threshold",
            "profiled_counter_adaptive",
        ],
        "players": [6],
        "seeds": [1, 2, 3, 4, 5],
        "track_opponents": True,
        "scenario_tests": ["tests/scenario"],
        "simple_min_bb100": 0.0,
        "min_delta_bb_per_100": -5.0,
        "catastrophic_floor_bb100": catastrophic_floor,
        "min_seed_pass_rate": 0.6,
        "counter_strategies": [
            "adaptive",
            "counter_adaptive",
            "threshold_pressure",
            "anti_threshold",
            "profiled_counter_adaptive",
        ],
    }
    if population_config is not None:
        payload["population_config"] = str(population_config)
    if profile is not None:
        payload["profile"] = profile
    if regime_overrides is not None:
        payload["regime_overrides"] = regime_overrides
    config_path.write_text(json.dumps(payload))
    return config_path


def write_population_config(tmp_path, *, min_worst=10.0):
    config_path = tmp_path / "population.json"
    config_path.write_text(
        json.dumps(
            {
                "hands": 100,
                "strategies": ["{candidate}"],
                "opponents": ["simple", "{champion}"],
                "players": [6],
                "seeds": [1],
                "track_opponents": True,
                "score_strategy": "{candidate}",
                "min_mean_bb100": 0.0,
                "min_worst_bb100": min_worst,
                "min_seed_bb100": min_worst,
                "min_seed_pass_rate": 0.5,
            }
        )
    )
    return config_path


def write_champion(tmp_path, strategy="champion"):
    champion_path = tmp_path / "champion.json"
    champion_path.write_text(json.dumps({"strategy": strategy}))
    return champion_path


def chips_for_bb100(bb100, hands):
    return int(round(bb100 * BIG_BLIND * hands / 100))


def fake_benchmark_runner(
    strat,
    *,
    hands,
    seed,
    opponent_name,
    players,
    track_opponents=False,
    opponent_db=None,
    **kwargs,
):
    candidate_bb_by_opponent = {
        "simple": 21 + seed,
        "adaptive": -4,
        "counter_adaptive": -7,
        "threshold_pressure": -9,
        "anti_threshold": -11,
        "profiled_counter_adaptive": -6,
        "champion": -9,
    }
    champion_bb_by_opponent = {
        "simple": 25,
        "adaptive": 0,
        "counter_adaptive": -3,
        "threshold_pressure": -5,
        "anti_threshold": -7,
        "profiled_counter_adaptive": -2,
        "champion": -5,
    }
    bb100 = (
        champion_bb_by_opponent[opponent_name]
        if strat == "champion"
        else candidate_bb_by_opponent[opponent_name]
    )
    net = chips_for_bb100(bb100, hands)
    return SelfPlayResult(
        hands=hands,
        strat=strat,
        opponent=opponent_name,
        wins=1,
        losses=0,
        pushes=0,
        net_chips=net,
        elapsed=0.01,
        players=players,
    )


def swingy_benchmark_runner(
    strat,
    *,
    hands,
    seed,
    opponent_name,
    players,
    track_opponents=False,
    opponent_db=None,
    **kwargs,
):
    net = 0
    if strat != "champion":
        net = 1200 if seed == 1 else -400
    return SelfPlayResult(
        hands=hands,
        strat=strat,
        opponent=opponent_name,
        wins=1,
        losses=0,
        pushes=0,
        net_chips=net,
        elapsed=0.01,
        players=players,
    )


def test_build_opponent_pool_includes_current_champion():
    config = promotion_gate.PromotionGateConfig(
        hands=100,
        opponents=("simple", "adaptive"),
        players=(6,),
        seeds=(1,),
        track_opponents=True,
        scenario_tests=("tests/scenario",),
        simple_min_bb100=0.0,
        min_delta_bb_per_100=-2.5,
        catastrophic_floor_bb100=-50.0,
        counter_strategies=("adaptive",),
        min_seed_pass_rate=0.6,
        population_config=None,
    )

    config_with_placeholder = promotion_gate.PromotionGateConfig(
        **{
            **config.__dict__,
            "opponents": (
                "simple",
                "counter_adaptive+threshold_pressure+anti_threshold"
                "+profiled_counter_adaptive+{champion}",
            ),
        }
    )

    assert promotion_gate.build_opponent_pool(config_with_placeholder, "champion") == (
        "simple",
        "counter_adaptive+threshold_pressure+anti_threshold"
        "+profiled_counter_adaptive+champion",
        "champion",
    )


def test_scenario_failure_skips_benchmark_and_does_not_update_champion(tmp_path):
    config_path = write_config(tmp_path)
    champion_path = write_champion(tmp_path)
    called = False

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True

    report = promotion_gate.run_promotion_gate(
        "candidate",
        config_path=config_path,
        champion_json=champion_path,
        benchmark_runner=runner,
        scenario_runner=lambda *_args, **_kwargs: completed(returncode=1),
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        history_index=tmp_path / "index.jsonl",
    )

    assert report.passed is False
    assert report.benchmark_report is None
    assert called is False
    assert json.loads(champion_path.read_text()) == {"strategy": "champion"}
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["benchmark"] is None
    assert payload["champion_metadata"] == {"strategy": "champion"}
    history_rows = (tmp_path / "index.jsonl").read_text().splitlines()
    assert len(history_rows) == 1
    assert json.loads(history_rows[0])["passed"] is False


def test_passing_promotion_gate_writes_reports_and_updates_champion(tmp_path):
    config_path = write_config(tmp_path)
    champion_path = write_champion(tmp_path)

    report = promotion_gate.run_promotion_gate(
        "candidate",
        config_path=config_path,
        champion_json=champion_path,
        benchmark_runner=fake_benchmark_runner,
        scenario_runner=lambda *_args, **_kwargs: completed(),
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        history_index=tmp_path / "index.jsonl",
    )

    assert report.passed is True
    assert report.champion_updated is True
    assert report.seed_variance
    champion_payload = json.loads(champion_path.read_text())
    assert champion_payload["strategy"] == "candidate"
    assert champion_payload["previous_strategy"] == "champion"
    assert champion_payload["summary"]["status"] == "promoted"
    report_payload = json.loads((tmp_path / "report.json").read_text())
    assert report_payload["passed"] is True
    assert report_payload["champion_updated"] is True
    assert report_payload["checks"][0]["name"] == "positive bb/100 vs simple"
    assert report_payload["seed_variance"][0]["seed_count"] == 5
    assert [check["name"] for check in report_payload["checks"]] == [
        "positive bb/100 vs simple",
        "champion regression margin",
        "no catastrophic counter loss",
        "seed consistency",
    ]
    route_diagnostics = report_payload["route_diagnostics"]
    assert route_diagnostics["schema_version"] == 1
    assert len(route_diagnostics["candidate"]) == len(
        report.benchmark_report.aggregates
    )
    assert len(route_diagnostics["baseline"]) == len(
        report.benchmark_report.baseline_aggregates
    )
    assert all(
        row["activation_fraction"] == 0.0
        for row in route_diagnostics["candidate"]
    )
    assert report_payload["reproducibility"]["strategies"]["candidate"]["module"] == (
        "poker_bot.strategies.candidate"
    )
    markdown = (tmp_path / "report.md").read_text()
    assert "Promotion gate: candidate" in markdown
    assert "## Route Diagnostics" in markdown
    history_payload = json.loads((tmp_path / "index.jsonl").read_text())
    assert history_payload["candidate"] == "candidate"
    assert history_payload["champion_updated"] is True


def test_failing_counter_floor_leaves_champion_unchanged(tmp_path):
    config_path = write_config(tmp_path, catastrophic_floor=-2.0)
    champion_path = write_champion(tmp_path)

    report = promotion_gate.run_promotion_gate(
        "candidate",
        config_path=config_path,
        champion_json=champion_path,
        benchmark_runner=fake_benchmark_runner,
        scenario_runner=lambda *_args, **_kwargs: completed(),
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        history_index=tmp_path / "index.jsonl",
    )

    assert report.passed is False
    assert report.champion_updated is False
    assert json.loads(champion_path.read_text()) == {"strategy": "champion"}
    assert any(check.name == "no catastrophic counter loss" for check in report.checks)


def test_seed_consistency_gate_catches_swingy_aggregate_pass(tmp_path):
    config_path = write_config(tmp_path)
    champion_path = write_champion(tmp_path)

    report = promotion_gate.run_promotion_gate(
        "candidate",
        config_path=config_path,
        champion_json=champion_path,
        benchmark_runner=swingy_benchmark_runner,
        scenario_runner=lambda *_args, **_kwargs: completed(),
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        history_index=tmp_path / "index.jsonl",
    )

    seed_gate = next(
        check for check in report.checks if check.name == "seed consistency"
    )
    assert report.passed is False
    assert seed_gate.passed is False
    assert json.loads(champion_path.read_text()) == {"strategy": "champion"}


def test_population_score_gate_can_block_promotion(tmp_path):
    population_config = write_population_config(tmp_path, min_worst=10.0)
    config_path = write_config(tmp_path, population_config=population_config)
    champion_path = write_champion(tmp_path)

    report = promotion_gate.run_promotion_gate(
        "candidate",
        config_path=config_path,
        champion_json=champion_path,
        benchmark_runner=fake_benchmark_runner,
        scenario_runner=lambda *_args, **_kwargs: completed(),
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        history_index=tmp_path / "index.jsonl",
    )

    population_gate = next(
        check for check in report.checks if check.name == "population score"
    )
    payload = json.loads((tmp_path / "report.json").read_text())

    assert report.passed is False
    assert population_gate.passed is False
    assert report.population_report is not None
    assert payload["population"]["passed"] is False


def _make_aggregate(strat, opponent, players, seeds, hands, net_chips):
    return BenchmarkAggregate(
        strat=strat,
        opponent=opponent,
        players=players,
        seeds=seeds,
        hands=hands,
        wins=1,
        losses=0,
        pushes=0,
        net_chips=net_chips,
        elapsed=0.01,
    )


def _make_report(*, candidate_aggregates, baseline_aggregates, comparisons):
    return BenchmarkReport(
        strat="candidate",
        cases=(),
        results=(),
        aggregates=candidate_aggregates,
        elapsed=0.0,
        baseline_strat="champion",
        baseline_aggregates=baseline_aggregates,
        comparisons=comparisons,
        min_delta_bb_per_100=-5.0,
    )


def test_evaluate_gate_catastrophic_uses_delta_not_absolute():
    config = promotion_gate.PromotionGateConfig(
        hands=100,
        opponents=("simple", "adaptive"),
        players=(6,),
        seeds=(1,),
        track_opponents=True,
        scenario_tests=("tests/scenario",),
        simple_min_bb100=0.0,
        min_delta_bb_per_100=-5.0,
        catastrophic_floor_bb100=-10.0,
        counter_strategies=("adaptive",),
        min_seed_pass_rate=0.8,
        population_config=None,
    )

    # candidate loses 15 bb/100, champion loses 15 bb/100 -> delta 0
    # absolute check would fail (-15 < -10), delta check passes (0 >= -10)
    candidate_aggregates = (
        _make_aggregate("candidate", "simple", 6, (1,), 100, 3000),
        _make_aggregate("candidate", "adaptive", 6, (1,), 100, -3000),
    )
    baseline_aggregates = (
        _make_aggregate("champion", "simple", 6, (1,), 100, 3000),
        _make_aggregate("champion", "adaptive", 6, (1,), 100, -3000),
    )
    comparisons = (
        BenchmarkComparison(
            opponent="simple",
            players=6,
            candidate_bb_per_100=15.0,
            baseline_bb_per_100=15.0,
            delta_bb_per_100=0.0,
            min_delta_bb_per_100=-5.0,
        ),
        BenchmarkComparison(
            opponent="adaptive",
            players=6,
            candidate_bb_per_100=-15.0,
            baseline_bb_per_100=-15.0,
            delta_bb_per_100=0.0,
            min_delta_bb_per_100=-5.0,
        ),
    )
    report = _make_report(
        candidate_aggregates=candidate_aggregates,
        baseline_aggregates=baseline_aggregates,
        comparisons=comparisons,
    )

    checks = promotion_gate.evaluate_gate(report, config, "champion")
    catastrophic = next(c for c in checks if c.name == "no catastrophic counter loss")
    assert catastrophic.passed is True


def test_evaluate_gate_catastrophic_excludes_champion_from_counter_set():
    config = promotion_gate.PromotionGateConfig(
        hands=100,
        opponents=("simple", "champion"),
        players=(6,),
        seeds=(1,),
        track_opponents=True,
        scenario_tests=("tests/scenario",),
        simple_min_bb100=0.0,
        min_delta_bb_per_100=-5.0,
        catastrophic_floor_bb100=-10.0,
        counter_strategies=("adaptive",),
        min_seed_pass_rate=0.8,
        population_config=None,
    )

    # only "simple" in candidate aggregates; "champion" not in counter_strategies
    # and should NOT be auto-added (the old code added it).
    candidate_aggregates = (
        _make_aggregate("candidate", "simple", 6, (1,), 100, 5000),
        _make_aggregate("candidate", "champion", 6, (1,), 100, -5000),
    )
    baseline_aggregates = (_make_aggregate("champion", "simple", 6, (1,), 100, 5000),)
    comparisons = (
        BenchmarkComparison(
            opponent="simple",
            players=6,
            candidate_bb_per_100=25.0,
            baseline_bb_per_100=25.0,
            delta_bb_per_100=0.0,
            min_delta_bb_per_100=-5.0,
        ),
    )
    report = _make_report(
        candidate_aggregates=candidate_aggregates,
        baseline_aggregates=baseline_aggregates,
        comparisons=comparisons,
    )

    checks = promotion_gate.evaluate_gate(report, config, "champion")
    catastrophic = next(c for c in checks if c.name == "no catastrophic counter loss")
    # No "adaptive" rows in aggregates -> catastrophic has no deltas -> fails
    # (we just need to ensure the champion row was NOT evaluated)
    assert catastrophic.passed is False
    assert "worst delta n/a" in catastrophic.detail


def test_evaluate_seed_consistency_fails_on_low_ci95():
    config = promotion_gate.PromotionGateConfig(
        hands=100,
        opponents=("simple",),
        players=(6,),
        seeds=(1, 2),
        track_opponents=True,
        scenario_tests=("tests/scenario",),
        simple_min_bb100=0.0,
        min_delta_bb_per_100=-5.0,
        catastrophic_floor_bb100=-10.0,
        counter_strategies=("simple",),
        min_seed_pass_rate=0.8,
        population_config=None,
    )

    # Two seeds both clear the delta (>= -5) but CI95 low is deep below -5
    seed_variance = (
        promotion_gate.SeedVariance(
            opponent="simple",
            players=6,
            seeds=(1, 2),
            candidate_bb_per_100=(0.0, 0.0),
            baseline_bb_per_100=(0.0, 0.0),
            delta_bb_per_100=(-3.0, -3.0),
            candidate_mean_bb_per_100=0.0,
            candidate_stddev_bb_per_100=0.0,
            candidate_stderr_bb_per_100=0.0,
            candidate_ci95_low_bb_per_100=-10.0,
            candidate_ci95_high_bb_per_100=10.0,
            delta_mean_bb_per_100=-3.0,
            delta_stddev_bb_per_100=0.0,
            delta_stderr_bb_per_100=0.0,
            delta_ci95_low_bb_per_100=-20.0,
            delta_ci95_high_bb_per_100=14.0,
            seed_passes=2,
            seed_count=2,
            seed_pass_rate=1.0,
        ),
    )

    check = promotion_gate.evaluate_seed_consistency(seed_variance, config)
    assert check.passed is False
    assert "delta CI95 low" in check.detail


def test_format_markdown_includes_scenario_output_when_failed(tmp_path):
    config_path = write_config(tmp_path)
    champion_path = write_champion(tmp_path)

    long_stdout = "stdout-line\n" * 500
    long_stderr = "stderr-line\n" * 500

    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=("pytest", "tests/scenario"),
            returncode=1,
            stdout=long_stdout,
            stderr=long_stderr,
        )

    promotion_gate.run_promotion_gate(
        "candidate",
        config_path=config_path,
        champion_json=champion_path,
        benchmark_runner=lambda *a, **k: None,
        scenario_runner=runner,
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        history_index=tmp_path / "index.jsonl",
    )

    markdown = (tmp_path / "report.md").read_text()
    assert "Scenario stdout (last 2000 chars)" in markdown
    assert "Scenario stderr (last 2000 chars)" in markdown
    assert "stdout-line" in markdown
    assert "stderr-line" in markdown
    assert json.loads(champion_path.read_text()) == {"strategy": "champion"}


def test_promotion_gate_passes_workers_to_benchmark(tmp_path, monkeypatch):
    config_path = write_config(tmp_path)
    champion_path = write_champion(tmp_path)

    captured: dict = {}

    def fake_run_benchmark(*args, **kwargs):
        captured["workers"] = kwargs.get("workers")
        return _make_fake_benchmark_report()

    def _make_fake_benchmark_report():
        return promotion_gate.benchmark.BenchmarkReport(
            strat="candidate",
            cases=(),
            results=(),
            aggregates=(),
            elapsed=0.0,
            baseline_strat="champion",
            baseline_results=(),
            baseline_aggregates=(),
            comparisons=(),
            min_delta_bb_per_100=-5.0,
        )

    monkeypatch.setattr(
        promotion_gate.benchmark,
        "run_benchmark",
        fake_run_benchmark,
    )

    payload = json.loads(config_path.read_text())
    payload["workers"] = 4
    payload["profile"] = "smoke"
    config_path.write_text(json.dumps(payload))

    promotion_gate.run_promotion_gate(
        "candidate",
        config_path=config_path,
        champion_json=champion_path,
        benchmark_runner=fake_benchmark_runner,
        scenario_runner=lambda *_a, **_k: completed(),
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        history_index=tmp_path / "index.jsonl",
    )

    assert captured["workers"] == 4


def test_production_profile_promotion_allows_parallel_workers(tmp_path, monkeypatch):
    """Per-case profile isolation (see eval.benchmark._case_db_paths) makes
    every worker count safe for a track_opponents=True production
    promotion -- the old workers=1 restriction, which existed only to
    "preserve one sequential history" across cases sharing one DB, is
    obsolete now that no case's profile history can leak into another's
    regardless of dispatch order."""
    config_path = write_config(tmp_path)
    champion_path = write_champion(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["workers"] = 2
    config_path.write_text(json.dumps(payload))

    captured: dict = {}

    def fake_run_benchmark(*args, **kwargs):
        captured["workers"] = kwargs.get("workers")
        return promotion_gate.benchmark.BenchmarkReport(
            strat="candidate",
            cases=(),
            results=(),
            aggregates=(),
            elapsed=0.0,
            baseline_strat="champion",
            baseline_results=(),
            baseline_aggregates=(),
            comparisons=(),
            min_delta_bb_per_100=-5.0,
        )

    monkeypatch.setattr(
        promotion_gate.benchmark,
        "run_benchmark",
        fake_run_benchmark,
    )

    promotion_gate.run_promotion_gate(
        "candidate",
        config_path=config_path,
        champion_json=champion_path,
        benchmark_runner=fake_benchmark_runner,
        scenario_runner=lambda *_a, **_k: completed(),
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        history_index=tmp_path / "index.jsonl",
    )

    assert captured["workers"] == 2


# ── Per-regime non-inferiority / catastrophic floors ────────────────────────


def test_resolve_regime_floor_prefers_override_over_global():
    config = promotion_gate.PromotionGateConfig(
        hands=100,
        opponents=(),
        players=(2,),
        seeds=(1,),
        track_opponents=True,
        scenario_tests=(),
        simple_min_bb100=0.0,
        min_delta_bb_per_100=-5.0,
        catastrophic_floor_bb100=-10.0,
        counter_strategies=(),
        min_seed_pass_rate=0.8,
        population_config=None,
        regime_overrides={2: {"min_delta_bb_per_100": -1.0}},
    )

    assert (
        promotion_gate.resolve_regime_floor(config, 2, "min_delta_bb_per_100") == -1.0
    )
    assert (
        promotion_gate.resolve_regime_floor(config, 6, "min_delta_bb_per_100") == -5.0
    )
    assert (
        promotion_gate.resolve_regime_floor(config, 2, "catastrophic_floor_bb100")
        == -10.0
    )


def test_evaluate_gate_regime_floor_catches_hu_collapse_hidden_by_loose_global_floor():
    """A generous global floor alone lets a -3.0 bb/100 HU regression
    through; a tighter per-regime (HU) override must catch it even though
    6-max is a big win. This is the exact failure mode called out in the
    audit: a large 6-max win must not hide a HU or 3-handed collapse."""
    base_kwargs = dict(
        hands=100,
        opponents=("simple",),
        players=(2, 6),
        seeds=(1,),
        track_opponents=True,
        scenario_tests=("tests/scenario",),
        simple_min_bb100=0.0,
        min_delta_bb_per_100=-5.0,
        catastrophic_floor_bb100=-10.0,
        counter_strategies=(),
        min_seed_pass_rate=0.8,
        population_config=None,
    )
    comparisons = (
        BenchmarkComparison(
            opponent="simple",
            players=6,
            candidate_bb_per_100=30.0,
            baseline_bb_per_100=10.0,
            delta_bb_per_100=20.0,
            min_delta_bb_per_100=-5.0,
        ),
        BenchmarkComparison(
            opponent="simple",
            players=2,
            candidate_bb_per_100=2.0,
            baseline_bb_per_100=5.0,
            delta_bb_per_100=-3.0,
            min_delta_bb_per_100=-5.0,
        ),
    )
    report = _make_report(
        candidate_aggregates=(
            _make_aggregate(
                "candidate", "simple", 6, (1,), 100, chips_for_bb100(30.0, 100)
            ),
            _make_aggregate(
                "candidate", "simple", 2, (1,), 100, chips_for_bb100(2.0, 100)
            ),
        ),
        baseline_aggregates=(),
        comparisons=comparisons,
    )

    loose_config = promotion_gate.PromotionGateConfig(**base_kwargs)
    loose_margin = next(
        check
        for check in promotion_gate.evaluate_gate(report, loose_config, "champion")
        if check.name == "champion regression margin"
    )
    assert loose_margin.passed is True  # global floor alone lets the HU row through

    tight_config = promotion_gate.PromotionGateConfig(
        **{**base_kwargs, "regime_overrides": {2: {"min_delta_bb_per_100": -1.0}}}
    )
    tight_margin = next(
        check
        for check in promotion_gate.evaluate_gate(report, tight_config, "champion")
        if check.name == "champion regression margin"
    )
    assert tight_margin.passed is False  # per-regime HU floor catches the collapse
    assert "2p@1000 simple" in tight_margin.detail


def test_evaluate_gate_catastrophic_regime_floor_catches_hu_collapse():
    base_kwargs = dict(
        hands=100,
        opponents=("adaptive",),
        players=(2, 6),
        seeds=(1,),
        track_opponents=True,
        scenario_tests=("tests/scenario",),
        simple_min_bb100=0.0,
        min_delta_bb_per_100=-100.0,
        catastrophic_floor_bb100=-10.0,
        counter_strategies=("adaptive",),
        min_seed_pass_rate=0.8,
        population_config=None,
    )
    candidate_aggregates = (
        _make_aggregate(
            "candidate", "adaptive", 6, (1,), 100, chips_for_bb100(40.0, 100)
        ),
        _make_aggregate(
            "candidate", "adaptive", 2, (1,), 100, chips_for_bb100(-12.0, 100)
        ),
    )
    baseline_aggregates = (
        _make_aggregate(
            "champion", "adaptive", 6, (1,), 100, chips_for_bb100(10.0, 100)
        ),
        _make_aggregate(
            "champion", "adaptive", 2, (1,), 100, chips_for_bb100(-2.0, 100)
        ),
    )
    # Deltas: 6p = +30.0 (comfortably clears any floor); 2p = -10.0, exactly
    # at the global floor (passes globally) but must fail a tighter HU
    # override.
    report = _make_report(
        candidate_aggregates=candidate_aggregates,
        baseline_aggregates=baseline_aggregates,
        comparisons=(),
    )

    loose_config = promotion_gate.PromotionGateConfig(**base_kwargs)
    loose_catastrophic = next(
        check
        for check in promotion_gate.evaluate_gate(report, loose_config, "champion")
        if check.name == "no catastrophic counter loss"
    )
    assert loose_catastrophic.passed is True

    tight_config = promotion_gate.PromotionGateConfig(
        **{**base_kwargs, "regime_overrides": {2: {"catastrophic_floor_bb100": -5.0}}}
    )
    tight_catastrophic = next(
        check
        for check in promotion_gate.evaluate_gate(report, tight_config, "champion")
        if check.name == "no catastrophic counter loss"
    )
    assert tight_catastrophic.passed is False
    assert "2p@1000 adaptive" in tight_catastrophic.detail


def test_evaluate_seed_consistency_uses_regime_floor():
    config = promotion_gate.PromotionGateConfig(
        hands=100,
        opponents=(),
        players=(2, 6),
        seeds=(1, 2),
        track_opponents=True,
        scenario_tests=(),
        simple_min_bb100=0.0,
        min_delta_bb_per_100=-5.0,
        catastrophic_floor_bb100=-10.0,
        counter_strategies=(),
        min_seed_pass_rate=0.5,
        population_config=None,
        regime_overrides={2: {"min_delta_bb_per_100": -1.0}},
    )
    hu_row = promotion_gate.SeedVariance(
        opponent="simple",
        players=2,
        seeds=(1, 2),
        candidate_bb_per_100=(2.0, 2.0),
        baseline_bb_per_100=(5.0, 5.0),
        delta_bb_per_100=(-3.0, -3.0),
        candidate_mean_bb_per_100=2.0,
        candidate_stddev_bb_per_100=0.0,
        candidate_stderr_bb_per_100=0.0,
        candidate_ci95_low_bb_per_100=2.0,
        candidate_ci95_high_bb_per_100=2.0,
        delta_mean_bb_per_100=-3.0,
        delta_stddev_bb_per_100=0.0,
        delta_stderr_bb_per_100=0.0,
        delta_ci95_low_bb_per_100=-3.0,
        delta_ci95_high_bb_per_100=-3.0,
        seed_passes=2,
        seed_count=2,
        seed_pass_rate=1.0,
    )

    check = promotion_gate.evaluate_seed_consistency((hu_row,), config)

    # seed_pass_rate is 1.0 (>= 0.5) but delta_ci95_low (-3.0) fails the
    # tighter HU regime floor (-1.0), even though the global floor (-5.0)
    # would have let it through.
    assert check.passed is False


# ── Aggregate paired-delta confidence interval ──────────────────────────────


def test_calculate_aggregate_variance_pools_all_seed_deltas():
    row_a = promotion_gate.SeedVariance(
        opponent="simple",
        players=6,
        seeds=(1, 2),
        candidate_bb_per_100=(10.0, 12.0),
        baseline_bb_per_100=(8.0, 8.0),
        delta_bb_per_100=(2.0, 4.0),
        candidate_mean_bb_per_100=11.0,
        candidate_stddev_bb_per_100=0.0,
        candidate_stderr_bb_per_100=0.0,
        candidate_ci95_low_bb_per_100=11.0,
        candidate_ci95_high_bb_per_100=11.0,
        delta_mean_bb_per_100=3.0,
        delta_stddev_bb_per_100=0.0,
        delta_stderr_bb_per_100=0.0,
        delta_ci95_low_bb_per_100=3.0,
        delta_ci95_high_bb_per_100=3.0,
        seed_passes=2,
        seed_count=2,
        seed_pass_rate=1.0,
    )
    row_b = promotion_gate.SeedVariance(
        opponent="simple",
        players=2,
        seeds=(1,),
        candidate_bb_per_100=(1.0,),
        baseline_bb_per_100=(5.0,),
        delta_bb_per_100=(-4.0,),
        candidate_mean_bb_per_100=1.0,
        candidate_stddev_bb_per_100=0.0,
        candidate_stderr_bb_per_100=0.0,
        candidate_ci95_low_bb_per_100=1.0,
        candidate_ci95_high_bb_per_100=1.0,
        delta_mean_bb_per_100=-4.0,
        delta_stddev_bb_per_100=0.0,
        delta_stderr_bb_per_100=0.0,
        delta_ci95_low_bb_per_100=-4.0,
        delta_ci95_high_bb_per_100=-4.0,
        seed_passes=0,
        seed_count=1,
        seed_pass_rate=0.0,
    )

    aggregate = promotion_gate.calculate_aggregate_variance((row_a, row_b))

    # Pooled per-seed deltas: 2.0, 4.0, -4.0 -> mean = 2/3.
    assert aggregate.seed_count == 3
    assert aggregate.delta_mean_bb_per_100 == pytest.approx(2.0 / 3.0)
    assert aggregate.delta_stddev_bb_per_100 > 0.0
    assert aggregate.delta_ci95_low_bb_per_100 < aggregate.delta_mean_bb_per_100
    assert aggregate.delta_ci95_high_bb_per_100 > aggregate.delta_mean_bb_per_100


# ── Production vs smoke profile: config parsing, gating, unmistakability ───


def test_load_promotion_config_rejects_invalid_profile(tmp_path):
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps({"hands": 100, "profile": "nonsense"}))

    with pytest.raises(ValueError, match="profile"):
        promotion_gate.load_promotion_config(config_path)


def test_load_promotion_config_parses_regime_overrides(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "hands": 5000,
                "players": [2, 3, 4, 5, 6],
                "seeds": [1, 2, 3, 4, 5],
                "profile": "production",
                "regime_overrides": {
                    "2": {
                        "min_delta_bb_per_100": -1.0,
                        "catastrophic_floor_bb100": -6.0,
                    },
                    "3": {"min_delta_bb_per_100": -2.0},
                },
            }
        )
    )

    config = promotion_gate.load_promotion_config(config_path)

    assert config.profile == "production"
    assert config.regime_overrides == {
        2: {"min_delta_bb_per_100": -1.0, "catastrophic_floor_bb100": -6.0},
        3: {"min_delta_bb_per_100": -2.0},
    }


def test_run_promotion_gate_smoke_profile_never_updates_champion_even_when_passing(
    tmp_path,
):
    config_path = write_config(tmp_path, profile="smoke")
    champion_path = write_champion(tmp_path)

    report = promotion_gate.run_promotion_gate(
        "candidate",
        config_path=config_path,
        champion_json=champion_path,
        benchmark_runner=fake_benchmark_runner,
        scenario_runner=lambda *_args, **_kwargs: completed(),
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        history_index=tmp_path / "index.jsonl",
    )

    assert report.passed is True  # every gate genuinely passes...
    assert report.promoted is False  # ...but a smoke run is never promotion-quality
    assert report.champion_updated is False
    assert json.loads(champion_path.read_text()) == {"strategy": "champion"}
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["profile"] == "smoke"
    assert payload["promotion_quality"] is False
    markdown = (tmp_path / "report.md").read_text()
    assert "SMOKE PROFILE" in markdown


def test_passing_production_profile_marks_output_promotion_quality(tmp_path):
    config_path = write_config(tmp_path)  # profile omitted -> defaults to production
    champion_path = write_champion(tmp_path)

    report = promotion_gate.run_promotion_gate(
        "candidate",
        config_path=config_path,
        champion_json=champion_path,
        benchmark_runner=fake_benchmark_runner,
        scenario_runner=lambda *_args, **_kwargs: completed(),
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        history_index=tmp_path / "index.jsonl",
    )

    assert report.config.profile == "production"
    assert report.champion_updated is True
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["profile"] == "production"
    assert payload["promotion_quality"] is True
    assert "SMOKE PROFILE" not in (tmp_path / "report.md").read_text()


def test_shipped_production_and_smoke_configs_are_unmistakable():
    production = promotion_gate.load_promotion_config(promotion_gate.DEFAULT_CONFIG)
    smoke = promotion_gate.load_promotion_config(promotion_gate.DEFAULT_SMOKE_CONFIG)

    assert production.profile == "production"
    assert smoke.profile == "smoke"
    assert production.profile != smoke.profile
    # Per-case profile isolation (see eval.benchmark._case_db_paths) is
    # proven scheduling-equivalent between workers=1 and workers=12 --
    # chips, route diagnostics, and merged action profiles are byte-
    # identical either way -- so the shipped production gate uses
    # parallel workers for throughput.
    assert production.workers == 12
    assert production.profile_state_mode == "persistent"



    # The production gate is statistically meaningful: 2-6 players, at
    # least five seeds, and a hand count in line with the README's
    # documented full-matrix recommendation.
    assert set(production.players) == {2, 3, 4, 5, 6}
    assert len(production.seeds) >= 5
    assert production.hands >= 1000

    # The smoke config is cheap by comparison -- unmistakably not the
    # production gate.
    assert smoke.hands < production.hands
    assert len(smoke.seeds) < len(production.seeds)


def test_production_config_rejects_nonpersistent_profile_state(tmp_path):
    config_path = write_config(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["profile_state_mode"] = "sharded_research"
    config_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="requires profile_state_mode='persistent'"):
        promotion_gate.load_promotion_config(config_path)


def test_shipped_production_config_expands_every_case_without_lineup_error():
    """The shipped production config's full opponent/player/stack Cartesian
    product must resolve to a legal per-seat opponent lineup for every case.

    `build_cases` alone does not validate lineup shape -- a config whose
    opponent pool includes a heterogeneous (mixed multi-strategy) lineup
    sized for one player count, but combined with other player counts,
    builds fine and only raises `ValueError` deep inside a live benchmark
    run when `resolve_opponent_lineup` actually resolves it. This test
    resolves every shipped case up front so that class of bug is caught by
    a fast, deterministic test instead of a multi-hour production run."""
    production = promotion_gate.load_promotion_config(promotion_gate.DEFAULT_CONFIG)
    champion = "champion_placeholder"

    opponents = promotion_gate.build_opponent_pool(production, champion)
    counter_strategies = promotion_gate.resolve_champion_placeholders(
        production.counter_strategies, champion
    )

    cases = build_cases(
        "candidate",
        opponents,
        production.players,
        production.seeds,
        production.hands,
        production.initial_stacks,
    )
    assert cases  # sanity: the Cartesian product is non-empty

    for case in cases:
        lineup = resolve_opponent_lineup(case.opponent, case.players)
        assert len(lineup) == case.players - 1

    # counter_strategies must be resolvable at every configured player
    # count too, or the catastrophic-loss check silently degrades to an
    # always-empty (and therefore always-failing) row set.
    assert counter_strategies, "production config must define counter_strategies"
    for name in counter_strategies:
        for players in production.players:
            lineup = resolve_opponent_lineup(name, players)
            assert len(lineup) == players - 1

    # Every counter strategy must actually appear in the benchmarked
    # opponent pool -- otherwise the catastrophic-loss check would compare
    # against rows the gate never produced.
    assert set(counter_strategies) <= set(opponents)


def test_population_config_still_covers_heterogeneous_opponent_lineups():
    """Heterogeneous (mixed multi-strategy) lineups intentionally live in
    the population self-play matrix, not the promotion gate's own opponent
    pool, and that matrix must actually still exercise them at a player
    count each lineup is correctly sized for."""
    from eval.selfplay import parse_opponent_lineup

    production = promotion_gate.load_promotion_config(promotion_gate.DEFAULT_CONFIG)
    population_path = Path(production.population_config)
    assert population_path.exists()

    data = json.loads(population_path.read_text())
    opponents = data["opponents"]
    heterogeneous = [name for name in opponents if "+" in name or "," in name]
    assert heterogeneous, "population config must cover heterogeneous lineups"

    for name in heterogeneous:
        resolved_name = name.replace("{champion}", "champion_placeholder")
        token_count = len(parse_opponent_lineup(resolved_name))
        for players in data["players"]:
            assert token_count == players - 1, (
                f"heterogeneous opponent {name!r} has {token_count} tokens "
                f"but population config allows players={players}"
            )


def test_build_parser_smoke_flag_defaults():
    args = promotion_gate.build_parser().parse_args(["--strat", "candidate"])
    assert args.smoke is False
    assert args.config is None

    smoke_args = promotion_gate.build_parser().parse_args(
        ["--strat", "candidate", "--smoke"]
    )
    assert smoke_args.smoke is True


# ── Candidate/baseline comparisons must be real, not vacuously empty ───────


def regressing_benchmark_runner(
    strat,
    *,
    hands,
    seed,
    opponent_name,
    players,
    track_opponents=False,
    opponent_db=None,
    **kwargs,
):
    """Candidate collapses vs `simple` (delta -50 bb/100); everything else
    is a wash. Used to prove the champion-regression-margin gate actually
    inspects real per-row deltas rather than an empty comparisons tuple."""
    candidate_bb_by_opponent = {
        "simple": -30.0,
        "adaptive": 5.0,
        "counter_adaptive": 5.0,
        "threshold_pressure": 5.0,
        "anti_threshold": 5.0,
        "profiled_counter_adaptive": 5.0,
        "champion": 5.0,
    }
    champion_bb_by_opponent = {
        "simple": 20.0,
        "adaptive": 5.0,
        "counter_adaptive": 5.0,
        "threshold_pressure": 5.0,
        "anti_threshold": 5.0,
        "profiled_counter_adaptive": 5.0,
        "champion": 5.0,
    }
    bb100 = (
        champion_bb_by_opponent[opponent_name]
        if strat == "champion"
        else candidate_bb_by_opponent[opponent_name]
    )
    net = chips_for_bb100(bb100, hands)
    return SelfPlayResult(
        hands=hands,
        strat=strat,
        opponent=opponent_name,
        wins=1,
        losses=0,
        pushes=0,
        net_chips=net,
        elapsed=0.01,
        players=players,
    )


def test_run_promotion_gate_regressing_candidate_fails_champion_margin(tmp_path):
    """Runs the real sequential candidate-then-baseline pipeline
    (run_promotion_gate -> benchmark.run_benchmark, each called with
    baseline_strat=None) and proves a genuinely regressing candidate is
    caught by the champion-regression-margin gate. Before the fix, each
    benchmark.run_benchmark() call's own `.comparisons` was empty (since
    baseline_strat=None), and concatenating two empty tuples together left
    `report.comparisons` empty -- making this gate vacuously pass no
    matter how badly the candidate regressed."""
    config_path = write_config(tmp_path)  # min_delta_bb_per_100 = -5.0
    champion_path = write_champion(tmp_path)


    report = promotion_gate.run_promotion_gate(
        "candidate",
        config_path=config_path,
        champion_json=champion_path,
        benchmark_runner=regressing_benchmark_runner,
        scenario_runner=lambda *_args, **_kwargs: completed(),
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        history_index=tmp_path / "index.jsonl",
    )

    assert report.benchmark_report is not None
    # The bug this guards against silently emptied comparisons; assert
    # they are actually populated with one row per opponent.
    assert len(report.benchmark_report.comparisons) == 7
    assert all(
        row.strat == "candidate" for row in report.benchmark_report.aggregates
    ), "aggregates must be candidate-only, not mixed with baseline rows"

    margin_check = next(
        check for check in report.checks if check.name == "champion regression margin"
    )
    assert margin_check.passed is False
    assert "simple" in margin_check.detail
    assert report.passed is False
    assert report.champion_updated is False
    assert json.loads(champion_path.read_text()) == {"strategy": "champion"}



def test_series_stats_uses_student_t_critical_value_for_four_degrees_of_freedom():
    mean, stddev, stderr, low, high = promotion_gate._series_stats(
        (1.0, 2.0, 3.0, 4.0, 5.0)
    )

    assert mean == pytest.approx(3.0)
    assert stddev == pytest.approx(1.5811388300841898)
    assert stderr == pytest.approx(0.7071067811865476)
    assert (high - mean) / stderr == pytest.approx(2.7764451051977987)
    assert (mean - low) / stderr == pytest.approx(2.7764451051977987)


def test_series_stats_marks_one_paired_observation_as_insufficient():
    assert promotion_gate._series_stats((3.0,)) == (3.0, None, None, None, None)


# ── Bounded sequential paired CI sampling ───────────────────────────────────


def write_sequential_config(
    tmp_path,
    *,
    hands,
    batch_hands=None,
    max_hands=None,
    target=None,
):
    config_path = tmp_path / "sequential_promotion_gate.json"
    payload = {
        "hands": hands,
        "opponents": ["simple", "adaptive"],
        "players": [6],
        "seeds": [1, 2],
        "track_opponents": False,
        "scenario_tests": ["tests/scenario"],
        "simple_min_bb100": 0.0,
        "min_delta_bb_per_100": -5.0,
        "catastrophic_floor_bb100": -50.0,
        "min_seed_pass_rate": 0.5,
        "counter_strategies": ["adaptive"],
        "workers": 1,
    }
    if batch_hands is not None:
        payload["batch_hands"] = batch_hands
    if max_hands is not None:
        payload["max_hands"] = max_hands
    if target is not None:
        payload["target_delta_ci95_half_width_bb100"] = target
    config_path.write_text(json.dumps(payload))
    return config_path


def make_shrinking_ci_runner(
    candidate_base=30.0, baseline_base=10.0, noise_scale=200.0
):
    """A deterministic mocked benchmark runner whose CANDIDATE bb/100 has a
    seed-dependent noise term that shrinks as `hands` grows (noise_scale /
    hands), while the baseline is noise-free. The paired delta's CI95
    half-width is then exactly `1.96 * noise_scale / hands` (2 seeds,
    alternating +/- sign), giving full control over precision-vs-hands for
    deterministic tests -- without depending on real self-play variance.
    """
    hands_seen = set()

    def runner(
        strat,
        *,
        hands,
        seed,
        opponent_name,
        players,
        track_opponents=False,
        opponent_db=None,
        **kwargs,
    ):
        hands_seen.add(hands)
        sign = 1.0 if seed % 2 == 0 else -1.0
        if strat == "champion":
            bb100 = baseline_base
        else:
            bb100 = candidate_base + sign * noise_scale / hands
        net = chips_for_bb100(bb100, hands)
        return SelfPlayResult(
            hands=hands,
            strat=strat,
            opponent=opponent_name,
            wins=1,
            losses=0,
            pushes=0,
            net_chips=net,
            elapsed=0.01,
            players=players,
        )

    runner.hands_seen = hands_seen
    return runner


def test_sequential_evaluation_disabled_without_target_runs_single_stage(tmp_path):
    """A config without target_delta_ci95_half_width_bb100 (the existing,
    pre-sequential shape) must behave exactly as before: one run at
    `hands`, no sequential machinery engaged."""
    config_path = write_sequential_config(tmp_path, hands=123)
    champion_path = write_champion(tmp_path)
    runner = make_shrinking_ci_runner()

    report = promotion_gate.run_promotion_gate(
        "candidate",
        config_path=config_path,
        champion_json=champion_path,
        benchmark_runner=runner,
        scenario_runner=lambda *_args, **_kwargs: completed(),
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        history_index=tmp_path / "index.jsonl",
    )

    assert report.evaluated_hands == 123
    assert report.evaluation_stages == 1
    assert report.sequential_precision_achieved is None
    assert runner.hands_seen == {123}
    assert all(check.name != "sequential precision" for check in report.checks)


def test_sequential_evaluation_wide_first_stage_schedules_another_stage(tmp_path):
    """A wide (imprecise) first stage must trigger another paired stage at
    a larger cumulative hand count, and stop once precision is reached."""
    config_path = write_sequential_config(
        tmp_path, hands=100, batch_hands=400, max_hands=1000, target=5.5
    )
    champion_path = write_champion(tmp_path)
    runner = make_shrinking_ci_runner(noise_scale=200.0)

    report = promotion_gate.run_promotion_gate(
        "candidate",
        config_path=config_path,
        champion_json=champion_path,
        benchmark_runner=runner,
        scenario_runner=lambda *_args, **_kwargs: completed(),
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        history_index=tmp_path / "index.jsonl",
    )

    # With two paired seeds (df=1), Student-t makes the first half-width
    # 12.706*200/100 = 25.412 > target 5.5. At 500 hands it is 5.082,
    # which clears the target.
    assert runner.hands_seen == {100, 500}
    assert report.evaluation_stages == 2
    assert report.evaluated_hands == 500
    assert report.sequential_precision_achieved is True
    sequential_check = next(
        check for check in report.checks if check.name == "sequential precision"
    )
    assert sequential_check.passed is True
    assert report.passed is True
    assert report.champion_updated is True
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["evaluated_hands"] == 500
    assert payload["evaluation_stages"] == 2
    assert payload["sequential_precision_achieved"] is True


def test_sequential_precision_does_not_accept_normal_ci_with_two_seeds(tmp_path):
    config_path = write_sequential_config(
        tmp_path, hands=100, batch_hands=400, max_hands=1000, target=1.0
    )
    champion_path = write_champion(tmp_path)
    runner = make_shrinking_ci_runner(noise_scale=200.0)

    report = promotion_gate.run_promotion_gate(
        "candidate",
        config_path=config_path,
        champion_json=champion_path,
        benchmark_runner=runner,
        scenario_runner=lambda *_args, **_kwargs: completed(),
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        history_index=tmp_path / "index.jsonl",
    )

    # A normal 1.96 multiplier would have accepted at 500 hands. The valid
    # df=1 Student-t interval remains wider than 1.0 through max_hands.
    assert runner.hands_seen == {100, 500, 900, 1000}
    assert report.sequential_precision_achieved is False
    assert report.promoted is False


def test_sequential_evaluation_precise_first_stage_stops_immediately(tmp_path):
    """A first stage that already clears the (looser) target must not
    trigger any further stages."""
    config_path = write_sequential_config(
        tmp_path, hands=100, batch_hands=400, max_hands=1000, target=30.0
    )
    champion_path = write_champion(tmp_path)
    runner = make_shrinking_ci_runner(candidate_base=40.0, noise_scale=200.0)

    report = promotion_gate.run_promotion_gate(
        "candidate",
        config_path=config_path,
        champion_json=champion_path,
        benchmark_runner=runner,
        scenario_runner=lambda *_args, **_kwargs: completed(),
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        history_index=tmp_path / "index.jsonl",
    )

    # The df=1 Student-t half-width is 25.412 <= target 30.0, so stop here.
    assert runner.hands_seen == {100}
    assert report.evaluation_stages == 1
    assert report.evaluated_hands == 100
    assert report.sequential_precision_achieved is True
    assert report.passed is True
    assert report.champion_updated is True


def test_sequential_evaluation_max_hands_imprecision_fails_and_never_promotes(
    tmp_path,
):
    """If max_hands is exhausted without reaching the precision target,
    the gate must report an explicit failing/inconclusive check and never
    promote -- even though every other gate genuinely passes."""
    config_path = write_sequential_config(
        tmp_path, hands=100, batch_hands=100, max_hands=300, target=0.1
    )
    champion_path = write_champion(tmp_path)
    runner = make_shrinking_ci_runner(noise_scale=200.0)

    report = promotion_gate.run_promotion_gate(
        "candidate",
        config_path=config_path,
        champion_json=champion_path,
        benchmark_runner=runner,
        scenario_runner=lambda *_args, **_kwargs: completed(),
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        history_index=tmp_path / "index.jsonl",
    )

    # Half-width never drops below the unreachable target=0.1: stages run
    # at 100, 200, 300 hands and then stop at max_hands=300.
    assert runner.hands_seen == {100, 200, 300}
    assert report.evaluation_stages == 3
    assert report.evaluated_hands == 300
    assert report.sequential_precision_achieved is False

    sequential_check = next(
        check for check in report.checks if check.name == "sequential precision"
    )
    assert sequential_check.passed is False

    # Every other gate genuinely passes on this data (candidate clearly
    # beats the champion and clears the simple/catastrophic floors) --
    # only the unresolved precision blocks promotion.
    other_checks = [c for c in report.checks if c.name != "sequential precision"]
    assert other_checks and all(c.passed for c in other_checks)

    assert report.passed is False
    assert report.promoted is False
    assert report.champion_updated is False
    assert json.loads(champion_path.read_text()) == {"strategy": "champion"}
    markdown = (tmp_path / "report.md").read_text()
    assert "sequential precision" in markdown.lower() or "NOT achieved" in markdown


def test_load_promotion_config_requires_max_hands_with_target(tmp_path):
    config_path = tmp_path / "bad.json"
    config_path.write_text(
        json.dumps({"hands": 100, "target_delta_ci95_half_width_bb100": 1.0})
    )

    with pytest.raises(ValueError, match="max_hands"):
        promotion_gate.load_promotion_config(config_path)


def test_shipped_production_config_has_sequential_ci_sampling_configured():
    production = promotion_gate.load_promotion_config(promotion_gate.DEFAULT_CONFIG)
    smoke = promotion_gate.load_promotion_config(promotion_gate.DEFAULT_SMOKE_CONFIG)

    assert production.target_delta_ci95_half_width_bb100 is not None
    assert production.target_delta_ci95_half_width_bb100 > 0
    assert production.max_hands is not None
    assert production.max_hands >= production.hands
    assert production.batch_hands is not None and production.batch_hands > 0

    # Smoke stays a single fast, fixed-sample 200-hand run.
    assert smoke.target_delta_ci95_half_width_bb100 is None
    assert smoke.hands == 200

    # The population self-play matrix is part of the authoritative
    # production gate (per README); smoke intentionally skips it.
    assert production.population_config == "benchmarks/population_selfplay.json"
    assert Path(production.population_config).exists()
    assert smoke.population_config is None


# ── Multi-stack-depth matrix: no depth may mask another depth's collapse ───


def test_run_promotion_gate_regression_at_one_stack_depth_fails_despite_another_winning(
    tmp_path,
):
    """A candidate that wins big at one configured stack depth but
    collapses at another must still fail the gate -- exactly the
    per-player-count masking failure mode, but for the stack dimension:
    every promotion matching/grouping key must include initial_stack so
    no depth can hide behind another."""
    config_path = tmp_path / "stack_promotion_gate.json"
    config_path.write_text(
        json.dumps(
            {
                "hands": 100,
                "opponents": ["simple"],
                "players": [6],
                "seeds": [1],
                "initial_stacks": [500, 2000],
                "track_opponents": False,
                "scenario_tests": ["tests/scenario"],
                "simple_min_bb100": -1000.0,
                "min_delta_bb_per_100": -5.0,
                "catastrophic_floor_bb100": -1000.0,
                "min_seed_pass_rate": 0.0,
                "counter_strategies": ["simple"],
                "workers": 1,
            }
        )
    )
    champion_path = write_champion(tmp_path)

    def stack_regressing_runner(
        strat,
        *,
        hands,
        seed,
        opponent_name,
        players,
        track_opponents=False,
        opponent_db=None,
        initial_stack=None,
        **kwargs,
    ):
        assert initial_stack is not None
        if strat == "champion":
            bb100 = 10.0
        else:
            # Candidate crushes at the 2000-chip (deep) stack but
            # collapses at the 500-chip (short) stack.
            bb100 = 50.0 if initial_stack == 2000 else -60.0
        net = chips_for_bb100(bb100, hands)
        return SelfPlayResult(
            hands=hands,
            strat=strat,
            opponent=opponent_name,
            wins=1,
            losses=0,
            pushes=0,
            net_chips=net,
            elapsed=0.01,
            players=players,
            initial_stack=initial_stack,
        )

    report = promotion_gate.run_promotion_gate(
        "candidate",
        config_path=config_path,
        champion_json=champion_path,
        benchmark_runner=stack_regressing_runner,
        scenario_runner=lambda *_args, **_kwargs: completed(),
        output_json=tmp_path / "report.json",
        output_markdown=tmp_path / "report.md",
        history_index=tmp_path / "index.jsonl",
    )

    # 2 opponents (config's "simple" plus the auto-appended champion
    # mirror-match) x 2 stacks = 4 comparison rows; scope to "simple".
    assert len(report.benchmark_report.comparisons) == 4
    simple_comparisons = [
        c for c in report.benchmark_report.comparisons if c.opponent == "simple"
    ]
    by_stack = {c.initial_stack: c for c in simple_comparisons}
    assert set(by_stack) == {500, 2000}
    assert by_stack[2000].delta_bb_per_100 > 0
    assert by_stack[500].delta_bb_per_100 < -5.0
    margin_check = next(
        check for check in report.checks if check.name == "champion regression margin"
    )
    assert margin_check.passed is False
    assert "500" in margin_check.detail

    assert report.passed is False
    assert report.champion_updated is False
    assert json.loads(champion_path.read_text()) == {"strategy": "champion"}

    payload = json.loads((tmp_path / "report.json").read_text())
    seed_variance_stacks = {row["initial_stack"] for row in payload["seed_variance"]}
    assert seed_variance_stacks == {500, 2000}


def test_promotion_gate_config_default_initial_stacks_matches_module_constant():
    config = promotion_gate.PromotionGateConfig(
        hands=100,
        opponents=(),
        players=(6,),
        seeds=(1,),
        track_opponents=True,
        scenario_tests=(),
        simple_min_bb100=0.0,
        min_delta_bb_per_100=-5.0,
        catastrophic_floor_bb100=-10.0,
        counter_strategies=(),
        min_seed_pass_rate=0.8,
        population_config=None,
    )

    assert config.initial_stacks == (promotion_gate.INITIAL_STACK,)


def test_load_promotion_config_parses_initial_stacks(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"hands": 100, "initial_stacks": [500, 1000, 2000]})
    )

    config = promotion_gate.load_promotion_config(config_path)

    assert config.initial_stacks == (500, 1000, 2000)


def test_shipped_production_config_evaluates_defensible_stack_depths():
    production = promotion_gate.load_promotion_config(promotion_gate.DEFAULT_CONFIG)
    smoke = promotion_gate.load_promotion_config(promotion_gate.DEFAULT_SMOKE_CONFIG)

    # 50bb / 100bb / 200bb at the simulator's BIG_BLIND=10.
    assert production.initial_stacks == (500, 1000, 2000)

    # Smoke stays a single default-depth run.
    assert smoke.initial_stacks == (promotion_gate.INITIAL_STACK,)
