import json
import subprocess

from eval import promotion_gate
from eval.selfplay import BIG_BLIND, SelfPlayResult


def completed(returncode=0):
    return subprocess.CompletedProcess(
        args=("pytest", "tests/scenario"),
        returncode=returncode,
        stdout="scenario output",
        stderr="",
    )


def write_config(tmp_path, *, catastrophic_floor=-50.0, population_config=None):
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
        "seeds": [1, 2],
        "track_opponents": True,
        "scenario_tests": ["tests/scenario"],
        "simple_min_bb100": 0.0,
        "min_delta_bb100": -5.0,
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
        min_delta_bb100=-2.5,
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
    assert report_payload["seed_variance"][0]["seed_count"] == 2
    assert report_payload["reproducibility"]["strategies"]["candidate"]["module"] == (
        "poker_bot.strategies.candidate"
    )
    assert "Promotion gate: candidate" in (tmp_path / "report.md").read_text()
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
    assert json.loads(champion_path.read_text()) == {"strategy": "champion"}
