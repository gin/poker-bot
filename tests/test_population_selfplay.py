import json

from eval import population_selfplay
from eval.selfplay import SelfPlayResult


def write_champion(tmp_path, strategy="champion"):
    champion_path = tmp_path / "champion.json"
    champion_path.write_text(json.dumps({"strategy": strategy}))
    return champion_path


def write_config(tmp_path, *, min_worst=-20.0):
    config_path = tmp_path / "population.json"
    config_path.write_text(
        json.dumps(
            {
                "hands": 100,
                "strategies": ["{candidate}", "{champion}", "{candidate}"],
                "opponents": ["simple", "{champion}"],
                "players": [6],
                "seeds": [1, 2],
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


def fake_runner(
    strat,
    *,
    hands,
    seed,
    opponent_name,
    players,
    track_opponents=False,
    opponent_db=None,
):
    net = 0
    if strat == "candidate" and opponent_name == "simple":
        net = 400 + seed
    elif strat == "candidate" and opponent_name == "champion":
        net = -100
    elif strat == "champion" and opponent_name == "simple":
        net = 150
    return SelfPlayResult(
        hands=hands,
        strat=strat,
        opponent=opponent_name,
        wins=seed,
        losses=players,
        pushes=0,
        net_chips=net,
        elapsed=0.01,
        players=players,
    )


def test_load_population_config_resolves_placeholders_and_dedupes(tmp_path):
    champion_path = write_champion(tmp_path)
    config_path = write_config(tmp_path)

    config = population_selfplay.load_population_config(
        config_path,
        candidate="candidate",
        champion_json=champion_path,
    )

    assert config.strategies == ("candidate", "champion")
    assert config.opponents == ("simple", "champion")
    assert config.score_strategy == "candidate"


def test_run_population_selfplay_scores_candidate_and_writes_reports(tmp_path):
    champion_path = write_champion(tmp_path)
    config_path = write_config(tmp_path)

    report = population_selfplay.run_population_selfplay(
        candidate="candidate",
        config_path=config_path,
        champion_json=champion_path,
        runner=fake_runner,
        output_json=tmp_path / "population.json",
        output_markdown=tmp_path / "population.md",
    )

    assert len(report.cases) == 8
    assert report.passed is True
    candidate_score = report.target_score
    assert candidate_score.strat == "candidate"
    assert candidate_score.worst_row_bb_per_100 < 0
    payload = json.loads((tmp_path / "population.json").read_text())
    assert payload["passed"] is True
    assert payload["gate"]["score"]["strat"] == "candidate"
    assert "Population self-play" in (tmp_path / "population.md").read_text()


def test_population_score_fails_on_weak_worst_row(tmp_path):
    champion_path = write_champion(tmp_path)
    config_path = write_config(tmp_path, min_worst=0.0)

    report = population_selfplay.run_population_selfplay(
        candidate="candidate",
        config_path=config_path,
        champion_json=champion_path,
        runner=fake_runner,
    )

    gate = population_selfplay.population_gate_summary(report)

    assert report.passed is False
    assert gate["passed"] is False
    assert "worst" in gate["detail"]
