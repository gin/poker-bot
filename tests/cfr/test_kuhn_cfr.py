import json

from eval import cfr_train
from poker_bot.cfr.kuhn import (
    InfoSetNode,
    best_response_value,
    legal_actions,
    terminal_utility_player0,
    train_kuhn,
    write_json_report,
    write_markdown_report,
)


def test_kuhn_legal_actions_and_terminal_utilities():
    assert legal_actions("") == ("x", "b")
    assert legal_actions("x") == ("x", "b")
    assert legal_actions("b") == ("c", "f")
    assert legal_actions("xb") == ("c", "f")
    assert legal_actions("xx") == ()

    assert terminal_utility_player0(("K", "J"), "xx") == 1.0
    assert terminal_utility_player0(("J", "K"), "bc") == -2.0
    assert terminal_utility_player0(("Q", "K"), "bf") == 1.0
    assert terminal_utility_player0(("Q", "K"), "xbf") == -1.0


def test_regret_matching_uses_positive_regrets():
    node = InfoSetNode.create("K|-", ("x", "b"))
    node.regret_sum["x"] = -4.0
    node.regret_sum["b"] = 2.0

    strategy = node.strategy(realization_weight=3.0)

    assert strategy == {"x": 0.0, "b": 1.0}
    assert node.strategy_sum == {"x": 0.0, "b": 3.0}


def test_kuhn_cfr_training_is_deterministic_and_learns_sensible_policy():
    first = train_kuhn(1000)
    second = train_kuhn(1000)

    assert first.strategy == second.strategy
    assert abs(first.average_game_value - (-1 / 18)) < 0.04
    assert abs(first.player0_best_response - (-1 / 18)) < 0.03
    assert abs(first.player1_best_response - (1 / 18)) < 0.03
    assert first.nash_conv < 0.03
    assert first.exploitability < 0.02
    assert first.strategy["K|-"]["bet"] > 0.55
    assert first.strategy["J|b"]["fold"] > 0.95
    assert first.strategy["K|b"]["call"] > 0.95


def test_best_response_value_is_finite_for_average_strategy():
    report = train_kuhn(200)
    raw_strategy = {
        key: {
            {"check": "x", "bet": "b", "call": "c", "fold": "f"}[action]: value
            for action, value in row.items()
        }
        for key, row in report.strategy.items()
    }

    assert -2.0 <= best_response_value(raw_strategy, player=0) <= 2.0
    assert -2.0 <= best_response_value(raw_strategy, player=1) <= 2.0


def test_write_cfr_reports(tmp_path):
    report = train_kuhn(50)
    json_path = tmp_path / "kuhn.json"
    md_path = tmp_path / "kuhn.md"

    write_json_report(report, json_path)
    write_markdown_report(report, md_path)

    payload = json.loads(json_path.read_text())
    assert payload["iterations"] == 50
    assert "nash_conv" in payload
    assert "K|-" in payload["strategy"]
    assert "# Kuhn CFR" in md_path.read_text()


def test_cfr_train_cli_writes_default_reports(tmp_path):
    exit_code = cfr_train.main(
        [
            "--iterations",
            "25",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "kuhn-25.json").exists()
    assert (tmp_path / "kuhn-25.md").exists()
