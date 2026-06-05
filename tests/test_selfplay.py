import pytest

from eval.selfplay import (
    DEFAULT_HANDS,
    SelfPlayResult,
    build_parser,
    format_result,
    run_selfplay,
)
from poker_bot.strategies.all_in_everytime import choose_action as all_in
from poker_bot.strategies.loader import load_strategy


def test_load_strategy_loads_choose_action():
    assert load_strategy("all_in_everytime") is all_in


def test_load_strategy_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        load_strategy("does_not_exist")


def test_parser_defaults_hands_to_200():
    args = build_parser().parse_args(["--strat", "all_in_everytime"])

    assert args.strat == "all_in_everytime"
    assert args.opponent == "simple"
    assert args.hands == DEFAULT_HANDS
    assert args.seed is None


def test_parser_accepts_opponent_strategy():
    args = build_parser().parse_args(
        ["--strat", "all_in_everytime", "--opponent", "adaptive"]
    )

    assert args.opponent == "adaptive"


def test_run_selfplay_is_reproducible_with_seed():
    first = run_selfplay("all_in_everytime", hands=12, seed=7)
    second = run_selfplay("all_in_everytime", hands=12, seed=7)

    assert first.hands == 12
    assert first.wins == second.wins
    assert first.losses == second.losses
    assert first.pushes == second.pushes
    assert first.net_chips == second.net_chips


def test_run_selfplay_counts_every_hand():
    result = run_selfplay("all_in_everytime", hands=10, seed=3)

    assert result.wins + result.losses + result.pushes == 10


def test_format_result_matches_expected_shape():
    result = SelfPlayResult(
        hands=200,
        strat="all_in_everytime",
        opponent="simple",
        wins=148,
        losses=50,
        pushes=2,
        net_chips=144,
        elapsed=2.25,
    )

    assert format_result(result) == "\n".join(
        [
            "  hands       : 200",
            "  opponent    : simple x1",
            "  wins/losses : 148/50  (push: 2)",
            "  net chips   : +144",
            "  bb/100      : +1.4",
            "  elapsed     : 2.2s  (88 hands/s)",
        ]
    )
