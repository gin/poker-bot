import pytest

from eval.selfplay import (
    DEFAULT_HANDS,
    SelfPlayResult,
    build_parser,
    format_result,
    resolve_opponent_lineup,
    run_selfplay,
)
from poker_bot.opponent_store import connect, load_profile
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
    assert args.players == 2
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


def test_run_selfplay_supports_six_max():
    result = run_selfplay("survival_sixmax", hands=10, seed=3, players=6)

    assert result.players == 6
    assert result.opponent == "simple"
    assert result.wins + result.losses + result.pushes == 10


def test_run_selfplay_supports_mixed_multiway_lineup():
    result = run_selfplay(
        "survival_sixmax",
        hands=5,
        seed=3,
        opponent_name=(
            "simple+adaptive+royal_adaptive+threshold_pressure+anti_threshold"
        ),
        players=6,
    )

    assert result.players == 6
    assert result.opponent == (
        "simple+adaptive+royal_adaptive+threshold_pressure+anti_threshold"
    )
    assert result.wins + result.losses + result.pushes == 5


def test_resolve_opponent_lineup_repeats_single_opponent():
    assert resolve_opponent_lineup("simple", players=6) == (
        "simple",
        "simple",
        "simple",
        "simple",
        "simple",
    )


def test_resolve_opponent_lineup_requires_one_strategy_per_opponent_seat():
    with pytest.raises(ValueError, match="exactly one strategy per opponent seat"):
        resolve_opponent_lineup("simple+adaptive", players=6)


def test_run_selfplay_can_persist_opponent_profiles(tmp_path):
    db_path = tmp_path / "opponents.sqlite"

    result = run_selfplay(
        "survival_lookahead",
        opponent_name="simple",
        hands=5,
        seed=3,
        players=6,
        opponent_db=db_path,
    )
    profile = load_profile(connect(db_path), "selfplay", "bot-agent-1")

    assert result.hands == 5
    assert profile is not None
    assert profile.hands_seen == 5
    assert profile.calls + profile.bets + profile.raises + profile.folds > 0


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
            "  chips/hand  : +0.7",
            "  bb/100      : +7.2",
            "  elapsed     : 2.2s  (88 hands/s)",
            "  profile     : (not profiled)",
        ]
    )


def test_format_result_shows_mixed_lineup_without_repeat_suffix():
    result = SelfPlayResult(
        hands=20,
        strat="candidate",
        opponent="simple+adaptive",
        wins=10,
        losses=8,
        pushes=2,
        net_chips=50,
        elapsed=1.0,
        players=3,
    )

    output = format_result(result)
    assert "  opponent    : simple+adaptive" in output
    assert "simple+adaptive x2" not in output
    assert "profile     : (not profiled)" in output
