import pytest

import eval.selfplay as selfplay
from eval.profiler import PlayProfiler
from eval.selfplay import (
    DEFAULT_HANDS,
    SelfPlayResult,
    build_parser,
    format_result,
    infer_player_count,
    parse_opponent_lineup,
    resolve_opponent_lineup,
    run_selfplay,
    run_selfplay_parallel,
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
    assert args.players is None
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


def test_run_selfplay_profiler_tracks_preflop_in_six_max():
    profiler = PlayProfiler()
    result = run_selfplay(
        "all_in_everytime",
        hands=20,
        seed=3,
        players=6,
        profiler=profiler,
    )
    profile = profiler.compute_profile()

    assert result.players == 6
    assert profile.total_hands == 20
    assert profile.vpip_pct > 0
    assert profile.pfr_pct > 0


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


def test_run_selfplay_supports_comma_mixed_multiway_lineup_with_inferred_players():
    result = run_selfplay(
        "survival_sixmax",
        hands=5,
        seed=3,
        opponent_name="simple,adaptive",
    )

    assert result.players == 3
    assert result.opponent == "simple+adaptive"
    assert result.wins + result.losses + result.pushes == 5


def test_parse_opponent_lineup_accepts_comma_or_plus():
    assert parse_opponent_lineup("simple, adaptive+royal_adaptive") == (
        "simple",
        "adaptive",
        "royal_adaptive",
    )


def test_infer_player_count_uses_opponent_lineup_count():
    assert infer_player_count("simple,adaptive") == 3
    assert infer_player_count("simple") == 2


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


def test_run_selfplay_rejects_explicit_player_count_mismatch():
    with pytest.raises(ValueError, match="exactly one strategy per opponent seat"):
        run_selfplay(
            "survival_sixmax",
            hands=5,
            seed=3,
            opponent_name="simple,adaptive",
            players=2,
        )


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


def test_run_selfplay_heads_up_persists_matching_opponent_profile(tmp_path):
    db_path = tmp_path / "heads-up-opponents.sqlite"

    result = run_selfplay(
        "all_in_everytime",
        opponent_name="simple",
        hands=5,
        seed=3,
        players=2,
        opponent_db=db_path,
    )
    profile = load_profile(connect(db_path), "selfplay", "bot-agent")

    assert result.hands == 5
    assert profile is not None
    assert profile.hands_seen == 5
    assert profile.calls + profile.bets + profile.raises + profile.folds > 0


def test_run_selfplay_heads_up_telemetry_records_outcomes(tmp_path):
    db_path = tmp_path / "heads-up-telemetry.sqlite"

    result = run_selfplay(
        "all_in_everytime",
        opponent_name="simple",
        hands=5,
        seed=3,
        players=2,
        opponent_db=db_path,
        telemetry=True,
        telemetry_run_id="heads-up-run",
    )
    conn = connect(db_path)
    missing_outcomes = conn.execute(
        """
        select count(*) as count
        from decision_telemetry
        where run_id = ? and hero_net_chips is null
        """,
        ("heads-up-run",),
    ).fetchone()["count"]

    assert result.hands == 5
    assert missing_outcomes == 0


def test_run_selfplay_injects_live_opponent_profiles(monkeypatch):
    observed_bot_hands_seen = []
    observed_bot_actions = []

    def probe_strategy(table, seat):
        profile = (table.get("opponentProfiles") or {}).get("bot-agent")
        if profile is not None:
            observed_bot_hands_seen.append(profile.hands_seen)
            observed_bot_actions.append(
                profile.calls + profile.bets + profile.raises + profile.folds
            )
        allowed = table["allowedActions"]
        if "check" in allowed["availableActions"]:
            return "check", None, "probe check"
        if "call" in allowed["availableActions"]:
            return "call", None, "probe call"
        return "fold", None, "probe fold"

    def aggressive_opponent(table, seat):
        allowed = table["allowedActions"]
        if "raise" in allowed["availableActions"]:
            return "raise", allowed["minRaiseTo"], "probe raise"
        if "bet" in allowed["availableActions"]:
            return "bet", allowed["minBet"], "probe bet"
        if "call" in allowed["availableActions"]:
            return "call", None, "probe call"
        return "check", None, "probe check"

    def fake_load_strategy(name):
        if name == "profile_probe":
            return probe_strategy
        if name == "profile_opponent":
            return aggressive_opponent
        return load_strategy(name)

    monkeypatch.setattr(selfplay, "load_strategy", fake_load_strategy)

    result = run_selfplay(
        "profile_probe",
        opponent_name="profile_opponent",
        hands=4,
        seed=3,
        players=2,
        track_opponents=True,
    )

    assert result.hands == 4
    assert observed_bot_hands_seen
    assert max(observed_bot_hands_seen) >= 2
    assert max(observed_bot_actions) > 0


def test_run_selfplay_batches_db_commits(tmp_path):
    db_path = tmp_path / "opponents.sqlite"

    result = run_selfplay(
        "survival_lookahead",
        opponent_name="simple",
        hands=5,
        seed=3,
        players=6,
        opponent_db=db_path,
        db_commit_interval=0,
    )
    profile = load_profile(connect(db_path), "selfplay", "bot-agent-1")

    assert result.hands == 5
    assert profile is not None
    assert profile.hands_seen == 5


def test_run_selfplay_parallel_merges_worker_dbs(tmp_path):
    db_path = tmp_path / "parallel-opponents.sqlite"

    result = run_selfplay_parallel(
        "survival_lookahead",
        opponent_name="simple",
        hands=6,
        seed=3,
        players=6,
        opponent_db=db_path,
        workers=2,
        db_commit_interval=0,
    )
    profile = load_profile(connect(db_path), "selfplay", "bot-agent-1")

    assert result.hands == 6
    assert profile is not None
    assert profile.hands_seen == 6


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
