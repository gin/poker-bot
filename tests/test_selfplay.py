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


def test_selfplay_routes_canonical_v007_tag_as_alternate(monkeypatch):
    real_loader = selfplay.load_strategy
    simple_action = real_loader("simple")

    def tagged_action(table, my_seat):
        action, amount, _message = simple_action(table, my_seat)
        return action, amount, "[short_handed] [v007 canonical low-VPIP route]"

    monkeypatch.setattr(
        selfplay,
        "load_strategy",
        lambda name: tagged_action if name == "tagged" else real_loader(name),
    )

    result = run_selfplay(
        "tagged", hands=5, seed=7, opponent_name="simple", players=3
    )

    assert result.route_diagnostics.alternate_decisions > 0
    assert result.route_diagnostics.fallback_decisions == 0
    assert (
        result.route_diagnostics.alternate_hands
        == result.route_diagnostics.observed_hands
    )


def test_route_diagnostics_canonicalize_multiple_decisions_per_hand():
    collector = selfplay._RouteDiagnosticCollector()
    collector.observe(
        hand_id="hand-1",
        message="[short_handed] baseline decision",
    )
    collector.observe(
        hand_id="hand-1",
        message="[short_handed profile-gated s3v013] alternate decision",
    )
    collector.observe(
        hand_id="hand-1",
        message="[short_handed profile-gated s3v013] alternate decision",
    )
    collector.observe(hand_id="hand-2", message=None)

    diagnostics = collector.result()

    assert diagnostics.observed_hands == 2
    assert diagnostics.hero_decisions == 4
    assert diagnostics.alternate_decisions == 2
    assert diagnostics.fallback_decisions == 1
    assert diagnostics.unknown_decisions == 1
    assert diagnostics.alternate_hands == 1
    assert diagnostics.unknown_hands == 1
    assert diagnostics.activation_fraction == 0.5


def test_worker_result_aggregation_sums_route_diagnostics():
    first = SelfPlayResult(
        hands=2,
        strat="candidate",
        opponent="simple",
        wins=1,
        losses=1,
        pushes=0,
        net_chips=10,
        elapsed=0.1,
        route_diagnostics=selfplay.RouteDiagnostics(
            observed_hands=2,
            hero_decisions=3,
            alternate_decisions=2,
            fallback_decisions=1,
            alternate_hands=1,
            fallback_hands=1,
            profile_stats_schema_version=2,
            profile_stats_provenance="canonical",
        ),
    )
    second = SelfPlayResult(
        hands=3,
        strat="candidate",
        opponent="simple",
        wins=2,
        losses=1,
        pushes=0,
        net_chips=20,
        elapsed=0.1,
        route_diagnostics=selfplay.RouteDiagnostics(
            observed_hands=3,
            hero_decisions=4,
            unknown_decisions=4,
            unknown_hands=3,
            profile_stats_schema_version=2,
            profile_stats_provenance="canonical",
        ),
    )

    merged = selfplay._aggregate_worker_results((first, second), elapsed=0.2)

    assert merged.hands == 5
    assert merged.route_diagnostics.observed_hands == 5
    assert merged.route_diagnostics.alternate_decisions == 2
    assert merged.route_diagnostics.fallback_decisions == 1
    assert merged.route_diagnostics.unknown_decisions == 4
    assert merged.route_diagnostics.alternate_hands == 1
    assert merged.route_diagnostics.unknown_hands == 3
    assert merged.route_diagnostics.profile_stats_schema_version == 2
    assert merged.route_diagnostics.profile_stats_provenance == "canonical"


def test_run_selfplay_counts_every_hand():
    result = run_selfplay("all_in_everytime", hands=10, seed=3)

    assert result.wins + result.losses + result.pushes == 10


def test_run_selfplay_supports_six_max():
    result = run_selfplay("survival_sixmax", hands=10, seed=3, players=6)

    assert result.players == 6
    assert result.opponent == "simple"
    assert result.wins + result.losses + result.pushes == 10


@pytest.mark.parametrize(
    ("players", "opponent_id"),
    [(2, selfplay.BOT_AGENT_ID), (3, "bot-agent-1")],
)
def test_selfplay_persists_canonical_hand_profiles_and_route_metadata(
    tmp_path, players, opponent_id
):
    db_path = tmp_path / f"canonical-{players}.sqlite"

    result = run_selfplay(
        "all_in_everytime",
        hands=8,
        seed=19,
        players=players,
        track_opponents=True,
        opponent_db=db_path,
        db_commit_interval=0,
    )
    profile = load_profile(connect(db_path), "selfplay", opponent_id)

    assert profile is not None
    assert profile.has_canonical_preflop_stats is True
    assert profile.preflop_hands_seen == 8
    assert result.route_diagnostics.profile_stats_schema_version == 2
    assert result.route_diagnostics.profile_stats_provenance == "canonical"


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


def test_profiler_counts_only_voluntary_preflop_all_ins():
    profiler = PlayProfiler()
    profiler.start_hand("forced")
    profiler.observer(
        hand_id="forced",
        action="all-in",
        street="Preflop",
        voluntary=False,
        seat={"agentId": "player-agent"},
    )
    profiler.end_hand()
    profiler.start_hand("voluntary")
    profiler.observer(
        hand_id="voluntary",
        action="all-in",
        street="Preflop",
        voluntary=True,
        seat={"agentId": "player-agent"},
    )

    profile = profiler.compute_profile()
    assert (profile.vpip_pct, profile.pfr_pct) == (50.0, 50.0)


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
        profile_state_mode=selfplay.PROFILE_STATE_SHARDED_RESEARCH,
    )
    profile = load_profile(connect(db_path), "selfplay", "bot-agent-1")

    assert result.hands == 6
    assert profile is not None
    assert profile.hands_seen == 6
    assert profile.has_canonical_preflop_stats is True
    assert profile.preflop_hands_seen == 6
    assert 0 <= profile.pfr <= profile.vpip <= profile.preflop_hands_seen
    assert result.route_diagnostics.observed_hands == 6
    assert (
        result.route_diagnostics.hero_decisions
        >= result.route_diagnostics.observed_hands
    )
    assert (
        result.route_diagnostics.profile_state_mode
        == selfplay.PROFILE_STATE_SHARDED_RESEARCH
    )


def test_run_selfplay_parallel_rejects_tracked_profiles_without_research_opt_in():
    with pytest.raises(ValueError, match="one worker for persistent profiles"):
        run_selfplay_parallel(
            "simple",
            opponent_name="simple",
            hands=2,
            seed=3,
            players=2,
            track_opponents=True,
            workers=2,
        )




def test_sharded_profile_state_requires_parallel_workers():
    with pytest.raises(ValueError, match="requires more than one worker"):
        run_selfplay_parallel(
            "simple",
            opponent_name="simple",
            hands=2,
            seed=3,
            players=2,
            track_opponents=True,
            workers=1,
            profile_state_mode=selfplay.PROFILE_STATE_SHARDED_RESEARCH,
        )


def test_run_selfplay_parallel_preserves_sequential_profile_support(tmp_path):
    db_path = tmp_path / "sequential-opponents.sqlite"

    result = run_selfplay_parallel(
        "survival_lookahead",
        opponent_name="simple",
        hands=30,
        seed=3,
        players=2,
        opponent_db=db_path,
        workers=1,
        db_commit_interval=0,
    )
    profile = load_profile(connect(db_path), "selfplay", "bot-agent")

    assert profile is not None
    assert profile.hands_seen == 30
    assert (
        result.route_diagnostics.profile_state_mode
        == selfplay.PROFILE_STATE_PERSISTENT
    )


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
            "  route diag  : v1, 0/0 alternate hands (0.0%), 0/0/0 decisions "
            "(alternate/fallback/unknown)",
            "  profile state: untracked",
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


# ── Deal-stream pairing / deck RNG isolation ────────────────────────────────
#
# Promotion decisions compare a candidate strategy against the champion
# using the *same* seed for both runs. That is only a fair, paired
# comparison if the deck shuffle is fully independent of which strategy is
# playing -- i.e. the shared per-run RNG is consumed exactly once per hand
# (to shuffle the deck) and never touched by policy decisions. These tests
# verify that behaviorally (by actually running both strategies and
# diffing what got dealt), not by inspecting source text.


class _DealStreamRecorder(PlayProfiler):
    """Reuses the existing profiler action-observer hook to record the
    hole cards and board dealt each hand, keyed by hand_id."""

    def __init__(self):
        super().__init__()
        self.hole_cards_by_hand = {}
        self.board_by_hand = {}

    def _observe(self, **event):
        table = event.get("table")
        hand_id = event.get("hand_id")
        if table is not None and hand_id is not None:
            if hand_id not in self.hole_cards_by_hand:
                self.hole_cards_by_hand[hand_id] = tuple(
                    tuple(seat["holeCards"]) for seat in table["seats"]
                )
            board = table.get("boardCards") or []
            if len(board) > len(self.board_by_hand.get(hand_id, ())):
                self.board_by_hand[hand_id] = tuple(board)
        super()._observe(**event)




def test_deal_stream_is_identical_across_different_hero_strategies_same_seed():
    """The deck RNG must be isolated from policy-dependent RNG consumption:
    running two *different* strategies with the same seed/opponent/players
    must deal the exact same hole cards (and, whenever both reach a given
    street, the same board) hand by hand -- this is the property a paired
    candidate-vs-baseline promotion comparison depends on."""
    recorder_a = _DealStreamRecorder()
    recorder_b = _DealStreamRecorder()

    run_selfplay(
        "simple",
        hands=40,
        seed=99,
        opponent_name="all_in_everytime",
        players=3,
        profiler=recorder_a,
    )
    run_selfplay(
        "all_in_everytime",
        hands=40,
        seed=99,
        opponent_name="all_in_everytime",
        players=3,
        profiler=recorder_b,
    )

    assert recorder_a.hole_cards_by_hand, "expected at least one hand observed"
    assert set(recorder_a.hole_cards_by_hand) == set(recorder_b.hole_cards_by_hand)
    for hand_id, hole_cards in recorder_a.hole_cards_by_hand.items():
        assert hole_cards == recorder_b.hole_cards_by_hand[hand_id], (
            f"hole cards diverged for {hand_id}: policy decisions must never "
            "perturb the deck RNG"
        )

    # Whenever both runs reached the same (or a further) street for a hand,
    # the revealed board cards must match up to the shorter of the two --
    # i.e. the same shuffled deck order, only possibly truncated by an
    # earlier fold in one run.
    for hand_id, board_a in recorder_a.board_by_hand.items():
        board_b = recorder_b.board_by_hand.get(hand_id, ())
        common_len = min(len(board_a), len(board_b))
        assert board_a[:common_len] == board_b[:common_len]


def test_deal_stream_is_paired_between_two_run_selfplay_calls_like_promotion_gate():
    """Mirrors exactly how promotion_gate/benchmark pair candidate vs
    baseline: two independent run_selfplay calls, same seed/opponent
    lineup/player count, different hero strategy -- net_chips need not
    match, but the underlying deal stream (hole cards) must."""
    recorder_candidate = _DealStreamRecorder()
    recorder_baseline = _DealStreamRecorder()

    run_selfplay(
        "royal_adaptive",
        hands=25,
        seed=2024,
        opponent_name="simple",
        players=6,
        profiler=recorder_candidate,
    )
    run_selfplay(
        "adaptive",
        hands=25,
        seed=2024,
        opponent_name="simple",
        players=6,
        profiler=recorder_baseline,
    )

    assert recorder_candidate.hole_cards_by_hand == recorder_baseline.hole_cards_by_hand


# ── Configurable initial stack ──────────────────────────────────────────────


class _StackRecorder(PlayProfiler):
    """Reuses the profiler action-observer hook to capture the total chips
    in play (stack + current bet, summed across all seats) the first time
    a table is observed each hand."""

    def __init__(self):
        super().__init__()
        self.total_chips_by_hand = {}

    def _observe(self, **event):
        table = event.get("table")
        hand_id = event.get("hand_id")
        already_seen = hand_id in self.total_chips_by_hand
        if table is not None and hand_id is not None and not already_seen:
            self.total_chips_by_hand[hand_id] = sum(
                seat["stackChips"] + seat["currentBetChips"] for seat in table["seats"]
            )
        super()._observe(**event)


def test_run_selfplay_defaults_initial_stack_to_simulator_constant():
    result = run_selfplay("simple", hands=1, seed=1, opponent_name="simple", players=2)

    from simulator import INITIAL_STACK

    assert result.initial_stack == INITIAL_STACK


def test_run_selfplay_heads_up_uses_configured_initial_stack():
    recorder = _StackRecorder()

    result = run_selfplay(
        "simple",
        hands=3,
        seed=1,
        opponent_name="simple",
        players=2,
        initial_stack=2500,
        profiler=recorder,
    )

    assert result.initial_stack == 2500
    assert recorder.total_chips_by_hand
    assert all(total == 2500 * 2 for total in recorder.total_chips_by_hand.values())


def test_run_selfplay_multiway_uses_configured_initial_stack():
    recorder = _StackRecorder()

    result = run_selfplay(
        "simple",
        hands=3,
        seed=1,
        opponent_name="simple",
        players=4,
        initial_stack=3000,
        profiler=recorder,
    )

    assert result.initial_stack == 3000
    assert recorder.total_chips_by_hand
    assert all(total == 3000 * 4 for total in recorder.total_chips_by_hand.values())


def test_run_selfplay_hero_delta_measured_against_configured_initial_stack():
    """With all-in-everytime on both sides and a symmetric configured
    stack, the entire stack changes hands every hand -- net_chips must
    scale with initial_stack, not the module default."""
    small = run_selfplay(
        "all_in_everytime",
        hands=5,
        seed=7,
        opponent_name="all_in_everytime",
        players=2,
        initial_stack=500,
    )
    large = run_selfplay(
        "all_in_everytime",
        hands=5,
        seed=7,
        opponent_name="all_in_everytime",
        players=2,
        initial_stack=5000,
    )

    assert small.initial_stack == 500
    assert large.initial_stack == 5000
    # Every hand is a full double-up or bust at the configured stack size,
    # so |net_chips| is bounded by hands * initial_stack for each run, and
    # the large-stack run must show a proportionally larger swing given
    # the identical seed/strategy pairing.
    assert abs(small.net_chips) <= 5 * 500
    assert abs(large.net_chips) <= 5 * 5000
    assert abs(large.net_chips) > abs(small.net_chips)


def test_run_selfplay_rejects_non_positive_initial_stack():
    with pytest.raises(ValueError, match="initial-stack"):
        run_selfplay("simple", hands=1, initial_stack=0)


def test_run_selfplay_parallel_threads_initial_stack(monkeypatch):
    monkeypatch.setattr(selfplay, "_resolve_workers", lambda _w: 2)

    result = run_selfplay_parallel(
        "simple",
        hands=4,
        seed=1,
        opponent_name="simple",
        players=2,
        workers=2,
        initial_stack=1750,
    )

    assert result.initial_stack == 1750


def test_build_parser_initial_stack_flag_defaults():
    from simulator import INITIAL_STACK

    args = build_parser().parse_args(["--strat", "simple"])
    assert args.initial_stack == INITIAL_STACK

    args_custom = build_parser().parse_args(
        ["--strat", "simple", "--initial-stack", "4000"]
    )
    assert args_custom.initial_stack == 4000
