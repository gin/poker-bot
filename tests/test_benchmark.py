import json

import pytest

from eval import benchmark, selfplay
from eval.selfplay import SelfPlayResult
from poker_bot import opponent_store
from simulator import BOT_AGENT_ID


def fake_runner(
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
    net = seed * 10 if opponent_name == "simple" else -seed * 5
    if strat == "baseline":
        net -= seed * 4
    if strat == "weak_candidate":
        net -= seed * 20
    return SelfPlayResult(
        hands=hands,
        strat=strat,
        opponent=opponent_name,
        wins=seed,
        losses=players,
        pushes=1,
        net_chips=net,
        elapsed=0.01,
        players=players,
    )


def db_writing_fake_runner(
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
    """A picklable (module-level) fake runner that actually writes to
    whatever per-case ``opponent_db`` it's given -- used to prove the
    orchestrator hands each parallel worker a genuinely distinct,
    isolated file rather than one shared path."""
    if opponent_db is not None:
        conn = opponent_store.connect(opponent_db)
        for _ in range(hands):
            opponent_store.increment_hand_seen(conn, "selfplay", "bot-agent-1")
        conn.commit()
        conn.close()
    return fake_runner(
        strat,
        hands=hands,
        seed=seed,
        opponent_name=opponent_name,
        players=players,
        track_opponents=track_opponents,
        opponent_db=opponent_db,
        **kwargs,
    )


def test_parse_csv_helpers():
    assert benchmark.parse_csv_strings("simple, adaptive") == ("simple", "adaptive")
    assert benchmark.parse_csv_ints("1,2, 3") == (1, 2, 3)


def test_build_cases_crosses_opponents_players_and_seeds():
    cases = benchmark.build_cases(
        "candidate",
        opponents=("simple", "adaptive"),
        players=(2, 6),
        seeds=(1, 2),
        hands=100,
    )

    assert len(cases) == 8
    assert cases[0] == benchmark.BenchmarkCase(
        strat="candidate",
        opponent="simple",
        players=2,
        seed=1,
        hands=100,
    )
    assert cases[-1].opponent == "adaptive"
    assert cases[-1].players == 6
    assert cases[-1].seed == 2


def test_resolve_options_loads_config_and_allows_cli_overrides(tmp_path):
    config_path = tmp_path / "benchmark.json"
    config_path.write_text(
        json.dumps(
            {
                "hands": 500,
                "opponents": ["simple", "adaptive"],
                "players": [2, 6],
                "seeds": [7, 11],
                "track_opponents": True,
                "baseline": "baseline",
                "min_delta_bb_per_100": 1.5,
                "fail_under_bb100": -10.0,
            }
        )
    )
    args = benchmark.build_parser().parse_args(
        [
            "--strat",
            "candidate",
            "--config",
            str(config_path),
            "--players",
            "6",
        ]
    )

    options = benchmark.resolve_options(args)

    assert options["hands"] == 500
    assert options["opponents"] == ("simple", "adaptive")
    assert options["players"] == (6,)
    assert options["seeds"] == (7, 11)
    assert options["track_opponents"] is True
    assert options["baseline"] == "baseline"
    assert options["min_delta_bb_per_100"] == 1.5
    assert options["fail_under_bb100"] == -10.0


def test_resolve_options_h2h_defaults_to_heads_up_and_tracking():
    args = benchmark.build_parser().parse_args(
        [
            "--strat",
            "candidate",
            "--h2h",
            "--opponent",
            "simple,all_in_everytime,adaptive",
            "--hands",
            "50",
            "--seeds",
            "1",
        ]
    )

    options = benchmark.resolve_options(args)

    assert options["h2h"] is True
    assert options["opponents"] == ("simple", "all_in_everytime", "adaptive")
    assert options["players"] == (2,)
    assert options["track_opponents"] is True


def test_resolve_options_h2h_rejects_non_heads_up_players():
    args = benchmark.build_parser().parse_args(
        ["--strat", "candidate", "--h2h", "--players", "6"]
    )

    with pytest.raises(ValueError, match="heads-up"):
        benchmark.resolve_options(args)


def test_run_benchmark_aggregates_by_opponent_and_player_count():
    report = benchmark.run_benchmark(
        "candidate",
        opponents=("simple",),
        players=(2, 6),
        seeds=(1, 2),
        hands=100,
        runner=fake_runner,
    )

    assert len(report.cases) == 4
    assert len(report.results) == 4
    rows = {(row.opponent, row.players): row for row in report.aggregates}
    assert rows[("simple", 2)].hands == 200
    assert rows[("simple", 2)].seeds == (1, 2)
    assert rows[("simple", 2)].net_chips == 30
    assert rows[("simple", 6)].wins == 3
    assert rows[("simple", 6)].losses == 12


def test_run_benchmark_preserves_requested_opponent_order():
    report = benchmark.run_benchmark(
        "candidate",
        opponents=("adaptive", "simple", "all_in_everytime"),
        players=(2,),
        seeds=(1,),
        hands=100,
        runner=fake_runner,
        workers=1,
    )

    assert [row.opponent for row in report.aggregates] == [
        "adaptive",
        "simple",
        "all_in_everytime",
    ]
    text = benchmark.format_report(report)
    assert text.index("adaptive") < text.index("simple")
    assert text.index("simple") < text.index("all_in_everytime")


def test_run_benchmark_h2h_enables_tracking_and_isolated_platforms():
    calls = []

    def capture_runner(
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
        calls.append(
            {
                "opponent": opponent_name,
                "track_opponents": track_opponents,
                "platform": kwargs.get("platform"),
            }
        )
        return fake_runner(
            strat,
            hands=hands,
            seed=seed,
            opponent_name=opponent_name,
            players=players,
            track_opponents=track_opponents,
            opponent_db=opponent_db,
            **kwargs,
        )

    report = benchmark.run_benchmark(
        "candidate",
        opponents=("simple", "adaptive"),
        players=(2,),
        seeds=(1,),
        hands=100,
        runner=capture_runner,
        workers=1,
        h2h=True,
    )

    assert report.h2h is True
    assert all(call["track_opponents"] is True for call in calls)
    assert calls[0]["platform"] == "benchmark-h2h:candidate:vs:simple:p2:seed1"
    assert calls[1]["platform"] == "benchmark-h2h:candidate:vs:adaptive:p2:seed1"


def test_run_benchmark_compares_candidate_to_baseline():
    report = benchmark.run_benchmark(
        "candidate",
        opponents=("simple",),
        players=(2,),
        seeds=(1, 2),
        hands=100,
        baseline_strat="baseline",
        min_delta_bb_per_100=0.0,
        runner=fake_runner,
    )

    assert report.passed is True
    assert report.baseline_strat == "baseline"
    assert len(report.baseline_results) == 2
    assert len(report.comparisons) == 1
    assert report.comparisons[0].delta_bb_per_100 > 0


def test_run_benchmark_fails_when_candidate_lags_baseline():
    report = benchmark.run_benchmark(
        "weak_candidate",
        opponents=("simple",),
        players=(2,),
        seeds=(1,),
        hands=100,
        baseline_strat="baseline",
        min_delta_bb_per_100=0.0,
        runner=fake_runner,
    )

    assert report.passed is False
    assert report.comparisons[0].passed is False


def test_build_cases_accepts_mixed_lineup_expression():
    cases = benchmark.build_cases(
        "candidate",
        opponents=("simple+adaptive+royal_adaptive+threshold_pressure+anti_threshold",),
        players=(6,),
        seeds=(1,),
        hands=100,
    )

    assert cases[0].opponent == (
        "simple+adaptive+royal_adaptive+threshold_pressure+anti_threshold"
    )


def test_format_report_includes_gate_status():
    report = benchmark.run_benchmark(
        "candidate",
        opponents=("simple",),
        players=(2,),
        seeds=(1,),
        hands=100,
        fail_under_bb100=0.0,
        runner=fake_runner,
    )

    text = benchmark.format_report(report)

    assert "benchmark   : candidate" in text
    assert "simple" in text
    assert "gate        : PASS bb/100 >= 0.0" in text


def test_format_report_omits_gate_when_no_gate_configured():
    report = benchmark.run_benchmark(
        "candidate",
        opponents=("simple",),
        players=(2,),
        seeds=(1,),
        hands=100,
        runner=fake_runner,
    )

    text = benchmark.format_report(report)

    assert report.passed is None
    assert "gate        :" not in text


def test_format_report_includes_baseline_comparison():
    report = benchmark.run_benchmark(
        "candidate",
        opponents=("simple",),
        players=(2,),
        seeds=(1,),
        hands=100,
        baseline_strat="baseline",
        min_delta_bb_per_100=0.0,
        runner=fake_runner,
    )

    text = benchmark.format_report(report)

    assert "baseline    : baseline" in text
    assert "delta vs baseline >= 0.0" in text


def test_write_json_report(tmp_path):
    report = benchmark.run_benchmark(
        "candidate",
        opponents=("simple",),
        players=(2,),
        seeds=(1,),
        hands=100,
        fail_under_bb100=0.0,
        runner=fake_runner,
    )
    output_path = tmp_path / "report.json"

    benchmark.write_json_report(report, output_path)

    payload = json.loads(output_path.read_text())
    assert payload["strat"] == "candidate"
    assert payload["passed"] is True
    assert payload["aggregates"][0]["opponent"] == "simple"


def test_write_json_report_includes_baseline_comparison(tmp_path):
    report = benchmark.run_benchmark(
        "candidate",
        opponents=("simple",),
        players=(2,),
        seeds=(1,),
        hands=100,
        baseline_strat="baseline",
        min_delta_bb_per_100=0.0,
        runner=fake_runner,
        workers=1,
    )
    output_path = tmp_path / "report.json"

    benchmark.write_json_report(report, output_path)

    payload = json.loads(output_path.read_text())
    assert payload["baseline_strat"] == "baseline"
    assert payload["comparisons"][0]["passed"] is True
    assert payload["workers"] == 1


def test_run_benchmark_parallel_workers_returns_results_in_order(monkeypatch):
    monkeypatch.setattr(benchmark, "_resolve_workers", lambda _w: 2)

    report = benchmark.run_benchmark(
        "candidate",
        opponents=("simple", "adaptive"),
        players=(2,),
        seeds=(1, 2, 3),
        hands=100,
        runner=fake_runner,
        workers=2,
    )

    assert len(report.results) == 6
    assert report.workers == 2
    # Net chips should be deterministic per case; verify a couple of slots
    first = report.results[0]
    assert first.opponent in {"simple", "adaptive"}
    assert first.strat == "candidate"


def test_run_benchmark_parallel_workers_isolate_and_merge_per_case_profile_dbs(
    tmp_path,
):
    """Parallel (workers>1) dispatch must also give every case its own
    isolated opponent-profile DB, not just the sequential (workers=1)
    path -- and still transactionally merge every finished case's
    snapshot into the requested --db-path aggregate, summing across
    cases rather than one worker's file silently overwriting another's."""
    db_path = tmp_path / "aggregate.sqlite"
    report = benchmark.run_benchmark(
        "candidate",
        opponents=("simple", "adaptive"),
        players=(2,),
        seeds=(1,),
        hands=4,
        track_opponents=True,
        opponent_db=db_path,
        runner=db_writing_fake_runner,
        workers=2,
    )

    assert len(report.results) == 2
    conn = opponent_store.connect(db_path)
    profile = opponent_store.load_profile(conn, "selfplay", "bot-agent-1")
    assert profile is not None
    # Both cases wrote 4 hands each to their OWN isolated per-case file;
    # the merge sums both into the shared aggregate (4 + 4 == 8), proving
    # neither worker process's file was shared with or clobbered by the
    # other's.
    assert profile.hands_seen == 8


def test_run_benchmark_workers_safety_cap(monkeypatch):
    monkeypatch.setenv("POKER_BENCHMARK_ALLOW_HIGH_WORKERS", "0")
    with pytest.raises(ValueError, match="exceeds safety cap"):
        benchmark._resolve_workers(20)


def test_run_benchmark_workers_safety_cap_override(monkeypatch):
    monkeypatch.setenv("POKER_BENCHMARK_ALLOW_HIGH_WORKERS", "1")
    assert benchmark._resolve_workers(20) == 20


def test_merge_worker_db_sums_opponent_stats(tmp_path):
    main_db = tmp_path / "main.sqlite"
    worker_a = tmp_path / "worker_a.sqlite"
    worker_b = tmp_path / "worker_b.sqlite"

    # Initialize all three DBs
    opponent_store.connect(main_db)
    conn_a = opponent_store.connect(worker_a)
    conn_b = opponent_store.connect(worker_b)

    # Create the same opponent in each worker
    oid_a = opponent_store.upsert_opponent(conn_a, "selfplay", "villain", "v")
    opponent_store.increment_hand_seen(conn_a, "selfplay", "villain", "v")
    opponent_store.increment_hand_seen(conn_a, "selfplay", "villain", "v")
    conn_a.close()

    opponent_store.upsert_opponent(conn_b, "selfplay", "villain", "v")
    opponent_store.increment_hand_seen(conn_b, "selfplay", "villain", "v")
    conn_b.close()

    # Merge both into the main DB
    opponent_store.merge_worker_db(str(main_db), str(worker_a))
    opponent_store.merge_worker_db(str(main_db), str(worker_b))

    main_conn = opponent_store.connect(str(main_db))
    row = main_conn.execute(
        "SELECT * FROM opponent_stats WHERE opponent_id = ?",
        (oid_a,),
    ).fetchone()
    assert row is not None
    # Two increments in worker_a, one in worker_b -> total 3
    assert row["hands_seen"] == 3
    main_conn.close()


def test_merge_worker_db_maps_opponents_by_platform_and_agent(tmp_path):
    main_db = tmp_path / "main.sqlite"
    worker_db = tmp_path / "worker.sqlite"

    main_conn = opponent_store.connect(main_db)
    decoy_id = opponent_store.upsert_opponent(main_conn, "selfplay", "decoy")
    main_conn.close()

    worker_conn = opponent_store.connect(worker_db)
    worker_id = opponent_store.upsert_opponent(worker_conn, "selfplay", "villain")
    assert worker_id == decoy_id
    opponent_store.increment_hand_seen(
        worker_conn, "selfplay", "villain", hand_id="h1"
    )
    opponent_store.record_observed_action(
        worker_conn,
        platform="selfplay",
        agent_id="villain",
        hand_id="h1",
        street="Preflop",
        action="raise",
        amount=100,
        pot=75,
        voluntary=True,
    )
    worker_conn.close()

    opponent_store.merge_worker_db(str(main_db), str(worker_db))

    main_conn = opponent_store.connect(str(main_db))
    villain = main_conn.execute(
        """
        SELECT o.id, s.hands_seen, s.preflop_hands_seen, s.vpip, s.pfr,
               s.profile_stats_provenance, s.raises
        FROM opponents AS o
        JOIN opponent_stats AS s ON s.opponent_id = o.id
        WHERE o.agent_id = 'villain'
        """
    ).fetchone()
    action = main_conn.execute("SELECT opponent_id FROM opponent_actions").fetchone()

    assert villain is not None
    assert villain["id"] != decoy_id
    assert villain["hands_seen"] == 1
    assert villain["raises"] == 1
    assert villain["preflop_hands_seen"] == 1
    assert (villain["vpip"], villain["pfr"]) == (1, 1)
    assert villain["profile_stats_provenance"] == "canonical"
    assert action["opponent_id"] == villain["id"]
    main_conn.close()


# ── Multi-stack-depth matrix ─────────────────────────────────────────────────


def stack_aware_runner(
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
    """Deterministic runner whose net_chips depends on initial_stack, so
    tests can prove the stack value actually reaches the runner and that
    per-stack rows stay distinct."""
    assert initial_stack is not None
    net = initial_stack // 10 + seed
    if strat == "baseline":
        net -= 1000
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


def test_build_cases_defaults_to_single_module_initial_stack():
    cases = benchmark.build_cases(
        "candidate",
        opponents=("simple",),
        players=(2,),
        seeds=(1, 2),
        hands=100,
    )

    assert len(cases) == 2
    assert all(case.initial_stack == benchmark.INITIAL_STACK for case in cases)


def test_build_cases_crosses_initial_stacks():
    cases = benchmark.build_cases(
        "candidate",
        opponents=("simple", "adaptive"),
        players=(2, 6),
        seeds=(1, 2),
        hands=100,
        initial_stacks=(500, 1000, 2000),
    )

    # 2 opponents x 2 player counts x 3 stacks x 2 seeds
    assert len(cases) == 24
    assert {case.initial_stack for case in cases} == {500, 1000, 2000}
    # Every (opponent, players, stack) combination is covered by both seeds.
    for opponent in ("simple", "adaptive"):
        for player_count in (2, 6):
            for stack in (500, 1000, 2000):
                matching = [
                    case
                    for case in cases
                    if case.opponent == opponent
                    and case.players == player_count
                    and case.initial_stack == stack
                ]
                assert {case.seed for case in matching} == {1, 2}


def test_build_cases_rejects_non_positive_initial_stack():
    with pytest.raises(ValueError, match="initial-stacks"):
        benchmark.build_cases(
            "candidate",
            opponents=("simple",),
            players=(2,),
            seeds=(1,),
            hands=100,
            initial_stacks=(0,),
        )


def test_aggregate_results_groups_separately_by_initial_stack():
    cases = benchmark.build_cases(
        "candidate",
        opponents=("simple",),
        players=(6,),
        seeds=(1,),
        hands=100,
        initial_stacks=(500, 2000),
    )
    results = [
        stack_aware_runner(
            case.strat,
            hands=case.hands,
            seed=case.seed,
            opponent_name=case.opponent,
            players=case.players,
            initial_stack=case.initial_stack,
        )
        for case in cases
    ]

    rows = benchmark.aggregate_results(cases, results)

    assert len(rows) == 2
    by_stack = {row.initial_stack: row for row in rows}
    assert set(by_stack) == {500, 2000}
    # Distinct net_chips per stack prove the rows were not merged together.
    assert by_stack[500].net_chips != by_stack[2000].net_chips




def test_route_diagnostics_aggregate_and_serialize_per_seed():
    cases = benchmark.build_cases(
        "candidate",
        opponents=("adaptive",),
        players=(3,),
        seeds=(1, 2),
        hands=10,
    )
    results = (
        SelfPlayResult(
            hands=10,
            strat="candidate",
            opponent="adaptive",
            wins=1,
            losses=0,
            pushes=9,
            net_chips=20,
            elapsed=0.01,
            players=3,
            route_diagnostics=benchmark.RouteDiagnostics(
                observed_hands=10,
                hero_decisions=15,
                alternate_decisions=15,
                alternate_hands=10,
            ),
        ),
        SelfPlayResult(
            hands=10,
            strat="candidate",
            opponent="adaptive",
            wins=0,
            losses=1,
            pushes=9,
            net_chips=-10,
            elapsed=0.01,
            players=3,
            route_diagnostics=benchmark.RouteDiagnostics(
                observed_hands=10,
                hero_decisions=10,
                fallback_decisions=10,
                fallback_hands=10,
            ),
        ),
    )

    aggregate = benchmark.aggregate_results(cases, results)[0]
    report = benchmark.BenchmarkReport(
        strat="candidate",
        cases=cases,
        results=results,
        aggregates=(aggregate,),
        elapsed=0.02,
    )
    payload = benchmark.report_to_jsonable(report)

    assert aggregate.route_diagnostics.observed_hands == 20
    assert aggregate.route_diagnostics.alternate_hands == 10
    assert aggregate.route_diagnostics.fallback_hands == 10
    assert aggregate.route_diagnostics.activation_fraction == 0.5
    assert payload["aggregates"][0]["route_diagnostics"] == {
        "schema_version": 1,
        "observed_hands": 20,
        "hero_decisions": 25,
        "alternate_decisions": 15,
        "fallback_decisions": 10,
        "unknown_decisions": 0,
        "alternate_hands": 10,
        "fallback_hands": 10,
        "unknown_hands": 0,
        "profile_stats_schema_version": None,
        "profile_stats_provenance": None,
        "profile_state_mode": "untracked",
    }
    assert (
        "route diag v1: profile state untracked, alt 10/20 hands (50.0%)"
        in benchmark.format_report(report)
    )


def test_tracked_benchmark_reports_persistent_profile_state():
    report = benchmark.run_benchmark(
        "simple",
        opponents=("simple",),
        players=(2,),
        seeds=(1,),
        hands=2,
        track_opponents=True,
        workers=1,
    )

    assert report.aggregates[0].route_diagnostics.profile_state_mode == "persistent"
    assert (
        benchmark.report_to_jsonable(report)["aggregates"][0]["route_diagnostics"][
            "profile_state_mode"
        ]
        == "persistent"
    )
    assert "profile state persistent" in benchmark.format_report(report)


# ── Per-case opponent-profile DB isolation ──────────────────────────────────


def _probe_strategy(observed):
    def probe(table, seat):
        profile = (table.get("opponentProfiles") or {}).get(BOT_AGENT_ID)
        if profile is not None:
            observed.append(profile.hands_seen)
        allowed = table["allowedActions"]
        if "check" in allowed["availableActions"]:
            return "check", None, "probe check"
        if "call" in allowed["availableActions"]:
            return "call", None, "probe call"
        return "fold", None, "probe fold"

    return probe


def test_sequential_benchmark_cases_start_each_opponent_profile_cold(
    tmp_path, monkeypatch
):
    """Two sequential (workers=1) benchmark cases against DIFFERENT
    opponents share the exact same synthetic seat id at players=2
    (BOT_AGENT_ID). The second case must never inherit the first case's
    accumulated opponent-profile history -- even though both cases share
    track_opponents=True and a persistent --db-path aggregate -- proving
    every logical (strategy, opponent, players, stack, seed) run starts
    with a fresh, isolated profile DB rather than one shared sequential
    connection."""
    observed: list[int] = []
    real_load_strategy = selfplay.load_strategy

    def fake_load_strategy(name):
        if name == "profile_probe":
            return _probe_strategy(observed)
        return real_load_strategy(name)

    monkeypatch.setattr(selfplay, "load_strategy", fake_load_strategy)

    db_path = tmp_path / "aggregate.sqlite"
    benchmark.run_benchmark(
        "profile_probe",
        opponents=("simple", "adaptive"),
        players=(2,),
        seeds=(1,),
        hands=5,
        track_opponents=True,
        opponent_db=db_path,
        workers=1,
    )

    assert observed
    # If the second case inherited the first case's history, hands_seen
    # would climb past 5 (case 1's own hand count) into 6..10. Since both
    # cases are isolated, every observation stays within case 1's own
    # 1..5 range -- case 2 restarts at 1 rather than continuing at 6.
    assert max(observed) == 5
    assert observed.count(1) >= 2, (
        "expected at least two independent cold starts (one per opponent "
        f"case), got {observed}"
    )


def test_candidate_and_baseline_benchmark_runs_never_share_profile_state(
    tmp_path, monkeypatch
):
    """`run_benchmark(..., baseline_strat=...)` runs every candidate case
    and then every baseline case against the same opponent/players/seed --
    i.e. the same synthetic seat id. The baseline pass must never inherit
    the candidate pass's accumulated opponent-profile history."""
    observed: list[int] = []
    real_load_strategy = selfplay.load_strategy

    def fake_load_strategy(name):
        if name in ("profile_probe_candidate", "profile_probe_baseline"):
            return _probe_strategy(observed)
        return real_load_strategy(name)

    monkeypatch.setattr(selfplay, "load_strategy", fake_load_strategy)

    db_path = tmp_path / "aggregate.sqlite"
    report = benchmark.run_benchmark(
        "profile_probe_candidate",
        opponents=("simple",),
        players=(2,),
        seeds=(1,),
        hands=5,
        baseline_strat="profile_probe_baseline",
        track_opponents=True,
        opponent_db=db_path,
        workers=1,
    )

    assert report.baseline_strat == "profile_probe_baseline"
    assert observed
    # Candidate plays 5 hands first (hands_seen 1..5); if the baseline
    # pass inherited that state its own 5 hands would read 6..10 instead
    # of restarting at 1..5.
    assert max(observed) == 5
    assert observed.count(1) >= 2, (
        "expected independent cold starts for candidate and baseline, "
        f"got {observed}"
    )


def test_route_diagnostics_aggregate_across_isolated_per_case_dbs(tmp_path):
    """Route diagnostics must still aggregate correctly across multiple
    per-seed cases now that each case runs against its own isolated
    opponent-profile DB, and the requested --db-path aggregate must
    receive every case's merged profile data -- not just the last one."""
    db_path = tmp_path / "aggregate.sqlite"
    report = benchmark.run_benchmark(
        "survival_lookahead",
        opponents=("simple",),
        players=(6,),
        seeds=(1, 2),
        hands=10,
        track_opponents=True,
        opponent_db=db_path,
        workers=1,
    )

    assert len(report.aggregates) == 1
    aggregate = report.aggregates[0]
    assert aggregate.hands == 20
    assert aggregate.route_diagnostics.observed_hands > 0

    conn = opponent_store.connect(db_path)
    merged_hands = {
        agent_id: profile.hands_seen
        for agent_id in (
            "bot-agent-1",
            "bot-agent-2",
            "bot-agent-3",
            "bot-agent-4",
            "bot-agent-5",
        )
        if (profile := opponent_store.load_profile(conn, "selfplay", agent_id))
        is not None
    }
    assert merged_hands
    # Each opponent seat's hands are summed across BOTH per-seed
    # snapshots by the merge (10 + 10), proving neither seed's case was
    # dropped or overwritten by the other's merge.
    assert all(value == 20 for value in merged_hands.values())
def test_compare_aggregates_matches_candidate_to_baseline_by_stack():
    """A candidate/baseline pair must be compared within the SAME stack
    depth -- never cross-matched between a candidate row at one depth and
    a baseline row at a different depth."""
    candidate_rows = (
        benchmark.BenchmarkAggregate(
            strat="candidate",
            opponent="simple",
            players=6,
            seeds=(1,),
            hands=100,
            wins=1,
            losses=0,
            pushes=0,
            net_chips=1000,
            elapsed=0.01,
            initial_stack=500,
        ),
        benchmark.BenchmarkAggregate(
            strat="candidate",
            opponent="simple",
            players=6,
            seeds=(1,),
            hands=100,
            wins=1,
            losses=0,
            pushes=0,
            net_chips=-1000,
            elapsed=0.01,
            initial_stack=2000,
        ),
    )
    baseline_rows = (
        benchmark.BenchmarkAggregate(
            strat="baseline",
            opponent="simple",
            players=6,
            seeds=(1,),
            hands=100,
            wins=1,
            losses=0,
            pushes=0,
            net_chips=0,
            elapsed=0.01,
            initial_stack=500,
        ),
        benchmark.BenchmarkAggregate(
            strat="baseline",
            opponent="simple",
            players=6,
            seeds=(1,),
            hands=100,
            wins=1,
            losses=0,
            pushes=0,
            net_chips=0,
            elapsed=0.01,
            initial_stack=2000,
        ),
    )

    comparisons = benchmark.compare_aggregates(candidate_rows, baseline_rows, -5.0)

    assert len(comparisons) == 2
    by_stack = {row.initial_stack: row for row in comparisons}
    assert set(by_stack) == {500, 2000}
    # The 500-stack candidate beats its own baseline; the 2000-stack
    # candidate loses to ITS baseline -- proving each depth was matched
    # to its own baseline row, not some blended/cross-depth comparison.
    assert by_stack[500].delta_bb_per_100 > 0
    assert by_stack[2000].delta_bb_per_100 < 0


def test_run_benchmark_multi_stack_matrix_reports_distinct_per_stack_rows():
    report = benchmark.run_benchmark(
        "candidate",
        opponents=("simple",),
        players=(6,),
        seeds=(1,),
        hands=50,
        initial_stacks=(500, 1000, 2000),
        runner=stack_aware_runner,
        workers=1,
    )

    assert len(report.aggregates) == 3
    assert {row.initial_stack for row in report.aggregates} == {500, 1000, 2000}
    assert len({row.net_chips for row in report.aggregates}) == 3

    text = benchmark.format_report(report)
    for stack in (500, 1000, 2000):
        assert str(stack) in text

    payload = benchmark.report_to_jsonable(report)
    assert {row["initial_stack"] for row in payload["aggregates"]} == {500, 1000, 2000}


def test_run_benchmark_multi_stack_matrix_with_baseline_reports_per_stack_deltas():
    report = benchmark.run_benchmark(
        "candidate",
        opponents=("simple",),
        players=(6,),
        seeds=(1,),
        hands=50,
        initial_stacks=(500, 2000),
        baseline_strat="baseline",
        min_delta_bb_per_100=-5.0,
        runner=stack_aware_runner,
        workers=1,
    )

    assert len(report.comparisons) == 2
    assert {row.initial_stack for row in report.comparisons} == {500, 2000}


def test_resolve_options_parses_initial_stacks_from_cli_and_config(tmp_path):
    config_path = tmp_path / "benchmark.json"
    config_path.write_text(json.dumps({"initial_stacks": [500, 1000]}))

    args = benchmark.build_parser().parse_args(
        ["--strat", "candidate", "--config", str(config_path)]
    )
    options = benchmark.resolve_options(args)
    assert options["initial_stacks"] == (500, 1000)

    cli_args = benchmark.build_parser().parse_args(
        [
            "--strat",
            "candidate",
            "--config",
            str(config_path),
            "--initial-stacks",
            "250,750",
        ]
    )
    cli_options = benchmark.resolve_options(cli_args)
    assert cli_options["initial_stacks"] == (250, 750)


def test_resolve_options_defaults_initial_stacks_to_module_constant():
    args = benchmark.build_parser().parse_args(["--strat", "candidate"])
    options = benchmark.resolve_options(args)
    assert options["initial_stacks"] == (benchmark.INITIAL_STACK,)
