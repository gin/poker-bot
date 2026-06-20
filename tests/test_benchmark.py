import json

import pytest

from eval import benchmark
from eval.selfplay import SelfPlayResult
from poker_bot import opponent_store


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
    opponent_store.increment_hand_seen(worker_conn, "selfplay", "villain")
    opponent_store.record_observed_action(
        worker_conn,
        platform="selfplay",
        agent_id="villain",
        hand_id="h1",
        street="Preflop",
        action="raise",
        amount=100,
        pot=75,
    )
    worker_conn.close()

    opponent_store.merge_worker_db(str(main_db), str(worker_db))

    main_conn = opponent_store.connect(str(main_db))
    villain = main_conn.execute(
        """
        SELECT o.id, s.hands_seen, s.raises
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
    assert action["opponent_id"] == villain["id"]
    main_conn.close()
