import json
from types import SimpleNamespace

import pytest

from eval import profile_calibration
from poker_bot.opponents import OpponentProfile


def _seat(agent_id, *, vpip, pfr, actions=(4, 3, 3), folds=(2, 4)):
    calls, bets, raises = actions
    folded, opportunities = folds
    return {
        "agent_id": agent_id,
        "canonical": {
            "profile_stats_schema_version": 2,
            "profile_stats_provenance": "canonical",
            "hands_seen": 10,
            "preflop_hands_seen": 10,
        },
        "metrics": {
            "vpip": profile_calibration._metric(vpip, 10),
            "pfr": profile_calibration._metric(pfr, 10),
            "action_aggression": profile_calibration._metric(
                bets + raises, calls + bets + raises
            ),
            "fold_to_bet": profile_calibration._metric(folded, opportunities),
        },
    }


def _case(opponent, seat):
    return {
        "opponent": opponent,
        "players": 3,
        "initial_stack": 500,
        "seed": 1,
        "seats": [seat],
    }


def test_wilson_interval_matches_v007_formula():
    assert profile_calibration.wilson_interval(5, 10) == pytest.approx(
        (0.23658959361548731, 0.7634104063845126)
    )
    assert profile_calibration.wilson_interval(1, 0) is None
    assert profile_calibration.wilson_interval(11, 10) is None


def test_wilson_interval_supports_conservative_99_percent_bound():
    interval = profile_calibration.wilson_interval(
        5, 10, z=profile_calibration.WILSON_99_Z
    )
    assert interval is not None
    assert interval[1] == pytest.approx(0.8157744817527646)


def test_candidate_uses_only_gain_calibration_and_rejects_fallback_holdout():
    labels = {
        "gain-a": "gain",
        "gain-b": "gain",
        "fallback": "fallback",
        "unlabeled": "unlabeled_validation",
    }
    calibration = [
        _case("gain-a", _seat("bot-agent-1", vpip=6, pfr=3)),
        _case("gain-b", _seat("bot-agent-1", vpip=7, pfr=4)),
        _case("fallback", _seat("bot-agent-1", vpip=0, pfr=0)),
    ]
    holdout = [
        _case("fallback", _seat("bot-agent-1", vpip=6, pfr=3)),
        _case("unlabeled", _seat("bot-agent-1", vpip=6, pfr=3)),
    ]

    candidate = profile_calibration._candidate_assessment(calibration, holdout, labels)

    expected_max = max(
        profile_calibration.wilson_interval(
            sample["seats"][0]["metrics"]["vpip"]["numerator"],
            sample["seats"][0]["metrics"]["vpip"]["denominator"],
            z=profile_calibration.WILSON_99_Z,
        )[1]
        for sample in calibration[:2]
    )
    assert candidate["source"] == "labeled_gain_calibration_vpip_upper_99_only"
    assert candidate["limits"] == {"vpip": {"max": expected_max}}
    assert candidate["both_seats_required"] is True
    assert candidate["status"] == "rejected"
    assert candidate["rejection_reason"] == (
        "labeled fallback activated on holdout: fallback"
    )
    assert candidate["unlabeled_validation_activation"]["unlabeled"]["activated"] == 1


def test_candidate_rejects_partial_labeled_gain_holdout_activation():
    labels = {"gain": "gain", "fallback": "fallback"}
    calibration = [_case("gain", _seat("bot-agent-1", vpip=6, pfr=3))]
    holdout = [_case("gain", _seat("bot-agent-1", vpip=10, pfr=3))]

    candidate = profile_calibration._candidate_assessment(calibration, holdout, labels)

    assert candidate["status"] == "rejected"
    assert candidate["gain_support_failures"] == [
        "gain (profiles 0/1; both seats 0/1)"
    ]


def test_candidate_requires_both_opponent_seats_to_activate():
    labels = {"simple": "gain", "fallback": "fallback"}
    calibration = [
        {
            **_case("simple", _seat("bot-agent-1", vpip=1, pfr=0)),
            "seats": [
                _seat("bot-agent-1", vpip=1, pfr=0),
                _seat("bot-agent-2", vpip=1, pfr=0),
            ],
        }
    ]
    holdout = [
        {
            **_case("simple", _seat("bot-agent-1", vpip=1, pfr=0)),
            "seats": [
                _seat("bot-agent-1", vpip=1, pfr=0),
                _seat("bot-agent-2", vpip=10, pfr=0),
            ],
        }
    ]

    candidate = profile_calibration._candidate_assessment(calibration, holdout, labels)

    summary = candidate["holdout_activation"]["simple"]
    assert summary["activated"] == 1
    assert summary["profiles"] == 2
    assert summary["both_seats_activated"] == 0
    assert summary["cases"] == 1
    assert candidate["status"] == "rejected"


def test_profile_record_rejects_untrusted_and_invalid_canonical_profiles():
    with pytest.raises(ValueError, match="untrusted profile"):
        profile_calibration._profile_record(
            OpponentProfile(
                "bot-agent-1",
                preflop_hands_seen=10,
                profile_stats_provenance="legacy_untrusted",
            ),
            "bot-agent-1",
        )
    with pytest.raises(ValueError, match="canonical preflop invariant"):
        profile_calibration._profile_record(
            OpponentProfile("bot-agent-1", vpip=4, pfr=5, preflop_hands_seen=10),
            "bot-agent-1",
        )


def test_smoke_cli_runs_isolated_profiles_and_writes_deterministic_reports(
    tmp_path, capsys
):
    output_json = tmp_path / "report.json"
    output_markdown = tmp_path / "report.md"

    exit_code = profile_calibration.main(
        [
            "--smoke",
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ]
    )

    assert exit_code == 0
    report = json.loads(output_json.read_text())
    assert report["strategy"] == "multi_core_v007"
    assert len(report["calibration_cases"]) == 7
    assert len(report["holdout_cases"]) == 7
    assert report["config"]["workers"] == 1
    assert report["config"]["strategy"] == "multi_core_v007"
    assert report["config"]["profile_state_mode"] == "persistent"
    assert all(
        case["profile_state_mode"] == "persistent"
        for case in report["calibration_cases"] + report["holdout_cases"]
    )
    assert all(
        case["strategy"] == "multi_core_v007"
        for case in report["calibration_cases"] + report["holdout_cases"]
    )
    assert all(
        seat["canonical"]["profile_stats_schema_version"] == 2
        and seat["canonical"]["profile_stats_provenance"] == "canonical"
        for case in report["calibration_cases"] + report["holdout_cases"]
        for seat in case["seats"]
    )
    assert report["candidate"]["unlabeled_validation_activation"] == {}
    first_json = output_json.read_text()
    profile_calibration.write_json_report(report, output_json)
    assert output_json.read_text() == first_json
    assert "Candidate VPIP threshold" in output_markdown.read_text()
    assert "Per-seat canonical observations" in output_markdown.read_text()
    assert "Offline profile calibration" in capsys.readouterr().out


def test_default_config_has_required_offline_cohorts():
    config = profile_calibration.load_config("benchmarks/profile_calibration.json")

    assert config["stacks"] == [500, 1000, 2000]
    assert config["hands"] == 5000
    assert config["workers"] == 1
    assert config["profile_state_mode"] == "persistent"
    assert config["calibration_seeds"] == [501, 502, 503, 504, 505]
    assert config["holdout_seeds"] == [601, 602, 603, 604, 605]
    assert config["offline_labels"] == {
        "simple": "gain",
        "adaptive": "fallback",
        "royal_adaptive": "fallback",
        "counter_adaptive": "fallback",
        "threshold_pressure": "fallback",
        "anti_threshold": "fallback",
        "profiled_counter_adaptive": "fallback",
    }
    assert config["strategy"] == "multi_core_v007"


def test_config_rejects_missing_strategy():
    config = profile_calibration.load_config(
        "benchmarks/profile_calibration.smoke.json"
    )
    del config["strategy"]

    with pytest.raises(ValueError, match="strategy"):
        profile_calibration._validate_config(config)



def test_run_case_rejects_runner_strategy_mismatch(monkeypatch):
    monkeypatch.setattr(
        profile_calibration,
        "run_selfplay_parallel",
        lambda *_args, **_kwargs: SimpleNamespace(strat="other_strategy"),
    )

    with pytest.raises(ValueError, match="strategy mismatch"):
        profile_calibration._run_case(
            strategy="multi_core_v007",
            opponent="simple",
            stack=500,
            seed=1,
            hands=1,
            workers=1,
            profile_state_mode="persistent",
        )