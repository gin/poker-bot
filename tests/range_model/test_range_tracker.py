from poker_bot.range_model import (
    RANGE_STATE_DIR_ENV,
    BayesianRangeTracker,
    RangeTrackerState,
    average_summary,
    default_state_dir,
)


def test_default_state_dir_uses_environment_variable(monkeypatch, tmp_path):
    custom = tmp_path / "luigi-instance"
    monkeypatch.setenv(RANGE_STATE_DIR_ENV, str(custom))

    assert default_state_dir() == custom
    assert BayesianRangeTracker().state_dir == custom


def test_tracker_updates_posterior_for_agent(tmp_path):
    tracker = BayesianRangeTracker(tmp_path)
    state = tracker.update(
        "villain",
        position="BTN",
        situation="open",
        action="raise",
        amount=300,
        pot=200,
    )

    assert state.agent_id == "villain"
    assert state.posterior_range.total_weight() > 0
    assert state.action_history[-1]["action"] == "raise"
    assert state.confidence > 0


def test_tracker_removes_blockers_before_update(tmp_path):
    tracker = BayesianRangeTracker(tmp_path)
    state = tracker.update(
        "villain",
        position="BTN",
        situation="open",
        action="raise",
        known_cards=["AS"],
        amount=300,
        pot=200,
    )

    assert state.posterior_range.total_weight() > 0
    assert all("AS" not in combo for combo in state.posterior_range.weights)


def test_tracker_prior_depends_on_position(tmp_path):
    tracker = BayesianRangeTracker(tmp_path)
    utg = tracker.update(
        "utg",
        position="UTG",
        situation="open",
        action="raise",
        amount=300,
        pot=200,
    )
    btn = tracker.update(
        "btn",
        position="BTN",
        situation="open",
        action="raise",
        amount=300,
        pot=200,
    )

    assert btn.posterior_range.probability_of_class("76s") > 0
    assert utg.posterior_range.probability_of_class("76s") == 0


def test_tracker_persists_state_across_instances(tmp_path):
    first = BayesianRangeTracker(tmp_path)
    first.update(
        "villain",
        position="BTN",
        situation="open",
        action="raise",
        amount=300,
        pot=200,
    )
    first.save()

    second = BayesianRangeTracker(tmp_path)
    state = second.update(
        "villain",
        position="BTN",
        situation="open",
        action="call",
        amount=300,
        pot=200,
    )

    assert state.action_history[0]["action"] == "raise"
    assert state.action_history[1]["action"] == "call"


def test_summary_is_bounded_and_contains_top_classes(tmp_path):
    tracker = BayesianRangeTracker(tmp_path)
    tracker.update(
        "villain",
        position="BTN",
        situation="open",
        action="raise",
        amount=300,
        pot=200,
    )
    summary = tracker.summary("villain")

    assert 0.0 <= summary["posterior_strength"] <= 1.0
    assert 0.0 <= summary["prior_strength"] <= 1.0
    assert 0.0 <= summary["bluff_frequency"] <= 1.0
    assert 0.0 <= summary["value_frequency"] <= 1.0
    assert 0.0 <= summary["capped_probability"] <= 1.0
    assert summary["top_classes"]
    assert len(summary["top_classes"]) <= 8


def test_showdown_removes_impossible_combos_and_increases_confidence(tmp_path):
    tracker = BayesianRangeTracker(tmp_path)
    tracker.update(
        "villain",
        position="BTN",
        situation="open",
        action="raise",
        amount=300,
        pot=200,
    )
    state = tracker.record_showdown("villain", ["AS", "AD"])

    assert ("AD", "AS") in state.posterior_range.weights
    assert state.confidence > 0
    assert all(combo == ("AD", "AS") for combo in state.posterior_range.weights)


def test_average_summary_is_bounded(tmp_path):
    tracker = BayesianRangeTracker(tmp_path)
    tracker.update(
        "villain",
        position="BTN",
        situation="open",
        action="raise",
        amount=300,
        pot=200,
    )

    summary = average_summary([tracker.summary("villain")])

    assert 0.0 <= summary["tracker_strength"] <= 1.0
    assert 0.0 <= summary["tracker_capped_probability"] <= 1.0
    assert summary["tracker_samples"] == 1


def test_tracker_exports_from_package():
    assert BayesianRangeTracker is not None
    assert RangeTrackerState is not None
