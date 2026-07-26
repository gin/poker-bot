"""Focused contracts for the three-player canonical low-VPIP wrapper."""

import pytest

import poker_bot.strategies.multi_core_v007 as candidate
from poker_bot.opponents import OpponentProfile


def _profile(agent_id, *, hands=400, vpip=41, provenance="canonical"):
    return OpponentProfile(
        agent_id=agent_id,
        hands_seen=hands,
        preflop_hands_seen=hands,
        profile_stats_schema_version=2,
        profile_stats_provenance=provenance,
        vpip=vpip,
        pfr=min(vpip, 46),
    )


def _baseline_raise(*_args):
    return "raise", 40, "balanced value/open raise score 56"


def _war_table(first, second, *, raise_count=3):
    hero = {
        "agentId": "hero",
        "seatNumber": 1,
        "holeCards": ["Jh", "Ts"],
        "stackChips": 1000,
    }
    return {
        "buttonSeatNumber": 1,
        "street": "Preflop",
        "currentBet": 80,
        "seats": [
            hero,
            {"agentId": "first", "seatNumber": 2, "stackChips": 1000},
            {"agentId": "second", "seatNumber": 3, "stackChips": 1000},
        ],
        "allowedActions": {
            "availableActions": ["fold", "call", "raise"],
            "callAmount": 40,
            "minBet": 10,
        },
        "actionHistory": [
            {"agentId": "hero", "action": "raise", "street": "Preflop"}
            for _ in range(raise_count)
        ],
        "opponentProfiles": {"first": first, "second": second},
    }, hero


def _shallow_bb_table(first, second, *, largest_opponent_total=600):
    hero = {
        "agentId": "hero",
        "seatNumber": 3,
        "holeCards": ["Ah", "6h"],
        "stackChips": 990,
        "currentBetChips": 10,
    }
    return {
        "buttonSeatNumber": 1,
        "street": "Preflop",
        "currentBet": 20,
        "seats": [
            {
                "agentId": "first",
                "seatNumber": 1,
                "stackChips": largest_opponent_total - 20,
                "currentBetChips": 20,
            },
            {
                "agentId": "second",
                "seatNumber": 2,
                "stackChips": 500,
                "currentBetChips": 5,
            },
            hero,
        ],
        "allowedActions": {
            "availableActions": ["fold", "call", "raise"],
            "callAmount": 10,
            "minBet": 10,
        },
        "actionHistory": [
            {"agentId": "first", "action": "raise", "street": "Preflop"}
        ],
        "opponentProfiles": {"first": first, "second": second},
    }, hero


def test_war_cap_uses_unmodified_v005_override(monkeypatch):
    monkeypatch.setattr(candidate, "baseline_choose_action", _baseline_raise)
    table, hero = _war_table(_profile("first"), _profile("second"))

    assert candidate.choose_action(table, hero) == (
        "call",
        40,
        "[short_handed] [v007 canonical low-VPIP war cap] "
        "preflop war cap: call after 3 raise-backs",
    )


def _heads_up_war_table(opponent, *, effective_stack=1500):
    table, hero = _war_table(opponent, _profile("unused"))
    table["seats"].pop()
    table["opponentProfiles"].pop("second")
    hero["stackChips"] = effective_stack
    table["seats"][1]["stackChips"] = effective_stack
    return table, hero


@pytest.mark.parametrize(
    ("effective_stack", "expected_action"), [(1499, "raise"), (1500, "call")]
)
def test_heads_up_war_cap_has_exact_effective_stack_boundary(
    monkeypatch, effective_stack, expected_action
):
    monkeypatch.setattr(candidate, "baseline_choose_action", _baseline_raise)
    table, hero = _heads_up_war_table(
        _profile("first"), effective_stack=effective_stack
    )

    action, amount, message = candidate.choose_action(table, hero)
    assert action == expected_action
    if expected_action == "call":
        assert (amount, message) == (
            40,
            "[heads_up] [v007 canonical low-VPIP war cap] "
            "preflop war cap: call after 3 raise-backs",
        )
    else:
        assert (amount, message) == (40, "balanced value/open raise score 56")


@pytest.mark.parametrize(
    "opponent",
    [
        _profile("first", hands=399, vpip=40),
        _profile("first", vpip=42),
        _profile("first", provenance="legacy_untrusted"),
    ],
)
def test_heads_up_war_cap_fails_closed_for_unqualified_opponent(
    monkeypatch, opponent
):
    monkeypatch.setattr(candidate, "baseline_choose_action", _baseline_raise)
    table, hero = _heads_up_war_table(opponent)

    assert candidate.choose_action(table, hero) == _baseline_raise()


def test_heads_up_war_cap_preserves_guarded_baseline(monkeypatch):
    guarded = ("raise", 40, "balanced value/open raise score 56 [guard:war]")
    monkeypatch.setattr(candidate, "baseline_choose_action", lambda *_: guarded)
    table, hero = _heads_up_war_table(_profile("first"))

    assert candidate.choose_action(table, hero) == guarded



@pytest.mark.parametrize(
    ("largest_opponent_total", "expected_action"), [(600, "call"), (601, "raise")]
)
def test_shallow_bb_cap_has_exact_effective_stack_boundary(
    monkeypatch, largest_opponent_total, expected_action
):
    monkeypatch.setattr(candidate, "baseline_choose_action", _baseline_raise)
    table, hero = _shallow_bb_table(
        _profile("first"),
        _profile("second"),
        largest_opponent_total=largest_opponent_total,
    )

    action, amount, message = candidate.choose_action(table, hero)
    assert action == expected_action
    if expected_action == "call":
        assert amount == 10
        assert "[v007 canonical low-VPIP shallow BB cap]" in message
    else:
        assert (amount, message) == (40, "balanced value/open raise score 56")


def test_classifier_has_exact_calibrated_threshold_and_support_boundaries(
    monkeypatch,
):
    monkeypatch.setattr(candidate, "baseline_choose_action", _baseline_raise)
    assert candidate._LOW_VPIP_UPPER_99 == 0.1485555135752894
    assert candidate._is_canonical_low_vpip_profile(_profile("first", vpip=41))
    assert not candidate._is_canonical_low_vpip_profile(_profile("first", vpip=42))
    assert not candidate._is_canonical_low_vpip_profile(
        _profile("first", hands=399, vpip=40)
    )
    assert not candidate._is_canonical_low_vpip_profile(
        _profile("first", provenance="legacy_untrusted")
    )

    table, hero = _war_table(_profile("first"), _profile("second", vpip=42))
    assert candidate.choose_action(table, hero) == _baseline_raise()

    table, hero = _war_table(_profile("first"), _profile("second"))
    table["opponentProfiles"] = {}
    assert candidate.choose_action(table, hero) == _baseline_raise()


def test_calibration_metadata_is_compact_and_pinned():
    assert candidate._LOW_VPIP_CALIBRATION == {
        "schema_version": 2,
        "config_path": "benchmarks/profile_calibration.json",
        "calibration_seeds": (501, 502, 503, 504, 505),
        "holdout_seeds": (601, 602, 603, 604, 605),
        "workers": 1,
        "profile_state": "persistent",
        "artifact_path": "artifacts/multi_core_v007_profile_calibration.json",
        "artifact_sha256": (
            "d4ff6978ea90eed36912982aab0dd2137af1d402cbecf1239a7cc0d028c22f6f"
        ),
        "promotion_status": "screened_candidate",
    }


def test_guard_tag_preserves_baseline_precedence(monkeypatch):
    guarded = ("raise", 40, "balanced value/open raise score 56 [guard:war]")
    monkeypatch.setattr(candidate, "baseline_choose_action", lambda *_: guarded)
    table, hero = _war_table(_profile("first"), _profile("second"))

    assert candidate.choose_action(table, hero) == guarded


@pytest.mark.parametrize(
    "mutate",
    [
        lambda table, hero: table["actionHistory"].append(
            {"agentId": "second", "action": "raise", "street": "Preflop"}
        ),
        lambda table, hero: table["actionHistory"].append(
            {"agentId": "hero", "action": "call", "street": "Preflop"}
        ),
        lambda table, hero: table["allowedActions"].update(
            {"availableActions": ["fold", "raise"]}
        ),
        lambda table, hero: table.update({"street": "Flop"}),
    ],
)
def test_shallow_bb_cap_fails_closed(monkeypatch, mutate):
    monkeypatch.setattr(candidate, "baseline_choose_action", _baseline_raise)
    table, hero = _shallow_bb_table(_profile("first"), _profile("second"))
    mutate(table, hero)

    assert candidate.choose_action(table, hero) == _baseline_raise()




def test_non_three_player_and_royal_like_profiles_return_baseline(monkeypatch):
    monkeypatch.setattr(candidate, "baseline_choose_action", _baseline_raise)
    table, hero = _war_table(
        _profile("first"), _profile("second", hands=5000, vpip=1000)
    )
    assert candidate.choose_action(table, hero) == _baseline_raise()

    table, hero = _war_table(_profile("first"), _profile("second"))
    table["seats"].pop()
    table["opponentProfiles"].pop("second")
    assert candidate.choose_action(table, hero) == _baseline_raise()
