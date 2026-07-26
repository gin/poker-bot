"""Heads-up and three-player canonical low-VPIP baseline wrapper.

``multi_core`` remains the production policy everywhere. This module caps
qualified preflop raise wars in heads-up only at deep effective stacks; the
shallow-BB cap remains exclusive to three-player play.
"""

from __future__ import annotations

from math import sqrt
from typing import Any

from poker_bot.hand_utils import (
    blind_size,
    call_amount,
    count_dealt_in_players,
    profile_value,
    seat_is_live,
)
from poker_bot.strategies.adaptive import preflop_score
from poker_bot.strategies.multi_core import choose_action as baseline_choose_action
from poker_bot.strategies.s3base import preflop_min_raise_war_cap

ActionDecision = tuple[str | None, int | None, str]

_REQUIRED_PROFILE_STATS_SCHEMA = 2
_LOW_VPIP_MIN_PREFLOP_HANDS = 400
_LOW_VPIP_UPPER_99 = 0.1485555135752894
_LOW_VPIP_CALIBRATION = {
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
_SHALLOW_BB_EFFECTIVE_STACK_CAP = 600
_HU_WAR_MIN_EFFECTIVE_STACK = 1500


def _canonical_count(profile: object, name: str) -> int | None:
    value = profile_value(profile, name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None









def _wilson_upper_99(successes: int, trials: int) -> float | None:
    """Return the two-sided 99% Wilson upper bound for a canonical counter."""
    if trials <= 0 or successes < 0 or successes > trials:
        return None
    z = 2.5758293035489004
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    radius = (
        z
        * sqrt((proportion * (1 - proportion) + z * z / (4 * trials)) / trials)
        / denominator
    )
    return min(1.0, centre + radius)


def _is_canonical_low_vpip_profile(profile: object) -> bool:
    """Identify the calibrated low-VPIP cohort without opponent identity."""
    if (
        profile_value(profile, "profile_stats_provenance") != "canonical"
        or _canonical_count(profile, "profile_stats_schema_version")
        != _REQUIRED_PROFILE_STATS_SCHEMA
    ):
        return False
    preflop_hands = _canonical_count(profile, "preflop_hands_seen")
    vpip = _canonical_count(profile, "vpip")
    if (
        preflop_hands is None
        or vpip is None
        or preflop_hands < _LOW_VPIP_MIN_PREFLOP_HANDS
    ):
        return False
    upper = _wilson_upper_99(vpip, preflop_hands)
    return upper is not None and upper <= _LOW_VPIP_UPPER_99


def _dealt_in_seats(table: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the stable dealt-in seats without deriving a live-player count."""
    return [
        seat
        for seat in table.get("seats", [])
        if seat.get("agentId")
        and (
            seat_is_live(seat)
            or seat.get("folded", False)
            or seat.get("hasFolded", False)
            or seat.get("status") == "Folded"
        )
    ]


def _has_canonical_low_vpip_opponents(
    table: dict[str, Any],
    my_seat: dict[str, Any],
    *,
    expected_opponents: int,
) -> bool:
    """Require every dealt-in opponent to belong to the calibrated cohort."""
    profiles = table.get("opponentProfiles")
    my_id = my_seat.get("agentId")
    opponent_ids = [
        seat.get("agentId")
        for seat in _dealt_in_seats(table)
        if seat.get("agentId") != my_id
    ]
    if not isinstance(profiles, dict) or len(opponent_ids) != expected_opponents:
        return False
    return all(
        opponent_id in profiles
        and _is_canonical_low_vpip_profile(profiles[opponent_id])
        for opponent_id in opponent_ids
    )




def _heads_up_effective_stack_at_least(
    table: dict[str, Any], my_seat: dict[str, Any], minimum: int
) -> bool:
    """Require a live heads-up opponent and the requested effective stack."""
    hero_id = my_seat.get("agentId")
    opponents = [
        seat
        for seat in _dealt_in_seats(table)
        if seat.get("agentId") != hero_id and seat_is_live(seat)
    ]
    if len(opponents) != 1:
        return False
    hero_total = int(my_seat.get("stackChips") or 0) + int(
        my_seat.get("currentBetChips") or 0
    )
    opponent = opponents[0]
    opponent_total = int(opponent.get("stackChips") or 0) + int(
        opponent.get("currentBetChips") or 0
    )
    return min(hero_total, opponent_total) >= minimum


def _is_true_big_blind(table: dict[str, Any], my_seat: dict[str, Any]) -> bool:
    """Identify BB from the dealt-in button/seat order, not a position label."""
    button = table.get("buttonSeatNumber")
    hero_number = my_seat.get("seatNumber")
    seats = sorted(
        _dealt_in_seats(table),
        key=lambda seat: int(seat.get("seatNumber") or -1),
    )
    seat_numbers = [seat.get("seatNumber") for seat in seats]
    if len(seats) != 3 or button not in seat_numbers or hero_number not in seat_numbers:
        return False
    ordered = seats[seat_numbers.index(button) :] + seats[: seat_numbers.index(button)]
    return ordered[2].get("seatNumber") == hero_number


def _shallow_bb_open_three_bet_call(
    table: dict[str, Any], my_seat: dict[str, Any], baseline: ActionDecision
) -> ActionDecision | None:
    """Cap only the calibrated shallow-BB single-open re-raise response."""
    if table.get("street") != "Preflop" or not _is_true_big_blind(table, my_seat):
        return None
    action, _amount, message = baseline
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if (
        action != "raise"
        or "balanced value/open" not in message
        or "call" not in available
    ):
        return None
    score = preflop_score(my_seat.get("holeCards", []))
    if not 56 <= score <= 59:
        return None
    blind = blind_size(allowed, table)
    if int(table.get("currentBet") or 0) <= blind:
        return None
    history = table.get("actionHistory") or table.get("action_history") or []
    hero_id = my_seat.get("agentId")
    if any(
        event.get("agentId") == hero_id
        and event.get("street") in (None, "Preflop")
        for event in history
    ):
        return None
    prior_opponent_raises = sum(
        1
        for event in history
        if event.get("agentId") != hero_id
        and event.get("action") == "raise"
        and event.get("street") in (None, "Preflop")
    )
    if prior_opponent_raises != 1:
        return None
    hero_total = int(my_seat.get("stackChips") or 0) + int(
        my_seat.get("currentBetChips") or 0
    )
    opponent_totals = [
        int(seat.get("stackChips") or 0) + int(seat.get("currentBetChips") or 0)
        for seat in _dealt_in_seats(table)
        if seat.get("agentId") != hero_id and seat_is_live(seat)
    ]
    if (
        not opponent_totals
        or min(hero_total, max(opponent_totals)) > _SHALLOW_BB_EFFECTIVE_STACK_CAP
    ):
        return None
    return "call", call_amount(allowed), "shallow BB cap: call one-open 3-bet"





def choose_action(table: dict[str, Any], my_seat: dict[str, Any]) -> ActionDecision:
    """Return baseline except for unguarded qualified HU/3p preflop war caps."""
    dealt_in = count_dealt_in_players(table)
    if dealt_in not in (2, 3):
        return baseline_choose_action(table, my_seat)

    baseline = baseline_choose_action(table, my_seat)
    if "[guard:" in baseline[2] or not _has_canonical_low_vpip_opponents(
        table, my_seat, expected_opponents=dealt_in - 1
    ):
        return baseline

    if dealt_in == 2 and not _heads_up_effective_stack_at_least(
        table, my_seat, _HU_WAR_MIN_EFFECTIVE_STACK
    ):
        return baseline

    war_cap = preflop_min_raise_war_cap(table, my_seat, baseline)
    if war_cap is not None:
        action, amount, message = war_cap
        prefix = "[heads_up]" if dealt_in == 2 else "[short_handed]"
        return action, amount, f"{prefix} [v007 canonical low-VPIP war cap] {message}"

    if dealt_in != 3:
        return baseline
    shallow_cap = _shallow_bb_open_three_bet_call(table, my_seat, baseline)
    if shallow_cap is None:
        return baseline
    action, amount, message = shallow_cap
    return (
        action,
        amount,
        f"[short_handed] [v007 canonical low-VPIP shallow BB cap] {message}",
    )
