"""Profile-gated three-handed multi-core candidate.

The default is the production ``multi_core`` policy.  Only a locked,
three-handed hand with two mature, empirically gain-group profiles may use
``s3v013``.  The condition deliberately uses observed behavior only; it never
uses an opponent name or strategy identifier.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from poker_bot.guards.context import GuardContext
from poker_bot.guards.guard_post import guard_post
from poker_bot.guards.guard_pre import guard_pre
from poker_bot.guards.telemetry import clear_events, record_guard_error
from poker_bot.hand_utils import (
    REGIME_THREE_HANDED,
    count_dealt_in_players,
    player_regime,
    profile_value,
)
from poker_bot.strategies.multi_core import choose_action as baseline_choose_action
from poker_bot.strategies.s3v013 import choose_action as s3_choose_action

ActionDecision = tuple[str | None, int | None, str]

# The telemetry gate is intentionally conservative.  At >=250 observed hands,
# all gain-group observations were VPIP <=0.624, PFR <=0.242, aggression
# <=0.498, and fold-to-bet >=0.332; fallback observations were respectively
# >=1.0, >=0.982, >=0.751, and <=0.200.
_MIN_HANDS = 250
_MAX_VPIP = 0.70
_MAX_PFR = 0.30
_MAX_AGGRESSION = 0.60
_MIN_FOLD_TO_BET = 0.25
_MAX_LOCKS = 4096
_route_locks: OrderedDict[tuple[object, ...], bool] = OrderedDict()


def _frequency(
    profile: object, frequency_name: str, numerator_name: str, denominator_name: str
) -> float | None:
    value = profile_value(profile, frequency_name)
    if value is not None:
        return float(value)
    numerator = profile_value(profile, numerator_name)
    denominator = profile_value(profile, denominator_name)
    if numerator is None or denominator is None or not denominator:
        return None
    return float(numerator) / float(denominator)


def _is_gain_group_profile(profile: object) -> bool:
    """Return whether one mature profile fits the measured gain-group bounds."""
    hands_seen = profile_value(profile, "hands_seen")
    if hands_seen is None or int(hands_seen) < _MIN_HANDS:
        return False

    vpip = _frequency(profile, "vpip_frequency", "vpip", "hands_seen")
    pfr = _frequency(profile, "pfr_frequency", "pfr", "hands_seen")
    aggression = _frequency(profile, "aggression_frequency", "bets", "actions")
    if aggression is None:
        calls = profile_value(profile, "calls")
        bets = profile_value(profile, "bets")
        raises = profile_value(profile, "raises")
        folds = profile_value(profile, "folds")
        if None in (calls, bets, raises, folds):
            return False
        total_actions = int(calls) + int(bets) + int(raises) + int(folds)
        if not total_actions:
            return False
        aggression = (int(bets) + int(raises)) / total_actions
    fold_to_bet = _frequency(
        profile,
        "fold_to_bet_frequency",
        "fold_to_bet",
        "opportunities_to_fold_to_bet",
    )
    if None in (vpip, pfr, aggression, fold_to_bet):
        return False
    return (
        vpip <= _MAX_VPIP
        and pfr <= _MAX_PFR
        and aggression <= _MAX_AGGRESSION
        and fold_to_bet >= _MIN_FOLD_TO_BET
    )


def _hand_key(
    table: dict[str, Any],
    my_seat: dict[str, Any],
    profiles: dict[str, object],
    opponent_ids: list[str],
) -> tuple[object, ...] | None:
    """Stable available hand identity for an across-street route lock.

    The platform table protocol has no hand id. Opponent id plus the
    pre-deal-incremented hand count distinguishes successive hands even when
    profile mappings are reconstructed per decision. Hole cards, button, and
    dealt-in ids keep the key stable through every street of the same hand.
    """
    hole_cards = my_seat.get("holeCards") or table.get("holeCards")
    button = table.get("buttonSeatNumber")
    if (
        not hole_cards
        or button is None
        or any(opponent_id not in profiles for opponent_id in opponent_ids)
    ):
        return None
    opponent_hand_counts = tuple(
        sorted(
            (
                opponent_id,
                int(profile_value(profiles[opponent_id], "hands_seen") or 0),
            )
            for opponent_id in opponent_ids
        )
    )
    dealt_ids = tuple(
        sorted(
            str(seat.get("agentId"))
            for seat in table.get("seats", [])
            if seat.get("agentId")
        )
    )
    return (
        button,
        dealt_ids,
        opponent_hand_counts,
        tuple(sorted(map(repr, hole_cards))),
    )


def _locked_use_s3v013(table: dict[str, Any], my_seat: dict[str, Any]) -> bool:
    """Lock the profile decision once for each hand and fail closed."""
    profiles = table.get("opponentProfiles")
    my_id = my_seat.get("agentId")
    opponent_ids = [
        seat.get("agentId")
        for seat in table.get("seats", [])
        if seat.get("agentId") and seat.get("agentId") != my_id
    ]
    if not profiles or len(opponent_ids) != 2:
        return False
    key = _hand_key(table, my_seat, profiles, opponent_ids)
    if key is None:
        return False
    cached = _route_locks.get(key)
    if cached is not None:
        _route_locks.move_to_end(key)
        return cached

    use_s3 = all(
        _is_gain_group_profile(profiles[opponent_id]) for opponent_id in opponent_ids
    )
    _route_locks[key] = use_s3
    if len(_route_locks) > _MAX_LOCKS:
        _route_locks.popitem(last=False)
    return use_s3


def _s3_with_production_guards(
    table: dict[str, Any], my_seat: dict[str, Any]
) -> ActionDecision:
    """Run the alternate core through the same production guard pipeline."""
    clear_events()
    try:
        ctx = GuardContext.build(table, my_seat, regime=REGIME_THREE_HANDED)
    except Exception as exc:
        ctx = None
        record_guard_error("context", exc)

    if ctx is not None:
        try:
            pre_result = guard_pre.run_pre(ctx)
            if pre_result is not None:
                (action, amount, message), guard_id = pre_result
                return action, float(amount or 0), f"{message} [guard:{guard_id}]"
        except Exception as exc:
            record_guard_error("pre", exc)

    action, amount, message = s3_choose_action(table, my_seat)
    message = f"[short_handed profile-gated s3v013] {message}"

    if ctx is not None:
        try:
            post_result = guard_post.run_post(ctx, (action, amount, message))
            if post_result[1] != "approved":
                (action, amount, message), guard_id = post_result
                return action, float(amount or 0), f"{message} [guard:{guard_id}]"
        except Exception as exc:
            record_guard_error("post", exc)
    return action, amount, message


def choose_action(table: dict[str, Any], my_seat: dict[str, Any]) -> ActionDecision:
    """Use s3v013 only for a locked mature gain-group three-handed hand."""
    if player_regime(count_dealt_in_players(table)) != REGIME_THREE_HANDED:
        return baseline_choose_action(table, my_seat)
    if not _locked_use_s3v013(table, my_seat):
        return baseline_choose_action(table, my_seat)
    return _s3_with_production_guards(table, my_seat)
