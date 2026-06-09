"""Range update helpers from observed actions."""

from __future__ import annotations

from poker_bot.range_model.hand_range import class_strength, combo_class
from poker_bot.range_model.preflop import default_preflop_range


def _pressure_ratio(amount=None, pot=None):
    amount = int(amount or 0)
    pot = int(pot or 0)
    if amount <= 0:
        return 0.0
    return amount / max(1, pot + amount)


def _raise_factor(strength, pressure):
    return max(0.02, 0.15 + strength * 1.85 + pressure * 0.75)


def _call_factor(strength, pressure):
    medium_bonus = max(0.0, 1.0 - abs(strength - 0.60) * 1.4)
    trap_bonus = 0.35 if strength >= 0.84 else 0.0
    pressure_penalty = pressure * max(0.0, 0.65 - strength)
    return max(
        0.03,
        0.30 + medium_bonus + trap_bonus + strength * 0.35 - pressure_penalty,
    )


def _check_factor(strength, _pressure):
    return max(0.10, 1.05 - strength * 0.35)


def _fold_factor(strength, pressure):
    return max(0.02, 1.25 - strength + pressure * max(0.0, 0.55 - strength))


def action_factor(action, hand_class, *, amount=None, pot=None):
    action = str(action or "").strip().lower().replace("_", "-")
    strength = class_strength(hand_class)
    pressure = _pressure_ratio(amount=amount, pot=pot)
    if action in {"raise", "reraise", "3bet", "all-in", "allin"}:
        return _raise_factor(strength, pressure)
    if action == "bet":
        return _raise_factor(strength, pressure * 0.8)
    if action == "call":
        return _call_factor(strength, pressure)
    if action == "check":
        return _check_factor(strength, pressure)
    if action == "fold":
        return _fold_factor(strength, pressure)
    return 1.0


def apply_action_update(hand_range, action, *, amount=None, pot=None, normalize=True):
    updated = hand_range.scale(
        lambda combo: action_factor(
            action,
            combo_class(combo),
            amount=amount,
            pot=pot,
        )
    )
    return updated.normalized() if normalize else updated


def remove_blockers(hand_range, known_cards):
    return hand_range.without_blockers(known_cards)


def estimate_action_range(
    *,
    position="MP",
    situation="open",
    action=None,
    known_cards=None,
    amount=None,
    pot=None,
):
    hand_range = default_preflop_range(position, situation)
    hand_range = remove_blockers(hand_range, known_cards)
    if action is not None:
        hand_range = apply_action_update(
            hand_range,
            action,
            amount=amount,
            pot=pot,
        )
    return hand_range
