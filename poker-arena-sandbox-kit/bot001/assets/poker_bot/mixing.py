"""Deterministic action mixing helpers.

These helpers let strategies use mixed frequencies without runtime randomness.
The same table state, hero seat, namespace, and salt always produce the same
roll, which keeps tests and self-play reproducible.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributionDecision:
    selected: object
    roll: float
    total_weight: float
    options: tuple[tuple[object, float], ...]

    @property
    def probabilities(self):
        if self.total_weight <= 0:
            return ()
        return tuple(
            (value, weight / self.total_weight) for value, weight in self.options
        )

    def probability_for(self, value):
        for option_value, probability in self.probabilities:
            if option_value == value:
                return probability
        return 0.0

    def summary(self):
        return "/".join(
            f"{value}:{probability:.0%}" for value, probability in self.probabilities
        )


def _cards(cards):
    return tuple(str(card).upper() for card in cards or [])


def _stable_payload(namespace, table=None, my_seat=None, *, strategy=None, extra=()):
    table = table or {}
    my_seat = my_seat or {}
    allowed = table.get("allowedActions") or {}
    payload = {
        "namespace": namespace,
        "strategy": strategy,
        "hand_id": table.get("handId") or table.get("tableId") or table.get("id"),
        "street": table.get("street"),
        "pot": table.get("potChips"),
        "current_bet": table.get("currentBet"),
        "button": table.get("buttonSeatNumber"),
        "acting": table.get("actingSeatNumber"),
        "hero_seat": my_seat.get("seatNumber") or table.get("selfSeatNumber"),
        "hero_cards": _cards(my_seat.get("holeCards")),
        "board": _cards(table.get("boardCards")),
        "available": tuple(allowed.get("availableActions") or ()),
        "call": allowed.get("callAmount") or allowed.get("callChips"),
        "min_raise": allowed.get("minRaiseTo"),
        "extra": tuple(str(part) for part in extra or ()),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def mix_value(namespace, table=None, my_seat=None, *, strategy=None, extra=()):
    """Return a deterministic float in [0.0, 1.0)."""
    payload = _stable_payload(
        namespace,
        table,
        my_seat,
        strategy=strategy,
        extra=extra,
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def mix_percent(namespace, table=None, my_seat=None, *, strategy=None, extra=()):
    value = mix_value(namespace, table, my_seat, strategy=strategy, extra=extra)
    return int(value * 100)


def should_take(
    probability,
    namespace,
    table=None,
    my_seat=None,
    *,
    strategy=None,
    extra=(),
):
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    return mix_value(namespace, table, my_seat, strategy=strategy, extra=extra) < float(
        probability
    )


def choose_weighted(
    options: Sequence[tuple[object, float]],
    namespace,
    table=None,
    my_seat=None,
    *,
    strategy=None,
    extra=(),
) -> object:
    return resolve_distribution(
        options,
        namespace,
        table,
        my_seat,
        strategy=strategy,
        extra=extra,
    ).selected


def resolve_distribution(
    options: Sequence[tuple[object, float]],
    namespace,
    table=None,
    my_seat=None,
    *,
    strategy=None,
    extra=(),
) -> DistributionDecision:
    weighted = [(value, float(weight)) for value, weight in options if weight > 0]
    if not weighted:
        raise ValueError("resolve_distribution requires at least one positive weight")

    total = sum(weight for _value, weight in weighted)
    roll_value = mix_value(namespace, table, my_seat, strategy=strategy, extra=extra)
    roll = roll_value * total
    cumulative = 0.0
    for value, weight in weighted:
        cumulative += weight
        if roll < cumulative:
            selected = value
            break
    else:
        selected = weighted[-1][0]
    return DistributionDecision(
        selected=selected,
        roll=roll_value,
        total_weight=total,
        options=tuple(weighted),
    )


def frequency_bucket(
    thresholds: Iterable[tuple[object, float]],
    namespace,
    table=None,
    my_seat=None,
    *,
    strategy=None,
    extra=(),
) -> object:
    roll = mix_value(namespace, table, my_seat, strategy=strategy, extra=extra)
    last_value = None
    for value, threshold in thresholds:
        last_value = value
        if roll < threshold:
            return value
    if last_value is None:
        raise ValueError("frequency_bucket requires at least one threshold")
    return last_value
