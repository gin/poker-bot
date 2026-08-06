"""Hand-combo range representation for Texas Hold'em."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

RANKS = "23456789TJQKA"
SUITS = "CDHS"
RANK_VALUE = {rank: index for index, rank in enumerate(RANKS, start=2)}

Combo = tuple[str, str]


def normalize_card(card):
    card = str(card).strip().upper()
    if len(card) != 2 or card[0] not in RANK_VALUE or card[1] not in SUITS:
        raise ValueError(f"invalid card: {card!r}")
    return card


def _card_sort_key(card):
    card = normalize_card(card)
    return (-RANK_VALUE[card[0]], card[1])


def normalize_combo(cards) -> Combo:
    if len(cards) != 2:
        raise ValueError("a combo must contain exactly two cards")
    first, second = sorted((normalize_card(card) for card in cards), key=_card_sort_key)
    if first == second:
        raise ValueError("a combo cannot contain duplicate cards")
    return first, second


def combo_class(cards):
    first, second = normalize_combo(cards)
    rank_a, rank_b = first[0], second[0]
    if rank_a == rank_b:
        return rank_a + rank_b
    suited = first[1] == second[1]
    return f"{rank_a}{rank_b}{'s' if suited else 'o'}"


def normalize_hand_class(hand_class):
    hand_class = str(hand_class).strip().upper()
    if len(hand_class) == 2:
        rank = hand_class[0]
        if rank != hand_class[1] or rank not in RANK_VALUE:
            raise ValueError(f"invalid pair class: {hand_class!r}")
        return rank + rank

    if len(hand_class) != 3:
        raise ValueError(f"invalid hand class: {hand_class!r}")
    high, low, suitedness = hand_class
    if high not in RANK_VALUE or low not in RANK_VALUE or high == low:
        raise ValueError(f"invalid hand class: {hand_class!r}")
    if RANK_VALUE[high] < RANK_VALUE[low]:
        high, low = low, high
    if suitedness not in {"S", "O"}:
        raise ValueError(f"invalid suitedness in hand class: {hand_class!r}")
    return high + low + suitedness.lower()


def combos_for_class(hand_class):
    hand_class = normalize_hand_class(hand_class)
    if len(hand_class) == 2:
        rank = hand_class[0]
        return tuple(
            normalize_combo((rank + first_suit, rank + second_suit))
            for first_suit, second_suit in combinations(SUITS, 2)
        )

    high, low, suitedness = hand_class
    if suitedness == "s":
        return tuple(normalize_combo((high + suit, low + suit)) for suit in SUITS)
    if suitedness == "o":
        return tuple(
            normalize_combo((high + high_suit, low + low_suit))
            for high_suit, low_suit in product(SUITS, SUITS)
            if high_suit != low_suit
        )
    raise AssertionError("unreachable suitedness")


def all_starting_combos():
    deck = [rank + suit for rank in RANKS for suit in SUITS]
    return tuple(normalize_combo(combo) for combo in combinations(deck, 2))


def class_strength(hand_class):
    """Return a rough 0-1 preflop strength prior for a canonical hand class."""
    try:
        hand_class = normalize_hand_class(hand_class)
    except ValueError:
        return 0.0
    if len(hand_class) == 2:
        rank = RANK_VALUE.get(hand_class[0], 0)
        return min(1.0, 0.42 + rank / 18)

    high = RANK_VALUE.get(hand_class[0], 0)
    low = RANK_VALUE.get(hand_class[1], 0)
    gap = max(0, high - low - 1)
    suited_bonus = 0.06 if hand_class[2] == "s" else 0.0
    connector_bonus = max(0.0, 0.05 - gap * 0.012)
    broadway_bonus = 0.08 if high >= 12 and low >= 10 else 0.0
    raw = high * 0.038 + low * 0.016 + suited_bonus + connector_bonus + broadway_bonus
    return max(0.0, min(1.0, raw))


@dataclass(frozen=True)
class HandRange:
    weights: dict[Combo, float]

    @classmethod
    def empty(cls):
        return cls({})

    @classmethod
    def all(cls, weight=1.0):
        return cls({combo: float(weight) for combo in all_starting_combos()})

    @classmethod
    def from_classes(cls, classes, weight=1.0):
        return cls.from_weighted_classes({hand_class: weight for hand_class in classes})

    @classmethod
    def from_weighted_classes(cls, class_weights):
        weights = {}
        for hand_class, class_weight in class_weights.items():
            if class_weight <= 0:
                continue
            for combo in combos_for_class(hand_class):
                weights[combo] = weights.get(combo, 0.0) + float(class_weight)
        return cls(weights)

    def total_weight(self):
        return sum(max(0.0, weight) for weight in self.weights.values())

    def normalized(self):
        total = self.total_weight()
        if total <= 0:
            return HandRange.empty()
        return HandRange(
            {
                combo: weight / total
                for combo, weight in self.weights.items()
                if weight > 0
            }
        )

    def without_blockers(self, known_cards):
        blockers = {normalize_card(card) for card in known_cards or []}
        if not blockers:
            return self
        return HandRange(
            {
                combo: weight
                for combo, weight in self.weights.items()
                if not blockers.intersection(combo)
            }
        )

    def scale(self, factor_fn):
        scaled = {}
        for combo, weight in self.weights.items():
            new_weight = weight * float(factor_fn(combo))
            if new_weight > 0:
                scaled[combo] = new_weight
        return HandRange(scaled)

    def class_weights(self):
        classes = {}
        for combo, weight in self.weights.items():
            hand_class = combo_class(combo)
            classes[hand_class] = classes.get(hand_class, 0.0) + weight
        return classes

    def probability_of_class(self, hand_class):
        total = self.total_weight()
        if total <= 0:
            return 0.0
        return self.class_weights().get(normalize_hand_class(hand_class), 0.0) / total

    def top_classes(self, limit=10):
        items = sorted(
            self.class_weights().items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return tuple(items[:limit])
