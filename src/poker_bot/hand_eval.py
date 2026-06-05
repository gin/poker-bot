"""Card and hand evaluation helpers for Texas Hold'em."""

import itertools
import random

RANKS = "23456789TJQKA"
SUITS = "CDHS"
DECK = [rank + suit for rank in RANKS for suit in SUITS]


def build_deck(rng=None):
    deck = DECK.copy()
    shuffler = rng if rng is not None else random
    shuffler.shuffle(deck)
    return deck


def deal_cards(deck):
    return [deck.pop(), deck.pop()]


def rank_five(cards):
    values = {rank: index for index, rank in enumerate(RANKS, start=2)}
    ranks = sorted((values[card[0]] for card in cards), reverse=True)
    counts = {}
    suits = {}
    for card in cards:
        rank = values[card[0]]
        suit = card[1]
        counts[rank] = counts.get(rank, 0) + 1
        suits[suit] = suits.get(suit, 0) + 1

    sorted_counts = sorted(
        counts.items(), key=lambda item: (item[1], item[0]), reverse=True
    )
    flush_suit = next((suit for suit, count in suits.items() if count >= 5), None)
    flush_ranks = []
    if flush_suit:
        flush_ranks = sorted(
            [values[card[0]] for card in cards if card[1] == flush_suit], reverse=True
        )

    unique_ranks = sorted(set(ranks), reverse=True)
    wheel_ranks = [5, 4, 3, 2, 1] if 14 in unique_ranks else []

    def straight_top(rank_list):
        for index in range(len(rank_list) - 4):
            window = rank_list[index : index + 5]
            if window[0] - window[4] == 4 and len(set(window)) == 5:
                return window[0]
        return None

    straight_top_rank = straight_top(unique_ranks)
    if (
        straight_top_rank is None
        and wheel_ranks
        and all(rank in unique_ranks for rank in [5, 4, 3, 2, 14])
    ):
        straight_top_rank = 5

    straight_flush_top = None
    if flush_suit:
        flush_cards = [card for card in cards if card[1] == flush_suit]
        flush_ranks_unique = sorted(
            {values[card[0]] for card in flush_cards}, reverse=True
        )
        straight_flush_top = straight_top(flush_ranks_unique)
        if (
            straight_flush_top is None
            and wheel_ranks
            and all(rank in flush_ranks_unique for rank in [5, 4, 3, 2, 14])
        ):
            straight_flush_top = 5

    if straight_flush_top:
        return (8, straight_flush_top)

    if sorted_counts[0][1] == 4:
        quad_rank = sorted_counts[0][0]
        kicker = max(rank for rank in unique_ranks if rank != quad_rank)
        return (7, quad_rank, kicker)

    if sorted_counts[0][1] == 3 and len(sorted_counts) > 1 and sorted_counts[1][1] >= 2:
        triple_rank = sorted_counts[0][0]
        pair_rank = sorted_counts[1][0]
        return (6, triple_rank, pair_rank)

    if flush_ranks:
        return (5,) + tuple(flush_ranks[:5])

    if straight_top_rank:
        return (4, straight_top_rank)

    if sorted_counts[0][1] == 3:
        triple_rank = sorted_counts[0][0]
        kickers = [rank for rank in unique_ranks if rank != triple_rank][:2]
        return (3, triple_rank, *kickers)

    if sorted_counts[0][1] == 2 and len(sorted_counts) > 1 and sorted_counts[1][1] == 2:
        pair_high = sorted_counts[0][0]
        pair_low = sorted_counts[1][0]
        kicker = [rank for rank in unique_ranks if rank not in (pair_high, pair_low)][0]
        return (2, pair_high, pair_low, kicker)

    if sorted_counts[0][1] == 2:
        pair_rank = sorted_counts[0][0]
        kickers = [rank for rank in unique_ranks if rank != pair_rank][:3]
        return (1, pair_rank, *kickers)

    return (0, *unique_ranks[:5])


def evaluate_hand(cards):
    return max(rank_five(combo) for combo in itertools.combinations(cards, 5))


def compare_hands(hand1, hand2):
    if hand1 > hand2:
        return 1
    if hand2 > hand1:
        return -1
    return 0
