"""Shared hand/board/pot/table utilities for poker strategies and guards.

Extracted from hubase.py to allow guards to depend on a small utility module
rather than the full 8000-line strategy file. All functions are pure.
"""

from __future__ import annotations

import itertools as _itertools
from collections import deque as _deque
from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
from datetime import datetime as _datetime
from typing import Any as _Any

_RANKS = "23456789TJQKA"
_SUITS = "CDHS"
_DECK = [rank + suit for rank in _RANKS for suit in _SUITS]

BIG_BLIND = 2
RANK_VALUES = {rank: index for index, rank in enumerate(_RANKS, start=2)}


def _rank_five(cards):
    """Evaluate a 5-card poker hand. Returns (category, tiebreakers...)."""
    values = {rank: index for index, rank in enumerate(_RANKS, start=2)}
    ranks = sorted((values[card[0]] for card in cards), reverse=True)
    counts = {}
    suits = {}
    for card in cards:
        rank = values[card[0]]
        suit = card[1]
        counts[rank] = counts.get(rank, 0) + 1
        suits[suit] = suits.get(suit, 0) + 1
    sorted_counts = sorted(counts.items(), key=lambda i: (i[1], i[0]), reverse=True)
    flush_suit = next((s for s, c in suits.items() if c >= 5), None)
    flush_ranks = []
    if flush_suit:
        flush_ranks = sorted(
            [values[c[0]] for c in cards if c[1] == flush_suit], reverse=True
        )
    unique_ranks = sorted(set(ranks), reverse=True)
    wheel_ranks = [5, 4, 3, 2, 1] if 14 in unique_ranks else []

    def _st(rank_list):
        for i in range(len(rank_list) - 4):
            w = rank_list[i : i + 5]
            if w[0] - w[4] == 4 and len(set(w)) == 5:
                return w[0]
        return None

    straight_top = _st(unique_ranks)
    if (
        straight_top is None
        and wheel_ranks
        and all(r in unique_ranks for r in [5, 4, 3, 2, 14])
    ):
        straight_top = 5
    sf_top = None
    if flush_suit:
        fr = sorted({values[c[0]] for c in cards if c[1] == flush_suit}, reverse=True)
        sf_top = _st(fr)
        if sf_top is None and wheel_ranks and all(r in fr for r in [5, 4, 3, 2, 14]):
            sf_top = 5
    if sf_top:
        return (8, sf_top)
    if sorted_counts[0][1] == 4:
        qr = sorted_counts[0][0]
        return (7, qr, max(r for r in unique_ranks if r != qr))
    if sorted_counts[0][1] == 3 and len(sorted_counts) > 1 and sorted_counts[1][1] >= 2:
        return (6, sorted_counts[0][0], sorted_counts[1][0])
    if flush_ranks:
        return (5,) + tuple(flush_ranks[:5])
    if straight_top:
        return (4, straight_top)
    if sorted_counts[0][1] == 3:
        tr = sorted_counts[0][0]
        k = [r for r in unique_ranks if r != tr][:2]
        return (3, tr, *k)
    if sorted_counts[0][1] == 2 and len(sorted_counts) > 1 and sorted_counts[1][1] == 2:
        ph, pl = sorted_counts[0][0], sorted_counts[1][0]
        k = [r for r in unique_ranks if r not in (ph, pl)][0]
        return (2, ph, pl, k)
    if sorted_counts[0][1] == 2:
        pr = sorted_counts[0][0]
        k = [r for r in unique_ranks if r != pr][:3]
        return (1, pr, *k)
    return (0, *unique_ranks[:5])


def evaluate_hand(cards):
    """Evaluate the best 5-card hand from a list of cards (>=5 cards)."""
    return max(_rank_five(combo) for combo in _itertools.combinations(cards, 5))


def _choose_dummy_card(cards):
    present_ranks = {card[0] for card in cards}
    for rank in _RANKS:
        if rank not in present_ranks:
            return f"{rank}c"
    for candidate in [f"{r}{s}" for r in _RANKS for s in "cdhs"]:
        if candidate not in set(cards):
            return candidate
    return "2c"


def _best_hand_without(cards, drop_cards):
    pool = [c for c in cards if c not in drop_cards]
    dummy = _choose_dummy_card(pool)
    pool.extend([dummy] * len(drop_cards))
    best_rank, best_combo = None, None
    for combo in _itertools.combinations(pool, 5):
        r = _rank_five(combo)
        if best_rank is None or r > best_rank:
            best_rank = r
            best_combo = combo
    return best_rank, best_combo


# ── Card / Rank Utilities ──────────────────────────────────────────────────


def card_values(cards):
    return [RANK_VALUES.get(card[0], 0) for card in cards]


def rank_counts(cards):
    counts = {}
    for value in card_values(cards):
        counts[value] = counts.get(value, 0) + 1
    return counts


def hole_pair_rank(hole_cards):
    values = card_values(hole_cards)
    if len(values) == 2 and values[0] == values[1]:
        return values[0]
    return None


def made_hand_rank(hole_cards, board_cards):
    if len(board_cards) < 3:
        return 0
    board_rank = evaluate_hand(board_cards) if len(board_cards) >= 5 else (0,)
    full_rank = evaluate_hand(list(hole_cards) + list(board_cards))
    category = full_rank[0]
    if len(board_cards) >= 5 and full_rank == board_rank:
        return 0
    return category


# ── Pot / Odds ─────────────────────────────────────────────────────────────


def pot_odds(call_amount, pot):
    if call_amount <= 0:
        return 0.0
    return call_amount / (pot + call_amount)


def effective_pot(table):
    pot = int(table.get("potChips") or 0)
    live_bets = sum(
        int(seat.get("currentBetChips") or 0) for seat in table.get("seats", [])
    )
    return pot + live_bets


# ── Board Texture ──────────────────────────────────────────────────────────


def board_texture(board_cards):
    suits = [card[1] for card in board_cards]
    values = sorted(set(card_values(board_cards)))
    max_suit_count = max((suits.count(s) for s in set(suits)), default=0)
    connected = any(values[i + 2] - values[i] <= 4 for i in range(len(values) - 2))
    paired = len(values) < len(board_cards)
    return {
        "wet": max_suit_count >= 3 or connected,
        "paired": paired,
        "high": any(v >= 12 for v in values),
    }


def has_top_pair_or_better(hole_cards, board_cards):
    if not board_cards:
        return False
    board_high = max(card_values(board_cards))
    hole_values = card_values(hole_cards)
    all_values = card_values(list(hole_cards) + list(board_cards))
    return any(v == board_high and all_values.count(v) >= 2 for v in hole_values)


def top_pair_kicker_value(hole_cards, board_cards):
    if len(hole_cards) != 2 or not board_cards:
        return None
    board_values = card_values(board_cards)
    board_high = max(board_values)
    hole_values = card_values(hole_cards)
    if board_values.count(board_high) != 1 or hole_values.count(board_high) != 1:
        return None
    if board_high not in hole_values:
        return None
    return max(v for v in hole_values if v != board_high)


# ── Paired Board / Two Pair Fragility ──────────────────────────────────────


def paired_board_ranks(board_cards):
    return {v for v, c in rank_counts(board_cards).items() if c >= 2}


def board_has_two_pair(board_cards):
    return len(paired_board_ranks(board_cards)) >= 2


def board_has_pair(board_cards):
    return bool(paired_board_ranks(board_cards))


def board_dominated_two_pair(hole_cards, board_cards, rank):
    if rank != 2 or not board_has_two_pair(board_cards):
        return False
    hole_values = set(card_values(hole_cards))
    return not hole_values.intersection(paired_board_ranks(board_cards))


def paired_board_rank_two(hole_cards, board_cards, rank):
    if rank != 2:
        return False
    return bool(paired_board_ranks(board_cards))


def fragile_rank_two(hole_cards, board_cards, rank):
    return board_dominated_two_pair(hole_cards, board_cards, rank) or (
        paired_board_rank_two(hole_cards, board_cards, rank)
        and not has_top_pair_or_better(hole_cards, board_cards)
    )


def fragile_rank_two_on_paired_board(hole_cards, board_cards):
    rank = made_hand_rank(hole_cards, board_cards)
    if rank != 2 or not board_has_pair(board_cards):
        return False
    return not has_top_pair_or_better(hole_cards, board_cards)


def trips_board_ranks(board_cards):
    return {v for v, c in rank_counts(board_cards).items() if c >= 3}


def non_nut_trips_board_full_house(hole_cards, board_cards):
    if len(board_cards) < 4:
        return False
    trip_ranks = trips_board_ranks(board_cards)
    if not trip_ranks:
        return False
    full_rank = evaluate_hand(list(hole_cards) + list(board_cards))
    if full_rank[0] != 6:
        return False
    triple_rank, pair_rank = full_rank[1], full_rank[2]
    if triple_rank not in trip_ranks:
        return False
    return pair_rank < RANK_VALUES["A"]


# ── Draw Detection ─────────────────────────────────────────────────────────


def has_flush_draw(hole_cards, board_cards):
    if len(board_cards) not in {3, 4}:
        return False
    suits = [c[1] for c in list(hole_cards) + list(board_cards)]
    hole_suits = {c[1] for c in hole_cards}
    return any(suits.count(s) >= 4 and s in hole_suits for s in hole_suits)


def has_open_ended_straight_draw(hole_cards, board_cards):
    if len(board_cards) not in {3, 4}:
        return False
    values = set(card_values(list(hole_cards) + list(board_cards)))
    if 14 in values:
        values.add(1)
    for low in range(2, 11):
        if set(range(low, low + 4)).issubset(values):
            return True
    return False


def has_good_draw(hole_cards, board_cards):
    return has_flush_draw(hole_cards, board_cards) or has_open_ended_straight_draw(
        hole_cards, board_cards
    )


# ── Table / Seat Utilities ─────────────────────────────────────────────────


def active_seat_numbers(table):
    return [
        int(s.get("seatNumber"))
        for s in table.get("seats", [])
        if not s.get("folded", False)
        and not s.get("hasFolded", False)
        and s.get("seatNumber") is not None
    ]


def active_players(table):
    return max(1, len(active_seat_numbers(table)))


def seated_players(table):
    return sum(1 for s in table.get("seats", []) if s.get("agentId"))


def active_opponents(table, my_seat):
    my_id = (my_seat or {}).get("agentId")
    return sum(
        1
        for s in table.get("seats", [])
        if s.get("agentId") != my_id
        and not s.get("folded", False)
        and not s.get("hasFolded", False)
    )


def live_opponent_seats(table, my_seat):
    hero_id = (my_seat or {}).get("agentId")
    hero_seat = (my_seat or {}).get("seatNumber")
    return [
        s
        for s in table.get("seats", [])
        if s.get("agentId") != hero_id
        and s.get("seatNumber") != hero_seat
        and not s.get("folded", False)
        and not s.get("hasFolded", False)
    ]


# ── Action Helpers ─────────────────────────────────────────────────────────


def call_amount(allowed):
    return int(allowed.get("callAmount") or allowed.get("callChips") or 0)


def min_raise_to(allowed):
    if allowed.get("minRaiseTo") is not None:
        return int(allowed["minRaiseTo"])
    raise_range = allowed.get("raiseRange") or {}
    value = raise_range.get("min")
    return int(value) if value is not None else None


def min_bet(allowed):
    if allowed.get("minBet") is not None:
        return int(allowed["minBet"])
    bet_range = allowed.get("betRange") or {}
    return int(bet_range.get("min") or BIG_BLIND)


def max_commit(allowed, default=0):
    if allowed.get("maxCommit") is not None:
        return int(allowed["maxCommit"])
    raise_range = allowed.get("raiseRange") or {}
    bet_range = allowed.get("betRange") or {}
    return int(raise_range.get("max") or bet_range.get("max") or default)


def capped(amount, allowed):
    return max(0, min(int(amount), int(allowed.get("maxCommit", amount))))


def blind_size(allowed, table=None):
    if table is not None and table.get("bigBlindChips") is not None:
        return max(1, int(table["bigBlindChips"]))
    minimum = allowed.get("minBet")
    if minimum is None:
        bet_range = allowed.get("betRange") or {}
        minimum = bet_range.get("min")
    return max(1, int(minimum or 0))


def no_one_has_bet(table, allowed):
    return call_amount(allowed) == 0 and int(table.get("currentBet") or 0) == 0


def no_large_preflop_raise(table, allowed):
    blind = blind_size(allowed, table)
    return int(table.get("currentBet") or 0) <= blind and call_amount(allowed) <= blind


# ── OpponentProfile ────────────────────────────────────────────────────────


@_dataclass
class OpponentProfile:
    """Opponent profiling data. Mirrors poker_bot.opponents.OpponentProfile
    so guards can work with either version."""

    agent_id: str
    name: str | None = None
    hands_seen: int = 0
    vpip: int = 0
    pfr: int = 0
    calls: int = 0
    bets: int = 0
    raises: int = 0
    folds: int = 0
    fold_to_bet: int = 0
    opportunities_to_fold_to_bet: int = 0
    showdowns: int = 0
    weak_aggressive_showdowns: int = 0
    api_stats: dict | None = None
    api_fetched_at: _datetime | None = None
    api_source_used: bool = False
    api_sample_size: int = 0
    api_aggr_freq: float | None = None
    recent_actions: _deque = _field(default_factory=lambda: _deque(maxlen=20))

    @property
    def aggression_frequency(self):
        actions = self.calls + self.bets + self.raises + self.folds
        return (self.bets + self.raises) / actions if actions else 0.0

    @property
    def call_frequency(self):
        actions = self.calls + self.bets + self.raises + self.folds
        return self.calls / actions if actions else 0.0

    @property
    def fold_to_bet_frequency(self):
        if self.opportunities_to_fold_to_bet == 0:
            return 0.0
        return self.fold_to_bet / self.opportunities_to_fold_to_bet

    @property
    def vpip_frequency(self):
        return self.vpip / self.hands_seen if self.hands_seen else 0.0

    @property
    def pfr_frequency(self):
        return self.pfr / self.hands_seen if self.hands_seen else 0.0

    @property
    def weak_aggressive_showdown_frequency(self):
        return (
            self.weak_aggressive_showdowns / self.showdowns if self.showdowns else 0.0
        )

    @property
    def api_label(self):
        if not isinstance(self.api_stats, dict):
            return None
        style = self.api_stats.get("playingStyle")
        return style.get("label") if isinstance(style, dict) else None

    @property
    def api_vpip(self):
        return self.api_stats.get("vpip") if isinstance(self.api_stats, dict) else None

    @property
    def api_bluff_pct(self):
        return (
            self.api_stats.get("bluffPct") if isinstance(self.api_stats, dict) else None
        )

    def _local_label(self):
        if self.hands_seen < 5 and len(self.recent_actions) < 8:
            return "unknown"
        if self.weak_aggressive_showdown_frequency >= 0.35:
            return "bluffer"
        if self.vpip_frequency >= 0.45 and self.aggression_frequency >= 0.35:
            return "loose_aggressive"
        if self.vpip_frequency >= 0.45 and self.call_frequency >= 0.45:
            return "calling_station"
        if self.vpip_frequency <= 0.18 and self.pfr_frequency <= 0.08:
            return "patient_methodical"
        if self.pfr_frequency >= 0.22 and self.aggression_frequency >= 0.30:
            return "tight_aggressive"
        return "balanced"

    def label(self):
        local = self._local_label()
        api = self.api_label
        if api is None:
            return local
        if self.hands_seen >= 50 and local != api:
            return local
        return api

    def is_bluffer(self):
        if self.api_bluff_pct is not None and self.api_bluff_pct >= 0.35:
            return True
        return self.label() in {"bluffer", "loose_aggressive"}


# ── Profile Helpers ────────────────────────────────────────────────────────


def profile_value(profile, name):
    """Get a value from a profile (OpponentProfile object or dict)."""
    value = getattr(profile, name, None)
    if value is not None:
        return value
    if isinstance(profile, dict):
        return profile.get(name)
    return None


def profile_call_frequency(profile):
    value = profile_value(profile, "call_frequency")
    if value is not None:
        return float(value)
    calls = int(profile_value(profile, "calls") or 0)
    bets = int(profile_value(profile, "bets") or 0)
    raises = int(profile_value(profile, "raises") or 0)
    folds = int(profile_value(profile, "folds") or 0)
    actions = calls + bets + raises + folds
    return calls / actions if actions > 0 else 0.0


def profile_aggression_frequency_merged(profile):
    if profile is None:
        return 0.0
    api_freq = getattr(profile, "api_aggr_freq", None)
    api_used = getattr(profile, "api_source_used", False)
    if api_freq is not None and api_used:
        return float(api_freq)
    local = profile_value(profile, "aggression_frequency")
    return float(local) if local is not None else 0.0


def profile_fold_to_bet_frequency(profile):
    value = profile_value(profile, "fold_to_bet_frequency")
    if value is not None:
        return float(value)
    folds_val = int(profile_value(profile, "fold_to_bet") or 0)
    opportunities = int(profile_value(profile, "opportunities_to_fold_to_bet") or 0)
    return folds_val / opportunities if opportunities > 0 else 0.0


def profile_vpip_frequency(profile):
    value = profile_value(profile, "vpip_frequency")
    if value is not None:
        return float(value)
    hands = int(profile_value(profile, "hands_seen") or 0)
    if hands <= 0:
        return 0.0
    return int(profile_value(profile, "vpip") or 0) / hands


def single_opponent_profile(table, min_hands=10):
    """Return the first opponent profile with >= min_hands observed, else None."""
    for profile in (table.get("opponentProfiles") or {}).values():
        if profile is None:
            continue
        if int(profile_value(profile, "hands_seen") or 0) >= min_hands:
            return profile
    return None


def opponent_is_bluffy(profile, min_wasd=0.30):
    """True when the opponent demonstrably bluffs."""
    if profile is None:
        return False
    wasd = profile_value(profile, "weak_aggressive_showdown_frequency")
    if wasd is not None and float(wasd) >= min_wasd:
        return True
    is_bluffer_method = getattr(profile, "is_bluffer", None)
    if callable(is_bluffer_method):
        try:
            if is_bluffer_method():
                return True
        except Exception:
            pass
    api_bluff = profile_value(profile, "api_bluff_pct")
    if api_bluff is not None and float(api_bluff) >= 0.35:
        return True
    return False


def is_tight_opponent(
    table,
    vpip_threshold=0.25,
    fold_to_bet_threshold=0.55,
    aggression_threshold=0.35,
    min_hands=10,
    use_frequency_signal=False,
    dict_only=True,
):
    for profile in (table.get("opponentProfiles") or {}).values():
        if profile is None:
            continue
        if dict_only and not isinstance(profile, dict):
            continue
        hands = int(profile_value(profile, "hands_seen") or 0)
        if hands < min_hands:
            continue
        vpip = int(profile_value(profile, "vpip") or 0)
        if hands > 0 and (vpip / hands) < vpip_threshold:
            return True
        if use_frequency_signal:
            fold_to_bet = profile_fold_to_bet_frequency(profile)
            aggression = float(profile_aggression_frequency_merged(profile))
            if (
                fold_to_bet >= fold_to_bet_threshold
                and aggression <= aggression_threshold
            ):
                return True
    return False


def observed_profiles(table, minimum_hands=25, active_only=False):
    raw = table.get("opponentProfiles") or {}
    profiles = list(raw.values())
    if active_only:
        active_ids = [
            s.get("agentId")
            for s in table.get("seats", [])
            if not s.get("folded", False) and not s.get("hasFolded", False)
        ]
        active_profiles = [raw[a_id] for a_id in active_ids if a_id in raw]
        if active_profiles:
            profiles = active_profiles
    return [
        p for p in profiles if int(profile_value(p, "hands_seen") or 0) >= minimum_hands
    ]


# ── Overpair Detection ─────────────────────────────────────────────────────


def has_overpair_to_board(hole_cards, board_cards):
    """True when hero has a pocket pair higher than any board card."""
    pair_rank = hole_pair_rank(hole_cards)
    if pair_rank is None or not board_cards:
        return False
    return pair_rank > max(card_values(board_cards), default=0)


# ── Board-Made / Kicker-Vulnerable Detection ───────────────────────────────


def board_trips_with_kicker_only(hole_cards, board_cards) -> bool:
    """True when the board has trips and hero's hole cards are just a kicker
    (hero doesn't hold the trips rank). E.g. Qh Kd on 33385."""
    if len(board_cards) != 5:
        return False
    trip_ranks = [
        rank for rank, count in rank_counts(board_cards).items() if count == 3
    ]
    if len(trip_ranks) != 1:
        return False
    trip_rank = trip_ranks[0]
    if trip_rank in card_values(hole_cards):
        return False
    full_rank = evaluate_hand(list(hole_cards) + list(board_cards))
    return full_rank[0] == 3


def is_board_made_or_kicker_vulnerable(hole_cards, board_cards) -> bool:
    """True when hero's hand is entirely board-made (playing the board) or
    board-trips-with-kicker-only (no real private edge)."""
    if len(board_cards) != 5:
        return False
    if evaluate_hand(list(hole_cards) + list(board_cards)) == evaluate_hand(
        board_cards
    ):
        return True
    return board_trips_with_kicker_only(hole_cards, board_cards)


# ── Flush Utilities ────────────────────────────────────────────────────────


def flush_ranks(hole_cards, board_cards):
    """Return the ranks of the best flush (5+ same suit), or None."""
    cards = list(hole_cards) + list(board_cards)
    by_suit = {}
    for card in cards:
        if len(card) < 2:
            continue
        by_suit.setdefault(card[1].lower(), []).append(card)
    best_suit = max(by_suit, key=lambda suit: len(by_suit[suit]), default=None)
    if best_suit is None or len(by_suit[best_suit]) < 5:
        return None
    return sorted([card_values([card])[0] for card in by_suit[best_suit]], reverse=True)


def vulnerable_non_nut_flush_on_paired_board(hole_cards, board_cards) -> bool:
    """True when hero has a non-nut flush (Q-high or worse) on a paired board.
    These have severe reverse implied odds (full house possible)."""
    texture = board_texture(board_cards)
    if not texture.get("paired", False):
        return False
    ranks = flush_ranks(hole_cards, board_cards)
    if ranks is None:
        return False
    highest = max(ranks[:5])
    return highest < RANK_VALUES["K"]


# ── Royal Flush Detection ──────────────────────────────────────────────────

_ROYAL_RANKS = {"T", "J", "Q", "K", "A"}
_ROYAL_SUITS = {"S", "H", "D", "C"}


def royal_flush_possible(hole_cards, board_cards) -> bool:
    """True when a royal flush is still possible given hole + board cards."""
    known_cards = list(hole_cards) + list(board_cards)
    remaining_board_slots = max(0, 5 - len(board_cards))
    for suit in _ROYAL_SUITS:
        hole_royals = {
            card[0]
            for card in hole_cards
            if len(card) >= 2 and card[0] in _ROYAL_RANKS and card[1] == suit
        }
        if not hole_royals:
            continue
        known_royals = {
            card[0]
            for card in known_cards
            if len(card) >= 2 and card[0] in _ROYAL_RANKS and card[1] == suit
        }
        if len(known_royals) < 2:
            continue
        missing = len(_ROYAL_RANKS - known_royals)
        if missing <= remaining_board_slots:
            return True
    return False


def is_aks(hole_cards) -> bool:
    """True when hero holds suited Ace-King."""
    if len(hole_cards) != 2:
        return False
    ranks = {c[0] for c in hole_cards}
    suits = {c[1] for c in hole_cards}
    return ranks == {"A", "K"} and len(suits) == 1
