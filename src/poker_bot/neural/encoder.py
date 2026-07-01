"""State encoder: converts raw poker table state into normalized feature vector for NN."""

import math
from typing import Any

import numpy as np

from poker_bot.hand_eval import best_hand_rank_without, evaluate_hand

RANKS = "23456789TJQKA"
RANK_VALUES = {rank: index for index, rank in enumerate(RANKS, start=2)}

BLIND_SIZE = 10  # BIG_BLIND, used for normalization


def card_values(cards: list[str]) -> list[int]:
    return [RANK_VALUES.get(card[0], 0) for card in cards]


def _made_hand_rank(hole_cards: list[str], board_cards: list[str]) -> int:
    """Mirror hubase.made_hand_rank: 0 if preflop or board-only hand, else 0-8."""
    if len(board_cards) < 3:
        return 0
    board_rank = evaluate_hand(board_cards) if len(board_cards) >= 5 else (0,)
    full_rank = evaluate_hand(list(hole_cards) + list(board_cards))
    category = full_rank[0]
    if len(board_cards) >= 5 and full_rank == board_rank:
        return 0
    return category


def _hand_uses_hole_card(hole_cards: list[str], board_cards: list[str]) -> bool:
    """Does the best hand actually need at least one hole card?"""
    if not hole_cards or not board_cards:
        return False
    pool = list(hole_cards) + list(board_cards)
    if len(pool) < 5:
        return False
    full_rank = evaluate_hand(pool)
    for h in hole_cards:
        without_rank = best_hand_rank_without(pool, [h])
        if without_rank is None:
            continue
        if full_rank[0] > without_rank[0]:
            return True
    return False


def _preflop_score(hole_cards: list[str]) -> float:
    """
    Simplified preflop hand strength (0-1). Uses Chen-formula-like scoring
    that the full hubase strategy uses; this is a lightweight approximation
    sufficient for NN feature input.
    """
    if len(hole_cards) != 2:
        return 0.0
    vals = sorted(card_values(hole_cards), reverse=True)
    suited = hole_cards[0][1] == hole_cards[1][1]
    paired = vals[0] == vals[1]

    # Base: highest card value normalized
    score = vals[0] / 14.0

    # Pair bonus
    if paired:
        score = max(score, 0.5 + vals[0] / 28.0)

    # Suited bonus
    if suited:
        score += 0.1

    # Connected bonus
    gap = vals[0] - vals[1]
    if gap <= 2:
        score += 0.05 * (3 - gap)

    # High card synergy
    if vals[0] >= 10 and vals[1] >= 10:
        score += 0.15

    return min(score, 1.0)


def _board_texture(board_cards: list[str]) -> dict[str, bool]:
    """Mirror hubase.board_texture."""
    if not board_cards:
        return {"wet": False, "paired": False, "high": False}
    suits = [c[1] for c in board_cards]
    values = sorted(set(card_values(board_cards)))
    max_suit_count = max((suits.count(s) for s in set(suits)), default=0)
    connected = (
        any(values[i + 2] - values[i] <= 4 for i in range(len(values) - 2))
        if len(values) >= 3
        else False
    )
    return {
        "wet": max_suit_count >= 3 or connected,
        "paired": len(values) < len(board_cards),
        "high": any(v >= 12 for v in values),
    }


def _count_raises_this_street(table: dict[str, Any]) -> int:
    """Count raises/bets/all-ins in action history for the current street."""
    history = table.get("actionHistory") or table.get("action_history") or []
    street = table.get("street", "")
    count = 0
    for event in history:
        event_street = event.get("street", "")
        action = str(event.get("action", "")).lower()
        if event_street == street and action in {"bet", "raise", "all-in", "allin"}:
            count += 1
    return count


def encode_state(
    table: dict[str, Any],
    hero_seat: dict[str, Any],
    opponent_profile: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Converts poker table state into a 31-dimensional normalized feature vector.
    All values are in [0, 1] range for stable NN training.

    opponent_profile (optional): dict with keys —
        vpip, pfr, fold_to_bet, aggressiveness, showdown_rate, hands_seen
        When None (e.g. during imitation training), these default to neutral 0.5.
    """
    features: list[float] = []

    hole_cards: list[str] = hero_seat.get("holeCards", []) or []
    board_cards: list[str] = table.get("boardCards", []) or []
    allowed = table.get("allowedActions") or {}
    street = table.get("street", "")
    pot = table.get("potChips", 0) or 0
    current_bet = table.get("currentBet", 0) or 0
    call_amt = allowed.get("callAmount", 0) or 0
    available = allowed.get("availableActions", []) or []
    stack = hero_seat.get("stackChips", 0) or 0
    num_active = len([s for s in table.get("seats", []) if not s.get("folded", False)])
    button = table.get("buttonSeatNumber") or 0
    hero_num = hero_seat.get("seatNumber", 0) or 0

    # ── Hand strength ──────────────────────────────────────────────────
    # [0] Made hand rank category (0-8 → 0-1)
    hand_rank = _made_hand_rank(hole_cards, board_cards)
    features.append(hand_rank / 8.0)

    # [1] Hand uses hole card (0/1)
    uses_hole = _hand_uses_hole_card(hole_cards, board_cards)
    features.append(1.0 if uses_hole else 0.0)

    # [2] Preflop score (0-1)
    preflop = _preflop_score(hole_cards)
    features.append(preflop)

    # ── Pot geometry ───────────────────────────────────────────────────
    # [3] Pot odds
    pot_odds = call_amt / (pot + call_amt + 1e-6)
    features.append(float(pot_odds))

    # [4] SPR (stack-to-pot ratio, clipped)
    spr = stack / (pot + 1e-6)
    features.append(float(np.clip(spr / 50.0, 0, 1)))

    # [5] Facing bet (0/1)
    facing_bet = 1.0 if call_amt > 0 else 0.0
    features.append(facing_bet)

    # [6] Can check (0/1)
    features.append(1.0 if "check" in available else 0.0)

    # [7] Can raise (0/1)
    features.append(1.0 if "raise" in available else 0.0)

    # [8] Raises this street (normalized, capped at 4)
    raises = _count_raises_this_street(table)
    features.append(min(raises / 4.0, 1.0))

    # [9] Active players (normalized to max 6)
    features.append(min(max(num_active - 2, 0) / 4.0, 1.0))

    # [10] Position offset (0-1)
    if button > 0 and hero_num > 0:
        pos_diff = (hero_num - button) % 6
        features.append(pos_diff / 5.0)
    else:
        features.append(0.5)  # unknown → neutral

    # ── Board texture ──────────────────────────────────────────────────
    texture = _board_texture(board_cards)

    # [11] Paired board
    features.append(1.0 if texture["paired"] else 0.0)

    # [12] Monotone (3+ same suit)
    if board_cards:
        suits = [c[1] for c in board_cards]
        max_suit = max((suits.count(s) for s in set(suits)), default=0)
        features.append(1.0 if max_suit >= 3 else 0.0)
    else:
        features.append(0.0)

    # [13] Connected (3+ sequential)
    if board_cards:
        values = sorted(set(card_values(board_cards)))
        conn = (
            any(values[i + 2] - values[i] <= 4 for i in range(len(values) - 2))
            if len(values) >= 3
            else False
        )
        features.append(1.0 if conn else 0.0)
    else:
        features.append(0.0)

    # [14] High board (any Q+)
    features.append(1.0 if texture["high"] else 0.0)

    # [15] Wet board (monotone OR connected)
    features.append(1.0 if texture["wet"] else 0.0)

    # [16-19] Street one-hot
    for s in ("Preflop", "Flop", "Turn", "River"):
        features.append(1.0 if street == s else 0.0)

    # ── Additional context ─────────────────────────────────────────────
    # [20] Pot relative to BB (capped at 200 BB)
    features.append(min(pot / (BLIND_SIZE * 200), 1.0))

    # [21] Stack commitment (call / stack)
    stack_commit = call_amt / (stack + 1e-6)
    features.append(float(np.clip(stack_commit, 0, 1)))

    # [22] Aggression ratio (current_bet / pot)
    agg_ratio = current_bet / (pot + 1e-6)
    features.append(float(np.clip(agg_ratio / 3.0, 0, 1)))

    # [23] Hole cards suited
    if len(hole_cards) == 2:
        features.append(1.0 if hole_cards[0][1] == hole_cards[1][1] else 0.0)
    else:
        features.append(0.0)

    # [24] Hole card gap (normalized: 0 = paired, 1 = max gap A-2)
    if len(hole_cards) == 2:
        vals = card_values(hole_cards)
        gap = abs(vals[0] - vals[1])
        features.append(gap / 12.0)
    else:
        features.append(0.0)

    # ── Opponent modeling (NN-IMPROVE-006) ──────────────────────────
    # [25] VPIP — % of hands opp voluntarily puts money in pot
    # [26] PFR — % of hands opp raises preflop
    # [27] Fold-to-bet — % opp folds when facing a bet
    # [28] Aggression — bet+raise / (bet+raise+call)
    # [29] Showdown rate — % of hands reaching showdown
    # [30] Hands seen — log-scaled observation count
    if opponent_profile is not None:
        hands_seen = opponent_profile.get("hands_seen", 0)
        vpip = opponent_profile.get("vpip", 0.5)
        pfr = opponent_profile.get("pfr", 0.3)
        fold_to_bet = opponent_profile.get("fold_to_bet", 0.5)
        # aggression = bets+raises / total actions
        total_actions = opponent_profile.get("action_count", 1)
        bet_raise = opponent_profile.get("bet_raise_count", 0)
        aggro = bet_raise / total_actions if total_actions > 0 else 0.5
        showdown_rate = opponent_profile.get("showdown_rate", 0.3)
        hands_log = min(math.log1p(hands_seen) / math.log1p(1000), 1.0)

        features.extend(
            [
                float(np.clip(vpip, 0, 1)),  # [25]
                float(np.clip(pfr, 0, 1)),  # [26]
                float(np.clip(fold_to_bet, 0, 1)),  # [27]
                float(np.clip(aggro, 0, 1)),  # [28]
                float(np.clip(showdown_rate, 0, 1)),  # [29]
                float(hands_log),  # [30]
            ]
        )
    else:
        # No profile data (imitation training) → neutral defaults
        features.extend([0.5, 0.3, 0.5, 0.5, 0.3, 0.0])  # [25-30]

    return np.array(features, dtype=np.float32)
