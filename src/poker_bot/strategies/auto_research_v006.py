"""Auto-research candidate v006.

This candidate keeps v005 as the champion baseline and adds a bounded
decision-time local-search evaluator for close postflop spots. The evaluator
estimates action EV from deterministic rollouts against a light population
model and only overrides the baseline when the sampled edge is material.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from poker_bot.hand_eval import DECK, compare_hands, evaluate_hand
from poker_bot.strategies import auto_research_v005 as champion
from poker_bot.strategies.adaptive import (
    BIG_BLIND,
    board_texture,
    has_top_pair_or_better,
    made_hand_rank,
    pot_odds,
)

ActionDecision = tuple[str | None, int | None, str]

ROLLOUTS = 24
MAX_OPPONENTS = 3
MIN_EDGE_CHIPS = 16
MIN_AGGRESSIVE_EDGE_CHIPS = 64
MIN_EDGE_POT_FRACTION = 0.055
MAX_EDGE_POT_FRACTION = 0.12
CLOSE_STRENGTH_LOW = 0.31
CLOSE_STRENGTH_HIGH = 0.69


@dataclass(frozen=True)
class CandidateAction:
    action: str
    amount: int | None


def _clamp(value, low, high):
    return max(low, min(high, value))


def _call_amount(allowed):
    return int(allowed.get("callAmount") or allowed.get("callChips") or 0)


def _action_key(action):
    return (action.action, action.amount)


def _stable_seed(table, my_seat):
    payload = repr(
        (
            table.get("street"),
            tuple(my_seat.get("holeCards", ())),
            tuple(table.get("boardCards", ())),
            table.get("potChips"),
            table.get("currentBet"),
            tuple(table.get("allowedActions", {}).get("availableActions", ())),
            my_seat.get("seatNumber"),
            my_seat.get("stackChips"),
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def active_opponents(table, my_seat):
    hero_id = (my_seat or {}).get("agentId")
    opponents = 0
    for seat in table.get("seats", []):
        if seat.get("agentId") == hero_id or seat.get("folded"):
            continue
        opponents += 1
    return max(1, opponents)


def candidate_actions(table, my_seat, base) -> tuple[CandidateAction, ...]:
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    pot = int(table.get("potChips") or 0)
    stack = int(my_seat.get("stackChips") or 0)
    candidates = []
    for action in ("fold", "check", "call", "bet", "raise"):
        if action not in available:
            continue
        if action == "fold" and "check" in available:
            continue
        amount = None
        if action == "call":
            amount = _call_amount(allowed)
        elif action == "bet":
            minimum = int(allowed.get("minBet") or BIG_BLIND)
            maximum = int(allowed.get("maxCommit") or stack or minimum)
            amount = _clamp(max(minimum, int(pot * 0.40)), minimum, maximum)
        elif action == "raise":
            minimum = allowed.get("minRaiseTo")
            if minimum is None:
                continue
            maximum = int(allowed.get("maxCommit") or stack or minimum)
            target = int(table.get("currentBet") or 0) + int(pot * 0.55)
            amount = _clamp(max(int(minimum), target), int(minimum), maximum)
        candidates.append(CandidateAction(action, amount))

    base_action, base_amount, _message = base
    if base_action in available:
        base_candidate = CandidateAction(base_action, base_amount)
        candidate_keys = {_action_key(action) for action in candidates}
        if _action_key(base_candidate) not in candidate_keys:
            candidates.append(base_candidate)
    return tuple(candidates)


def _profile_fold_to_bet(table):
    profiles = table.get("opponentProfiles") or {}
    values = []
    for profile in profiles.values():
        if not isinstance(profile, dict):
            continue
        explicit = profile.get("fold_to_bet_frequency")
        if explicit is not None:
            values.append(_clamp(float(explicit), 0.0, 1.0))
    if not values:
        return 0.38
    return sum(values) / len(values)


def _has_explicit_fold_to_bet_profile(table, minimum=0.65):
    profiles = table.get("opponentProfiles") or {}
    for profile in profiles.values():
        if not isinstance(profile, dict):
            continue
        explicit = profile.get("fold_to_bet_frequency")
        if explicit is not None and float(explicit) >= minimum:
            return True
    return False


def _population_response(table, my_seat, action, equity, required):
    opponents = active_opponents(table, my_seat)
    texture = board_texture(table.get("boardCards", []))
    fold_to_bet = _profile_fold_to_bet(table)
    pressure = 0.0
    if action.amount:
        pot = max(1, int(table.get("potChips") or 0))
        pressure = action.amount / (pot + action.amount)
    fold_probability = fold_to_bet
    fold_probability += (pressure - 0.28) * 0.65
    fold_probability -= max(0, opponents - 1) * 0.07
    fold_probability -= max(0.0, equity - 0.58) * 0.20
    if texture.get("wet", False):
        fold_probability -= 0.05
    if action.action == "raise":
        fold_probability += 0.06
    if required >= 0.34:
        fold_probability += 0.04
    return _clamp(fold_probability, 0.12, 0.72)


def _sample_opponent_hands(rng, deck, opponents):
    sampled = []
    for _index in range(opponents):
        sampled.append((deck.pop(), deck.pop()))
    return sampled


def rollout_equity(table, my_seat, *, rollouts=ROLLOUTS):
    hole_cards = list(my_seat.get("holeCards", []))
    board_cards = list(table.get("boardCards", []))
    if len(hole_cards) != 2 or len(board_cards) < 3:
        return None

    known = set(hole_cards + board_cards)
    opponents = min(MAX_OPPONENTS, active_opponents(table, my_seat))
    rng = random.Random(_stable_seed(table, my_seat))
    wins = 0.0
    for _index in range(rollouts):
        deck = [card for card in DECK if card not in known]
        rng.shuffle(deck)
        opponent_hands = _sample_opponent_hands(rng, deck, opponents)
        runout = board_cards + [deck.pop() for _ in range(5 - len(board_cards))]
        hero_rank = evaluate_hand(hole_cards + runout)
        results = [
            compare_hands(hero_rank, evaluate_hand(list(hand) + runout))
            for hand in opponent_hands
        ]
        if all(result > 0 for result in results):
            wins += 1.0
        elif any(result < 0 for result in results):
            continue
        else:
            tied = 1 + sum(1 for result in results if result == 0)
            wins += 1.0 / tied
    return wins / rollouts


def has_flush_draw(hole_cards, board_cards):
    if len(board_cards) >= 5:
        return False
    suits = [card[1] for card in [*hole_cards, *board_cards]]
    return any(suits.count(suit) >= 4 for suit in set(suits))


def cheap_close_spot_filter(table, my_seat, base):
    if table.get("street", "Preflop") == "Preflop":
        return False
    board_cards = table.get("boardCards", [])
    hole_cards = my_seat.get("holeCards", [])
    if len(hole_cards) != 2 or len(board_cards) < 3:
        return False

    allowed = table.get("allowedActions", {})
    call_price = _call_amount(allowed)
    pot = int(table.get("potChips") or 0)
    required = pot_odds(call_price, pot)
    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    draw = has_flush_draw(hole_cards, board_cards)
    base_action, _amount, _message = base

    if base_action in {"fold", "call"} and call_price > 0:
        bad_price_call = base_action == "call" and required >= 0.34
        return bad_price_call or rank >= 1 or top_pair or draw or required <= 0.22
    if base_action in {"check", "bet", "raise"}:
        return rank in {1, 2} or top_pair or draw
    return False


def estimate_action_ev(table, my_seat, action, equity):
    allowed = table.get("allowedActions", {})
    pot = int(table.get("potChips") or 0)
    call_price = _call_amount(allowed)
    required = pot_odds(call_price, pot)
    showdown_ev = equity * pot - (1.0 - equity) * call_price

    if action.action == "fold":
        return 0.0
    if action.action == "check":
        return equity * pot
    if action.action == "call":
        return showdown_ev
    if action.action in {"bet", "raise"}:
        wager = int(action.amount or 0)
        if action.action == "raise" and equity + 0.16 < required:
            return -float(wager)
        fold_probability = _population_response(
            table,
            my_seat,
            action,
            equity,
            required,
        )
        called_pot = pot + wager + (call_price if action.action == "raise" else 0)
        called_ev = equity * called_pot - (1.0 - equity) * wager
        fold_ev = pot + (call_price if action.action == "raise" else 0)
        return fold_probability * fold_ev + (1.0 - fold_probability) * called_ev
    return float("-inf")


def is_close_spot(table, my_seat, base, equity):
    if table.get("street", "Preflop") == "Preflop":
        return False
    if equity is None:
        return False
    allowed = table.get("allowedActions", {})
    call_price = _call_amount(allowed)
    pot = int(table.get("potChips") or 0)
    required = pot_odds(call_price, pot)
    rank = made_hand_rank(my_seat.get("holeCards", []), table.get("boardCards", []))
    top_pair = has_top_pair_or_better(
        my_seat.get("holeCards", []),
        table.get("boardCards", []),
    )
    base_action, _amount, _message = base
    if base_action in {"fold", "call"} and call_price > 0:
        bad_price_call = base_action == "call" and equity + 0.16 < required
        return bad_price_call or abs(equity - required) <= 0.22 or rank >= 1 or top_pair
    if base_action in {"check", "bet", "raise"}:
        return CLOSE_STRENGTH_LOW <= equity <= CLOSE_STRENGTH_HIGH or rank in {1, 2}
    return False


def local_search_adjustment(table, my_seat, base) -> ActionDecision | None:
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if len(available) < 2:
        return None
    if not cheap_close_spot_filter(table, my_seat, base):
        return None
    equity = rollout_equity(table, my_seat)
    if not is_close_spot(table, my_seat, base, equity):
        return None

    actions = candidate_actions(table, my_seat, base)
    if len(actions) < 2:
        return None
    ev_by_action = {
        action: estimate_action_ev(table, my_seat, action, equity) for action in actions
    }
    best = max(actions, key=lambda action: ev_by_action[action])
    base_action = CandidateAction(base[0], base[1])
    base_ev = ev_by_action.get(base_action)
    if base_ev is None:
        matching = [action for action in actions if action.action == base[0]]
        base_ev = max(
            (ev_by_action[action] for action in matching),
            default=float("-inf"),
        )
    edge = ev_by_action[best] - base_ev
    pot = int(table.get("potChips") or 0)
    if base[0] in {"bet", "raise"} and best.action in {"check", "call", "fold"}:
        return None
    if base[0] in {"fold", "call"} and best.action in {"bet", "raise"}:
        return None
    if (
        base[0] == "check"
        and best.action == "bet"
        and not _has_explicit_fold_to_bet_profile(table)
    ):
        return None
    threshold = max(
        MIN_EDGE_CHIPS,
        min(
            MAX_EDGE_POT_FRACTION * max(pot, 1),
            MIN_EDGE_POT_FRACTION * max(pot, 1) + 12,
        ),
    )
    if best.action in {"bet", "raise"}:
        threshold = max(threshold, MIN_AGGRESSIVE_EDGE_CHIPS, 0.12 * max(pot, 1))
    if best.action == base[0] or edge < threshold:
        return None
    return (
        best.action,
        best.amount,
        (
            "v006 local search override: "
            f"{best.action} EV {ev_by_action[best]:+.1f} vs baseline "
            f"{base[0]} {base_ev:+.1f}, edge {edge:+.1f}, equity {equity:.0%}"
        ),
    )


def sixmax_adjustment(table, my_seat, base) -> ActionDecision | None:
    decision = local_search_adjustment(table, my_seat, base)
    if decision is not None:
        return decision
    return None


def choose_action(table, my_seat) -> ActionDecision:
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if not available:
        return None, None, "No legal actions available"
    if not my_seat:
        if "fold" in available:
            return "fold", None, "Fallback: seat not found"
        return None, None, "No matching seat found"

    base = champion.choose_action(table, my_seat)
    adjusted = sixmax_adjustment(table, my_seat, base)
    if adjusted is not None:
        return adjusted

    action, amount, message = base
    return action, amount, f"6:{message}"
