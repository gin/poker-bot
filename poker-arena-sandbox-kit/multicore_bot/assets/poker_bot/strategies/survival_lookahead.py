"""Survival lookup strategy with bounded Pluribus-style lookahead.

The lookup strategy is the blueprint. This module only searches a small action
abstraction in high-impact postflop spots, then uses a heuristic leaf score that
balances chip gain against bust risk.
"""

from __future__ import annotations

from poker_bot.strategies import survival_lookup
from poker_bot.strategies.adaptive import (
    BIG_BLIND,
    board_texture,
    has_top_pair_or_better,
    made_hand_rank,
    pot_odds,
    preflop_score,
)

ActionDecision = tuple[str | None, int | None, str]
Candidate = tuple[str, int | None]

LOOKAHEAD_STYLES = {"unknown", "short_handed", "loose_aggressive", "tight"}


def clamp(value, low, high):
    return max(low, min(high, value))


def amount_delta(action, amount, my_seat):
    if action not in {"bet", "raise"} or amount is None:
        return 0
    current = int(my_seat.get("currentBetChips") or 0)
    return max(0, int(amount) - current)


def legal_amount(action, amount, allowed):
    if amount is None:
        return None
    maximum = int(allowed.get("maxCommit") or amount)
    if action == "bet":
        minimum = int(allowed.get("minBet") or BIG_BLIND)
    elif action == "raise":
        minimum = int(allowed.get("minRaiseTo") or amount)
    else:
        return int(amount)
    return clamp(int(amount), minimum, maximum)


def candidate_actions(table, my_seat, blueprint: Candidate):
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    pot = int(table.get("potChips") or 0)
    current_bet = int(table.get("currentBet") or 0)
    candidates: list[Candidate] = []

    def add(action, amount=None):
        if action not in available:
            return
        normalized = (action, legal_amount(action, amount, allowed))
        if normalized not in candidates:
            candidates.append(normalized)

    add(*blueprint)
    add("fold")
    add("check")
    if "call" in available:
        add("call", int(allowed.get("callAmount") or allowed.get("callChips") or 0))
    if "bet" in available:
        minimum = int(allowed.get("minBet") or BIG_BLIND)
        add("bet", minimum)
        add("bet", max(minimum, int(max(pot, BIG_BLIND) * 0.35)))
        add("bet", max(minimum, int(max(pot, BIG_BLIND) * 0.60)))
    if "raise" in available:
        minimum = int(allowed.get("minRaiseTo") or current_bet + BIG_BLIND)
        add("raise", minimum)
        add("raise", max(minimum, current_bet + int(max(pot, BIG_BLIND) * 0.45)))
        add("raise", max(minimum, current_bet + int(max(pot, BIG_BLIND) * 0.75)))

    return candidates


def hand_equity_proxy(table, my_seat):
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if not board_cards:
        return clamp(preflop_score(hole_cards) / 135, 0.05, 0.90)

    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards)
    if rank >= 5:
        equity = 0.93
    elif rank == 4:
        equity = 0.86
    elif rank == 3:
        equity = 0.78
    elif rank == 2:
        equity = 0.64
    elif rank == 1 or top_pair:
        equity = 0.48
    else:
        equity = 0.20

    if texture.get("wet", False) and rank < 3:
        equity -= 0.06
    if survival_lookup.active_players(table) > 3:
        equity -= 0.05
    return clamp(equity, 0.04, 0.96)


def style_fold_equity(style, active_count, action):
    base = {
        "tight": 0.44,
        "unknown": 0.24,
        "short_handed": 0.22,
        "loose_aggressive": 0.18,
        "loose_passive": 0.10,
    }.get(style, 0.20)
    if action == "raise":
        base += 0.04
    base -= max(0, active_count - 2) * 0.05
    return clamp(base, 0.04, 0.52)


def pressure_when_covering(table, my_seat):
    my_stack = int(my_seat.get("stackChips") or 0)
    for seat in table.get("seats", []):
        if seat.get("agentId") == my_seat.get("agentId"):
            continue
        if seat.get("folded", False) or seat.get("hasFolded", False):
            continue
        if int(seat.get("stackChips") or 0) > my_stack:
            return True
    return False


def score_candidate(table, my_seat, style, candidate: Candidate, blueprint: Candidate):
    action, amount = candidate
    allowed = table.get("allowedActions", {})
    pot = int(table.get("potChips") or 0)
    stack = int(my_seat.get("stackChips") or 0)
    call_amount = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    required = pot_odds(call_amount, pot)
    active = survival_lookup.active_players(table)
    equity = hand_equity_proxy(table, my_seat)
    rank = made_hand_rank(my_seat.get("holeCards", []), table.get("boardCards", []))
    bucket = survival_lookup.hand_bucket(
        my_seat.get("holeCards", []), table.get("boardCards", [])
    )

    if action == "fold":
        return -call_amount * 0.35 - equity * max(pot, BIG_BLIND) * 0.10
    if action == "check":
        return equity * pot * 0.18 + 4

    if action == "call":
        commit = call_amount
        stack_after = max(0, stack - commit)
        score = equity * (pot + commit) - (1 - equity) * commit
        if bucket == "medium" and required <= 0.26:
            score += 18
        if rank >= 2 and required <= 0.32:
            score += 26
        if pressure_when_covering(table, my_seat) and bucket != "air":
            score += 10
        score -= bust_risk_penalty(commit, stack, stack_after)
    else:
        commit = amount_delta(action, amount, my_seat)
        stack_after = max(0, stack - commit)
        fold_equity = style_fold_equity(style, active, action)
        called_value = equity * (pot + commit) - (1 - equity) * commit
        score = fold_equity * pot + (1 - fold_equity) * called_value
        if rank >= 3:
            score += 18
        elif rank == 2 and call_amount > 0:
            score -= 42
        elif bucket == "air" and style == "tight":
            score += 12
        elif bucket == "air":
            score -= 20
        score -= bust_risk_penalty(commit, stack, stack_after)

    if candidate == blueprint:
        score += 6
    return score


def bust_risk_penalty(commit, stack, stack_after):
    if stack <= 0 or commit <= 0:
        return 0
    commit_ratio = commit / stack
    penalty = commit_ratio * commit * 0.35
    if stack_after < BIG_BLIND * 8:
        penalty += 28
    if commit_ratio >= 0.50:
        penalty += 35
    return penalty


def should_lookahead(table, my_seat, style, blueprint: Candidate):
    street = table.get("street", "Preflop")
    if street != "Flop":
        return False
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "call" not in available:
        return False
    call_amount = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    required = pot_odds(call_amount, int(table.get("potChips") or 0))
    rank = made_hand_rank(my_seat.get("holeCards", []), table.get("boardCards", []))
    bucket = survival_lookup.hand_bucket(
        my_seat.get("holeCards", []), table.get("boardCards", [])
    )
    has_profiles = bool(table.get("opponentProfiles"))
    if not has_profiles or style != "loose_aggressive":
        return False
    if blueprint[0] == "fold":
        return rank >= 2 or (bucket == "medium" and required <= 0.24)
    if blueprint[0] == "raise":
        return rank == 2 and required <= 0.32
    return False


def lookahead_action(table, my_seat, blueprint_action, blueprint_amount):
    blueprint = (blueprint_action, blueprint_amount)
    style = survival_lookup.table_style(table, my_seat)
    if not should_lookahead(table, my_seat, style, blueprint):
        return None

    allowed = table.get("allowedActions", {})
    call_amount = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    candidates = [blueprint, ("fold", None), ("call", call_amount)]
    if not candidates:
        return None

    scored = [
        (score_candidate(table, my_seat, style, candidate, blueprint), candidate)
        for candidate in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    blueprint_score = score_candidate(table, my_seat, style, blueprint, blueprint)
    if best == blueprint or best_score < blueprint_score + 5:
        return None
    action, amount = best
    return (
        action,
        amount,
        (
            f"lookahead {style}: {action}"
            f" score {best_score:.1f} over blueprint {blueprint_score:.1f}"
        ),
    )


def choose_action(table, my_seat) -> ActionDecision:
    action, amount, message = survival_lookup.choose_action(table, my_seat)
    if action is None or not my_seat:
        return action, amount, message

    override = lookahead_action(table, my_seat, action, amount)
    if override is not None:
        return override
    return action, amount, f"lookahead blueprint: {message}"
