"""Strategy-local vetoes for decisions that commit the hero's full stack.

The helpers here are deliberately not registered on the global guard rails.
They implement an opt-in, strategy-specific policy: only a preflop pair of
aces or the current sole nuts may commit every remaining chip.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations

from poker_bot.hand_eval import DECK, evaluate_hand

ActionDecision = tuple[str | None, int | None, str]
_ALL_IN_ACTIONS = frozenset({"all-in", "allin", "all_in"})


def _normalize_cards(cards: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return validated canonical cards suitable for the shared evaluator."""
    normalized = tuple(str(card).strip().upper() for card in cards)
    if any(len(card) != 2 or card not in DECK for card in normalized):
        raise ValueError("cards must be unique standard two-character cards")
    if len(set(normalized)) != len(normalized):
        raise ValueError("known cards must not contain duplicates")
    return normalized


@lru_cache(maxsize=8_192)
def _has_current_sole_nuts(
    hole_cards: tuple[str, str], board_cards: tuple[str, ...]
) -> bool:
    """Return whether hero's made hand strictly beats every legal opponent hand.

    This evaluates the current board only. A shared board hand is intentionally
    not enough: a sole-nuts policy does not stack off for a forced chop.
    """
    hero_rank = evaluate_hand([*hole_cards, *board_cards])
    known_cards = frozenset((*hole_cards, *board_cards))
    remaining_cards = tuple(card for card in DECK if card not in known_cards)
    best_opponent_rank = max(
        evaluate_hand([*opponent_cards, *board_cards])
        for opponent_cards in combinations(remaining_cards, 2)
    )
    return hero_rank > best_opponent_rank


def has_current_sole_nuts(hole_cards: list[str], board_cards: list[str]) -> bool:
    """Return whether hero has the current, non-shared nuts postflop."""
    normalized_hole = _normalize_cards(hole_cards)
    normalized_board = _normalize_cards(board_cards)
    if len(normalized_hole) != 2:
        return False
    if len(normalized_board) < 3 or len(normalized_board) > 5:
        return False
    if set(normalized_hole).intersection(normalized_board):
        raise ValueError("hole and board cards must not overlap")
    return _has_current_sole_nuts(normalized_hole, normalized_board)


def has_all_in_permission(hole_cards: list[str], board_cards: list[str]) -> bool:
    """Return whether this hand may commit the hero's entire stack."""
    normalized_hole = _normalize_cards(hole_cards)
    normalized_board = _normalize_cards(board_cards)
    if len(normalized_hole) != 2:
        return False
    if not normalized_board:
        return normalized_hole[0][0] == normalized_hole[1][0] == "A"
    return has_current_sole_nuts(list(normalized_hole), list(normalized_board))


def commits_all_chips(table: dict, my_seat: dict, decision: ActionDecision) -> bool:
    """Return whether an action spends all of the hero's remaining stack."""
    action, amount, _message = decision
    action = str(action or "").lower()
    if action in _ALL_IN_ACTIONS:
        return True

    allowed = table.get("allowedActions") or {}
    stack = int(my_seat.get("stackChips") or 0)
    if stack <= 0:
        return False
    if action == "call":
        call = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
        return call >= stack
    if action not in {"bet", "raise"} or amount is None:
        return False

    total_stack = stack + int(my_seat.get("currentBetChips") or 0)
    return int(amount) >= total_stack


def veto_non_nut_all_in(
    table: dict, my_seat: dict, decision: ActionDecision
) -> ActionDecision:
    """Replace a forbidden stack-off with the safest legal passive action."""
    if not commits_all_chips(table, my_seat, decision):
        return decision

    hole_cards = my_seat.get("holeCards") or []
    board_cards = table.get("boardCards") or []
    if has_all_in_permission(hole_cards, board_cards):
        return decision

    action, _amount, message = decision
    allowed = table.get("allowedActions") or {}
    available = set(allowed.get("availableActions") or ())
    stack = int(my_seat.get("stackChips") or 0)
    call = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    reason = f"{message} [nut all-in veto]"

    if "check" in available and call <= 0:
        return "check", None, f"{reason}: checking without sole nuts"
    if "call" in available and 0 < call < stack:
        return "call", call, f"{reason}: calling without committing stack"
    if "fold" in available:
        return "fold", None, f"{reason}: folding non-nut stack-off"

    # The engine supplied no legal non-all-in action. Returning the original
    # decision is safer than manufacturing an amount that may be illegal.
    return decision
