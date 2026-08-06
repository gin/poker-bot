"""
Unified guardrails system for poker bot.

Provides pre-decision and post-decision guards that can be imported
into any strategy. All guard overrides are logged to the database.

Guard Types:
- Pre-decision: run BEFORE the NN/strategy proposes an action; can short-circuit.
- Post-decision: run AFTER the NN/strategy proposes; can override to a safer action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

GuardId = str
GuardFunc = Callable[[dict[str, Any], dict[str, Any]], Optional["GuardResult"]]
PostGuardFunc = Callable[[dict[str, Any], dict[str, Any], str], Optional["GuardResult"]]


@dataclass(frozen=True)
class GuardResult:
    fired: bool
    guard_id: GuardId
    original_action: str
    final_action: str
    reason: str
    pre_decision: bool = False


class GuardRail:
    """Central registry and runner for guard rules."""

    def __init__(self) -> None:
        self._pre_guards: list[GuardFunc] = []
        self._post_guards: list[PostGuardFunc] = []

    def register_pre(self, func: GuardFunc) -> GuardFunc:
        self._pre_guards.append(func)
        return func

    def register_post(self, func: PostGuardFunc) -> PostGuardFunc:
        self._post_guards.append(func)
        return func

    def run_pre(
        self, table: dict[str, Any], my_seat: dict[str, Any]
    ) -> Optional[GuardResult]:
        for guard_func in self._pre_guards:
            result = guard_func(table, my_seat)
            if result and result.fired:
                return result
        return None

    def run_post(
        self, table: dict[str, Any], my_seat: dict[str, Any], proposed_action: str
    ) -> GuardResult:
        for guard_func in self._post_guards:
            result = guard_func(table, my_seat, proposed_action)
            if result and result.fired:
                return result
        return GuardResult(
            fired=False,
            guard_id="",
            original_action=proposed_action,
            final_action=proposed_action,
            reason="approved",
            pre_decision=False,
        )

    def log_override(
        self,
        conn: Any,
        *,
        run_id: str,
        hand_id: str,
        decision_index: int,
        guard_result: GuardResult,
        table: dict[str, Any],
        seat: dict[str, Any],
    ) -> None:
        allowed = table.get("allowedActions") or {}
        conn.execute(
            """
            insert into guard_overrides(
                run_id, hand_id, decision_index, guard_id, pre_decision,
                original_action, final_action, reason,
                street, pot_chips, call_amount, available_actions
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                hand_id,
                decision_index,
                guard_result.guard_id,
                1 if guard_result.pre_decision else 0,
                guard_result.original_action,
                guard_result.final_action,
                guard_result.reason,
                table.get("street"),
                table.get("potChips"),
                allowed.get("callAmount") or allowed.get("callChips") or 0,
                _join_actions(allowed.get("availableActions", [])),
            ),
        )

    def _action_from_guard(
        self,
        guard_result: GuardResult,
        table: dict[str, Any],
    ) -> tuple[str, int | None, str]:
        """Convert a GuardResult into (action, amount, reason) tuple."""
        return _action_from_guard(guard_result, table)


RANK_VALUES = {rank: index for index, rank in enumerate("23456789TJQKA", start=2)}

ROYAL_RANKS = {"T", "J", "Q", "K", "A"}
ROYAL_SUITS = {"S", "H", "D", "C"}


def _card_values(cards: list[str]) -> list[int]:
    return [RANK_VALUES.get(card[0], 0) for card in cards]


def _join_actions(actions: list[str]) -> str:
    return ",".join(str(a) for a in actions)


def _join_cards(cards: list[str]) -> str:
    return ",".join(str(c) for c in cards)


def royal_flush_possible(hole_cards: list[str], board_cards: list[str]) -> bool:
    known_cards = list(hole_cards) + list(board_cards)
    remaining_board_slots = max(0, 5 - len(board_cards))

    for suit in ROYAL_SUITS:
        hole_royals = {
            card[0]
            for card in hole_cards
            if len(card) >= 2 and card[0] in ROYAL_RANKS and card[1] == suit
        }
        if not hole_royals:
            continue

        known_royals = {
            card[0]
            for card in known_cards
            if len(card) >= 2 and card[0] in ROYAL_RANKS and card[1] == suit
        }
        if len(known_royals) < 2:
            continue

        missing = len(ROYAL_RANKS - known_royals)
        if missing <= remaining_board_slots:
            return True

    return False


def _is_aks(hole_cards: list[str]) -> bool:
    if len(hole_cards) != 2:
        return False
    ranks = {c[0] for c in hole_cards}
    suits = {c[1] for c in hole_cards}
    return ranks == {"A", "K"} and len(suits) == 1


def _safe_call_amount(allowed: dict[str, Any]) -> int:
    return int(allowed.get("callAmount") or allowed.get("callChips") or 0)


def _safe_min_raise_to(allowed: dict[str, Any]) -> int:
    value = allowed.get("minRaiseTo")
    if value is None:
        raise_range = allowed.get("raiseRange") or {}
        value = raise_range.get("min")
    return int(value or 0)


def _action_from_guard(
    guard_result: GuardResult, table: dict[str, Any]
) -> tuple[str, int | None, str]:
    action = guard_result.final_action
    allowed = table.get("allowedActions") or {}

    if action in ("call", "check"):
        amount = _safe_call_amount(allowed) if action == "call" else None
    elif action == "raise":
        amount = _safe_min_raise_to(allowed)
    else:
        amount = None

    return action, amount, guard_result.reason


# ---------------------------------------------------------------------------
# PRE-DECISION GUARDS
# ---------------------------------------------------------------------------


def royal_flush_predecision_guard(
    table: dict[str, Any], my_seat: dict[str, Any]
) -> Optional[GuardResult]:
    """Force check or call when royal flush is possible.

    Preflop: if hero holds AKs, never fold or raise — only check or call.
    Postflop: if royal flush is still possible, never fold or raise.
    """
    hole_cards = my_seat.get("holeCards", [])
    street = table.get("street", "Preflop")

    # Preflop: AKs should never fold or raise
    if street == "Preflop" and len(hole_cards) == 2 and _is_aks(hole_cards):
        allowed = table.get("allowedActions", {})
        available = allowed.get("availableActions", [])

        if "check" in available:
            call_amt = _safe_call_amount(allowed)
            if call_amt == 0:
                return GuardResult(
                    fired=True,
                    guard_id="royal_flush_aks_preflop",
                    original_action="__pending__",
                    final_action="check",
                    reason="AKs preflop — royal flush guard forces check",
                    pre_decision=True,
                )

        if "call" in available:
            call_amt = _safe_call_amount(allowed)
            return GuardResult(
                fired=True,
                guard_id="royal_flush_aks_preflop",
                original_action="__pending__",
                final_action="call",
                reason=f"AKs preflop — royal flush guard forces call ({call_amt} chips)",
                pre_decision=True,
            )

        return None

    # Postflop: check if royal flush is still possible
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None

    if not royal_flush_possible(hole_cards, board_cards):
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])

    if "check" in available:
        call_amt = _safe_call_amount(allowed)
        if call_amt == 0:
            return GuardResult(
                fired=True,
                guard_id="royal_flush_possible_postflop",
                original_action="__pending__",
                final_action="check",
                reason="Royal flush possible — guard forces check",
                pre_decision=True,
            )

    if "call" in available:
        call_amt = _safe_call_amount(allowed)
        return GuardResult(
            fired=True,
            guard_id="royal_flush_possible_postflop",
            original_action="__pending__",
            final_action="call",
            reason=f"Royal flush possible — guard forces call ({call_amt} chips)",
            pre_decision=True,
        )

    return None


# ---------------------------------------------------------------------------
# POST-DECISION GUARDS
# ---------------------------------------------------------------------------


def sliver_shove_floor(
    table: dict[str, Any], my_seat: dict[str, Any], proposed_action: str
) -> Optional[GuardResult]:
    """If pot odds are < 15%, folding is almost always a mistake."""
    if proposed_action != "fold":
        return None

    allowed = table.get("allowedActions") or {}
    call_amt = allowed.get("callAmount", 0) or 0
    if call_amt <= 0:
        return None

    pot = table.get("potChips", 0) or 0
    pot_odds = call_amt / (pot + call_amt + 1e-6)

    if pot_odds < 0.15:
        return GuardResult(
            fired=True,
            guard_id="sliver_shove_floor",
            original_action="fold",
            final_action="call",
            reason=f"Pot odds {pot_odds:.2f} < 0.15 — override fold to call",
            pre_decision=False,
        )

    return None


def excessive_bet_size_cap(
    table: dict[str, Any], my_seat: dict[str, Any], proposed_action: str
) -> Optional[GuardResult]:
    """Prevent absurdly large raises."""
    if proposed_action != "raise":
        return None

    allowed = table.get("allowedActions") or {}
    call_amt = allowed.get("callAmount", 0) or 0
    pot = table.get("potChips", 0) or 0
    raise_range = allowed.get("raiseRange") or {}
    raise_amt = raise_range.get("min", 0) or 0

    if raise_amt > pot * 3:
        return GuardResult(
            fired=True,
            guard_id="excessive_bet_size",
            original_action="raise",
            final_action="call",
            reason=f"Min raise {raise_amt} > 3x pot {pot} — cap to call",
            pre_decision=False,
        )

    return None


def trips_on_paired_board_cap(
    table: dict[str, Any], my_seat: dict[str, Any], proposed_action: str
) -> Optional[GuardResult]:
    """Prevent massive overbets with trips on paired board.

    If we have trips (made_rank == 3) on a paired board and the raise
    would be > 1.5x pot, cap to call instead.
    """
    if proposed_action != "raise":
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(hole_cards) != 2 or len(board_cards) < 3:
        return None

    # Check if board is paired
    board_values = [c[0] for c in board_cards]
    if len(set(board_values)) == len(board_values):
        return None  # board not paired

    # Compute made hand rank (trips = rank 3)
    # Reuse the hand_eval logic
    from poker_bot.hand_eval import evaluate_hand
    from poker_bot.strategies.adaptive import made_hand_rank

    made_rank = made_hand_rank(hole_cards, board_cards)
    if made_rank != 3:  # trips = rank 3
        return None

    # We have trips on a paired board - vulnerable hand
    allowed = table.get("allowedActions") or {}
    raise_range = allowed.get("raiseRange") or {}
    raise_amt = raise_range.get("min", 0) or 0
    pot = table.get("potChips", 0) or 0

    if raise_amt > pot * 1.5:
        # Cap to call instead of massive overbet
        call_amt = _safe_call_amount(allowed)
        return GuardResult(
            fired=True,
            guard_id="trips_paired_board_cap",
            original_action="raise",
            final_action="call",
            reason=f"Trips on paired board: raise {raise_amt} > 1.5x pot {pot} — cap to call",
            pre_decision=False,
        )

    return None


def nut_hand_protection(
    table: dict[str, Any], my_seat: dict[str, Any], proposed_action: str
) -> Optional[GuardResult]:
    """Never fold the nuts or near-nuts.

    Placeholder for future encoder integration.
    """
    return None


# ---------------------------------------------------------------------------
# GLOBAL GUARD RAIL INSTANCE
# ---------------------------------------------------------------------------

default_guardrail: Optional[GuardRail] = None


def get_guardrail() -> GuardRail:
    global default_guardrail
    if default_guardrail is None:
        default_guardrail = GuardRail()
        default_guardrail.register_pre(royal_flush_predecision_guard)
        default_guardrail.register_post(sliver_shove_floor)
        default_guardrail.register_post(excessive_bet_size_cap)
        default_guardrail.register_post(trips_on_paired_board_cap)
        default_guardrail.register_post(nut_hand_protection)
    return default_guardrail
