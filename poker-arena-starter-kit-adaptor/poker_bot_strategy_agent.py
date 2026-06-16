"""Adapter for poker_bot.strategies.* strategies.

Set POKER_BOT_STRATEGY before launching the benchmark:

    POKER_BOT_STRATEGY=auto_research_v008 ./pokerkit run --agent examples/poker_bot_strategy_agent.py --max-hands 50

The selected module must expose:

    choose_action(table, my_seat) -> (action, amount, message)
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Optional


DEFAULT_STRATEGY = "auto_research_v008"


def _strategy_name() -> str:
    name = os.getenv("POKER_BOT_STRATEGY", DEFAULT_STRATEGY).strip()
    if not name:
        name = DEFAULT_STRATEGY
    if any(part in name for part in ("/", "\\", ".")):
        raise SystemExit(
            "POKER_BOT_STRATEGY must be a module name like auto_research_v008, "
            f"not {name!r}"
        )
    return name


def _load_choose_action(strategy_name: str) -> Any:
    try:
        module = importlib.import_module(f"poker_bot.strategies.{strategy_name}")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Could not import poker_bot.strategies.{strategy_name}. "
            "Check POKER_BOT_STRATEGY and ensure the poker-bot dependency is installed."
        ) from exc

    choose_action = getattr(module, "choose_action", None)
    if not callable(choose_action):
        raise SystemExit(
            f"poker_bot.strategies.{strategy_name} must define callable choose_action(table, my_seat)."
        )
    return choose_action


STRATEGY_NAME = _strategy_name()
choose_action = _load_choose_action(STRATEGY_NAME)


def _hero_seat(table: dict) -> dict:
    self_num = table.get("selfSeatNumber")
    seats = table.get("seats") or []

    if self_num is not None:
        for seat in seats:
            if seat.get("seatNumber") == self_num:
                return seat

    acting_num = table.get("actingSeatNumber")
    if acting_num is not None:
        for seat in seats:
            if seat.get("seatNumber") == acting_num:
                return seat

    return {}


def _clean_card(card: object) -> object:
    if isinstance(card, str):
        stripped = card.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return stripped[1:-1]
    return card


def _clean_cards(cards: Any) -> list[object]:
    return [_clean_card(card) for card in (cards or [])]


def _normalize_table_for_strategy(table: dict) -> dict:
    """Normalize local selfplay card reprs like '[Ah]' to Arena-style 'Ah'."""
    normalized = dict(table)
    normalized["boardCards"] = _clean_cards(table.get("boardCards"))

    seats = []
    for seat in table.get("seats") or []:
        normalized_seat = dict(seat)
        normalized_seat["holeCards"] = _clean_cards(normalized_seat.get("holeCards"))
        seats.append(normalized_seat)
    normalized["seats"] = seats
    return normalized


def _reasoning(action: str, table: dict, allowed: dict) -> str:
    board = table.get("boardCards") or []
    street = table.get("street") or "Preflop"
    seat = table.get("selfSeatNumber") or table.get("actingSeatNumber") or 0
    pos = "IP" if seat and int(seat) % 2 == 0 else "OOP"

    plan = {"Preflop": "ranges", "Flop": "spot", "Turn": "pressure", "River": "resolve"}
    pp = f"{pos} {plan.get(street, 'pot ctrl')}"[:30]

    if not board:
        bf = "[pf]"
    else:
        suits = [str(card)[-1].lower() for card in board]
        ranks = [str(card)[0].upper() for card in board]
        features = []
        for suit in set(suits):
            if suits.count(suit) >= 2:
                features.append(f"FD-{suit}")
        if len(set(ranks)) < len(ranks):
            features.append("paired")
        bf = "[" + ",".join(features[:3]) + "]" if features else "[dry]"

    parts = [
        f'vr: "{STRATEGY_NAME}"',
        f'ke: "legal {action}"',
        f"bf: {bf}",
        f'pp: "{pp}"',
    ]
    yaml = "{" + ", ".join(parts) + "}"
    if len(yaml) <= 150:
        return yaml
    return f'{{vr: "{STRATEGY_NAME}", ke: "legal", pp: "pot control"}}'


def _payload(action: str, amount: Optional[int], table: dict, message: str) -> dict:
    allowed = table.get("allowedActions") or {}
    available = allowed.get("availableActions") or []
    if action not in available and available:
        action = available[0]

    if action in {"fold", "check", "call"}:
        amount = None
    elif action == "all-in":
        amount = int(allowed.get("allInToAmount") or amount or 0)
    elif action == "raise":
        rr = allowed.get("raiseRange") or {}
        if rr:
            lo, hi = int(rr.get("min") or 1), int(rr.get("max") or int(amount or 1))
            amount = max(lo, min(int(amount or lo), hi))
        else:
            amount = int(amount or allowed.get("minRaiseTo") or 1)
    elif action == "bet":
        br = allowed.get("betRange") or {}
        if br:
            lo, hi = int(br.get("min") or 1), int(br.get("max") or int(amount or 1))
            amount = max(lo, min(int(amount or lo), hi))
        else:
            amount = int(amount or allowed.get("minBet") or 1)
    else:
        amount = None

    payload = {
        "action": action,
        "message": (message or f"{STRATEGY_NAME} chose {action}")[:500],
        "reasoning": _reasoning(action, table, allowed),
    }
    if amount is not None:
        payload["amount"] = int(amount)
    return payload


def _deadline_fallback(table: dict) -> dict:
    allowed = table.get("allowedActions") or {}
    available = allowed.get("availableActions") or []
    for action in ("check", "call", "fold"):
        if action in available:
            return _payload(action, None, table, "deadline tight, safe legal fallback")
    if available:
        return _payload(available[0], None, table, "deadline tight, legal fallback")
    return _payload("fold", None, table, "no legal action available")


def decide(
    table: dict, deadline_s: float = 10.0, research_context: Optional[dict] = None
) -> dict:
    """Return one Arena-compatible action using the selected poker_bot strategy."""
    if deadline_s < 2.0:
        return _deadline_fallback(table)

    normalized = _normalize_table_for_strategy(table)
    hero = _hero_seat(normalized)
    action, amount, message = choose_action(normalized, hero)
    return _payload(action or "fold", amount, table, message or STRATEGY_NAME)
