"""Guard registry for the sandbox."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from poker_bot.guards.context import GuardContext

ActionDecision = tuple[str, int | None, str]
GuardFunc = Callable[[GuardContext], Optional[ActionDecision]]
PostGuardFunc = Callable[[GuardContext, ActionDecision], Optional[ActionDecision]]


@dataclass
class GuardMeta:
    """Guard metadata."""

    guard_id: str
    phase: str
    precedence: int
    table_sizes: list[str]
    description: str
    func: Callable


class GuardRail:
    """Guard rail."""

    def __init__(self) -> None:
        self._guards: list[GuardMeta] = []


    def register(self, guard_id: str, phase: str, precedence: int,
                table_sizes: list[str] | None = None, description: str = ""):
        """Register a guard."""
        def decorator(func: Callable) -> Callable:
            self._guards.append(GuardMeta(
                guard_id=guard_id,
                phase=phase,
                precedence=precedence,
                table_sizes=table_sizes or ["hu", "6max"],
                description=description,
                func=func,
            ))
            return func
        return decorator


    def run_pre(self, ctx: GuardContext) -> Optional[tuple[ActionDecision, str]]:
        """Run pre-decision guards."""
        for meta in sorted(self._guards, key=lambda m: m.precedence):
            if meta.phase != "pre":
                continue
            result = meta.func(ctx)
            if result is not None:
                return result, meta.guard_id
        return None
