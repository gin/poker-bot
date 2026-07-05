"""GuardRail registry — precedence-based guard cascade.

Guards are registered with metadata (id, phase, precedence, table_sizes).
The registry runs them in precedence order (lowest number = highest priority
= fires first). This prevents the "general guard swallows specific guard"
problem: specific guards (lower precedence number) always fire before general
guards (higher precedence number).

Guard protocol:
    - Each guard receives a GuardContext and returns an ActionDecision | None
    - Pre-guards: run BEFORE the core. If any fires, the core is skipped.
    - Post-guards: run AFTER the core proposes an action. They receive the
      core's proposed action and can override it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from poker_bot.guards.context import GuardContext

ActionDecision = tuple[str, int | None, str]
GuardFunc = Callable[[GuardContext], Optional[ActionDecision]]
PostGuardFunc = Callable[[GuardContext, ActionDecision], Optional[ActionDecision]]


@dataclass
class GuardMeta:
    """Metadata for a registered guard."""
    guard_id: str
    phase: str              # "pre" or "post"
    precedence: int         # 0=highest (fires first), 4=lowest
    table_sizes: list[str]  # ["hu"], ["6max"], ["hu", "6max"]
    description: str
    func: Callable


class GuardRail:
    """Central registry and runner for guard rules."""

    def __init__(self) -> None:
        self._guards: list[GuardMeta] = []

    def register(
        self,
        guard_id: str,
        phase: str,
        precedence: int,
        table_sizes: list[str] | None = None,
        description: str = "",
    ):
        """Decorator to register a guard with metadata."""
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

    def _applicable(self, meta: GuardMeta, ctx: GuardContext) -> bool:
        """Check if a guard applies to the current table size."""
        if "all" in meta.table_sizes:
            return True
        if ctx.is_heads_up and "hu" in meta.table_sizes:
            return True
        if not ctx.is_heads_up and "6max" in meta.table_sizes:
            return True
        return False

    def run_pre(self, ctx: GuardContext) -> Optional[tuple[ActionDecision, str]]:
        """Run pre-decision guards. Returns (decision, guard_id) or None."""
        for meta in sorted(self._guards, key=lambda m: m.precedence):
            if meta.phase != "pre":
                continue
            if not self._applicable(meta, ctx):
                continue
            result = meta.func(ctx)
            if result is not None:
                return result, meta.guard_id
        return None

    def run_post(
        self, ctx: GuardContext, proposed: ActionDecision
    ) -> tuple[ActionDecision, str]:
        """Run post-decision guards. Returns (final_decision, guard_id_or_approved)."""
        for meta in sorted(self._guards, key=lambda m: m.precedence):
            if meta.phase != "post":
                continue
            if not self._applicable(meta, ctx):
                continue
            result = meta.func(ctx, proposed)
            if result is not None:
                return result, meta.guard_id
        return proposed, "approved"

    def list_guards(self, phase: str | None = None) -> list[GuardMeta]:
        """List registered guards, optionally filtered by phase."""
        guards = sorted(self._guards, key=lambda m: (m.phase, m.precedence))
        if phase:
            guards = [g for g in guards if g.phase == phase]
        return guards
