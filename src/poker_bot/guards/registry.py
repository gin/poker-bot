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
from poker_bot.guards.telemetry import GuardEvent, guard_mode, record_event

ActionDecision = tuple[str, int | None, str]
GuardFunc = Callable[[GuardContext], Optional[ActionDecision]]
PostGuardFunc = Callable[[GuardContext, ActionDecision], Optional[ActionDecision]]


@dataclass
class GuardMeta:
    """Metadata for a registered guard."""

    guard_id: str
    phase: str  # "pre" or "post"
    precedence: int  # 0=highest (fires first), 4=lowest
    table_sizes: list[str]  # ["hu"], ["6max"], ["hu", "6max"]
    description: str
    func: Callable
    shadow: bool = False  # shadow guards log fires but never override


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
        shadow: bool = False,
    ):
        """Decorator to register a guard with metadata."""

        def decorator(func: Callable) -> Callable:
            self._guards.append(
                GuardMeta(
                    guard_id=guard_id,
                    phase=phase,
                    precedence=precedence,
                    table_sizes=table_sizes or ["hu", "6max"],
                    description=description,
                    func=func,
                    shadow=shadow,
                )
            )
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

    def _record_fire(
        self,
        meta: GuardMeta,
        ctx: GuardContext,
        result: ActionDecision,
        *,
        applied: bool,
        mode: str,
        original: ActionDecision | None,
    ) -> None:
        action, amount, reason = result
        orig_action, orig_amount = "__pending__", None
        if original is not None:
            orig_action, orig_amount = original[0], original[1]
        record_event(
            GuardEvent(
                guard_id=meta.guard_id,
                phase=meta.phase,
                precedence=meta.precedence,
                shadow=(mode == "shadow"),
                applied=applied,
                original_action=orig_action,
                original_amount=orig_amount,
                final_action=action,
                final_amount=amount,
                reason=reason,
                street=ctx.street,
                pot=ctx.pot,
                call_price=ctx.call_price,
                available_actions=",".join(ctx.available_actions),
            )
        )

    def _run_phase(
        self,
        phase: str,
        ctx: GuardContext,
        proposed: ActionDecision | None,
    ) -> tuple[ActionDecision, str] | None:
        """Evaluate every applicable guard in precedence order.

        Every fire is recorded as a GuardEvent; only the first fire from an
        *active* guard is applied (returned). Shadow fires and fires after
        the applied one are logged with applied=False, which gives overlap
        data (co-firing guards) for free. A guard that raises is skipped so
        one broken guard cannot kill the whole cascade.
        """
        winner: tuple[ActionDecision, str] | None = None
        for meta in sorted(self._guards, key=lambda m: m.precedence):
            if meta.phase != phase:
                continue
            if not self._applicable(meta, ctx):
                continue
            mode = guard_mode(meta.guard_id, shadow_default=meta.shadow)
            if mode == "off":
                continue
            try:
                result = (
                    meta.func(ctx) if phase == "pre" else meta.func(ctx, proposed)
                )
            except Exception:
                continue
            if result is None:
                continue
            applies = mode == "active" and winner is None
            self._record_fire(
                meta, ctx, result, applied=applies, mode=mode, original=proposed
            )
            if applies:
                winner = (result, meta.guard_id)
        return winner

    def run_pre(self, ctx: GuardContext) -> Optional[tuple[ActionDecision, str]]:
        """Run pre-decision guards. Returns (decision, guard_id) or None."""
        return self._run_phase("pre", ctx, None)

    def run_post(
        self, ctx: GuardContext, proposed: ActionDecision
    ) -> tuple[ActionDecision, str]:
        """Run post-decision guards. Returns (final_decision, guard_id_or_approved)."""
        winner = self._run_phase("post", ctx, proposed)
        if winner is not None:
            return winner
        return proposed, "approved"

    def list_guards(self, phase: str | None = None) -> list[GuardMeta]:
        """List registered guards, optionally filtered by phase."""
        guards = sorted(self._guards, key=lambda m: (m.phase, m.precedence))
        if phase:
            guards = [g for g in guards if g.phase == phase]
        return guards
