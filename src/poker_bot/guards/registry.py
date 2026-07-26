"""GuardRail registry — precedence-based guard cascade.

Guards are registered with metadata (id, phase, precedence, regimes).
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
from poker_bot.guards.telemetry import (
    GuardEvent,
    guard_mode,
    record_event,
    record_guard_error,
)

ActionDecision = tuple[str, int | None, str]
GuardFunc = Callable[[GuardContext], Optional[ActionDecision]]
PostGuardFunc = Callable[[GuardContext, ActionDecision], Optional[ActionDecision]]


@dataclass
class GuardMeta:
    """Metadata for a registered guard."""

    guard_id: str
    phase: str  # "pre" or "post"
    precedence: int  # 0=highest (fires first), 4=lowest
    # Canonical hand-level player regimes this guard applies to: any of
    # "heads_up", "three_handed", "full_table", or the sentinel "all" for
    # guards genuinely unrestricted by table size. Matched against
    # ctx.regime (poker_bot.hand_utils.player_regime), never seat counts.
    regimes: list[str]
    description: str
    func: Callable
    shadow: bool = False  # shadow guards log fires but never override


class GuardRail:
    """Central registry and runner for guard rules.

    default_shadow=True makes every guard registered on this rail run in
    shadow mode (log fires, never override) unless the registration passes
    an explicit shadow=False. Pruning sweep 2026-07-06: no guard showed a
    positive counterfactual on the benchmark pools, so both rails default
    to shadow; activate individual guards only after they pass a pool-wide
    on/off gate (or force with POKER_GUARD_ACTIVATE for experiments).
    """

    def __init__(self, default_shadow: bool = False) -> None:
        self._guards: list[GuardMeta] = []
        self._default_shadow = default_shadow

    def register(
        self,
        guard_id: str,
        phase: str,
        precedence: int,
        regimes: list[str] | None = None,
        description: str = "",
        shadow: bool | None = None,
    ):
        """Decorator to register a guard with metadata."""

        def decorator(func: Callable) -> Callable:
            self._guards.append(
                GuardMeta(
                    guard_id=guard_id,
                    phase=phase,
                    precedence=precedence,
                    regimes=regimes or ["all"],
                    description=description,
                    func=func,
                    shadow=self._default_shadow if shadow is None else shadow,
                )
            )
            return func

        return decorator

    def _applicable(self, meta: GuardMeta, ctx: GuardContext) -> bool:
        """Check if a guard applies to the hand's canonical player regime."""
        return "all" in meta.regimes or ctx.regime in meta.regimes

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
        one broken guard cannot kill the whole cascade, but the failure is
        still recorded as an observable error event (poker_bot.guards.
        telemetry.record_guard_error) -- multi_core's outer run_pre/run_post
        try/except can never see an exception raised inside a single guard's
        func here, so this is the only place that failure can be reported.
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
            except Exception as exc:
                record_guard_error(phase, exc)
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
