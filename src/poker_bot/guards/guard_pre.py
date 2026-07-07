"""Pre-decision guards — run BEFORE the core heuristic/neural network proposes
an action. These catch obvious/structural/mathematical certainties that don't
need the core to run at all, saving compute and preventing the core from
wasting a decision on a known spot.

Guards are registered with precedence (lower = fires first = more specific).
"""

from __future__ import annotations

from poker_bot.guards.context import GuardContext
from poker_bot.guards.registry import GuardRail
from poker_bot.hand_utils import (
    pot_odds,
    effective_pot,
    call_amount,
    live_opponent_seats,
    is_board_made_or_kicker_vulnerable,
    royal_flush_possible,
    is_aks,
    card_values,
    rank_counts,
    evaluate_hand,
    board_texture,
    RANK_VALUES,
)
from poker_bot.strategies.adaptive import preflop_score

ActionDecision = tuple[str, int | None, str]
guard_rail = GuardRail()
guard_pre = guard_rail
# ── Thresholds ──────────────────────────────────────────────────────────────
SLIVER_SHOVE_POT_ODDS_FLOOR = 0.10
# preflop_commit_cap: max total preflop commitment (in big blinds) by hand
# tier before we stop feeding a raise war. Premium (score >= 80) is uncapped.
PREFLOP_COMMIT_CAP_MEDIUM_BB = 6
PREFLOP_COMMIT_CAP_AIR_BB = 2

# ══════════════════════════════════════════════════════════════════════════════
# Pre-Guard 0: preflop_commit_cap (Layer 0, multiway only) — SHADOW
# History (guard_audit 2026-07-06): written against 6-max preflop call-wars,
# which turned out to be an artifact of the router bug that sent every local
# decision through the heads-up core (count_active_players saw no 'status'
# field in simulator seats and returned 0). With routing fixed, s4v002 does
# not call-feed wars and leave-one-out benchmarking measured this guard as
# neutral (-0.4 bb/100, within noise). Kept in SHADOW to collect evidence;
# activate only if a pool-wide counterfactual gate shows >= 0 delta.
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "preflop_commit_cap",
    "pre",
    2,
    ["6max"],
    "Preflop commit cap: fold non-premium hands once total commitment "
    "exceeds tier cap (stops call-feeding raise wars)",
    shadow=True,
)
def preflop_commit_cap(ctx: GuardContext) -> ActionDecision | None:
    if ctx.street != "Preflop":
        return None
    if not ctx.facing_bet or ctx.call_price <= 0:
        return None
    if "fold" not in ctx.available_actions:
        return None

    score = preflop_score(ctx.hole_cards)
    if score >= 80:
        return None  # premium: let the core play the war

    blind = max(1, ctx.blind)
    cap_bb = (
        PREFLOP_COMMIT_CAP_MEDIUM_BB if score >= 50 else PREFLOP_COMMIT_CAP_AIR_BB
    )
    committed = int(ctx.my_seat.get("currentBetChips") or 0)
    if committed < cap_bb * blind:
        return None

    return (
        "fold",
        None,
        f"preflop commit cap: {committed} committed >= {cap_bb}bb cap "
        f"(score {score}), folding out of raise war",
    )

# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "spr_commitment_lock",
    "pre",
    0,
    ["hu", "6max"],
    "Pot-committed: call with strong hand (two pair+) when SPR is low",
)
def spr_commitment_lock(ctx: GuardContext) -> ActionDecision | None:
    if ctx.street == "Preflop":
        return None
    if "call" not in ctx.available_actions or ctx.call_price <= 0:
        return None
    if len(ctx.board_cards) < 3:
        return None

    # Only rescue two pair or better.
    if ctx.made_rank < 2:
        return None

    # Cannot call if call exceeds stack.
    if ctx.call_price > ctx.stack:
        return None

    # SPR threshold: tighter in multi-way pots.
    spr_threshold = 1.5 if ctx.num_active_opponents >= 4 else 3.0

    # Calculate SPR after the contemplated call.
    hero_stack_after = max(0, ctx.stack - ctx.call_price)
    opp_stacks_after = [
        max(0, int(s.get("stackChips") or 0))
        for s in live_opponent_seats(ctx.table, ctx.my_seat)
    ]
    eff_stack_after = min(
        [hero_stack_after, *opp_stacks_after]
        if opp_stacks_after
        else [hero_stack_after]
    )
    pot_after_call = ctx.effective_pot + ctx.call_price
    spr = eff_stack_after / max(1, pot_after_call)

    if spr < spr_threshold:
        return (
            "call",
            ctx.call_price,
            f"spr commitment lock: spr {spr:.2f} < {spr_threshold}, calling with strong hand vs {ctx.num_active_opponents} opps",
        )
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Pre-Guard 2: sliver_shove_guard (Layer 0, all sizes)
# Override folds when the call is priced in as a sliver (<= 10% pot odds).
# River only (no future streets, equity realization ~100%).
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "sliver_shove_guard",
    "pre",
    5,
    ["hu", "6max"],
    "River sliver: call when pot odds <= 10% (any two cards have enough equity)",
)
def sliver_shove_guard(ctx: GuardContext) -> ActionDecision | None:
    if ctx.street != "River":
        return None
    if "call" not in ctx.available_actions:
        return None
    if not ctx.facing_bet or ctx.call_price <= 0:
        return None

    required = pot_odds(ctx.call_price, max(ctx.pot, 1))
    if required <= SLIVER_SHOVE_POT_ODDS_FLOOR:
        return ("call", ctx.call_price, f"sliver-shove floor: {required:.1%} pot odds")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Pre-Guard 3: royal_flush_predecision (Layer 1, all sizes)
# Force check/call when royal flush is possible (AKs preflop or postflop).
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "royal_flush_predecision",
    "pre",
    10,
    ["hu", "6max"],
    "Royal flush possible: force check/call only (never fold/raise)",
)
def royal_flush_predecision(ctx: GuardContext) -> ActionDecision | None:
    # Preflop: AKs should never fold or raise — only check or call.
    if ctx.street == "Preflop":
        if not ctx.is_aks:
            return None
        if "check" in ctx.available_actions and ctx.no_one_bet:
            return ("check", None, "royal flush guard: AKs preflop, checking")
        if "call" in ctx.available_actions:
            return ("call", ctx.call_price, "royal flush guard: AKs preflop, calling")
        return None

    # Postflop: check if royal flush is still possible.
    if len(ctx.board_cards) < 3:
        return None
    if not ctx.royal_flush_possible:
        return None

    if "check" in ctx.available_actions and ctx.no_one_bet:
        return ("check", None, "royal flush guard: royal flush possible, checking")
    if "call" in ctx.available_actions:
        return (
            "call",
            ctx.call_price,
            "royal flush guard: royal flush possible, calling",
        )
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Pre-Guard 4: board_made_hand_guard (Layer 1, all sizes)
# Avoid value-raising or stacking off when hero has no real private edge.
# Board-made hands (trips on paired board, board flush, board straight) and
# board-trips-with-kicker-only (Qh Kd on 33385).
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "board_made_hand_guard",
    "pre",
    15,
    ["hu", "6max"],
    "Board-made hand: no private value to raise/bet (check/fold instead)",
)
def board_made_hand_guard(ctx: GuardContext) -> ActionDecision | None:
    if ctx.street == "Preflop":
        return None
    if not ctx.is_board_made_or_kicker:
        return None

    # This is a pre-guard: it fires before the core proposes anything.
    # We can only decide based on whether hero is facing a bet.
    available = ctx.available_actions

    if ctx.facing_bet and ctx.call_price > 0:
        # Facing a bet: fold if the price is expensive (>= 33% pot odds).
        required = pot_odds(ctx.call_price, max(ctx.pot, 1))
        if required >= 0.33 and "fold" in available:
            return (
                "fold",
                None,
                f"board-made hand: folded large bet at {required:.0%} price",
            )
        if "call" in available:
            return ("call", ctx.call_price, "board-made hand: calling cheap bet")
    else:
        # Not facing a bet: check back.
        if "check" in available:
            return ("check", None, "board-made hand: check back shared strength")
    return None
