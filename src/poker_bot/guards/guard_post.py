"""Post-decision guards — run AFTER the core heuristic/neural network proposes
an action. Each guard receives a GuardContext (pre-computed data) and the
core's proposed action, and can override it.

Guards are registered with precedence (lower = fires first = more specific).
Table-size filtering prevents HU guards from firing in 6-max and vice versa.

Phase 3: The 6 hu009-hu012 guards (most recently validated).
"""

from __future__ import annotations

from poker_bot.guards.context import GuardContext
from poker_bot.guards.registry import GuardRail
from poker_bot.hand_utils import (
    card_values,
    evaluate_hand,
    board_has_pair,
    board_dominated_two_pair,
    paired_board_ranks,
    pot_odds,
    profile_value,
    profile_vpip_frequency,
    profile_call_frequency,
    profile_fold_to_bet_frequency,
    profile_aggression_frequency_merged,
    single_opponent_profile,
    opponent_is_bluffy,
    is_tight_opponent,
    call_amount,
    no_one_has_bet,
)

ActionDecision = tuple[str, int | None, str]

# ── Thresholds (from hu009-hu012, benchmark-validated) ─────────────────────
_TURN_TWO_PAIR_SUPPRESS_VPIP = 0.30
_TURN_TWO_PAIR_SUPPRESS_FOLD_TO_BET = 0.55
_TURN_TWO_PAIR_SUPPRESS_AGGR = 0.35
_TURN_TWO_PAIR_STATION_CALL = 0.50
_TURN_TWO_PAIR_LOOSE_VPIP = 0.45
_TURN_TWO_PAIR_MIN_SPR = 3.0
_TURN_WEAK_FOLD_MIN_POT_ODDS = 0.25
_FLOP_BLUFFCATCH_MAX_POT_ODDS = 0.25
_FLOP_BLUFFCATCH_MIN_WASD = 0.30

# Default-shadow: pruning sweep 2026-07-06 found every firing guard was a
# net tax on the healthy cores (see artifacts/GUARD_AUDIT_2026-07-06.md).
guard_rail = GuardRail(default_shadow=True)
guard_post = guard_rail


# ══════════════════════════════════════════════════════════════════════════════
# Guard 1: two_pair_paired_board_overfold (Layer 2, HU only)
# From hu009. Don't fold genuine two pair (pocket pair + board pair) on paired
# boards. Excludes board-dominated and fragile (weak pocket pair vs high board).
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "two_pair_paired_board_overfold",
    "post",
    20,
    ["hu"],
    "Don't fold genuine two pair on paired boards (HU only, excludes fragile)",
)
def two_pair_paired_board_overfold(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    action, _amount, _message = proposed
    if action not in ("raise", "bet"):
        return None
    if ctx.street == "Preflop":
        return None
    if len(ctx.board_cards) < 4:
        return None
    if ctx.made_rank != 2:
        return None
    if not ctx.board_texture.get("paired"):
        return None

    hole_values = card_values(ctx.hole_cards)
    is_pocket_pair = hole_values[0] == hole_values[1]
    if not is_pocket_pair:
        return None

    if board_dominated_two_pair(ctx.hole_cards, ctx.board_cards, 2):
        return None

    paired_ranks = paired_board_ranks(ctx.board_cards)
    if paired_ranks:
        board_high = max(paired_ranks)
        pocket_low = min(hole_values)
        if pocket_low < board_high - 7:
            return None

    if "call" not in ctx.available_actions:
        return None
    price = ctx.call_price if ctx.facing_bet else call_amount(ctx.allowed)
    if price <= 0:
        return None
    return (
        "call",
        price,
        "two pair on paired board: call instead of fold (real pocket pair)",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Guard 2: board_assisted_two_pair (Layer 2, all sizes)
# From hu009. Board-assisted two pair on paired boards: suppress raises vs
# tight opponents (the hand is a bluff-catcher at best).
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "board_assisted_two_pair",
    "post",
    25,
    ["hu", "6max"],
    "Board-assisted two pair: suppress raises vs tight opponents",
)
def board_assisted_two_pair(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    action, _amount, _message = proposed
    if ctx.street == "Preflop":
        return None
    if len(ctx.board_cards) < 3:
        return None
    if not ctx.board_texture.get("paired"):
        return None
    if ctx.hand_rank[0] != 2:
        return None

    pair_high = ctx.pair_high_rank
    pair_low = ctx.pair_low_rank
    hole_ranks = {c[0] for c in ctx.hole_cards}
    rank_to_value = {rank: idx for idx, rank in enumerate("23456789TJQKA", start=2)}
    hole_values_set = {rank_to_value.get(r, 0) for r in hole_ranks}

    high_in_hole = pair_high in hole_values_set
    low_in_hole = pair_low in hole_values_set
    if high_in_hole and low_in_hole:
        return None

    is_tight = ctx.opponent_is_tight
    available = ctx.available_actions

    if action in ("raise", "bet") and is_tight:
        if "check" in available and ctx.no_one_bet:
            return ("check", None, "board-assisted two pair: check back vs tight")
        if "call" in available and ctx.facing_bet:
            price = ctx.call_price
            required = pot_odds(price, ctx.pot)
            if required > 0.25 and "fold" in available:
                return ("fold", None, "folded board-assisted two pair vs tight")
            return ("call", price, "board-assisted two pair: call vs tight")

    if action == "call" and is_tight and ctx.facing_bet:
        required = pot_odds(ctx.call_price, ctx.pot)
        if required > 0.25 and "fold" in available:
            return ("fold", None, "folded board-assisted two pair vs tight bet")

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Guard 3: turn_two_pair_bet_suppression (Layer 3, HU only)
# From hu009. Suppress turn two-pair value bet vs tight/passive opponents.
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "turn_two_pair_bet_suppression",
    "post",
    30,
    ["hu"],
    "Turn two-pair: check back vs tight/passive (metric-driven)",
)
def turn_two_pair_bet_suppression(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    action, _amount, _message = proposed
    if ctx.street != "Turn":
        return None
    if action != "bet":
        return None
    if ctx.facing_bet:
        return None
    if "check" not in ctx.available_actions:
        return None
    if ctx.made_rank != 2:
        return None

    if ctx.pot > 0 and ctx.min_effective_stack / ctx.pot < _TURN_TWO_PAIR_MIN_SPR:
        return None

    if ctx.opponent_hands_seen < 10:
        return None

    vpip = ctx.opponent_vpip or 0.0
    call_freq = ctx.opponent_call_freq or 0.0
    fold_to_bet = ctx.opponent_fold_to_bet or 0.0
    aggression = ctx.opponent_aggression or 0.0

    if call_freq >= _TURN_TWO_PAIR_STATION_CALL:
        return None
    if vpip >= _TURN_TWO_PAIR_LOOSE_VPIP:
        return None

    value_owned = vpip <= _TURN_TWO_PAIR_SUPPRESS_VPIP or (
        fold_to_bet >= _TURN_TWO_PAIR_SUPPRESS_FOLD_TO_BET
        and aggression <= _TURN_TWO_PAIR_SUPPRESS_AGGR
    )
    if not value_owned:
        return None

    return ("check", None, "turn two pair: check back vs tight/passive (metric-driven)")


# ══════════════════════════════════════════════════════════════════════════════
# Guard 4: turn_weak_hand_fold_vs_tight_raise (Layer 3, HU only)
# From hu009. Fold weak hands (high card / non-top-pair) vs tight opponent's
# turn raise.
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "turn_weak_hand_fold_vs_tight_raise",
    "post",
    30,
    ["hu"],
    "Fold weak hand (non-top-pair / high card) vs tight turn raise",
)
def turn_weak_hand_fold_vs_tight_raise(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    action, _amount, _message = proposed
    if ctx.street != "Turn":
        return None
    if action not in ("call", "raise"):
        return None
    if not ctx.facing_bet:
        return None
    if "fold" not in ctx.available_actions:
        return None
    if len(ctx.board_cards) < 4:
        return None

    rank = ctx.hand_rank[0]
    if rank >= 2:
        return None
    if rank == 1:
        pair_rank = ctx.hand_rank[1]
        if pair_rank >= ctx.max_board_rank:
            return None

    if (
        ctx.pot > 0
        and ctx.call_price / (ctx.pot + ctx.call_price) < _TURN_WEAK_FOLD_MIN_POT_ODDS
    ):
        return None

    if ctx.stack > 0 and ctx.call_price >= ctx.stack:
        return None

    if ctx.opponent_hands_seen < 10:
        return None
    call_freq = ctx.opponent_call_freq or 0.0
    vpip = ctx.opponent_vpip or 0.0
    if call_freq >= _TURN_TWO_PAIR_STATION_CALL:
        return None
    if vpip >= _TURN_TWO_PAIR_LOOSE_VPIP:
        return None
    fold_to_bet = ctx.opponent_fold_to_bet or 0.0
    aggression = ctx.opponent_aggression or 0.0
    value_owned = vpip <= _TURN_TWO_PAIR_SUPPRESS_VPIP or (
        fold_to_bet >= _TURN_TWO_PAIR_SUPPRESS_FOLD_TO_BET
        and aggression <= _TURN_TWO_PAIR_SUPPRESS_AGGR
    )
    if not value_owned:
        return None

    return (
        "fold",
        None,
        "turn weak hand (non-top pair / high card) vs tight raise: fold",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Guard 5: flop_hu_bluffcatch (Layer 3, HU only)
# From hu010/hu011. Bluff-catch weak hands on dry boards vs bluffy opponents.
# Fires on Flop (rank 0 or 1) and Turn (rank 1 only).
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "flop_hu_bluffcatch",
    "post",
    35,
    ["hu"],
    "Flop/Turn HU bluff-catch vs bluffy opponents on dry boards",
)
def flop_hu_bluffcatch(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    action, _amount, _message = proposed
    if ctx.street not in ("Flop", "Turn"):
        return None
    if action != "fold":
        return None
    if "call" not in ctx.available_actions:
        return None
    if not ctx.facing_bet or ctx.call_price <= 0:
        return None
    if len(ctx.board_cards) < 3:
        return None

    if ctx.board_texture.get("wet") or ctx.board_texture.get("paired"):
        return None

    rank = ctx.made_rank
    if ctx.street == "Flop":
        if rank not in (0, 1):
            return None
    else:
        if rank != 1:
            return None
    if rank == 1:
        pair_rank = ctx.hand_rank[1]
        if pair_rank >= ctx.max_board_rank:
            return None

    if ctx.pot_odds is None or ctx.pot_odds > _FLOP_BLUFFCATCH_MAX_POT_ODDS:
        return None

    if ctx.stack > 0 and ctx.call_price > ctx.stack * 0.12:
        return None

    if ctx.opponent_hands_seen < 15:
        return None
    if not ctx.opponent_is_bluffy:
        return None

    return (
        "call",
        ctx.call_price,
        f"{ctx.street.lower()} HU bluff-catch rank {rank} vs bluffy opponent (dry board)",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Guard 6: river_two_pair_facing_bet_call (Layer 3, HU only)
# From hu012. Call (don't fold) river two pair facing a bet on paired boards
# at <= 40% pot odds. Excludes fragile two pair.
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "river_two_pair_facing_bet_call",
    "post",
    35,
    ["hu"],
    "River two-pair: call instead of fold on paired boards (non-fragile)",
)
def river_two_pair_facing_bet_call(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    action, _amount, _message = proposed
    if ctx.street != "River":
        return None
    if action != "fold":
        return None
    if "call" not in ctx.available_actions:
        return None
    if not ctx.facing_bet or ctx.call_price <= 0:
        return None

    if ctx.hand_rank[0] != 2:
        return None

    if ctx.is_fragile_two_pair:
        return None

    if not ctx.board_texture.get("paired"):
        return None

    if ctx.pot_odds is None or ctx.pot_odds > 0.40:
        return None

    return (
        "call",
        ctx.call_price,
        "river two pair: call instead of fold on paired board",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Guard 7: rank_two_facing_bet_guard (Layer 2, all sizes)
# Convert rank-2 (two pair) raises to calls when facing a bet (flop/turn).
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "rank_two_facing_bet",
    "post",
    22,
    ["hu", "6max"],
    "Two pair facing bet: call instead of raise (flop/turn)",
)
def rank_two_facing_bet(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    action, _amount, _message = proposed
    if ctx.street not in ("Flop", "Turn"):
        return None
    if action not in ("raise", "bet"):
        return None
    if not ctx.facing_bet:
        return None
    if ctx.made_rank != 2:
        return None

    # When raise_count >= 3, let postflop_marginal_hand_war_cap handle it.
    history = ctx.table.get("actionHistory") or ctx.table.get("action_history") or []
    my_id = (ctx.my_seat or {}).get("agentId")
    raise_count = sum(
        1
        for h in history
        if h.get("agentId") == my_id
        and h.get("action") in ("raise", "bet")
        and h.get("street") == ctx.street
    )
    if raise_count >= 3:
        return None

    if "call" in ctx.available_actions:
        return (
            "call",
            ctx.call_price,
            f"{ctx.street.lower()} rank-2 facing bet: call instead of raise",
        )
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Guard 8: board_dominated_trips_guard (Layer 2, all sizes)
# Suppress value raises when trips are fully on the board (flop only).
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "board_dominated_trips",
    "post",
    23,
    ["hu", "6max"],
    "Board-dominated trips: suppress value raise, check back (flop)",
)
def board_dominated_trips(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    action, _amount, _message = proposed
    if action != "raise":
        return None
    if ctx.street != "Flop":
        return None
    if len(ctx.board_cards) < 3:
        return None
    if not ctx.board_texture.get("paired"):
        return None
    if ctx.hand_rank[0] != 3:
        return None

    trips_rank_value = ctx.hand_rank[1]
    hole_values = set(card_values(ctx.hole_cards))
    if trips_rank_value in hole_values:
        return None  # real trips, allow raise

    if "check" in ctx.available_actions and ctx.no_one_bet:
        return (
            "check",
            None,
            "board-dominated trips: suppress value raise, check back",
        )
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Guard 9: river_one_pair_over_call (Layer 2, all sizes)
# Fold one pair on river/turn vs >30% pot bet.
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "river_one_pair_over_call",
    "post",
    28,
    ["hu", "6max"],
    "One pair on river/turn: fold vs >30% pot bet",
)
def river_one_pair_over_call(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    action, _amount, _message = proposed
    if ctx.street not in ("River", "Turn"):
        return None
    if action != "call":
        return None
    if ctx.made_rank != 1:
        return None
    if ctx.call_price <= 0 or ctx.pot <= 0:
        return None

    if ctx.call_price / ctx.pot > 0.30:
        return (
            "fold",
            None,
            f"{ctx.street.lower()} one pair: fold vs {ctx.call_price / ctx.pot:.0%} pot bet",
        )
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Guard 10: vulnerable_flush_guard (Layer 2, all sizes)
# Non-nut flush on paired board: suppress raises/bets (reverse implied odds).
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "vulnerable_flush",
    "post",
    27,
    ["hu", "6max"],
    "Non-nut flush on paired board: suppress raises (reverse implied odds)",
)
def vulnerable_flush(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    from poker_bot.hand_utils import vulnerable_non_nut_flush_on_paired_board

    action, _amount, _message = proposed
    if ctx.street == "Preflop":
        return None
    if action == "fold":
        return None
    if not vulnerable_non_nut_flush_on_paired_board(ctx.hole_cards, ctx.board_cards):
        return None

    available = ctx.available_actions
    if action == "raise":
        if int(ctx.table.get("facing_bet") or 0) == 1:
            if "call" in available:
                return (
                    "call",
                    ctx.call_price,
                    "non-nut flush on paired board: bluff catch",
                )
        if "fold" in available:
            return ("fold", None, "non-nut flush on paired board: folded value-raise")
    if action == "bet" and "check" in available:
        return ("check", None, "non-nut flush on paired board: check back")
    if action == "call" and ctx.call_price > 0:
        required = pot_odds(ctx.call_price, max(ctx.pot, 1))
        if required >= 0.33 and "fold" in available:
            return (
                "fold",
                None,
                f"non-nut flush on paired board: folded large bet at {required:.0%} price",
            )
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Guard 11: paired_board_pot_control (Layer 2, all sizes)
# Fragile two pair or non-nut full house on paired board: pot control.
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "paired_board_pot_control",
    "post",
    26,
    ["hu", "6max"],
    "Fragile two pair / non-nut full house on paired board: pot control",
)
def paired_board_pot_control(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    from poker_bot.hand_utils import (
        fragile_rank_two_on_paired_board,
        non_nut_trips_board_full_house,
    )

    action, _amount, _message = proposed
    if ctx.street == "Preflop":
        return None
    if len(ctx.board_cards) < 3:
        return None

    frag2 = fragile_rank_two_on_paired_board(ctx.hole_cards, ctx.board_cards)
    non_nut_fh = non_nut_trips_board_full_house(ctx.hole_cards, ctx.board_cards)
    if not frag2 and not non_nut_fh:
        return None

    available = ctx.available_actions

    if action == "bet" and "check" in available and ctx.no_one_bet:
        return ("check", None, "pot control: fragile paired-board value hand")

    if action == "raise" and "call" in available:
        required = pot_odds(ctx.call_price, max(ctx.pot, 1))
        stack = ctx.stack
        if board_dominated_two_pair(ctx.hole_cards, ctx.board_cards, 2):
            if "fold" in available:
                return ("fold", None, "folded board-dominated two pair on paired board")
            return None
        if frag2 and (required > 0.35 or ctx.call_price > max(stack, 1)):
            if "fold" in available:
                return (
                    "fold",
                    None,
                    f"folded fragile paired-board hand at {required:.0%} price",
                )
            return None
        if (
            frag2
            and ctx.board_texture.get("high")
            and ctx.board_texture.get("paired")
            and required > 0.25
        ):
            if "fold" in available:
                return (
                    "fold",
                    None,
                    f"folded vulnerable two pair on A-high paired board at {required:.0%}",
                )
            return None
        descriptor = "non-nut full house" if non_nut_fh else "fragile two pair"
        wet_suffix = " wet" if ctx.board_texture.get("wet", False) else ""
        return (
            "call",
            ctx.call_price,
            f"capped paired-board aggression with {descriptor}{wet_suffix}",
        )

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Guard 12: preflop_min_raise_war_cap (Layer 4, all sizes)
# Cap preflop min-raise wars after 3 raise-backs.
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "preflop_min_raise_war_cap",
    "post",
    40,
    ["hu", "6max"],
    "Preflop war cap: call/check after 3 raise-backs",
)
def preflop_min_raise_war_cap(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    action, _amount, _message = proposed
    if ctx.street != "Preflop":
        return None
    if action not in ("raise", "bet"):
        return None

    history = ctx.table.get("actionHistory") or ctx.table.get("action_history") or []
    my_id = (ctx.my_seat or {}).get("agentId")
    raise_count = sum(
        1
        for h in history
        if h.get("agentId") == my_id
        and h.get("action") in ("raise", "bet")
        and h.get("street") == "Preflop"
    )
    if raise_count < 3:
        return None

    available = ctx.available_actions
    if action == "raise" and "call" in available:
        return ("call", ctx.call_price, "preflop war cap: call after 3 raise-backs")
    if action == "bet" and "check" in available:
        return ("check", None, "preflop war cap: check after 3 raise-backs")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Guard 13: postflop_marginal_hand_war_cap (Layer 4, all sizes)
# Cap postflop raises with marginal hands after 3+ raises this street.
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "postflop_marginal_hand_war_cap",
    "post",
    40,
    ["hu", "6max"],
    "Postflop war cap: call/check marginal hands after 3 raises",
)
def postflop_marginal_hand_war_cap(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    action, _amount, _message = proposed
    if ctx.street == "Preflop":
        return None
    if action not in ("raise", "bet"):
        return None

    history = ctx.table.get("actionHistory") or ctx.table.get("action_history") or []
    my_id = (ctx.my_seat or {}).get("agentId")
    raise_count = sum(
        1
        for h in history
        if h.get("agentId") == my_id
        and h.get("action") in ("raise", "bet")
        and h.get("street") == ctx.street
    )
    if raise_count < 3:
        return None
    if len(ctx.board_cards) < 3:
        return None
    if ctx.made_rank >= 3:
        return None  # strong hand — allow raise

    available = ctx.available_actions
    if action == "raise":
        if "call" in available:
            if ctx.made_rank == 2:
                required = pot_odds(ctx.call_price, ctx.pot)
                if required > 0.33:
                    return (
                        "fold",
                        None,
                        f"{ctx.street.lower()} marginal war cap: fold rank 2 after {raise_count} raises (price {required:.0%})",
                    )
            return (
                "call",
                ctx.call_price,
                f"{ctx.street.lower()} marginal war cap: call after {raise_count} raises (rank {ctx.made_rank})",
            )
    if action == "bet" and "check" in available:
        return (
            "check",
            None,
            f"{ctx.street.lower()} marginal war cap: check after {raise_count} bets (rank {ctx.made_rank})",
        )
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Guard 14: small_pair_multiway_fold_guard (Layer 2, all sizes)
# Fold small pocket pairs (<= 77) preflop in multi-way pots at bad stack prices.
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "small_pair_multiway_fold",
    "post",
    45,
    ["hu", "6max"],
    "Small pair multiway: fold below-77 pair at >5% stack in 3+ way pots",
)
def small_pair_multiway_fold(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    action, _amount, _message = proposed
    if ctx.street != "Preflop":
        return None
    raise_seat = ctx.table.get("raiseSeatNumber")
    hero_seat_num = (ctx.my_seat or {}).get("seatNumber")
    if raise_seat is None or raise_seat == hero_seat_num:
        return None
    if action not in ("fold", "raise", "call"):
        return None
    if "fold" not in ctx.available_actions:
        return None
    if ctx.num_active < 3:
        return None

    from poker_bot.hand_utils import hole_pair_rank, RANK_VALUES

    rank = hole_pair_rank(ctx.hole_cards)
    if rank is None or rank > RANK_VALUES["7"]:
        return None

    call = ctx.call_price
    if call <= 0 or ctx.stack <= 0:
        return None
    price_to_stack = call / ctx.stack
    if price_to_stack <= 0.05:
        return None

    return (
        "fold",
        None,
        f"small-pair multiway guard: {rank}{ctx.hole_cards[0][1]}{ctx.hole_cards[1][1]} folded at {price_to_stack:.1%} stack in {ctx.num_active}-way pot",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Guard 15: medium_pair_paired_board_fold_guard (Layer 2, all sizes)
# Fold 77 on paired boards (crushed against any reasonable range).
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "medium_pair_paired_board_fold",
    "post",
    45,
    ["hu", "6max"],
    "77 on paired board: fold (crushed against any reasonable range)",
)
def medium_pair_paired_board_fold(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    action, _amount, _message = proposed
    if ctx.street == "Preflop":
        return None
    if action not in ("call", "raise"):
        return None
    if "fold" not in ctx.available_actions:
        return None

    from poker_bot.hand_utils import hole_pair_rank, RANK_VALUES, evaluate_hand

    rank = hole_pair_rank(ctx.hole_cards)
    if rank != RANK_VALUES["7"]:
        return None
    if len(ctx.board_cards) < 3 or not ctx.board_texture.get("paired"):
        return None
    if ctx.call_price <= 0:
        return None

    # If 77 makes a full house (board ranks include 7), allow play.
    full_rank = evaluate_hand(list(ctx.hole_cards) + list(ctx.board_cards))
    if full_rank[0] >= 5:
        return None

    return ("fold", None, "77 on paired board: crushed against any reasonable range")


# ══════════════════════════════════════════════════════════════════════════════
# Guard 16: medium_hand_multiway_fold_guard (Layer 2, all sizes)
# Fold medium-strength hands (one pair) postflop in 3+ player pots at bad prices.
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "medium_hand_multiway_fold",
    "post",
    45,
    ["hu", "6max"],
    "Medium hand multiway: fold one pair in 3+ way pots at >20% price",
)
def medium_hand_multiway_fold(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    from poker_bot.hand_utils import has_overpair_to_board

    action, _amount, _message = proposed
    if ctx.street == "Preflop":
        return None
    if action not in ("call", "raise"):
        return None
    if "fold" not in ctx.available_actions:
        return None
    if ctx.num_active < 3:
        return None
    if len(ctx.board_cards) < 3:
        return None
    if ctx.made_rank != 1:
        return None
    if has_overpair_to_board(ctx.hole_cards, ctx.board_cards):
        return None
    if ctx.call_price <= 0:
        return None

    required = pot_odds(ctx.call_price, max(ctx.pot, 1))
    if required <= 0.20:
        return None

    return (
        "fold",
        None,
        f"medium-hand multiway guard: 1-pair folded at {required:.0%} price in {ctx.num_active}-way pot",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Guard 17: cheap_postflop_continue (Layer 4, all sizes)
# Call cheap postflop bets (<=16% pot odds, <=8% stack) when hand has equity.
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "cheap_postflop_continue",
    "post",
    48,
    ["hu", "6max"],
    "Cheap postflop continue: call cheap bets with equity",
)
def cheap_postflop_continue(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    action, _amount, _message = proposed
    if ctx.street == "Preflop":
        return None
    if action != "fold":
        return None
    if "call" not in ctx.available_actions or ctx.call_price <= 0:
        return None

    required = pot_odds(ctx.call_price, max(ctx.pot, 1))
    if required > 0.16 or ctx.call_price > max(ctx.blind, int(ctx.stack * 0.08)):
        return None

    if len(ctx.board_cards) < 3:
        return None
    rank = ctx.made_rank
    draw = ctx.has_good_draw

    if rank >= 2 and not ctx.is_fragile_two_pair:
        return ("call", ctx.call_price, f"cheap continue made rank {rank}")
    if ctx.num_active_opponents <= 3 and (ctx.has_top_pair or rank == 1):
        return ("call", ctx.call_price, f"cheap bluff catch rank {rank}")
    if draw and not ctx.board_texture.get("paired", False) and required <= 0.12:
        return ("call", ctx.call_price, "cheap draw continue")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Guard 18: postflop_draw_continue (Layer 4, all sizes)
# Call postflop bets with flush/straight draws at favorable pot odds.
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "postflop_draw_continue",
    "post",
    48,
    ["hu", "6max"],
    "Draw continue: call flush/OESD draws at favorable odds",
)
def postflop_draw_continue(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    action, _amount, _message = proposed
    if ctx.street == "Preflop":
        return None
    if action != "fold":
        return None
    if "call" not in ctx.available_actions:
        return None
    if ctx.call_price <= 0 or len(ctx.board_cards) < 3:
        return None

    if not ctx.has_flush_draw and not ctx.has_oesd:
        return None
    if ctx.street not in ("Flop", "Turn"):
        return None

    if ctx.street == "Flop":
        cap = 0.30 if ctx.num_active_opponents <= 2 else 0.22
    else:
        cap = 0.20 if ctx.num_active_opponents <= 2 else 0.15

    required = pot_odds(ctx.call_price, max(ctx.effective_pot, 1))
    if required > cap:
        return None
    if ctx.stack > 0 and ctx.call_price > ctx.stack * 0.20:
        return None

    draw_label = "+".join(
        label
        for label, present in (("FD", ctx.has_flush_draw), ("OESD", ctx.has_oesd))
        if present
    )
    return (
        "call",
        ctx.call_price,
        f"draw continue {draw_label} street {ctx.street} opp {ctx.num_active_opponents} required {required:.0%} cap {cap:.0%}",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Guard 19: excessive_bet_size_cap (Layer 4, all sizes)
# Prevent absurdly large raises (> 3x pot).
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "excessive_bet_size_cap",
    "post",
    50,
    ["hu", "6max"],
    "Prevent absurdly large raises (> 3x pot)",
)
def excessive_bet_size_cap(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    action, _amount, _message = proposed
    if action != "raise":
        return None

    raise_range = ctx.allowed.get("raiseRange") or {}
    raise_amt = raise_range.get("min", 0) or 0
    if raise_amt > ctx.pot * 3 and ctx.pot > 0:
        if "call" in ctx.available_actions:
            return (
                "call",
                ctx.call_price,
                f"Min raise {raise_amt} > 3x pot {ctx.pot} — cap to call",
            )
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Guard 20: trips_on_paired_board_cap (Layer 4, all sizes)
# Prevent massive overbets with trips on paired board (> 1.5x pot).
# ══════════════════════════════════════════════════════════════════════════════


@guard_rail.register(
    "trips_on_paired_board_cap",
    "post",
    50,
    ["hu", "6max"],
    "Cap massive overbets (>1.5x pot) with trips on paired board",
)
def trips_on_paired_board_cap(
    ctx: GuardContext, proposed: ActionDecision
) -> ActionDecision | None:
    action, _amount, _message = proposed
    if action != "raise":
        return None
    if len(ctx.board_cards) < 3:
        return None
    if ctx.hand_rank[0] != 3:
        return None
    if not ctx.board_texture.get("paired"):
        return None

    raise_range = ctx.allowed.get("raiseRange") or {}
    raise_amt = raise_range.get("min", 0) or 0
    if raise_amt > ctx.pot * 1.5 and ctx.pot > 0:
        if "call" in ctx.available_actions:
            return (
                "call",
                ctx.call_price,
                f"Trips on paired board: raise {raise_amt} > 1.5x pot {ctx.pot} — cap to call",
            )
    return None
