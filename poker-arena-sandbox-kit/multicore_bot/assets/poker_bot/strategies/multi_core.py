"""
Multi-core strategy: selects core logic based on players dealt into the hand.

- 2 dealt-in players -> hubase.py (heads-up aggressive)
- 3+ dealt-in players -> s4v002.py (multi-way conservative)

Routing on DEALT-IN players (live + folded-this-hand), not current actives:
handing a 6-max hand to the HU core mid-hand once opponents fold cost
~26 bb/100 in the 2026-07-06 decomposition (the cores plan whole hands, and
hubase's aggression is tuned for true heads-up tables). Dealt-in count keeps
one core per hand AND still routes tournament tables that are truly
short-handed (busted/sitting-out players don't count) to the HU core.

Guard system wraps both paths uniformly (rails are default-shadow).
"""

import os
from typing import Any

from poker_bot.guards.context import GuardContext
from poker_bot.guards.guard_post import guard_post
from poker_bot.guards.guard_pre import guard_pre
from poker_bot.guards.telemetry import clear_events
from poker_bot.hand_utils import profile_value, seat_is_live
from poker_bot.strategies.hubase import choose_action as hubase_choose_action
from poker_bot.strategies.s5base import choose_action as non_hu_choose_action


def count_active_players(table: dict[str, Any]) -> int:
    """Count players still contesting the hand.

    Arena seats carry status ('Active' vs Folded/Waiting/SittingOut, etc.);
    the local simulator's seats carry no status at all, only 'folded' flags.
    Counting status=='Active' alone returns 0 in the simulator, which used
    to route EVERY local decision through the heads-up core regardless of
    table size. seat_is_live handles both schemas.
    """
    return sum(
        1
        for s in table.get("seats", [])
        if s.get("agentId") and seat_is_live(s)
    )


def count_dealt_in_players(table: dict[str, Any]) -> int:
    """Count players dealt into this hand: still live, or folded this hand.

    Excludes seats that never took part (Busted/SittingOut/Waiting in the
    arena; the simulator deals in every seat). This is the routing count:
    it is stable for the whole hand, unlike the active count.
    """
    dealt = 0
    for s in table.get("seats", []):
        if not s.get("agentId"):
            continue
        if seat_is_live(s) or (
            s.get("folded", False)
            or s.get("hasFolded", False)
            or s.get("status") == "Folded"
        ):
            dealt += 1
    return dealt


def _extract_opponent_profile_msg(
    table: dict[str, Any], my_seat: dict[str, Any]
) -> str:
    """Extract opponent profile stats for HU telemetry."""
    profiles = table.get("opponentProfiles")
    if not profiles:
        return "[no profile] "

    my_id = my_seat.get("agentId", "hero")
    opp_profile = None

    for seat in table.get("seats", []):
        if seat.get("agentId") == my_id:
            continue
        if seat.get("status") == "Folded":
            continue
        opp_id = seat.get("agentId")
        if opp_id in profiles:
            opp_profile = profiles[opp_id]
            break

    if opp_profile is None:
        return "[no profile] "

    try:
        # Frequencies, not raw counters: formatting the vpip COUNTER with
        # :.0% is what produced "VPIP: 383500%" in live telemetry.
        # profile_value handles both OpponentProfile objects and the
        # sandbox tracker's dict profiles.
        vpip = float(profile_value(opp_profile, "vpip_frequency") or 0.0)
        pfr = float(profile_value(opp_profile, "pfr_frequency") or 0.0)
        f2b = float(profile_value(opp_profile, "fold_to_bet_frequency") or 0.0)

        return f"[VPIP: {vpip:.0%} | PFR: {pfr:.0%} | FoldToBet: {f2b:.0%}] "
    except Exception:
        return "[no profile] "


def _lookup_table_style(table: dict[str, Any], my_seat: dict[str, Any]) -> str:
    """Lookup table style for multi-way telemetry."""
    player_count = count_active_players(table)
    if player_count <= 3:
        return "short_handed"
    if player_count <= 6:
        return "full_ring"
    return "large_table"


def choose_action(
    table: dict[str, Any], my_seat: dict[str, Any]
) -> tuple[str, float, str]:
    """
    Multi-core action selection.

    1. Build GuardContext (pre-computed data)
    2. Run pre-guards (before core logic)
    3. Select core based on active player count
    4. Enrich message with diagnostic info
    5. Run post-guards (after core logic)
    """

    # Buffer only ever holds the current decision's guard events; the
    # selfplay observer drains it right after each action.
    clear_events()

    # Step 1: Build GuardContext
    try:
        ctx = GuardContext.build(table, my_seat)
    except Exception:
        ctx = None

    # Step 2: Run pre-guards
    if ctx is not None:
        try:
            pre_result = guard_pre.run_pre(ctx)
            if pre_result is not None:
                (pre_action, pre_amount, pre_msg), guard_id = pre_result
                return (
                    pre_action,
                    float(pre_amount or 0),
                    f"{pre_msg} [guard:{guard_id}]",
                )
        except Exception:
            pass

    # Step 3: Select core logic. Default: dealt-in players (stable for the
    # whole hand — no mid-hand core hand-offs). MULTI_CORE_ROUTER=active
    # restores the old per-decision active-count routing for experiments.
    if os.environ.get("MULTI_CORE_ROUTER") == "active":
        active_players = count_active_players(table)
    else:
        active_players = count_dealt_in_players(table)

    if active_players <= 2:
        # Heads-up: use aggressive hubase logic
        core_action, core_amount, core_msg = hubase_choose_action(table, my_seat)
        # Step 4a: Enrich with opponent profile for HU
        profile_prefix = _extract_opponent_profile_msg(table, my_seat)
        if profile_prefix:
            core_msg = profile_prefix + core_msg
    else:
        # Multi-way: use conservative s4v002 logic
        core_action, core_amount, core_msg = non_hu_choose_action(table, my_seat)
        # Step 4b: Enrich with table_style for multi-way
        style = _lookup_table_style(table, my_seat)
        core_msg = f"[{style}] {core_msg}"

    # Step 5: Run post-guards
    if ctx is not None:
        try:
            post_result = guard_post.run_post(ctx, (core_action, core_amount, core_msg))
            if post_result[1] != "approved":
                (post_action, post_amount, post_msg), guard_id = post_result
                return (
                    post_action,
                    float(post_amount or core_amount or 0),
                    f"{post_msg} [guard:{guard_id}]",
                )
        except Exception:
            pass

    # Pass the core's amount through untouched: coercing None -> 0.0 made
    # the simulator resolve default-amount bets/raises differently than the
    # bare core (persistent -0.2 bb/100 vs bare s4v002 across seeds).
    return core_action, core_amount, core_msg
