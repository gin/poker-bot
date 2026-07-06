"""GuardContext — pre-computed data object shared by all guards and the core.

Computes hand evaluation, board texture, pot odds, and opponent profile data
ONCE per decision, eliminating redundant computation across 32+ guards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poker_bot.hand_utils import (
    evaluate_hand,
    made_hand_rank,
    board_texture,
    pot_odds,
    effective_pot,
    seated_players,
    active_players,
    active_opponents,
    call_amount,
    blind_size,
    no_one_has_bet,
    has_top_pair_or_better,
    fragile_rank_two,
    board_dominated_two_pair,
    has_flush_draw,
    has_open_ended_straight_draw,
    has_good_draw,
    profile_value,
    profile_vpip_frequency,
    profile_call_frequency,
    profile_fold_to_bet_frequency,
    profile_aggression_frequency_merged,
    single_opponent_profile,
    opponent_is_bluffy,
    is_tight_opponent,
    card_values,
    is_board_made_or_kicker_vulnerable,
    royal_flush_possible,
    is_aks,
)


@dataclass
class GuardContext:
    """Pre-computed decision context. All fields are read-only after build()."""

    table: dict
    my_seat: dict
    hole_cards: list
    board_cards: list
    hand_rank: tuple
    made_rank: int
    board_texture: dict
    has_top_pair: bool
    is_fragile_two_pair: bool
    is_board_dominated_two_pair: bool
    is_board_made: bool
    has_flush_draw: bool
    has_oesd: bool
    has_good_draw: bool
    pot: int
    effective_pot: int
    stack: int
    call_price: int
    facing_bet: bool
    pot_odds: float | None
    blind: int
    street: str
    num_seated: int
    num_active: int
    num_active_opponents: int
    is_heads_up: bool
    no_one_bet: bool
    available_actions: list[str]
    allowed: dict
    opponent_profile: Any = None
    opponent_hands_seen: int = 0
    opponent_vpip: float | None = None
    opponent_call_freq: float | None = None
    opponent_fold_to_bet: float | None = None
    opponent_aggression: float | None = None
    opponent_wasd: float | None = None
    opponent_bluff_pct: float | None = None
    opponent_is_bluffy: bool = False
    opponent_is_tight: bool = False
    opponent_label: str = "unknown"
    # ── Derived hand/board fields (for guard convenience) ──
    pair_high_rank: int | None = None  # hand_rank[1] if two pair
    pair_low_rank: int | None = None  # hand_rank[2] if two pair
    max_board_rank: int = 0  # highest card value on board
    min_effective_stack: int = 0  # min(hero_stack, max_opp_stack) for SPR
    is_board_made_or_kicker: bool = False  # board-made or kicker-only hand
    royal_flush_possible: bool = False  # royal flush still possible
    is_aks: bool = False  # hero holds suited AK

    @classmethod
    def build(cls, table: dict, my_seat: dict) -> "GuardContext":
        """Build a GuardContext from a table state and hero seat."""
        hole_cards = my_seat.get("holeCards", [])
        board_cards = table.get("boardCards", [])
        allowed = table.get("allowedActions") or {}
        available = allowed.get("availableActions", [])

        all_cards = list(hole_cards) + list(board_cards)
        hand_rank = evaluate_hand(all_cards) if len(all_cards) >= 5 else (0,)
        made_rank = made_hand_rank(hole_cards, board_cards)
        texture = (
            board_texture(board_cards)
            if board_cards
            else {"wet": False, "paired": False, "high": False}
        )
        has_top = has_top_pair_or_better(hole_cards, board_cards)
        frag2 = (
            fragile_rank_two(hole_cards, board_cards, made_rank)
            if made_rank == 2
            else False
        )
        bdom = (
            board_dominated_two_pair(hole_cards, board_cards, made_rank)
            if made_rank == 2
            else False
        )
        board_made = made_rank == 0 and len(board_cards) >= 5
        hfd = has_flush_draw(hole_cards, board_cards)
        hoesd = has_open_ended_straight_draw(hole_cards, board_cards)

        pot = int(table.get("potChips") or 0)
        eff_pot = effective_pot(table)
        stack = int(my_seat.get("stackChips") or 0)
        price = call_amount(allowed)
        facing = price > 0
        odds = pot_odds(price, pot) if facing and pot > 0 else None
        blind = blind_size(allowed, table)

        num_seated = seated_players(table)
        num_active = active_players(table)
        num_opp = active_opponents(table, my_seat)
        street = table.get("street", "Preflop")
        no_bet = no_one_has_bet(table, allowed)

        opp_profile = single_opponent_profile(table, min_hands=1)
        opp_hands = 0
        opp_vpip = opp_call = opp_ftb = opp_aggr = opp_wasd = opp_bluff = None
        opp_is_bluffy_val = opp_is_tight_val = False
        opp_label = "unknown"

        if opp_profile is not None:
            opp_hands = int(profile_value(opp_profile, "hands_seen") or 0)
            opp_vpip = profile_vpip_frequency(opp_profile)
            opp_call = profile_call_frequency(opp_profile)
            opp_ftb = profile_fold_to_bet_frequency(opp_profile)
            opp_aggr = profile_aggression_frequency_merged(opp_profile)
            wasd_val = profile_value(opp_profile, "weak_aggressive_showdown_frequency")
            opp_wasd = float(wasd_val) if wasd_val is not None else None
            bluff_val = profile_value(opp_profile, "api_bluff_pct")
            opp_bluff = float(bluff_val) if bluff_val is not None else None
            opp_is_bluffy_val = opponent_is_bluffy(opp_profile)
            opp_is_tight_val = is_tight_opponent(table)
            label_method = getattr(opp_profile, "label", None)
            if callable(label_method):
                try:
                    opp_label = label_method()
                except Exception:
                    pass

        # Derived hand/board fields
        pair_high = hand_rank[1] if hand_rank[0] == 2 else None
        pair_low = hand_rank[2] if hand_rank[0] == 2 else None
        max_board = max(card_values(board_cards)) if board_cards else 0
        my_id = (my_seat or {}).get("agentId")
        max_opp_stack = 0
        for s in table.get("seats", []):
            if s.get("agentId") != my_id:
                max_opp_stack = max(max_opp_stack, int(s.get("stackChips") or 0))
        min_eff_stack = min(stack, max_opp_stack)
        board_made_kicker = is_board_made_or_kicker_vulnerable(hole_cards, board_cards)
        rf_possible = royal_flush_possible(hole_cards, board_cards)
        aks = is_aks(hole_cards)

        return cls(
            table=table,
            my_seat=my_seat,
            hole_cards=hole_cards,
            board_cards=board_cards,
            hand_rank=hand_rank,
            made_rank=made_rank,
            board_texture=texture,
            has_top_pair=has_top,
            is_fragile_two_pair=frag2,
            is_board_dominated_two_pair=bdom,
            is_board_made=board_made,
            has_flush_draw=hfd,
            has_oesd=hoesd,
            has_good_draw=hfd or hoesd,
            pot=pot,
            effective_pot=eff_pot,
            stack=stack,
            call_price=price,
            facing_bet=facing,
            pot_odds=odds,
            blind=blind,
            street=street,
            num_seated=num_seated,
            num_active=num_active,
            num_active_opponents=num_opp,
            is_heads_up=(num_seated < 4),
            no_one_bet=no_bet,
            available_actions=available,
            allowed=allowed,
            opponent_profile=opp_profile,
            opponent_hands_seen=opp_hands,
            opponent_vpip=opp_vpip,
            opponent_call_freq=opp_call,
            opponent_fold_to_bet=opp_ftb,
            opponent_aggression=opp_aggr,
            opponent_wasd=opp_wasd,
            opponent_bluff_pct=opp_bluff,
            opponent_is_bluffy=opp_is_bluffy_val,
            opponent_is_tight=opp_is_tight_val,
            opponent_label=opp_label,
            pair_high_rank=pair_high,
            pair_low_rank=pair_low,
            max_board_rank=max_board,
            min_effective_stack=min_eff_stack,
            is_board_made_or_kicker=board_made_kicker,
            royal_flush_possible=rf_possible,
            is_aks=aks,
        )
