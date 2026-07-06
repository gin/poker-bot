"""GuardContext for the sandbox."""

from __future__ import annotations

from dataclasses import dataclass

from poker_bot.hand_utils import (
    evaluate_hand,
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
    """Guard context for the sandbox."""

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
    opponent_profile: any = None

    @classmethod
    def build(cls, table: dict, my_seat: dict) -> "GuardContext":
        """Build GuardContext."""
        hole_cards = my_seat.get("holeCards", [])
        board_cards = table.get("boardCards", [])
        allowed = table.get("allowedActions") or {}
        combined_cards = list(hole_cards) + list(board_cards)
        hand_rank = evaluate_hand(combined_cards) if len(combined_cards) >= 5 else (0,)
        made_rank = int(hand_rank[0]) if hand_rank else 0
        pot = int(table.get("potChips") or 0)
        eff_pot = effective_pot(table)
        stack = int(my_seat.get("stackChips") or my_seat.get("stack") or 0)
        price = call_amount(allowed)
        profile = single_opponent_profile(table, min_hands=1)

        return cls(
            table=table,
            my_seat=my_seat,
            hole_cards=hole_cards,
            board_cards=board_cards,
            hand_rank=hand_rank,
            made_rank=made_rank,
            board_texture=board_texture(board_cards),
            has_top_pair=has_top_pair_or_better(hole_cards, board_cards),
            is_fragile_two_pair=fragile_rank_two(hole_cards, board_cards, made_rank),
            is_board_dominated_two_pair=board_dominated_two_pair(
                hole_cards, board_cards, made_rank
            ),
            is_board_made=is_board_made_or_kicker_vulnerable(hole_cards, board_cards),
            has_flush_draw=has_flush_draw(hole_cards, board_cards),
            has_oesd=has_open_ended_straight_draw(hole_cards, board_cards),
            has_good_draw=has_good_draw(hole_cards, board_cards),
            pot=pot,
            effective_pot=eff_pot,
            stack=stack,
            call_price=price,
            facing_bet=price > 0,
            pot_odds=pot_odds(price, eff_pot) if price > 0 else 0.0,
            blind=blind_size(allowed, table),
            street=str(table.get("street") or "Preflop"),
            num_seated=seated_players(table),
            num_active=active_players(table),
            num_active_opponents=active_opponents(table, my_seat),
            is_heads_up=active_players(table) <= 2,
            no_one_bet=no_one_has_bet(table, allowed),
            available_actions=list(allowed.get("availableActions") or []),
            allowed=allowed,
            opponent_profile=profile,
        )
