"""GuardContext for the sandbox."""

from __future__ import annotations

from dataclasses import dataclass

from poker_bot.hand_utils import (
    evaluate_hand, make_hand_rank, board_texture, pot_odds, effective_pot,
    seated_players, active_players, active_opponents, call_amount,
    blind_size, no_one_has_bet, has_top_pair_or_better, fragile_rank_two,
    board_dominated_two_pair, has_flush_draw, has_open_ended_straight_draw,
    has_good_draw, profile_value, profile_vpip_frequency,
    profile_call_frequency, profile_fold_to_bet_frequency,
    profile_aggression_frequency_merged, single_opponent_profile,
    opponent_is_bluffy, is_tight_opponent, card_values,
    is_board_made_or_kicker_vulnerable, royal_flush_possible, is_aks,
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
        
        return cls(
            table=table, my_seat=my_seat,
            hole_cards=hole_cards, board_cards=board_cards,
            hand_rank=(0,), made_rank=0,
            board_texture={"wet": False, "paired": False, "high": False},
            has_top_pair=False, is_fragile_two_pair=False,
            is_board_dominated_two_pair=False, is_board_made=False,
            has_flush_draw=False, has_oesd=False, has_good_draw=False,
            pot=0, effective_pot=0, stack=0, call_price=0,
            facing_bet=False, pot_odds=None, blind=1,
            street="Preflop", num_seated=0, num_active=1,
            num_active_opponents=0, is_heads_up=True, no_one_bet=True,
            available_actions=[], allowed={},
            opponent_profile=None,
        )
