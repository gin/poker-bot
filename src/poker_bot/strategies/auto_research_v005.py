"""Auto-research candidate v005.

This candidate keeps v004 as the champion baseline and adds paired-board
nut-awareness. Live telemetry showed v004 overvaluing fragile rank-two hands on
paired boards and raise-warring with non-nut full houses on trips boards.
"""

from __future__ import annotations

from collections import Counter

from poker_bot.hand_eval import evaluate_hand
from poker_bot.strategies import auto_research_v004 as champion
from poker_bot.strategies.adaptive import (
    RANK_VALUES,
    board_texture,
    card_values,
    has_top_pair_good_kicker,
    has_top_pair_or_better,
    made_hand_rank,
    pot_odds,
    top_pair_defense_price_cap,
)

ActionDecision = tuple[str | None, int | None, str]


def call_amount(allowed):
    return int(allowed.get("callAmount") or allowed.get("callChips") or 0)


def no_one_has_bet(table, allowed):
    return int(table.get("currentBet") or 0) == 0 and call_amount(allowed) == 0


def board_rank_counts(board_cards):
    return Counter(card_values(board_cards))


def paired_board_ranks(board_cards):
    return {
        rank for rank, count in board_rank_counts(board_cards).items() if count >= 2
    }


def trips_board_ranks(board_cards):
    return {
        rank for rank, count in board_rank_counts(board_cards).items() if count >= 3
    }


def board_has_pair(board_cards):
    return bool(paired_board_ranks(board_cards))


def active_opponents(table, my_seat):
    hero_agent_id = (my_seat or {}).get("agentId")
    hero_seat_number = (my_seat or {}).get("seatNumber")
    opponents = 0
    for seat in table.get("seats", []):
        if seat.get("folded"):
            continue
        if hero_agent_id is not None and seat.get("agentId") == hero_agent_id:
            continue
        if hero_seat_number is not None and seat.get("seatNumber") == hero_seat_number:
            continue
        opponents += 1
    return max(1, opponents)


def hole_pair_rank(hole_cards):
    values = card_values(hole_cards)
    if len(values) == 2 and values[0] == values[1]:
        return values[0]
    return None


def has_overpair_to_board(hole_cards, board_cards):
    pair_rank = hole_pair_rank(hole_cards)
    if pair_rank is None or not board_cards:
        return False
    return pair_rank > max(card_values(board_cards), default=0)


def fragile_rank_two_on_paired_board(hole_cards, board_cards):
    rank = made_hand_rank(hole_cards, board_cards)
    if rank != 2 or not board_has_pair(board_cards):
        return False
    return not has_top_pair_or_better(hole_cards, board_cards)


def non_nut_trips_board_full_house(hole_cards, board_cards):
    if len(board_cards) < 4:
        return False
    trip_ranks = trips_board_ranks(board_cards)
    if not trip_ranks:
        return False

    full_rank = evaluate_hand(list(hole_cards) + list(board_cards))
    if full_rank[0] != 6:
        return False

    triple_rank, pair_rank = full_rank[1], full_rank[2]
    if triple_rank not in trip_ranks:
        return False
    return pair_rank < RANK_VALUES["A"]


def paired_board_pot_control(table, my_seat, base) -> ActionDecision | None:
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None

    fragile_rank_two = fragile_rank_two_on_paired_board(hole_cards, board_cards)
    non_nut_full_house = non_nut_trips_board_full_house(hole_cards, board_cards)
    if not fragile_rank_two and not non_nut_full_house:
        return None

    if action == "bet" and "check" in available and no_one_has_bet(table, allowed):
        return (
            "check",
            None,
            "v005 pot control: fragile paired-board value hand",
        )

    if action == "raise" and "call" in available:
        price = call_amount(allowed)
        pot = int(table.get("potChips") or 0)
        required = pot_odds(price, pot)
        stack = int(my_seat.get("stackChips") or 0)
        texture = board_texture(board_cards)
        if fragile_rank_two and (required > 0.42 or price > max(stack, 1)):
            if "fold" in available:
                return (
                    "fold",
                    None,
                    f"v005 folded fragile paired-board hand at {required:.0%} price",
                )
            return None
        descriptor = "non-nut full house" if non_nut_full_house else "fragile two pair"
        wet_suffix = " wet" if texture.get("wet", False) else ""
        return (
            "call",
            price,
            f"v005 capped paired-board aggression with {descriptor}{wet_suffix}",
        )

    return None


def strong_top_pair_defense(table, my_seat, base) -> ActionDecision | None:
    if table.get("street", "Preflop") == "Preflop":
        return None

    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    price = call_amount(allowed)
    if action != "fold" or "call" not in available or price <= 0:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if len(board_cards) < 3:
        return None
    if made_hand_rank(hole_cards, board_cards) != 1:
        return None
    if not has_top_pair_good_kicker(hole_cards, board_cards):
        return None

    pot = int(table.get("potChips") or 0)
    required = pot_odds(price, pot)
    cap = top_pair_defense_price_cap(
        hole_cards,
        board_cards,
        street=table.get("street", "Flop"),
        active_opponents=active_opponents(table, my_seat),
    )
    if required > cap:
        return None

    return (
        "call",
        price,
        f"v005 strong top-pair defense required {required:.0%} cap {cap:.0%}",
    )


def sixmax_adjustment(table, my_seat, base) -> ActionDecision | None:
    for adjustment in (
        paired_board_pot_control,
        strong_top_pair_defense,
    ):
        decision = adjustment(table, my_seat, base)
        if decision is not None:
            return decision
    return None


def choose_action(table, my_seat) -> ActionDecision:
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if not available:
        return None, None, "No legal actions available"
    if not my_seat:
        if "fold" in available:
            return "fold", None, "Fallback: seat not found"
        return None, None, "No matching seat found"

    base = champion.choose_action(table, my_seat)
    adjusted = sixmax_adjustment(table, my_seat, base)
    if adjusted is not None:
        return adjusted

    action, amount, message = base
    return action, amount, f"5:{message}"
