"""Balanced survival strategy with controlled postflop pressure + wider preflop defense.

Same base as survival_balanced_postflop_pressure but fixes the preflop raise-defense
logic: the old `required <= 0.14` condition was mathematically impossible to meet in
most situations, making the bot fold nearly every raise even with playable hands.
This version relaxes defense to defend more correctly against min-raises and small
raises when getting reasonable pot odds.
"""

from __future__ import annotations

from poker_bot.strategies import survival_lookahead, survival_lookup
from poker_bot.strategies.adaptive import (
    BIG_BLIND,
    board_texture,
    capped,
    card_values,
    has_top_pair_or_better,
    made_hand_rank,
    pot_odds,
    preflop_score,
)

ActionDecision = tuple[str | None, int | None, str]


def active_seat_numbers(table):
    return [
        int(seat.get("seatNumber"))
        for seat in table.get("seats", [])
        if not seat.get("folded", False)
        and not seat.get("hasFolded", False)
        and seat.get("seatNumber") is not None
    ]


def position_bucket(table, my_seat):
    seats = active_seat_numbers(table)
    player_count = max(2, len(seats))
    button = int(table.get("buttonSeatNumber") or seats[0] if seats else 1)
    seat_number = int(my_seat.get("seatNumber") or button)
    offset = (seat_number - button) % player_count

    if player_count <= 3:
        return "short"
    if offset == 0 or offset == player_count - 1:
        return "late"
    if offset in {1, 2}:
        return "blind"
    if offset == 3:
        return "early"
    return "middle"


def preflop_thresholds(table, my_seat):
    active = survival_lookup.active_players(table)
    position = position_bucket(table, my_seat)
    if active <= 3 or position == "short":
        return 56, 46
    if position == "late":
        return 62, 49
    if position == "middle":
        return 68, 52
    if position == "blind":
        return 70, 50
    return 74, 56


def balanced_raise_amount(table, allowed, score):
    minimum = allowed.get("minRaiseTo")
    if minimum is None:
        return None
    if score < 74:
        return int(minimum)
    pot = int(table.get("potChips") or 0)
    current_bet = int(table.get("currentBet") or 0)
    fraction = 0.45 if score >= 90 else 0.25
    target = max(int(minimum), current_bet + int(max(pot, BIG_BLIND) * fraction))
    return capped(target, allowed)


def balanced_bet_amount(table, allowed, strong=False):
    minimum = int(allowed.get("minBet") or BIG_BLIND)
    pot = int(table.get("potChips") or 0)
    fraction = 0.50 if strong else 0.34
    return capped(max(minimum, int(max(pot, BIG_BLIND) * fraction)), allowed)


def probe_bet_amount(table, allowed, active):
    minimum = int(allowed.get("minBet") or BIG_BLIND)
    pot = int(table.get("potChips") or 0)
    fraction = 0.32 if active >= 4 else 0.38
    return capped(max(minimum, int(max(pot, BIG_BLIND) * fraction)), allowed)


def balanced_preflop_action(table, my_seat):
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    hole_cards = my_seat.get("holeCards", [])
    score = preflop_score(hole_cards)
    pot = int(table.get("potChips") or 0) + sum(
        int(seat.get("currentBetChips") or 0) for seat in table.get("seats", [])
    )
    call_amount = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    required = pot_odds(call_amount, pot)
    raise_threshold, call_threshold = preflop_thresholds(table, my_seat)
    blind_size = max(int(allowed.get("minBet") or 0), BIG_BLIND)
    facing_raise = (
        int(table.get("currentBet") or 0) > blind_size or call_amount > blind_size
    )

    if "raise" in available and score >= raise_threshold:
        amount = balanced_raise_amount(table, allowed, score)
        return "raise", amount, f"balanced value/open raise score {score}"

    if "call" in available:
        if facing_raise:
            if score >= call_threshold + 6 or (
                score >= call_threshold + 2 and required <= 0.30
            ):
                return "call", call_amount, f"balanced defend score {score}"
        elif score >= call_threshold and required <= 0.38:
            return "call", call_amount, f"balanced selective call score {score}"

        if "check" in available:
            return "check", None, f"balanced preflop check score {score}"
        return "fold", None, f"balanced preflop fold score {score}"

    if "bet" in available and score >= call_threshold + 6:
        amount = balanced_bet_amount(table, allowed, strong=score >= raise_threshold)
        return "bet", amount, f"balanced preflop bet score {score}"

    if "check" in available:
        return "check", None, f"balanced preflop check score {score}"
    if "fold" in available:
        return "fold", None, f"balanced preflop fold score {score}"
    return None


def has_flush_draw(hole_cards, board_cards):
    if len(board_cards) not in {3, 4}:
        return False
    suits = [card[1] for card in list(hole_cards) + list(board_cards)]
    hole_suits = {card[1] for card in hole_cards}
    return any(suits.count(suit) >= 4 and suit in hole_suits for suit in hole_suits)


def has_open_ended_straight_draw(hole_cards, board_cards):
    if len(board_cards) not in {3, 4}:
        return False
    values = set(card_values(list(hole_cards) + list(board_cards)))
    if 14 in values:
        values.add(1)
    for low in range(1, 11):
        window = set(range(low, low + 4))
        if window.issubset(values):
            return True
    return False


def has_good_draw(hole_cards, board_cards):
    return has_flush_draw(hole_cards, board_cards) or has_open_ended_straight_draw(
        hole_cards, board_cards
    )


def covered_by_larger_stack(table, my_seat):
    my_stack = int(my_seat.get("stackChips") or 0)
    for seat in table.get("seats", []):
        if seat.get("agentId") == my_seat.get("agentId"):
            continue
        if seat.get("folded", False) or seat.get("hasFolded", False):
            continue
        if int(seat.get("stackChips") or 0) > my_stack:
            return True
    return False


def has_preflop_advantage(table, my_seat):
    score = preflop_score(my_seat.get("holeCards", []))
    position = position_bucket(table, my_seat)
    if position in {"late", "short"}:
        return score >= 62
    if position == "middle":
        return score >= 68
    return score >= 72


def no_one_has_bet(allowed, table):
    call_amount = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    return call_amount == 0 and int(table.get("currentBet") or 0) == 0


def balanced_postflop_adjustment(table, my_seat, blueprint):
    action, _amount, _message = blueprint
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if action != "check" or "bet" not in available:
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards) if board_cards else {"wet": False}
    active = survival_lookup.active_players(table)

    if rank >= 2 or top_pair:
        amount = balanced_bet_amount(table, allowed, strong=rank >= 3)
        return "bet", amount, f"balanced value pressure rank {rank}"

    if (
        active <= 2
        and not texture.get("wet", False)
        and preflop_score(hole_cards) >= 70
    ):
        amount = balanced_bet_amount(table, allowed)
        return "bet", amount, "balanced dry-board continuation"

    if (
        no_one_has_bet(allowed, table)
        and not texture.get("wet", False)
        and active <= 4
        and (rank == 1 or has_preflop_advantage(table, my_seat))
    ):
        amount = probe_bet_amount(table, allowed, active)
        return "bet", amount, f"postflop pressure dry-board probe rank {rank}"

    return None


def anti_bully_defense(table, my_seat, blueprint):
    action, _amount, _message = blueprint
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if action != "fold" or "call" not in available:
        return None
    if not covered_by_larger_stack(table, my_seat):
        return None

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    if not board_cards:
        return None

    pot = int(table.get("potChips") or 0)
    call_amount = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    stack = int(my_seat.get("stackChips") or 0)
    required = pot_odds(call_amount, pot)
    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    medium = rank == 1 or top_pair
    draw = has_good_draw(hole_cards, board_cards)
    cheap_stack_price = stack <= 0 or call_amount / stack <= 0.18

    if medium and required <= 0.26 and cheap_stack_price:
        return "call", call_amount, f"anti-bully cheap medium defense rank {rank}"
    if draw and required <= 0.20 and cheap_stack_price:
        return "call", call_amount, "anti-bully cheap draw defense"
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

    if table.get("street", "Preflop") == "Preflop":
        preflop = balanced_preflop_action(table, my_seat)
        if preflop is not None:
            return preflop

    blueprint = survival_lookahead.choose_action(table, my_seat)
    defense = anti_bully_defense(table, my_seat, blueprint)
    if defense is not None:
        return defense

    postflop = balanced_postflop_adjustment(table, my_seat, blueprint)
    if postflop is not None:
        return postflop

    action, amount, message = blueprint
    return action, amount, f"balanced postflop-pressure blueprint: {message}"
