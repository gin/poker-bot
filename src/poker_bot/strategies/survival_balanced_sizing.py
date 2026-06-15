"""Balanced survival strategy with profile-aware thresholds and mixed sizing."""

from __future__ import annotations

import hashlib

from poker_bot.strategies import survival_balanced, survival_lookahead, survival_lookup
from poker_bot.strategies.adaptive import (
    BIG_BLIND,
    board_texture,
    capped,
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
    return survival_balanced.position_bucket(table, my_seat)


def profile_adjusted_thresholds(table, my_seat):
    raise_threshold, call_threshold = survival_balanced.preflop_thresholds(
        table, my_seat
    )
    style = survival_lookup.table_style(table, my_seat)

    if style == "tight":
        raise_threshold -= 3
        call_threshold -= 1
    elif style == "loose_passive":
        raise_threshold += 2
        call_threshold += 4
    elif style == "loose_aggressive":
        raise_threshold += 3
        call_threshold += 6
    elif style == "short_handed":
        raise_threshold -= 2
        call_threshold -= 1

    return max(38, raise_threshold), max(36, call_threshold), style


def mix_roll(table, my_seat, salt):
    parts = [
        str(table.get("street", "")),
        ",".join(my_seat.get("holeCards", [])),
        ",".join(table.get("boardCards", [])),
        str(table.get("potChips", 0)),
        str(table.get("currentBet", 0)),
        str(my_seat.get("seatNumber", "")),
        salt,
    ]
    digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=2).digest()
    return int.from_bytes(digest, "big") / 65535


def mixed_fraction(table, my_seat, options, salt):
    roll = mix_roll(table, my_seat, salt)
    cumulative = 0.0
    for fraction, weight in options:
        cumulative += weight
        if roll <= cumulative:
            return fraction
    return options[-1][0]


def amount_from_fraction(table, allowed, fraction, action="bet"):
    pot = int(table.get("potChips") or 0)
    current_bet = int(table.get("currentBet") or 0)
    if action == "raise":
        minimum = allowed.get("minRaiseTo")
        if minimum is None:
            return None
        target = max(int(minimum), current_bet + int(max(pot, BIG_BLIND) * fraction))
        return capped(target, allowed)

    minimum = int(allowed.get("minBet") or BIG_BLIND)
    target = max(minimum, int(max(pot, BIG_BLIND) * fraction))
    return capped(target, allowed)


def sizing_fraction(table, my_seat, style, purpose, strong=False):
    texture = (
        board_texture(table.get("boardCards", [])) if table.get("boardCards") else {}
    )

    if purpose == "preflop_raise":
        if strong:
            return mixed_fraction(table, my_seat, [(0.34, 0.65), (0.48, 0.35)], purpose)
        return mixed_fraction(table, my_seat, [(0.20, 0.70), (0.30, 0.30)], purpose)

    if purpose == "value":
        if style == "loose_passive":
            options = [(0.48, 0.65), (0.62, 0.35)]
        elif texture.get("wet", False):
            options = [(0.52, 0.60), (0.66, 0.40)]
        elif style == "tight":
            options = [(0.34, 0.70), (0.48, 0.30)]
        else:
            options = [(0.42, 0.65), (0.58, 0.35)]
        return mixed_fraction(table, my_seat, options, purpose)

    if purpose == "thin_value":
        if style == "loose_passive":
            options = [(0.42, 0.65), (0.55, 0.35)]
        else:
            options = [(0.30, 0.70), (0.42, 0.30)]
        return mixed_fraction(table, my_seat, options, purpose)

    options = (
        [(0.28, 0.70), (0.40, 0.30)] if not texture.get("wet", False) else [(0.18, 1.0)]
    )
    return mixed_fraction(table, my_seat, options, purpose)


def effective_preflop_pot(table):
    return int(table.get("potChips") or 0) + sum(
        int(seat.get("currentBetChips") or 0) for seat in table.get("seats", [])
    )


def balanced_sizing_raise_amount(table, my_seat, allowed, score, style):
    fraction = sizing_fraction(
        table,
        my_seat,
        style,
        "preflop_raise",
        strong=score >= 88 or style == "loose_passive",
    )
    return amount_from_fraction(table, allowed, fraction, action="raise")


def balanced_sizing_bet_amount(table, my_seat, allowed, style, purpose, strong=False):
    fraction = sizing_fraction(table, my_seat, style, purpose, strong=strong)
    return amount_from_fraction(table, allowed, fraction)


def balanced_sizing_preflop_action(table, my_seat):
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    hole_cards = my_seat.get("holeCards", [])
    score = preflop_score(hole_cards)
    call_amount = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    required = pot_odds(call_amount, effective_preflop_pot(table))
    raise_threshold, call_threshold, style = profile_adjusted_thresholds(table, my_seat)
    blind_size = max(int(allowed.get("minBet") or 0), BIG_BLIND)
    facing_raise = (
        int(table.get("currentBet") or 0) > blind_size or call_amount > blind_size
    )

    if "raise" in available and score >= raise_threshold:
        amount = balanced_sizing_raise_amount(table, my_seat, allowed, score, style)
        return "raise", amount, f"sizing {style} value/open raise score {score}"

    if "call" in available:
        if facing_raise:
            if score >= raise_threshold + 2 or (
                score >= call_threshold + 8 and required <= 0.13
            ):
                return "call", call_amount, f"sizing {style} defend score {score}"
        elif score >= call_threshold and required <= 0.34:
            return "call", call_amount, f"sizing {style} selective call score {score}"

        if "check" in available:
            return "check", None, f"sizing {style} preflop check score {score}"
        return "fold", None, f"sizing {style} preflop fold score {score}"

    if "bet" in available and score >= call_threshold + 6:
        amount = balanced_sizing_bet_amount(
            table,
            my_seat,
            allowed,
            style,
            "value",
            strong=score >= raise_threshold,
        )
        return "bet", amount, f"sizing {style} preflop bet score {score}"

    if "check" in available:
        return "check", None, f"sizing {style} preflop check score {score}"
    if "fold" in available:
        return "fold", None, f"sizing {style} preflop fold score {score}"
    return None


def covered_by_large_stack(table, my_seat):
    my_stack = int(my_seat.get("stackChips") or 0)
    return any(
        seat.get("agentId") != my_seat.get("agentId")
        and not seat.get("folded", False)
        and not seat.get("hasFolded", False)
        and int(seat.get("stackChips") or 0) >= my_stack + BIG_BLIND * 8
        for seat in table.get("seats", [])
    )


def stack_pressure_defense(table, my_seat, blueprint, style):
    action, _amount, _message = blueprint
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if "call" not in available:
        return None
    if not covered_by_large_stack(table, my_seat) and style != "loose_aggressive":
        return None

    call_amount = int(allowed.get("callAmount") or allowed.get("callChips") or 0)
    required = pot_odds(call_amount, int(table.get("potChips") or 0))
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)

    if action == "raise" and rank == 2 and required <= 0.32:
        return "call", call_amount, "sizing stack-pressure pot control"
    if action == "fold" and (rank >= 2 or (top_pair and required <= 0.23)):
        return "call", call_amount, "sizing stack-pressure defense"
    return None


def balanced_sizing_postflop_adjustment(table, my_seat, blueprint, style):
    pressure = stack_pressure_defense(table, my_seat, blueprint, style)
    if pressure is not None:
        return pressure

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

    if rank >= 2:
        amount = balanced_sizing_bet_amount(
            table,
            my_seat,
            allowed,
            style,
            "value",
            strong=rank >= 3,
        )
        return "bet", amount, f"sizing {style} value pressure rank {rank}"

    if top_pair:
        amount = balanced_sizing_bet_amount(
            table,
            my_seat,
            allowed,
            style,
            "thin_value",
        )
        return "bet", amount, f"sizing {style} top-pair pressure"

    if (
        active <= 2
        and not texture.get("wet", False)
        and preflop_score(hole_cards) >= 70
    ):
        amount = balanced_sizing_bet_amount(
            table,
            my_seat,
            allowed,
            style,
            "bluff",
        )
        return "bet", amount, f"sizing {style} dry-board continuation"

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
        preflop = balanced_sizing_preflop_action(table, my_seat)
        if preflop is not None:
            return preflop

    style = survival_lookup.table_style(table, my_seat)
    blueprint = survival_lookahead.choose_action(table, my_seat)
    postflop = balanced_sizing_postflop_adjustment(table, my_seat, blueprint, style)
    if postflop is not None:
        return postflop

    action, amount, message = blueprint
    return action, amount, f"sizing blueprint: {message}"
