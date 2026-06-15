"""Royal-flush protected strategy with opponent-style adaptation."""

from poker_bot.analysis.opponent_tendencies import summarize_tendencies
from poker_bot.analysis.table_context import (
    active_opponents,
    call_amount,
    pot_size,
    table_agent_stats,
    table_profiles,
)
from poker_bot.strategies import royal_flush
from poker_bot.strategies.adaptive import (
    BIG_BLIND,
    board_texture,
    has_top_pair_or_better,
    made_hand_rank,
    pot_odds,
    preflop_score,
)

ActionDecision = tuple[str | None, int | None, str]


def _has_strong_value(hole_cards, board_cards, opponents):
    if not board_cards:
        return False
    rank = made_hand_rank(hole_cards, board_cards)
    return rank >= (2 if opponents <= 2 else 3)


def _pressure_raise(table, allowed, score):
    amount = royal_flush.balanced_raise_amount(table, allowed, score)
    if amount is not None:
        return amount
    raise_range = allowed.get("raiseRange") or {}
    return allowed.get("minRaiseTo") or raise_range.get("min")


def _adaptive_preflop(table, my_seat, base, tendencies, opponents):
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    hole_cards = my_seat.get("holeCards", [])
    score = preflop_score(hole_cards)
    price = call_amount(allowed)
    required = pot_odds(price, pot_size(table))
    facing_raise = price > BIG_BLIND or int(table.get("currentBet") or 0) > BIG_BLIND

    if (
        "raise" in available
        and not facing_raise
        and tendencies.all_patient
        and score >= (58 if opponents <= 2 else 66)
    ):
        amount = _pressure_raise(table, allowed, score)
        return "raise", amount, f"royal adaptive steal vs patient table score {score}"

    if "raise" in available and tendencies.has_calling_station and score >= 84:
        amount = _pressure_raise(table, allowed, score)
        return "raise", amount, f"royal adaptive value raise vs callers score {score}"

    action, _amount, _message = base
    if (
        action == "fold"
        and "call" in available
        and tendencies.has_bluffer
        and opponents <= 2
        and score >= 58
        and required <= 0.22
    ):
        return "call", price, f"royal adaptive defend vs bluffer score {score}"

    return None


def _adaptive_postflop(table, my_seat, base, tendencies, opponents):
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = pot_size(table)
    price = call_amount(allowed)
    required = pot_odds(price, pot)
    rank = made_hand_rank(hole_cards, board_cards) if board_cards else 0
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    showdown_value = rank >= 1 or top_pair
    strong_value = _has_strong_value(hole_cards, board_cards, opponents)
    draw = royal_flush.has_good_draw(hole_cards, board_cards)
    action, _amount, _message = base

    if tendencies.has_calling_station:
        if "raise" in available and strong_value:
            amount = royal_flush.postflop_raise_amount(
                table,
                allowed,
                strong=rank >= 4,
            )
            return "raise", amount, f"royal adaptive value raise vs station rank {rank}"
        if "bet" in available and (strong_value or top_pair):
            amount = royal_flush.balanced_bet_amount(
                table,
                allowed,
                strong=strong_value,
            )
            return "bet", amount, f"royal adaptive value bet vs station rank {rank}"
        if action in {"bet", "raise"} and not showdown_value and "check" in available:
            return "check", None, "royal adaptive avoid bluffing calling station"

    if (
        action == "fold"
        and "call" in available
        and tendencies.has_bluffer
        and (showdown_value or draw)
        and required <= (0.32 if opponents <= 2 else 0.22)
    ):
        return "call", price, f"royal adaptive bluff-catch rank {rank}"

    texture = board_texture(board_cards) if board_cards else {"wet": False}
    if (
        "bet" in available
        and tendencies.all_patient
        and opponents <= 3
        and not texture.get("wet", False)
        and not showdown_value
        and action == "check"
    ):
        amount = royal_flush.probe_bet_amount(table, allowed, opponents + 1)
        return "bet", amount, "royal adaptive dry-board pressure vs patient table"

    if (
        opponents >= 3
        and action in {"bet", "raise"}
        and not strong_value
        and "check" in available
    ):
        return "check", None, "royal adaptive multiway pot control"

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

    royal_override = royal_flush.royal_flush_override(table, my_seat)
    if royal_override is not None:
        return royal_override

    base = royal_flush.choose_action(table, my_seat)
    profiles = table_profiles(table, my_seat)
    agent_stats = table_agent_stats(table, my_seat)
    tendencies = summarize_tendencies(profiles, agent_stats)
    opponents = active_opponents(table, my_seat)
    if tendencies.confidence < 0.20:
        action, amount, message = base
        return action, amount, f"royal adaptive low-confidence base: {message}"

    if table.get("street", "Preflop") == "Preflop":
        adapted = _adaptive_preflop(table, my_seat, base, tendencies, opponents)
    else:
        adapted = _adaptive_postflop(table, my_seat, base, tendencies, opponents)

    if adapted is not None:
        return adapted

    action, amount, message = base
    label_summary = ",".join(tendencies.labels[:3]) or "unknown"
    return action, amount, f"royal adaptive base vs {label_summary}: {message}"
