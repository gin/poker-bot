"""Exploitative adaptive strategy tuned against the simple baseline."""

from poker_bot.hand_eval import evaluate_hand

ActionDecision = tuple[str | None, int | None, str]

BIG_BLIND = 2
RANK_VALUES = {rank: index for index, rank in enumerate("23456789TJQKA", start=2)}


def card_values(cards):
    return [RANK_VALUES.get(card[0], 0) for card in cards]


def preflop_score(hole_cards):
    if len(hole_cards) != 2:
        return 0

    first, second = card_values(hole_cards)
    high = max(first, second)
    low = min(first, second)
    score = high * 3 + low
    if first == second:
        score += 34 + high * 2
    if hole_cards[0][1] == hole_cards[1][1]:
        score += 8
    if abs(first - second) == 1:
        score += 5
    if high >= 12 and low >= 10:
        score += 12
    return score


def board_texture(board_cards):
    suits = [card[1] for card in board_cards]
    values = sorted(set(card_values(board_cards)))
    max_suit_count = max((suits.count(suit) for suit in set(suits)), default=0)
    connected = any(
        values[index + 2] - values[index] <= 4 for index in range(len(values) - 2)
    )
    paired = len(values) < len(board_cards)
    return {
        "wet": max_suit_count >= 3 or connected,
        "paired": paired,
        "high": any(value >= 12 for value in values),
    }


def made_hand_rank(hole_cards, board_cards):
    if len(board_cards) < 3:
        return 0
    board_rank = evaluate_hand(board_cards) if len(board_cards) >= 5 else (0,)
    full_rank = evaluate_hand(list(hole_cards) + list(board_cards))
    category = full_rank[0]
    if len(board_cards) >= 5 and full_rank == board_rank:
        return 0
    return category


def has_top_pair_or_better(hole_cards, board_cards):
    if not board_cards:
        return False
    board_high = max(card_values(board_cards))
    hole_values = card_values(hole_cards)
    all_values = card_values(list(hole_cards) + list(board_cards))
    return any(
        value == board_high and all_values.count(value) >= 2 for value in hole_values
    )


def top_pair_kicker_value(hole_cards, board_cards):
    if len(hole_cards) != 2 or not board_cards:
        return None

    board_values = card_values(board_cards)
    board_high = max(board_values)
    hole_values = card_values(hole_cards)
    if board_values.count(board_high) != 1 or hole_values.count(board_high) != 1:
        return None
    if board_high not in hole_values:
        return None
    return max(value for value in hole_values if value != board_high)


def has_top_pair_good_kicker(hole_cards, board_cards):
    kicker = top_pair_kicker_value(hole_cards, board_cards)
    return kicker is not None and kicker >= RANK_VALUES["Q"]


def top_pair_defense_price_cap(
    hole_cards,
    board_cards,
    *,
    street="Flop",
    active_opponents=1,
):
    kicker = top_pair_kicker_value(hole_cards, board_cards)
    if kicker is None:
        return 0.24

    cap = 0.30
    if kicker == RANK_VALUES["A"]:
        cap += 0.06
    elif kicker >= RANK_VALUES["Q"]:
        cap += 0.03

    if street == "Flop":
        cap += 0.01
    elif street == "Turn":
        cap += 0.02
    elif street == "River":
        cap -= 0.02

    active_opponents = max(1, int(active_opponents or 1))
    if active_opponents <= 2:
        cap += 0.02
    elif active_opponents >= 4:
        cap -= 0.02

    texture = board_texture(board_cards)
    if texture.get("wet", False):
        cap -= 0.03
    if texture.get("paired", False):
        cap -= 0.04

    return max(0.24, min(cap, 0.40))


def has_overcard_pressure(hole_cards, board_cards):
    if not board_cards:
        return False
    board_high = max(card_values(board_cards))
    return max(card_values(hole_cards), default=0) > board_high


def pot_odds(call_amount, pot):
    if call_amount <= 0:
        return 0.0
    return call_amount / (pot + call_amount)


def capped(amount, allowed):
    return max(0, min(amount, allowed.get("maxCommit", amount)))


def pressure_bet_amount(pot, allowed, texture):
    min_bet = allowed.get("minBet", BIG_BLIND)
    if pot <= 0:
        return capped(max(min_bet, BIG_BLIND), allowed)
    fraction = 0.65 if texture["wet"] else 0.45
    return capped(max(min_bet, int(pot * fraction)), allowed)


def value_bet_amount(pot, allowed, strong=False):
    min_bet = allowed.get("minBet", BIG_BLIND)
    if pot <= 0:
        return capped(max(min_bet, BIG_BLIND), allowed)
    return capped(max(min_bet, int(pot * (0.62 if strong else 0.38))), allowed)


def raise_amount(table, allowed, strong=False):
    minimum = allowed.get("minRaiseTo")
    if minimum is None:
        return None
    pot = table.get("potChips", 0)
    current_bet = table.get("currentBet", 0)
    pressure = int(max(pot, BIG_BLIND) * (0.62 if strong else 0.38))
    return capped(max(minimum, current_bet + pressure), allowed)


def choose_action(table, my_seat) -> ActionDecision:
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if not available:
        return None, None, "No legal actions available"
    if not my_seat:
        if "fold" in available:
            return "fold", None, "Fallback: seat not found"
        return None, None, "No matching seat found"

    street = table.get("street", "Preflop")
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = table.get("potChips", 0)
    call_amount = allowed.get("callAmount", 0)
    required = pot_odds(call_amount, pot)

    if street == "Preflop":
        score = preflop_score(hole_cards)
        if "raise" in available and score >= 78:
            amount = raise_amount(table, allowed, strong=score >= 95)
            return "raise", amount, f"Premium preflop score {score}, raising"
        if "call" in available:
            if score >= 46 or required <= 0.12:
                return "call", call_amount, f"Playable preflop score {score}, calling"
            if "check" in available:
                return "check", None, f"Weak preflop score {score}, checking"
            return "fold", None, f"Weak preflop score {score}, folding"
        if "bet" in available and score >= 62:
            amount = value_bet_amount(pot, allowed, strong=score >= 88)
            return "bet", amount, f"Strong preflop score {score}, betting"
        if "check" in available:
            return "check", None, f"Preflop score {score}, checking"

    made_rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards)
    strong = made_rank >= 2
    medium = made_rank == 1 or top_pair

    if "raise" in available and strong:
        amount = raise_amount(table, allowed, strong=made_rank >= 3)
        return "raise", amount, f"Made hand rank {made_rank}, raising simple"

    if "call" in available:
        if strong or (medium and required <= 0.24) or required <= 0.10:
            return "call", call_amount, f"Defending with rank {made_rank}"
        if "check" in available:
            return "check", None, "Not advantageous, checking"
        return "fold", None, f"Rank {made_rank} below price, folding"

    if "bet" in available:
        if strong:
            amount = value_bet_amount(pot, allowed, strong=made_rank >= 3)
            return "bet", amount, f"Value betting rank {made_rank}"
        if medium:
            amount = value_bet_amount(pot, allowed)
            return "bet", amount, "Thin value against simple"
        if texture["high"] and not texture["wet"] and has_overcard_pressure(
            hole_cards, board_cards
        ):
            amount = pressure_bet_amount(pot, allowed, texture)
            return "bet", amount, "Pressure betting dry high-card board"

    if "check" in available:
        return "check", None, "Not advantageous, checking"
    if "fold" in available:
        return "fold", None, "No profitable action, folding"
    return None, None, "No supported legal action available"
