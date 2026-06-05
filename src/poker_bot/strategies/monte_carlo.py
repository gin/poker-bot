"""Monte Carlo equity strategy for no-limit heads-up Texas Hold'em."""

import hashlib
import random
import time

from poker_bot.hand_eval import DECK, compare_hands, evaluate_hand

type ActionDecision = tuple[str | None, int | None, str]

MAX_EQUITY_SECONDS = 10.0
DEFAULT_SIMULATIONS = 400
BIG_BLIND = 50


def stable_seed(cards):
    raw = "|".join(sorted(cards)).encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def quick_preflop_estimate(hole_cards):
    if len(hole_cards) != 2:
        return 0.5

    rank_values = {rank: index for index, rank in enumerate("23456789TJQKA", start=2)}
    first = rank_values.get(hole_cards[0][0], 0)
    second = rank_values.get(hole_cards[1][0], 0)
    high = max(first, second)
    low = min(first, second)
    suited_bonus = 0.04 if hole_cards[0][1] == hole_cards[1][1] else 0
    pair_bonus = 0.18 if first == second else 0
    connector_bonus = 0.03 if abs(first - second) <= 1 else 0
    broadway_bonus = 0.04 if high >= 12 and low >= 10 else 0
    estimate = 0.22 + high / 35 + low / 75
    return min(
        0.88,
        estimate + suited_bonus + pair_bonus + connector_bonus + broadway_bonus,
    )


def estimate_equity(hole_cards, board_cards, max_simulations=DEFAULT_SIMULATIONS):
    known_cards = list(hole_cards) + list(board_cards)
    if len(hole_cards) != 2 or len(set(known_cards)) != len(known_cards):
        return 0.0

    rng = random.Random(stable_seed(known_cards))
    wins = 0.0
    simulations = 0
    deadline = time.perf_counter() + MAX_EQUITY_SECONDS

    while simulations < max_simulations and time.perf_counter() < deadline:
        deck = [card for card in DECK if card not in known_cards]
        rng.shuffle(deck)
        opponent = [deck.pop(), deck.pop()]
        completed_board = list(board_cards)
        while len(completed_board) < 5:
            completed_board.append(deck.pop())

        our_score = evaluate_hand(list(hole_cards) + completed_board)
        opponent_score = evaluate_hand(opponent + completed_board)
        result = compare_hands(our_score, opponent_score)
        if result > 0:
            wins += 1
        elif result == 0:
            wins += 0.5
        simulations += 1

    if simulations == 0:
        return quick_preflop_estimate(hole_cards)
    return wins / simulations


def pot_odds(call_amount, pot):
    if call_amount <= 0:
        return 0.0
    return call_amount / (pot + call_amount)


def capped_amount(amount, allowed):
    max_commit = allowed.get("maxCommit", amount)
    return max(0, min(amount, max_commit))


def bet_size(equity, pot, allowed):
    min_bet = allowed.get("minBet", BIG_BLIND)
    max_commit = allowed.get("maxCommit", min_bet)
    if pot <= 0:
        return capped_amount(max(min_bet, BIG_BLIND), allowed)

    stack_to_pot = max_commit / pot
    if equity >= 0.74 and stack_to_pot <= 1.5:
        return max_commit
    if equity >= 0.70:
        return capped_amount(max(min_bet, int(pot * 0.75)), allowed)
    return capped_amount(max(min_bet, int(pot * 0.50)), allowed)


def raise_size(equity, table, allowed):
    min_raise_to = allowed.get("minRaiseTo")
    if min_raise_to is None:
        return None

    pot = table.get("potChips", 0)
    current_bet = table.get("currentBet", 0)
    max_commit = allowed.get("maxCommit", min_raise_to)
    if pot <= 0:
        return capped_amount(min_raise_to, allowed)

    stack_to_pot = max_commit / pot
    if equity >= 0.72 and stack_to_pot <= 1.5:
        return max_commit

    pressure = int(pot * (0.75 if equity >= 0.68 else 0.50))
    return capped_amount(max(min_raise_to, current_bet + pressure), allowed)


def choose_action(table, my_seat) -> ActionDecision:
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if not available:
        return None, None, "No legal actions available"
    if not my_seat:
        if "fold" in available:
            return "fold", None, "Fallback: seat not found"
        return None, None, "No matching seat found"

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = table.get("potChips", 0)
    call_amount = allowed.get("callAmount", 0)
    equity = estimate_equity(hole_cards, board_cards)
    required_equity = pot_odds(call_amount, pot)

    if "raise" in available and equity >= max(0.62, required_equity + 0.18):
        amount = raise_size(equity, table, allowed)
        if amount is not None:
            return "raise", amount, f"Equity {equity:.2f}, raising for value"

    if "call" in available:
        if equity >= required_equity + 0.03:
            return "call", call_amount, f"Equity {equity:.2f} beats pot odds"
        if "check" in available:
            return "check", None, f"Equity {equity:.2f}, checking"
        return "fold", None, f"Equity {equity:.2f} below pot odds"

    if "bet" in available and equity >= 0.58:
        amount = bet_size(equity, pot, allowed)
        return "bet", amount, f"Equity {equity:.2f}, betting"

    if "check" in available:
        return "check", None, f"Equity {equity:.2f}, checking"

    if "fold" in available:
        return "fold", None, f"Equity {equity:.2f}, folding"

    return None, None, "No supported legal action available"
