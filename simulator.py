#!/usr/bin/env python3
"""Terminal poker simulator for playing heads-up against the poker bot."""

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from poker_bot.hand_eval import (  # noqa: E402
    build_deck,
    deal_cards,
    evaluate_hand,
)
from poker_bot.strategies.loader import load_strategy  # noqa: E402

DEFAULT_STRATEGY = "simple"
choose_action = load_strategy(DEFAULT_STRATEGY)

SMALL_BLIND = 25
BIG_BLIND = 50
INITIAL_STACK = 2000
PLAYER_AGENT_ID = "player-agent"
BOT_AGENT_ID = "bot-agent"


def format_cards(cards):
    return " ".join(cards)


def build_allowed_actions(seat, current_bet, min_raise=BIG_BLIND, can_raise=True):
    call_shortfall = max(0, current_bet - seat["currentBetChips"])
    call_amount = min(call_shortfall, seat["stackChips"])
    max_commit = seat["currentBetChips"] + seat["stackChips"]
    if seat["stackChips"] == 0:
        available = []
    elif call_shortfall == 0:
        available = ["fold", "check"]
        if seat["stackChips"] > 0:
            available.append("bet")
    else:
        available = ["fold", "call"]
        min_raise_to = current_bet + min_raise
        if can_raise and max_commit >= min_raise_to:
            available.append("raise")

    min_raise_to = None
    if "raise" in available:
        min_raise_to = current_bet + min_raise

    return {
        "availableActions": available,
        "callAmount": call_amount,
        "minRaiseTo": min_raise_to,
        "minBet": min(BIG_BLIND, seat["stackChips"]),
        "maxCommit": max_commit,
    }


def collect_live_bets(pot, seats):
    if seats[0]["currentBetChips"] != seats[1]["currentBetChips"]:
        high = max(seats, key=lambda seat: seat["currentBetChips"])
        low = min(seats, key=lambda seat: seat["currentBetChips"])
        uncalled = high["currentBetChips"] - low["currentBetChips"]
        high["stackChips"] += uncalled
        high["currentBetChips"] -= uncalled

    pot += seats[0]["currentBetChips"] + seats[1]["currentBetChips"]
    seats[0]["currentBetChips"] = 0
    seats[1]["currentBetChips"] = 0
    return pot


def all_in_betting_is_closed(seats, current_bet):
    if not any(seat["stackChips"] == 0 for seat in seats):
        return False
    return not any(
        seat["stackChips"] > 0 and seat["currentBetChips"] < current_bet
        for seat in seats
    )


def format_money(amount):
    return f"${amount}"


def prompt_user_action(allowed, call_amount):
    print("Available actions:")
    for index, action in enumerate(allowed, start=1):
        if action == "call":
            print(f"  {index}. call {format_money(call_amount)}")
        else:
            print(f"  {index}. {action}")
    while True:
        choice = input("Choose action: ").strip().lower()
        if choice.isdigit() and 1 <= int(choice) <= len(allowed):
            return allowed[int(choice) - 1]
        if choice in allowed:
            return choice
        print("Invalid action, please select a valid option.")


def prompt_player_action(allowed):
    while True:
        action = prompt_user_action(allowed["availableActions"], allowed["callAmount"])
        amount = None
        if action not in ("bet", "raise"):
            return action, amount

        while True:
            action_label = "raise to" if action == "raise" else "bet"
            minimum = allowed["minRaiseTo"] if action == "raise" else allowed["minBet"]
            prompt = (
                f"Enter amount to {action_label} "
                f"(minimum {format_money(minimum)}. "
                "0 to go back and choose action again): "
            )
            entry = input(prompt).strip()
            if entry == "0":
                break
            if not entry:
                return action, amount
            if entry.isdigit() and int(entry) >= minimum:
                amount = int(entry)
                return action, amount
            print("Please enter a valid numeric amount.")


def resolve_action(seat, action, amount, current_bet, min_raise_to=None):
    if action == "fold":
        return current_bet

    if action == "call":
        if amount is None:
            amount = max(0, current_bet - seat["currentBetChips"])
        pay = min(amount, seat["stackChips"])
        seat["stackChips"] -= pay
        seat["currentBetChips"] += pay
        return current_bet

    if action == "check":
        return current_bet

    target = amount
    if target is None:
        target = current_bet + BIG_BLIND
    if action == "bet" and target < BIG_BLIND:
        target = min(BIG_BLIND, seat["currentBetChips"] + seat["stackChips"])
    if action == "raise":
        if min_raise_to is None:
            min_raise_to = current_bet + BIG_BLIND
        target = max(target, min_raise_to)

    if target < seat["currentBetChips"]:
        target = seat["currentBetChips"]

    delta = target - seat["currentBetChips"]
    delta = min(delta, seat["stackChips"])
    seat["stackChips"] -= delta
    seat["currentBetChips"] += delta
    return max(current_bet, seat["currentBetChips"])


def build_table(seats, board, pot, current_bet, street, acting_seat):
    return {
        "street": street,
        "seats": seats,
        "boardCards": board,
        "potChips": pot,
        "currentBet": current_bet,
        "actingSeatNumber": acting_seat,
    }


def print_table(seats, board, pot, street, current_bet, active_id):
    print("\n" + "=" * 60)
    print(f"Street: {street}")
    print(f"Board: {format_cards(board) if board else '(empty)'}")
    print(f"Pot: {format_money(pot)} | Current bet: {format_money(current_bet)}")
    for seat in seats:
        label = "YOU" if seat["agentId"] == PLAYER_AGENT_ID else "BOT"
        cards = (
            "??"
            if seat["agentId"] != PLAYER_AGENT_ID
            else format_cards(seat["holeCards"])
        )
        line = (
            f"{label:<3} stacks={format_money(seat['stackChips'])} "
            f"bet={format_money(seat['currentBetChips'])} "
            f"cards={cards}"
        )
        if seat["agentId"] == active_id:
            line += "  <-- to act"
        print(line)
    print("=" * 60)


def run_betting_round(
    seats,
    board,
    pot,
    current_bet,
    street,
    first_actor_idx,
    action_providers=None,
    verbose=True,
):
    action_providers = action_providers or {}
    active_idx = first_actor_idx
    min_raise = BIG_BLIND
    last_actions = {PLAYER_AGENT_ID: None, BOT_AGENT_ID: None}
    while True:
        seat = seats[active_idx]
        opponent = seats[1 - active_idx]
        if seat["stackChips"] == 0:
            pot = collect_live_bets(pot, seats)
            return None, pot

        allowed = build_allowed_actions(
            seat,
            current_bet,
            min_raise,
            can_raise=opponent["stackChips"] > 0,
        )
        table = build_table(seats, board, pot, current_bet, street, seat["seatNumber"])
        table["allowedActions"] = allowed

        if verbose:
            print_table(seats, board, pot, street, current_bet, seat["agentId"])

        provider = action_providers.get(seat["agentId"])
        if provider is not None:
            action, amount, message = provider(table, seat)
            if verbose:
                label = "YOU" if seat["agentId"] == PLAYER_AGENT_ID else "BOT"
                print(
                    f"{label}: {action.upper()}"
                    + (f" {format_money(amount)}" if amount else "")
                    + f" | {message}"
                )
        elif seat["agentId"] == PLAYER_AGENT_ID:
            action, amount = prompt_player_action(allowed)
        else:
            action, amount, message = choose_action(table, seat)
            if verbose:
                print(
                    f"BOT: {action.upper()}"
                    + (f" {format_money(amount)}" if amount else "")
                    + f" | {message}"
                )

        if action == "fold":
            if verbose:
                print(f"{seat['agentId']} folds.")
            return opponent["agentId"], pot + seat["currentBetChips"] + opponent[
                "currentBetChips"
            ]

        previous_current_bet = current_bet
        current_bet = resolve_action(
            seat, action, amount, current_bet, allowed["minRaiseTo"]
        )
        last_actions[seat["agentId"]] = action
        if action in ("bet", "raise") and current_bet > previous_current_bet:
            min_raise = current_bet - previous_current_bet

        if all_in_betting_is_closed(seats, current_bet):
            break

        # End the betting round when all active players have matched the
        # current bet and each has acted (covers bet -> call sequence).
        agents = [s["agentId"] for s in seats]
        if (
            seats[0]["currentBetChips"] == seats[1]["currentBetChips"] == current_bet
            and last_actions[agents[0]] is not None
            and last_actions[agents[1]] is not None
        ):
            break

        if action in ("bet", "raise"):
            # Opponent must respond to a new bet/raise
            last_actions[opponent["agentId"]] = None

        active_idx = 1 - active_idx

    pot = collect_live_bets(pot, seats)
    return None, pot


def showdown(seats, board):
    best_score = None
    ties = []
    for seat in seats:
        hand_value = evaluate_hand(seat["holeCards"] + board)
        if best_score is None or hand_value > best_score:
            best_score = hand_value
            ties = [seat]
        elif hand_value == best_score:
            ties.append(seat)

    return ties, best_score


def post_blind(stack, blind):
    posted = min(stack, blind)
    return stack - posted, posted


def play_hand(
    player_stack,
    bot_stack,
    player_is_small_blind=True,
    player_strategy=None,
    bot_strategy=None,
    rng=None,
    verbose=True,
):
    deck = build_deck(rng)
    board = []
    player_blind = SMALL_BLIND if player_is_small_blind else BIG_BLIND
    bot_blind = BIG_BLIND if player_is_small_blind else SMALL_BLIND
    player_remaining, player_current_bet = post_blind(player_stack, player_blind)
    bot_remaining, bot_current_bet = post_blind(bot_stack, bot_blind)
    player = {
        "agentId": PLAYER_AGENT_ID,
        "seatNumber": 1,
        "holeCards": deal_cards(deck),
        "stackChips": player_remaining,
        "currentBetChips": player_current_bet,
    }
    bot = {
        "agentId": BOT_AGENT_ID,
        "seatNumber": 2,
        "holeCards": deal_cards(deck),
        "stackChips": bot_remaining,
        "currentBetChips": bot_current_bet,
    }
    pot = 0
    current_bet = max(player_current_bet, bot_current_bet)
    street = "Preflop"
    preflop_first_actor_idx = 0 if player_is_small_blind else 1
    postflop_first_actor_idx = 1 - preflop_first_actor_idx
    action_providers = {}
    if player_strategy is not None:
        action_providers[PLAYER_AGENT_ID] = player_strategy
    if bot_strategy is not None:
        action_providers[BOT_AGENT_ID] = bot_strategy

    fold_winner, pot = run_betting_round(
        [player, bot],
        board,
        pot,
        current_bet,
        street,
        first_actor_idx=preflop_first_actor_idx,
        action_providers=action_providers,
        verbose=verbose,
    )
    if fold_winner:
        if fold_winner == PLAYER_AGENT_ID:
            return player["stackChips"] + pot, bot["stackChips"]
        return player["stackChips"], bot["stackChips"] + pot

    for street, new_cards in [("Flop", 3), ("Turn", 1), ("River", 1)]:
        board.extend(deck.pop() for _ in range(new_cards))
        current_bet = 0
        fold_winner, pot = run_betting_round(
            [player, bot],
            board,
            pot,
            current_bet,
            street,
            first_actor_idx=postflop_first_actor_idx,
            action_providers=action_providers,
            verbose=verbose,
        )
        if fold_winner:
            if fold_winner == PLAYER_AGENT_ID:
                return player["stackChips"] + pot, bot["stackChips"]
            return player["stackChips"], bot["stackChips"] + pot

    ties, best_score = showdown([player, bot], board)
    if verbose:
        print_table([player, bot], board, pot, 0, "Showdown", None)
        print(f"Your hand: {format_cards(player['holeCards'])}")
        print(f"Bot hand: {format_cards(bot['holeCards'])}")
    if len(ties) > 1:
        if verbose:
            print("The hand is a tie. Pot is split.")
        split = pot // len(ties)
        player_stack = player["stackChips"] + split
        bot_stack = bot["stackChips"] + split
    else:
        winner = ties[0]
        if winner["agentId"] == PLAYER_AGENT_ID:
            if verbose:
                print("You win the showdown!")
            player_stack = player["stackChips"] + pot
            bot_stack = bot["stackChips"]
        else:
            if verbose:
                print("Bot wins the showdown.")
            player_stack = player["stackChips"]
            bot_stack = bot["stackChips"] + pot
    return player_stack, bot_stack


def build_parser():
    parser = argparse.ArgumentParser(
        description="Play heads-up Texas Hold'em against a bot strategy."
    )
    parser.add_argument(
        "--strat",
        default=DEFAULT_STRATEGY,
        help=(
            "Bot strategy module under poker_bot.strategies. "
            f"Defaults to {DEFAULT_STRATEGY}."
        ),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    bot_strategy = load_strategy(args.strat)
    print("Welcome to the poker bot simulator!")
    print(f"You will play heads-up Texas Hold'em against the {args.strat} bot.")
    player_stack = INITIAL_STACK
    bot_stack = INITIAL_STACK
    player_is_small_blind = True

    while player_stack > 0 and bot_stack > 0:
        print("\n" + "#" * 60)
        player_money = format_money(player_stack)
        bot_money = format_money(bot_stack)
        print(f"Stacks -> You: {player_money} | Bot: {bot_money}")
        player_stack, bot_stack = play_hand(
            player_stack,
            bot_stack,
            player_is_small_blind,
            bot_strategy=bot_strategy,
        )
        player_is_small_blind = not player_is_small_blind
        player_money = format_money(player_stack)
        bot_money = format_money(bot_stack)
        print(f"Hand complete. New stacks -> You: {player_money} | Bot: {bot_money}")
        if player_stack <= 0:
            print("You are busted. Game over.")
            break
        if bot_stack <= 0:
            print("Bot is busted. You win!")
            break

    print("Thanks for playing!")


if __name__ == "__main__":
    main()
