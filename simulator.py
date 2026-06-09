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
from poker_bot.opponents import OpponentProfile, record_action  # noqa: E402
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
    min_bet = min(BIG_BLIND, seat["stackChips"])
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
    bet_range = {"min": min_bet, "max": max_commit} if "bet" in available else None
    raise_range = (
        {"min": min_raise_to, "max": max_commit} if "raise" in available else None
    )

    return {
        "availableActions": available,
        "callAmount": call_amount,
        "callChips": call_amount,
        "callToAmount": seat["currentBetChips"] + call_amount,
        "canCheck": "check" in available,
        "canBet": "bet" in available,
        "canRaise": "raise" in available,
        "minRaiseTo": min_raise_to,
        "minBet": min_bet,
        "maxCommit": max_commit,
        "betRange": bet_range,
        "raiseRange": raise_range,
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


def build_table(
    seats,
    board,
    pot,
    current_bet,
    street,
    acting_seat,
    button_seat_number=None,
):
    return {
        "street": street,
        "seats": seats,
        "boardCards": board,
        "potChips": pot,
        "currentBet": current_bet,
        "actingSeatNumber": acting_seat,
        "selfSeatNumber": acting_seat,
        "buttonSeatNumber": button_seat_number,
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


def live_seats(seats):
    return [seat for seat in seats if not seat.get("folded", False)]


def active_seats(seats):
    return [seat for seat in live_seats(seats) if seat["stackChips"] > 0]


def collect_live_bets_multiway(pot, seats):
    for seat in seats:
        pot += seat["currentBetChips"]
        seat["currentBetChips"] = 0
    return pot


def next_actor_index(seats, start_idx):
    if not seats:
        return None
    for offset in range(1, len(seats) + 1):
        idx = (start_idx + offset) % len(seats)
        seat = seats[idx]
        if not seat.get("folded", False) and seat["stackChips"] > 0:
            return idx
    return None


def first_active_from(seats, start_idx):
    if not seats:
        return None
    for offset in range(len(seats)):
        idx = (start_idx + offset) % len(seats)
        seat = seats[idx]
        if not seat.get("folded", False) and seat["stackChips"] > 0:
            return idx
    return None


def betting_round_closed_multiway(seats, current_bet, last_actions):
    for seat in live_seats(seats):
        if seat["stackChips"] == 0:
            continue
        if seat["currentBetChips"] < current_bet:
            return False
        if last_actions.get(seat["agentId"]) is None:
            return False
    return True


def run_betting_round_multiway(
    seats,
    board,
    pot,
    current_bet,
    street,
    first_actor_idx,
    button_seat_number,
    action_providers=None,
    opponent_profiles=None,
    action_observer=None,
    hand_id=None,
    verbose=False,
):
    action_providers = action_providers or {}
    active_idx = first_active_from(seats, first_actor_idx)
    if active_idx is None:
        return None, collect_live_bets_multiway(pot, seats)

    min_raise = BIG_BLIND
    last_actions = {
        seat["agentId"]: None for seat in seats if not seat.get("folded", False)
    }
    while True:
        if len(live_seats(seats)) <= 1:
            winner = live_seats(seats)[0]
            return winner["agentId"], pot + sum(s["currentBetChips"] for s in seats)

        seat = seats[active_idx]
        if seat.get("folded", False) or seat["stackChips"] == 0:
            next_idx = next_actor_index(seats, active_idx)
            if next_idx is None:
                break
            active_idx = next_idx
            continue

        can_raise = len(active_seats(seats)) > 1
        allowed = build_allowed_actions(
            seat,
            current_bet,
            min_raise,
            can_raise=can_raise,
        )
        table = build_table(
            seats,
            board,
            pot,
            current_bet,
            street,
            seat["seatNumber"],
            button_seat_number=button_seat_number,
        )
        table["allowedActions"] = allowed
        if opponent_profiles is not None:
            table["opponentProfiles"] = {
                agent_id: profile
                for agent_id, profile in opponent_profiles.items()
                if agent_id != seat["agentId"]
            }

        if verbose:
            print_table(seats, board, pot, street, current_bet, seat["agentId"])

        provider = action_providers.get(seat["agentId"])
        if provider is None:
            action, amount, message = choose_action(table, seat)
        else:
            action, amount, message = provider(table, seat)

        if action not in allowed["availableActions"]:
            action = "check" if "check" in allowed["availableActions"] else "fold"
            amount = None
            message = "Fallback legal action"

        if verbose:
            print(
                f"{seat['agentId']}: {action.upper()}"
                + (f" {format_money(amount)}" if amount else "")
                + f" | {message}"
            )

        facing_bet = int(allowed.get("callAmount") or allowed.get("callChips") or 0) > 0
        voluntary = action in {"call", "bet", "raise"}
        if opponent_profiles is not None:
            profile = opponent_profiles.setdefault(
                seat["agentId"], OpponentProfile(agent_id=seat["agentId"])
            )
            record_action(
                profile,
                action,
                street=street,
                facing_bet=facing_bet,
                voluntary=voluntary,
            )
        if action_observer is not None:
            action_observer(
                hand_id=hand_id,
                table=table,
                seat=seat,
                allowed=allowed,
                action=action,
                amount=amount,
                message=message,
                facing_bet=facing_bet,
                voluntary=voluntary,
            )

        if action == "fold":
            seat["folded"] = True
            last_actions[seat["agentId"]] = "fold"
            if len(live_seats(seats)) == 1:
                winner = live_seats(seats)[0]
                return winner["agentId"], pot + sum(
                    s["currentBetChips"] for s in seats
                )
        else:
            previous_current_bet = current_bet
            current_bet = resolve_action(
                seat, action, amount, current_bet, allowed["minRaiseTo"]
            )
            last_actions[seat["agentId"]] = action
            if action in ("bet", "raise") and current_bet > previous_current_bet:
                min_raise = current_bet - previous_current_bet
                for other in live_seats(seats):
                    if other["agentId"] != seat["agentId"] and other["stackChips"] > 0:
                        last_actions[other["agentId"]] = None

        if betting_round_closed_multiway(seats, current_bet, last_actions):
            break

        next_idx = next_actor_index(seats, active_idx)
        if next_idx is None:
            break
        active_idx = next_idx

    pot = collect_live_bets_multiway(pot, seats)
    return None, pot


def showdown(seats, board):
    best_score = None
    ties = []
    for seat in live_seats(seats):
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


def play_hand_multiway(
    stacks,
    strategies,
    button_index=0,
    rng=None,
    opponent_profiles=None,
    action_observer=None,
    hand_id=None,
    verbose=False,
):
    if len(stacks) < 2 or len(stacks) > 6:
        raise ValueError("play_hand_multiway supports 2 to 6 players")
    if len(strategies) != len(stacks):
        raise ValueError("strategies must match stacks")

    deck = build_deck(rng)
    board = []
    player_count = len(stacks)
    small_blind_idx = (button_index + 1) % player_count
    big_blind_idx = (button_index + 2) % player_count

    seats = []
    for index, stack in enumerate(stacks):
        blind = 0
        if index == small_blind_idx:
            blind = SMALL_BLIND
        elif index == big_blind_idx:
            blind = BIG_BLIND
        remaining, current_bet = post_blind(stack, blind)
        seats.append(
            {
                "agentId": PLAYER_AGENT_ID if index == 0 else f"bot-agent-{index}",
                "seatNumber": index + 1,
                "holeCards": deal_cards(deck),
                "stackChips": remaining,
                "currentBetChips": current_bet,
                "folded": remaining == 0 and current_bet == 0,
            }
        )

    if opponent_profiles is not None:
        for seat in seats:
            profile = opponent_profiles.setdefault(
                seat["agentId"], OpponentProfile(agent_id=seat["agentId"])
            )
            profile.hands_seen += 1

    action_providers = {
        seat["agentId"]: strategy
        for seat, strategy in zip(seats, strategies, strict=True)
        if strategy is not None
    }
    pot = 0
    current_bet = max(seat["currentBetChips"] for seat in seats)
    button_seat_number = button_index + 1

    fold_winner, pot = run_betting_round_multiway(
        seats,
        board,
        pot,
        current_bet,
        "Preflop",
        first_actor_idx=(big_blind_idx + 1) % player_count,
        button_seat_number=button_seat_number,
        action_providers=action_providers,
        opponent_profiles=opponent_profiles,
        action_observer=action_observer,
        hand_id=hand_id,
        verbose=verbose,
    )
    if fold_winner:
        for seat in seats:
            if seat["agentId"] == fold_winner:
                seat["stackChips"] += pot
                break
        return [seat["stackChips"] for seat in seats]

    for street, new_cards in [("Flop", 3), ("Turn", 1), ("River", 1)]:
        board.extend(deck.pop() for _ in range(new_cards))
        fold_winner, pot = run_betting_round_multiway(
            seats,
            board,
            pot,
            0,
            street,
            first_actor_idx=(button_index + 1) % player_count,
            button_seat_number=button_seat_number,
            action_providers=action_providers,
            opponent_profiles=opponent_profiles,
            action_observer=action_observer,
            hand_id=hand_id,
            verbose=verbose,
        )
        if fold_winner:
            for seat in seats:
                if seat["agentId"] == fold_winner:
                    seat["stackChips"] += pot
                    break
            return [seat["stackChips"] for seat in seats]

    ties, _best_score = showdown(seats, board)
    if ties:
        split = pot // len(ties)
        remainder = pot % len(ties)
        for index, seat in enumerate(ties):
            seat["stackChips"] += split + (1 if index < remainder else 0)
    return [seat["stackChips"] for seat in seats]


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
