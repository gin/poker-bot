#!/usr/bin/env python3
"""Terminal poker simulator for playing heads-up against the poker bot."""

import argparse
import re
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
from poker_bot.opponents import OpponentProfile, record_action, record_hand_seen  # noqa: E402
from poker_bot.strategies.loader import load_strategy  # noqa: E402

DEFAULT_STRATEGY = "simple"
choose_action = load_strategy(DEFAULT_STRATEGY)

SMALL_BLIND = 5
BIG_BLIND = 10
INITIAL_STACK = 1000
PLAYER_AGENT_ID = "player-agent"
BOT_AGENT_ID = "bot-agent"

# ── ANSI colour / style helpers ────────────────────────────────────────────

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_BLUE = "\033[94m"
_CYAN = "\033[96m"
_WHITE = "\033[97m"
_BG = "\033[30;47m"  # black-on-white highlight

SUIT_SYMBOLS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}

# Box-drawing characters
_H = "─"
_V = "│"
_TL = "┌"
_TR = "┐"
_BL = "└"
_BR = "┘"
_CA = "├"
_CB = "┤"
_CE = "═"
_VA = "║"


def _c(code, text):
    """Wrap text in an ANSI code, then reset."""
    return f"{code}{text}{_RESET}"


def _bold(text):
    return _c(_BOLD, text)


def _head(text):
    """Header-level styling."""
    return _c(_WHITE + _BOLD, text)


def colour_card(card_str):
    """Render a card string (e.g. 'As') with Unicode suit + colour."""
    if not card_str or card_str == "??":
        return card_str
    rank = card_str[:-1]
    suit_char = card_str[-1].lower()
    sym = SUIT_SYMBOLS.get(suit_char, suit_char)
    out = f"{rank}{sym}"
    if suit_char in ("h", "d"):
        return _c(_RED, out)
    return out


def colour_hand(cards, sep=" "):
    """Render a list of cards coloured by suit."""
    return sep.join(colour_card(c) for c in cards)


def _visible_len(text):
    return len(re.sub(r"\033\[[0-9;]*m", "", text))


def _ljust_visible(text, width):
    return f"{text}{' ' * max(0, width - _visible_len(text))}"


def street_icon(street):
    """Return a coloured street label."""
    icons = {
        "Preflop": _c(_BLUE, "♥ PRE-FLOP"),
        "Flop": _c(_CYAN, "♠ FLOP"),
        "Turn": _c(_YELLOW, "♦ TURN"),
        "River": _c(_RED, "♣ RIVER"),
        "Showdown": _bold("⚖ SHOWDOWN"),
    }
    return icons.get(street, street)


def blinder(player_is_sb):
    """Return (player_label, bot_label) position string."""
    if player_is_sb:
        return "(SB)", "(BB)"
    return "(BB)", "(SB)"


def format_cards(cards):
    return " ".join(cards)


def build_allowed_actions(
    seat,
    current_bet,
    min_raise=BIG_BLIND,
    can_raise=True,
    big_blind=BIG_BLIND,
    raise_reopened=True,
):
    call_shortfall = max(0, current_bet - seat["currentBetChips"])
    call_amount = min(call_shortfall, seat["stackChips"])
    max_commit = seat["currentBetChips"] + seat["stackChips"]
    min_bet = min(big_blind, seat["stackChips"])
    if seat["stackChips"] == 0:
        available = []
    elif call_shortfall == 0:
        available = ["fold", "check", "bet", "all-in"]
    else:
        available = ["fold", "call"]
        min_raise_to = current_bet + min_raise
        if can_raise and raise_reopened and max_commit >= min_raise_to:
            available.append("raise")
        # An all-in that would put in MORE than the current bet functions
        # as a raise, so it must be withheld exactly when this seat's
        # raise rights are closed (an earlier short all-in this player
        # already acted behind, per NLHE reopening rules) -- unrelated to
        # `can_raise`, which reflects whether anyone at the table could
        # still respond to a raise at all. An all-in that does not exceed
        # the current bet is just a short call and is always legal.
        if raise_reopened or max_commit <= current_bet:
            available.append("all-in")

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
        "canAllIn": "all-in" in available,
        "allInToAmount": max_commit,
        "minRaiseTo": min_raise_to,
        "minBet": min_bet,
        "maxCommit": max_commit,
        "betRange": bet_range,
        "raiseRange": raise_range,
    }


def _refund_uncalled_bet(seats):
    """Return an unmatched wager to its owner before it enters the pot.

    If exactly one seat's ``currentBetChips`` this street strictly exceeds
    every other seat's, the excess no other seat could match was never
    genuinely at risk and is refunded immediately -- mirroring live-table
    rules and preventing a later fold from stranding the excess in a pot
    layer nobody is eligible to win.
    """
    if not seats:
        return
    top_bet = max(seat["currentBetChips"] for seat in seats)
    if top_bet == 0:
        return
    top_seats = [seat for seat in seats if seat["currentBetChips"] == top_bet]
    if len(top_seats) != 1:
        return
    second_bet = max(
        (seat["currentBetChips"] for seat in seats if seat is not top_seats[0]),
        default=0,
    )
    uncalled = top_bet - second_bet
    if uncalled <= 0:
        return
    top_seats[0]["stackChips"] += uncalled
    top_seats[0]["currentBetChips"] -= uncalled


def collect_live_bets(pot, seats):
    """Heads-up entry point; multiway settlement already generalizes to
    N=2 seats, so this simply delegates."""
    return collect_live_bets_multiway(pot, seats)


def all_in_betting_is_closed(seats, current_bet):
    if not any(seat["stackChips"] == 0 for seat in seats):
        return False
    return not any(
        seat["stackChips"] > 0 and seat["currentBetChips"] < current_bet
        for seat in seats
    )


def format_money(amount):
    return _c(_GREEN, f"${amount}")


def prompt_user_action(acts, call_amount):
    """Show a numbered action menu. ``acts`` is a list of action names."""
    print(f"  {_c(_CYAN, 'Actions:')}")
    for index, action in enumerate(acts, start=1):
        if action == "call":
            print(f"    {_bold(str(index))}. call {format_money(call_amount)}")
        elif action == "raise":
            print(f"    {_bold(str(index))}. raise")
        elif action == "bet":
            print(f"    {_bold(str(index))}. bet")
        elif action == "all-in":
            print(f"    {_bold(str(index))}. all-in")
        else:
            print(f"    {_bold(str(index))}. {action}")
    while True:
        choice = input(f"  {_c(_GREEN, '?>')} ").strip().lower()
        if choice.isdigit() and 1 <= int(choice) <= len(acts):
            return acts[int(choice) - 1]
        if choice in acts:
            return choice
        print(f"  {_c(_RED, '!')} Invalid action. Pick a number or name.")


def prompt_player_action(allowed):
    while True:
        # Show min raise/bet info before the action menu
        min_raise_to = allowed.get("minRaiseTo")
        if min_raise_to:
            print(f"  {_c(_YELLOW, 'Min raise to:')} {format_money(min_raise_to)}+")
        elif allowed.get("minBet"):
            print(f"  {_c(_YELLOW, 'Min bet:')} {format_money(allowed['minBet'])}+")
        action = prompt_user_action(allowed["availableActions"], allowed["callAmount"])
        amount = None
        if action not in ("bet", "raise"):
            return action, amount

        while True:
            action_label = "raise to" if action == "raise" else "bet"
            minimum = allowed["minRaiseTo"] if action == "raise" else allowed["minBet"]
            max_allowed = allowed.get("maxCommit", minimum)
            prompt = (
                f"  {_c(_GREEN, '?>')} Enter amount to {action_label} "
                f"[{format_money(minimum)} – {format_money(max_allowed)}; "
                f"0 to go back]: "
            )
            entry = input(prompt).strip()
            if entry == "0":
                break
            if not entry:
                return action, amount
            if entry.isdigit() and int(entry) >= minimum and int(entry) <= max_allowed:
                amount = int(entry)
                return action, amount
            print(
                f"  {_c(_RED, '!')} Amount must be "
                f"{format_money(minimum)}–{format_money(max_allowed)}."
            )


def resolve_action(
    seat, action, amount, current_bet, min_raise_to=None, big_blind=BIG_BLIND
):
    # Chips are always discrete integers. Strategies occasionally compute a
    # call/bet/raise amount via float arithmetic (e.g. BIG_BLIND * 2.5, or
    # pot * a float fraction) without casting back to int. Left uncast,
    # that float would flow straight into stackChips/currentBetChips below
    # and permanently contaminate every downstream chip computation for
    # this seat (pot settlement, tournament stage deltas, report
    # formatting, ...) with float typing. resolve_action is the single
    # choke point every action passes through, so normalize any
    # strategy-supplied amount/target to an integer chip count here, using
    # the same int() truncation already used elsewhere in the engine (e.g.
    # the pot-relative sizing helpers strategies themselves call). This
    # preserves all legality/default logic below unchanged -- it only
    # fixes the numeric type, not the value, for any amount that already
    # represented a whole number of chips (the overwhelmingly common
    # case); a genuinely fractional amount is truncated toward zero, never
    # rounded up past what the strategy asked for or past a legal bound.
    if amount is not None:
        amount = int(amount)
    if min_raise_to is not None:
        min_raise_to = int(min_raise_to)
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

    if action == "all-in":
        target = seat["currentBetChips"] + seat["stackChips"]
    else:
        target = amount

    if target is None:
        target = current_bet + big_blind
    if action == "bet" and target < big_blind:
        target = min(big_blind, seat["currentBetChips"] + seat["stackChips"])
    if action == "raise":
        if min_raise_to is None:
            min_raise_to = current_bet + big_blind
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
    small_blind=SMALL_BLIND,
    big_blind=BIG_BLIND,
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
        "smallBlindChips": small_blind,
        "bigBlindChips": big_blind,
    }


def print_table(seats, board, pot, street, current_bet, active_id, min_raise_to=None):
    """Print a clean, colourised view of the table."""
    header = street_icon(street)
    board_str = colour_hand(board) if board else _c(_DIM, "(empty)")
    pot_str = format_money(pot)
    bet_str = format_money(current_bet) if current_bet else _c(_DIM, "$0")

    pot_row = f"Pot: {pot_str}   Bet: {bet_str}"
    player_rows = []
    for seat in seats:
        if seat["agentId"] == PLAYER_AGENT_ID:
            label = _c(_GREEN + _BOLD, "YOU")
        else:
            label = _c(_YELLOW + _BOLD, "BOT")
        hole = (
            colour_hand(seat["holeCards"])
            if seat["agentId"] == PLAYER_AGENT_ID
            else _c(_DIM, "?? ?")
        )
        stack_str = format_money(seat["stackChips"])
        if seat["currentBetChips"]:
            bet_str_s = format_money(seat["currentBetChips"])
        else:
            bet_str_s = _c(_DIM, "$0")
        row = f"{label}  stack={stack_str}  bet={bet_str_s}  cards={hole}"
        if seat["agentId"] == active_id:
            row += "  ◀"
        player_rows.append(row)

    width = max(_visible_len(pot_row), *(_visible_len(row) for row in player_rows))

    print()
    print(f"  {_c(_WHITE + _BOLD, header)}")
    print(f"  {_c(_CYAN, 'Board:')}  {board_str}")
    print()
    print(f"  {_V} {_ljust_visible(pot_row, width)} {_V}")
    print(f"  {_V} {_H * width} {_V}")
    for row in player_rows:
        print(f"  {_V} {_ljust_visible(row, width)} {_V}")
    print(f"  {_BL} {_H * width} {_BR}")

    # Minimum raise info
    if min_raise_to is not None and min_raise_to > 0:
        print(f"  {_c(_YELLOW, 'Min raise to:')} {format_money(min_raise_to)}")
    elif min_raise_to is None or min_raise_to == 0:
        # Show min bet from the first seat
        first = seats[0]
        min_bet = min(BIG_BLIND, first["stackChips"])
        if min_bet > 0 and first["currentBetChips"] == current_bet:
            print(f"  {_c(_YELLOW, 'Min bet:')} {format_money(min_bet)}")


def run_betting_round(
    seats,
    board,
    pot,
    current_bet,
    street,
    first_actor_idx,
    action_providers=None,
    verbose=True,
    action_observer=None,
    hand_id=None,
    opponent_profiles=None,
):
    action_providers = action_providers or {}
    active_idx = first_actor_idx
    min_raise = BIG_BLIND
    last_actions = {PLAYER_AGENT_ID: None, BOT_AGENT_ID: None}
    _round_action_history = []

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
        if opponent_profiles is not None:
            table["opponentProfiles"] = {
                agent_id: profile
                for agent_id, profile in opponent_profiles.items()
                if agent_id != seat["agentId"]
            }
        table["actionHistory"] = list(_round_action_history)

        if verbose:
            print_table(
                seats,
                board,
                pot,
                street,
                current_bet,
                seat["agentId"],
                min_raise_to=allowed.get("minRaiseTo"),
            )

        provider = action_providers.get(seat["agentId"])
        if provider is not None:
            action, amount, message = provider(table, seat)
            if verbose:
                if seat["agentId"] == PLAYER_AGENT_ID:
                    label = _c(_GREEN + _BOLD, "YOU")
                else:
                    label = _c(_YELLOW + _BOLD, "BOT")
                act_msg = f"{_bold(str(action).upper())}"
                if amount:
                    act_msg += f" {format_money(amount)}"
                if message:
                    act_msg += f"  {_c(_DIM, f'({message})')}"
                print(f"  {label}: {act_msg}")
        elif seat["agentId"] == PLAYER_AGENT_ID:
            action, amount = prompt_player_action(allowed)
        else:
            action, amount, message = choose_action(table, seat)
            if verbose:
                act_msg = f"{_bold(str(action).upper())}"
                if amount:
                    act_msg += f" {format_money(amount)}"
                if message:
                    act_msg += f"  {_c(_DIM, f'({message})')}"
                print(f"  {_c(_YELLOW + _BOLD, 'BOT')}: {act_msg}")

        facing_bet = int(allowed.get("callAmount") or 0) > 0
        voluntary = street == "Preflop" and action in {"call", "bet", "raise", "all-in"}
        is_preflop_raise = street == "Preflop" and (
            action in {"bet", "raise"}
            or (
                action == "all-in"
                and int(allowed.get("allInToAmount") or 0) > current_bet
            )
        )
        if opponent_profiles is not None:
            profile = opponent_profiles.setdefault(
                seat["agentId"], OpponentProfile(agent_id=seat["agentId"])
            )
            record_action(
                profile,
                action,
                hand_id=hand_id,
                street=street,
                facing_bet=facing_bet,
                voluntary=voluntary,
                is_preflop_raise=is_preflop_raise,
            )

        if action_observer is not None:
            action_observer(
                hand_id=hand_id,
                table=table,
                seat=seat,
                allowed=allowed,
                action=action,
                amount=amount,
                message=locals().get("message"),
                facing_bet=facing_bet,
                voluntary=voluntary,
                street=street,
            )

        _round_action_history.append(
            {
                "agentId": seat["agentId"],
                "action": action,
                "amount": amount,
                "street": street,
            }
        )

        if action == "fold":
            if verbose:
                if seat["agentId"] == PLAYER_AGENT_ID:
                    folded_label = _c(_GREEN + _BOLD, "YOU")
                else:
                    folded_label = _c(_YELLOW + _BOLD, "BOT")
                print(f"  {folded_label} {_c(_RED, 'folds')}.")
            return opponent["agentId"], pot + seat["currentBetChips"] + opponent[
                "currentBetChips"
            ]

        previous_current_bet = current_bet
        current_bet = resolve_action(
            seat, action, amount, current_bet, allowed["minRaiseTo"]
        )
        last_actions[seat["agentId"]] = action
        if action in ("bet", "raise", "all-in") and current_bet > previous_current_bet:
            # NLHE rule: a short all-in that is less than the current minimum
            # raise increment does NOT change the minimum raise for subsequent
            # actions. Only full raises update the minimum.
            raise_inc = current_bet - previous_current_bet
            if raise_inc >= min_raise:
                min_raise = raise_inc

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

        if action in ("bet", "raise", "all-in"):
            # Opponent must respond to a new bet/raise/all-in
            last_actions[opponent["agentId"]] = None

        active_idx = 1 - active_idx

    pot = collect_live_bets(pot, seats)
    return None, pot


def live_seats(seats):
    return [seat for seat in seats if not seat.get("folded", False)]


def active_seats(seats):
    return [seat for seat in live_seats(seats) if seat["stackChips"] > 0]


def collect_live_bets_multiway(pot, seats):
    """Sweep this street's bets into the pot.

    Refunds any unmatched excess first (see :func:`_refund_uncalled_bet`),
    then accumulates each seat's cumulative ``committedChips`` for the
    whole hand -- :func:`build_pots` needs this running total to build
    correct main/side pots at showdown, independent of which street the
    chips were committed on.
    """
    _refund_uncalled_bet(seats)
    for seat in seats:
        seat["committedChips"] = seat.get("committedChips", 0) + seat["currentBetChips"]
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
    small_blind=SMALL_BLIND,
    big_blind=BIG_BLIND,
):
    action_providers = action_providers or {}
    active_idx = first_active_from(seats, first_actor_idx)
    if active_idx is None:
        return None, collect_live_bets_multiway(pot, seats)

    min_raise = big_blind
    # Agent ids for whom raising is currently closed: they already acted
    # (called/raised/bet) at a prior betting level, and the most recent
    # increase was a short all-in (below min_raise) rather than a full
    # raise. NLHE: only a full raise reopens raising for players who have
    # already acted; a short all-in only obliges them to call the extra
    # amount or fold.
    raise_closed_for = set()
    last_actions = {
        seat["agentId"]: None for seat in seats if not seat.get("folded", False)
    }
    _round_action_history = []

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
        raise_reopened = seat["agentId"] not in raise_closed_for
        allowed = build_allowed_actions(
            seat,
            current_bet,
            min_raise,
            can_raise=can_raise,
            big_blind=big_blind,
            raise_reopened=raise_reopened,
        )
        table = build_table(
            seats,
            board,
            pot,
            current_bet,
            street,
            seat["seatNumber"],
            button_seat_number=button_seat_number,
            small_blind=small_blind,
            big_blind=big_blind,
        )
        table["allowedActions"] = allowed
        if opponent_profiles is not None:
            table["opponentProfiles"] = {
                agent_id: profile
                for agent_id, profile in opponent_profiles.items()
                if agent_id != seat["agentId"]
            }
        table["actionHistory"] = list(_round_action_history)

        if verbose:
            print_table(
                seats,
                board,
                pot,
                street,
                current_bet,
                seat["agentId"],
                min_raise_to=allowed.get("minRaiseTo"),
            )

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
        voluntary = street == "Preflop" and action in {"call", "bet", "raise", "all-in"}
        is_preflop_raise = street == "Preflop" and (
            action in {"bet", "raise"}
            or (
                action == "all-in"
                and int(allowed.get("allInToAmount") or 0) > current_bet
            )
        )
        if opponent_profiles is not None:
            profile = opponent_profiles.setdefault(
                seat["agentId"], OpponentProfile(agent_id=seat["agentId"])
            )
            record_action(
                profile,
                action,
                hand_id=hand_id,
                street=street,
                facing_bet=facing_bet,
                voluntary=voluntary,
                is_preflop_raise=is_preflop_raise,
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
                street=street,
            )

        _round_action_history.append(
            {
                "agentId": seat["agentId"],
                "action": action,
                "amount": amount,
                "street": street,
            }
        )

        if action == "fold":
            seat["folded"] = True
            last_actions[seat["agentId"]] = "fold"
            if len(live_seats(seats)) == 1:
                winner = live_seats(seats)[0]
                return winner["agentId"], pot + sum(s["currentBetChips"] for s in seats)
        else:
            previous_current_bet = current_bet
            current_bet = resolve_action(
                seat, action, amount, current_bet, allowed["minRaiseTo"], big_blind
            )
            last_actions[seat["agentId"]] = action
            if (
                action in ("bet", "raise", "all-in")
                and current_bet > previous_current_bet
            ):
                # NLHE: short all-in (< min_raise) does not change min raise
                raise_inc = current_bet - previous_current_bet
                if raise_inc >= min_raise:
                    min_raise = raise_inc
                    # A full raise reopens raising for every other player,
                    # even one a prior short all-in had closed it for.
                    raise_closed_for.clear()
                else:
                    # Short all-in: close raising only for players who had
                    # already acted at the previous level (captured from
                    # last_actions BEFORE the reset below clears it).
                    # Players who have not yet acted this round keep their
                    # normal raise rights when their turn comes.
                    for other_id, other_action in last_actions.items():
                        if other_id != seat["agentId"] and other_action is not None:
                            raise_closed_for.add(other_id)
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


def _seat_order_from_button(seats, button_seat_number):
    """Order seats starting immediately after the button.

    Used only to make odd-chip distribution deterministic: when a pot
    layer is split among tied hands, the leftover chip(s) go to the tied
    seat(s) closest to acting first next hand (i.e. immediately left of
    the button), matching common live-table convention.
    """
    if not seats:
        return []
    if button_seat_number is None:
        return list(seats)
    start = 0
    for index, seat in enumerate(seats):
        if seat["seatNumber"] == button_seat_number:
            start = (index + 1) % len(seats)
            break
    return seats[start:] + seats[:start]


def build_pots(seats):
    """Decompose per-seat cumulative contributions into main + side pots.

    Each seat must expose ``committedChips`` (its total chips put into the
    pot this hand, across every street, including blinds and anything
    contributed before folding) and ``folded``. Returns a tuple of
    ``(amount, eligible_agent_ids)`` layers ordered from the main pot
    outward. Chip-conserving: ``sum(amount for amount, _ in pots) ==
    sum(seat["committedChips"] for seat in seats)``.

    A layer contributed to by exactly one seat is returned to that seat
    whole -- this is defence in depth around :func:`_refund_uncalled_bet`,
    which already refunds a street's unmatched excess immediately, so this
    case should not arise from a legally played hand.
    """
    contributions = {seat["agentId"]: seat.get("committedChips", 0) for seat in seats}
    folded = {seat["agentId"] for seat in seats if seat.get("folded", False)}
    levels = sorted({amount for amount in contributions.values() if amount > 0})
    pots = []
    previous_level = 0
    for level in levels:
        layer = level - previous_level
        contributors = [
            agent_id for agent_id, amount in contributions.items() if amount >= level
        ]
        layer_amount = layer * len(contributors)
        if layer_amount > 0:
            eligible = tuple(
                agent_id for agent_id in contributors if agent_id not in folded
            )
            pots.append((layer_amount, eligible))
        previous_level = level
    return tuple(pots)


def settle_pots(seats, board, button_seat_number=None):
    """Award every pot layer to its best eligible hand(s).

    Splits a tied layer evenly and hands any odd chip(s) to the tied
    seat(s) ordered by :func:`_seat_order_from_button` (documented
    odd-chip convention: closest to acting first next hand, i.e.
    immediately left of the button).

    Returns ``{agentId: chips_won}``; callers apply the result to
    ``stackChips`` themselves. Every legally played hand has at least one
    live (non-folded) seat, so every pot layer has an eligible winner --
    an empty eligible set indicates a bookkeeping bug upstream and is
    raised rather than silently dropping chips.
    """
    ordered_seats = _seat_order_from_button(seats, button_seat_number)
    by_agent = {seat["agentId"]: seat for seat in seats}
    winnings = {seat["agentId"]: 0 for seat in seats}
    for amount, eligible_ids in build_pots(seats):
        if not eligible_ids:
            raise ValueError(
                "pot layer has no eligible (non-folded) contributor; this "
                "indicates a pot-accounting bug, not a legal hand"
            )
        if len(eligible_ids) == 1:
            winnings[eligible_ids[0]] += amount
            continue
        eligible_seats = [by_agent[agent_id] for agent_id in eligible_ids]
        best_score = None
        ties = []
        for seat in eligible_seats:
            score = evaluate_hand(seat["holeCards"] + board)
            if best_score is None or score > best_score:
                best_score = score
                ties = [seat]
            elif score == best_score:
                ties.append(seat)
        tie_ids = {seat["agentId"] for seat in ties}
        ordered_ties = [seat for seat in ordered_seats if seat["agentId"] in tie_ids]
        share, remainder = divmod(amount, len(ordered_ties))
        for index, seat in enumerate(ordered_ties):
            winnings[seat["agentId"]] += share + (1 if index < remainder else 0)
    return winnings


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
    action_observer=None,
    hand_id=None,
    opponent_profiles=None,
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
    if opponent_profiles is not None:
        for seat in (player, bot):
            profile = opponent_profiles.setdefault(
                seat["agentId"], OpponentProfile(agent_id=seat["agentId"])
            )
            record_hand_seen(profile, hand_id)

    fold_winner, pot = run_betting_round(
        [player, bot],
        board,
        pot,
        current_bet,
        street,
        first_actor_idx=preflop_first_actor_idx,
        action_providers=action_providers,
        verbose=verbose,
        action_observer=action_observer,
        hand_id=hand_id,
        opponent_profiles=opponent_profiles,
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
            action_observer=action_observer,
            hand_id=hand_id,
            opponent_profiles=opponent_profiles,
        )
        if fold_winner:
            if fold_winner == PLAYER_AGENT_ID:
                return player["stackChips"] + pot, bot["stackChips"]
            return player["stackChips"], bot["stackChips"] + pot

    if action_observer is not None:
        action_observer(action="showdown")
    if verbose:
        print(f"\n  {_bold('⚖ SHOWDOWN')}")
        print(f"  {_c(_DIM, '─────')}")
        print(f"  {_c(_GREEN + _BOLD, 'You')}: {colour_hand(player['holeCards'])}")
        print(f"  {_c(_YELLOW + _BOLD, 'Bot')}: {colour_hand(bot['holeCards'])}")
        print(f"  Board: {colour_hand(board)}")
        print(f"  Pot: {format_money(pot)}")
    button_seat_number = (
        player["seatNumber"] if player_is_small_blind else bot["seatNumber"]
    )
    winnings = settle_pots([player, bot], board, button_seat_number=button_seat_number)
    player_won = winnings.get(PLAYER_AGENT_ID, 0)
    bot_won = winnings.get(BOT_AGENT_ID, 0)
    if player_won > 0 and bot_won > 0:
        if verbose:
            print(f"  {_c(_CYAN, 'Split pot')} — hand is a tie.")
    elif player_won > 0:
        if verbose:
            print(f"  {_c(_GREEN + _BOLD, '✓ You win!')}")
    else:
        if verbose:
            print(f"  {_c(_RED + _BOLD, '✗ Bot wins.')}")
    player_stack = player["stackChips"] + player_won
    bot_stack = bot["stackChips"] + bot_won
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
    agent_ids=None,
    small_blind=SMALL_BLIND,
    big_blind=BIG_BLIND,
):
    if len(stacks) < 2 or len(stacks) > 6:
        raise ValueError("play_hand_multiway supports 2 to 6 players")
    if len(strategies) != len(stacks):
        raise ValueError("strategies must match stacks")
    if agent_ids is None:
        agent_ids = [
            PLAYER_AGENT_ID if index == 0 else f"bot-agent-{index}"
            for index in range(len(stacks))
        ]
    elif len(agent_ids) != len(stacks):
        raise ValueError("agent_ids must match stacks")

    deck = build_deck(rng)
    board = []
    player_count = len(stacks)
    if player_count == 2:
        # Heads-up: the button posts the small blind and acts first
        # preflop (last postflop). This is the opposite of the 3+ player
        # indexing below, where the button itself never posts a blind.
        small_blind_idx = button_index
        big_blind_idx = (button_index + 1) % player_count
    else:
        small_blind_idx = (button_index + 1) % player_count
        big_blind_idx = (button_index + 2) % player_count

    seats = []
    for index, stack in enumerate(stacks):
        blind = 0
        if index == small_blind_idx:
            blind = small_blind
        elif index == big_blind_idx:
            blind = big_blind
        remaining, current_bet = post_blind(stack, blind)
        seats.append(
            {
                "agentId": agent_ids[index],
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
            record_hand_seen(profile, hand_id)

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
        small_blind=small_blind,
        big_blind=big_blind,
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
            small_blind=small_blind,
            big_blind=big_blind,
        )
        if fold_winner:
            for seat in seats:
                if seat["agentId"] == fold_winner:
                    seat["stackChips"] += pot
                    break
            return [seat["stackChips"] for seat in seats]

    if action_observer is not None:
        action_observer(action="showdown")
    winnings = settle_pots(seats, board, button_seat_number=button_seat_number)
    for seat in seats:
        seat["stackChips"] += winnings.get(seat["agentId"], 0)
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

    print()
    title = "No-Limit Texas Hold'em Simulator"
    print(f"  {_bold(title)}")
    print(f"  {_c(_DIM, '────' + '─' * (22 + len(args.strat)))}")
    print(
        f"  Bot: {_c(_YELLOW, args.strat)}  |  "
        f"Blinds: {format_money(SMALL_BLIND)}/{format_money(BIG_BLIND)}"
    )
    print()
    player_stack = INITIAL_STACK
    bot_stack = INITIAL_STACK
    player_is_small_blind = True
    hand_num = 0

    while player_stack > 0 and bot_stack > 0:
        hand_num += 1
        player_pos, bot_pos = blinder(player_is_small_blind)
        print(
            f"  {_bold(f'─── Hand #{hand_num} ───')}  "
            f"{_c(_BLUE, f'You {player_pos}')}  "
            f"{_c(_YELLOW, f'Bot {bot_pos}')}"
        )
        print(
            f"  {_c(_DIM, 'Stacks:')} "
            f"{format_money(player_stack)}  vs  {format_money(bot_stack)}"
        )
        print()

        player_stack, bot_stack = play_hand(
            player_stack,
            bot_stack,
            player_is_small_blind,
            bot_strategy=bot_strategy,
        )

        player_is_small_blind = not player_is_small_blind

        if player_stack <= 0:
            print(f"\n  {_c(_RED + _BOLD, '✗ You are busted. Game over.')}")
            break
        if bot_stack <= 0:
            print(f"\n  {_c(_GREEN + _BOLD, '✓ Bot is busted. You win!')}")
            break

    print(
        f"\n  {_bold('Final:')}  "
        f"{format_money(player_stack)}  vs  {format_money(bot_stack)}"
    )
    print(f"  {_c(_DIM, 'Thanks for playing!')}")


if __name__ == "__main__":
    main()
