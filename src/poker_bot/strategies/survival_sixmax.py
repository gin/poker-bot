"""6-max survival strategy optimized for chip count over hand win rate."""

from poker_bot.strategies import adaptive as adaptive_strategy
from poker_bot.strategies import profiled_counter_adaptive as profiled_strategy
from poker_bot.strategies.adaptive import (
    BIG_BLIND,
    board_texture,
    capped,
    has_top_pair_or_better,
    made_hand_rank,
    pot_odds,
    preflop_score,
)
from poker_bot.strategies.profiled_counter_adaptive import active_opponents
from poker_bot.strategies.simple import choose_action as simple_choose_action

ActionDecision = tuple[str | None, int | None, str]

OPEN_RANGES = {
    "UTG": {"AA", "KK", "QQ", "JJ", "TT", "99", "AKs", "AKo", "AQs", "AJs", "KQs"},
    "MP": {
        "AA",
        "KK",
        "QQ",
        "JJ",
        "TT",
        "99",
        "88",
        "AKs",
        "AKo",
        "AQs",
        "AQo",
        "AJs",
        "ATs",
        "KQs",
        "KJs",
        "QJs",
    },
    "CO": {
        "AA",
        "KK",
        "QQ",
        "JJ",
        "TT",
        "99",
        "88",
        "77",
        "66",
        "AKs",
        "AKo",
        "AQs",
        "AQo",
        "AJs",
        "AJo",
        "ATs",
        "A9s",
        "KQs",
        "KQo",
        "KJs",
        "KTs",
        "QJs",
        "QTs",
        "JTs",
        "T9s",
    },
    "BTN": {
        "AA",
        "KK",
        "QQ",
        "JJ",
        "TT",
        "99",
        "88",
        "77",
        "66",
        "55",
        "44",
        "33",
        "22",
        "AKs",
        "AKo",
        "AQs",
        "AQo",
        "AJs",
        "AJo",
        "ATs",
        "ATo",
        "A9s",
        "A8s",
        "A7s",
        "A6s",
        "A5s",
        "A4s",
        "A3s",
        "A2s",
        "KQs",
        "KQo",
        "KJs",
        "KJo",
        "KTs",
        "K9s",
        "QJs",
        "QTs",
        "Q9s",
        "JTs",
        "J9s",
        "T9s",
        "98s",
        "87s",
        "76s",
    },
    "SB": {
        "AA",
        "KK",
        "QQ",
        "JJ",
        "TT",
        "99",
        "88",
        "AKs",
        "AKo",
        "AQs",
        "AQo",
        "AJs",
        "ATs",
        "KQs",
        "KJs",
        "QJs",
    },
    "BB": {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs", "AQo", "AJs"},
}

DEFEND_RANGES = {
    "UTG": {"AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs"},
    "MP": {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs", "AQo", "AJs", "KQs"},
    "CO": OPEN_RANGES["MP"] | {"77", "ATs", "KTs", "QTs", "JTs"},
    "BTN": OPEN_RANGES["CO"] | {"55", "44", "33", "22", "A5s", "A4s", "K9s"},
    "SB": {"AA", "KK", "QQ", "JJ", "TT", "99", "AKs", "AKo", "AQs", "AJs", "KQs"},
    "BB": OPEN_RANGES["BTN"] | {"K8s", "Q8s", "J8s", "T8s", "97s", "86s", "75s"},
}

CANONICAL_6MAX = {1: "BTN", 2: "SB", 3: "BB", 4: "UTG", 5: "MP", 6: "CO"}
BUTTON_POSITIONS = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["BTN", "SB", "BB", "CO"],
    5: ["BTN", "SB", "BB", "MP", "CO"],
    6: ["BTN", "SB", "BB", "UTG", "MP", "CO"],
}
PREMIUMS = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
AGGRESSIVE_LABELS = {"bluffer", "loose_aggressive"}


def _rank_index(card):
    ranks = "23456789TJQKA"
    return ranks.index(card[0].upper()) if card and card[0].upper() in ranks else -1


def hand_class(hole_cards):
    if len(hole_cards) != 2:
        return ""
    first, second = hole_cards
    r1, r2 = first[0].upper(), second[0].upper()
    s1, s2 = first[-1].upper(), second[-1].upper()
    if _rank_index(first) < _rank_index(second):
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return r1 + r2
    return f"{r1}{r2}{'s' if s1 == s2 else 'o'}"


def active_seat_numbers(table):
    return [
        seat.get("seatNumber")
        for seat in table.get("seats", [])
        if not seat.get("folded", False)
        and not seat.get("hasFolded", False)
        and seat.get("seatNumber") is not None
    ]


def position_label(table, my_seat):
    seat_number = my_seat.get("seatNumber")
    seats = sorted(active_seat_numbers(table))
    button = table.get("buttonSeatNumber")
    if button in seats and seat_number in seats:
        ordered = seats[seats.index(button) :] + seats[: seats.index(button)]
        labels = BUTTON_POSITIONS.get(len(ordered), BUTTON_POSITIONS[6])
        return labels[ordered.index(seat_number)]
    return CANONICAL_6MAX.get(seat_number, "MP")


def call_amount(allowed):
    return int(allowed.get("callAmount") or allowed.get("callChips") or 0)


def min_raise_to(allowed):
    if allowed.get("minRaiseTo") is not None:
        return int(allowed["minRaiseTo"])
    raise_range = allowed.get("raiseRange") or {}
    if raise_range.get("min") is not None:
        return int(raise_range["min"])
    return None


def max_commit(allowed, default=0):
    if allowed.get("maxCommit") is not None:
        return int(allowed["maxCommit"])
    raise_range = allowed.get("raiseRange") or {}
    bet_range = allowed.get("betRange") or {}
    return int(raise_range.get("max") or bet_range.get("max") or default)


def min_bet(allowed):
    if allowed.get("minBet") is not None:
        return int(allowed["minBet"])
    bet_range = allowed.get("betRange") or {}
    return int(bet_range.get("min") or BIG_BLIND)


def raise_to_amount(table, allowed, target, all_in=False):
    minimum = min_raise_to(allowed)
    if minimum is None:
        return None
    cap = max_commit(allowed, minimum)
    if all_in:
        return cap
    return capped(max(minimum, int(target)), allowed)


def bet_amount(table, allowed, fraction):
    pot = int(table.get("potChips") or 0)
    minimum = min_bet(allowed)
    return capped(max(minimum, int(max(pot, BIG_BLIND) * fraction)), allowed)


def is_late_position(position):
    return position in {"CO", "BTN"}


def guarded_baseline(table, my_seat, max_call_fraction=0.10):
    action, amount, message = simple_choose_action(table, my_seat)
    available = table.get("allowedActions", {}).get("availableActions", [])
    if action not in available:
        return None
    if action == "check":
        return action, amount, f"Baseline free option: {message}"
    if action == "call":
        stack = int(my_seat.get("stackChips") or 0)
        price = call_amount(table.get("allowedActions", {}))
        if price <= max(BIG_BLIND, int(stack * max_call_fraction)):
            return action, amount, f"Baseline cheap continue: {message}"
    return None


def stack_total(seat):
    return int(seat.get("stackChips") or 0) + int(seat.get("currentBetChips") or 0)


def pressure_seats(table, my_seat):
    my_bet = int(my_seat.get("currentBetChips") or 0)
    my_id = my_seat.get("agentId")
    return [
        seat
        for seat in table.get("seats", [])
        if seat.get("agentId") != my_id
        and not seat.get("folded", False)
        and not seat.get("hasFolded", False)
        and int(seat.get("currentBetChips") or 0) > my_bet
    ]


def aggressive_profile_ids(table, my_seat):
    aggressive_ids = set()
    for profile in profiled_strategy.table_profiles(table, my_seat):
        if profile.label() in AGGRESSIVE_LABELS:
            aggressive_ids.add(profile.agent_id)
    return aggressive_ids


def bully_context(table, my_seat):
    """Return info about large-stack pressure we should not overfold to."""
    pressured_by = pressure_seats(table, my_seat)
    if not pressured_by:
        return None

    hero_total = max(stack_total(my_seat), 1)
    aggressive_ids = aggressive_profile_ids(table, my_seat)
    large_pressure = [
        seat
        for seat in pressured_by
        if stack_total(seat) >= hero_total * 1.2
        or seat.get("agentId") in aggressive_ids
    ]
    if not large_pressure:
        return None

    largest = max(large_pressure, key=stack_total)
    return {
        "seat": largest,
        "known_aggressive": largest.get("agentId") in aggressive_ids,
        "stack_ratio": stack_total(largest) / hero_total,
    }


def anti_bully_action(table, my_seat) -> ActionDecision | None:
    ctx = bully_context(table, my_seat)
    if ctx is None:
        return None

    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    street = table.get("street", "Preflop")
    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = int(table.get("potChips") or 0)
    price = call_amount(allowed)
    required = pot_odds(price, pot)
    score = preflop_score(hole_cards)
    made_rank = made_hand_rank(hole_cards, board_cards) if board_cards else 0
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    medium = made_rank == 1 or top_pair
    strong = made_rank >= 2
    hand = hand_class(hole_cards)

    if street == "Preflop":
        if "raise" in available and (hand in PREMIUMS or score >= 88):
            target = max(BIG_BLIND * 5, int(pot * 1.25))
            return (
                "raise",
                raise_to_amount(table, allowed, target),
                f"anti-bully value backraise {hand}",
            )
        if (
            "call" in available
            and (score >= 58 or hand in DEFEND_RANGES["BB"])
            and (required <= 0.24 or ctx["known_aggressive"])
        ):
            return "call", price, f"anti-bully preflop defend {hand}"
        return None

    if "raise" in available and strong:
        current_bet = int(table.get("currentBet") or 0)
        target = current_bet + max(BIG_BLIND * 3, int(max(pot, BIG_BLIND) * 0.65))
        return (
            "raise",
            raise_to_amount(table, allowed, target),
            f"anti-bully value raise rank {made_rank}",
        )
    if "call" in available:
        if strong and required <= 0.45:
            return "call", price, f"anti-bully continue rank {made_rank}"
        if medium and required <= (0.32 if ctx["known_aggressive"] else 0.24):
            return "call", price, "anti-bully bluff catch"
        if required <= 0.10:
            return "call", price, "anti-bully tiny price"
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

    street = table.get("street", "Preflop")
    bully_action = anti_bully_action(table, my_seat)
    if bully_action is not None:
        return bully_action

    if street == "Preflop":
        action, amount, message = profiled_strategy.choose_action(table, my_seat)
        return action, amount, f"survival preflop: {message}"

    if "bet" in available or "raise" in available:
        action, amount, message = adaptive_strategy.choose_action(table, my_seat)
        return action, amount, f"survival value pressure: {message}"

    action, amount, message = profiled_strategy.choose_action(table, my_seat)
    return action, amount, f"survival defense: {message}"

    hole_cards = my_seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = int(table.get("potChips") or 0)
    current_bet = int(table.get("currentBet") or 0)
    stack = int(my_seat.get("stackChips") or 0)
    stack_bb = stack / BIG_BLIND if BIG_BLIND else 0
    required = pot_odds(call_amount(allowed), pot)
    opponents = active_opponents(table, my_seat)
    position = position_label(table, my_seat)
    hand = hand_class(hole_cards)

    baseline_action, baseline_amount, baseline_message = simple_choose_action(
        table, my_seat
    )
    if baseline_action in available:
        if street == "Preflop" and baseline_action == "raise" and hand in PREMIUMS:
            return baseline_action, baseline_amount, f"Premium {hand} from {position}"
        if street == "Preflop" and baseline_action == "fold":
            return baseline_action, baseline_amount, (
                f"Preserving stack with {hand} from {position}"
            )
        if street != "Preflop" and baseline_action == "fold":
            return baseline_action, baseline_amount, (
                f"Avoiding chip leak rank {made_hand_rank(hole_cards, board_cards)}"
            )
        return (
            baseline_action,
            baseline_amount,
            f"6-max chip baseline {position}: {baseline_message}",
        )

    if street == "Preflop":
        unopened = current_bet <= BIG_BLIND
        open_range = OPEN_RANGES.get(position, OPEN_RANGES["MP"])
        defend_range = DEFEND_RANGES.get(position, DEFEND_RANGES["MP"])

        if "raise" in available and hand in PREMIUMS:
            target = BIG_BLIND * (3.2 if opponents >= 4 else 2.6)
            all_in = stack_bb <= 10
            return (
                "raise",
                raise_to_amount(table, allowed, target, all_in=all_in),
                f"Premium {hand} from {position}, building chip EV",
            )

        if unopened and "raise" in available and hand in open_range:
            target = BIG_BLIND * (2.5 if is_late_position(position) else 3.0)
            return (
                "raise",
                raise_to_amount(table, allowed, target),
                f"Opening {hand} from {position}",
            )

        if "call" in available:
            cheap = required <= (0.08 if opponents >= 3 else 0.12)
            stack_safe = call_amount(allowed) <= max(BIG_BLIND, int(stack * 0.08))
            cheap_unopened = unopened and call_amount(allowed) <= BIG_BLIND
            score = preflop_score(hole_cards)
            if cheap_unopened and stack_safe and score >= 35:
                return "call", call_amount(allowed), f"Cheap unopened continue {hand}"
            if hand in defend_range and (cheap or stack_safe):
                return "call", call_amount(allowed), f"Controlled defend {hand}"
            if "check" in available:
                return "check", None, f"Free option with {hand}"
            baseline = guarded_baseline(table, my_seat, max_call_fraction=0.08)
            if baseline is not None:
                return baseline
            return "fold", None, f"Preserving stack with {hand} from {position}"

        if "check" in available:
            return "check", None, f"Checking {hand} from {position}"

    made_rank = made_hand_rank(hole_cards, board_cards)
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    texture = board_texture(board_cards) if board_cards else {"wet": False}
    strong_threshold = 2
    strong = made_rank >= strong_threshold
    monster = made_rank >= 4
    medium = made_rank == 1 or top_pair
    dry = not texture.get("wet", False)

    if "raise" in available and (strong or monster):
        fraction = 0.85 if monster else 0.55
        target = current_bet + max(BIG_BLIND, int(max(pot, BIG_BLIND) * fraction))
        return (
            "raise",
            raise_to_amount(table, allowed, target, all_in=monster and stack_bb <= 18),
            f"Value raise rank {made_rank}",
        )

    if "call" in available:
        price = call_amount(allowed)
        if strong and required <= 0.42:
            return "call", price, f"Continue strong rank {made_rank}"
        if medium and opponents <= 2 and required <= 0.24 and price <= stack * 0.18:
            return "call", price, "Heads-up medium hand at fair price"
        if medium and opponents <= 4 and required <= 0.18 and price <= stack * 0.14:
            return "call", price, "Small multiway price for medium hand"
        if top_pair and required <= 0.16 and price <= stack * 0.12:
            return "call", price, "Top-pair price is acceptable"
        if required <= 0.06 and price <= stack * 0.06:
            return "call", price, "Tiny price, preserving optionality"
        if "check" in available:
            return "check", None, "Free pot control"
        baseline = guarded_baseline(table, my_seat, max_call_fraction=0.12)
        if baseline is not None:
            return baseline
        return "fold", None, f"Avoiding chip leak rank {made_rank}"

    if "bet" in available:
        if strong:
            return (
                "bet",
                bet_amount(table, allowed, 0.72 if monster else 0.55),
                f"Value bet rank {made_rank}",
            )
        if medium and opponents <= 2:
            return "bet", bet_amount(table, allowed, 0.38), "Thin value heads-up"
        if medium and dry:
            return "bet", bet_amount(table, allowed, 0.34), "Small dry-board value"
        if opponents <= 1 and dry and is_late_position(position):
            return "bet", bet_amount(table, allowed, 0.25), "Low-risk late c-bet"

    if "check" in available:
        return "check", None, "Pot control, survival-first"
    if "fold" in available:
        return "fold", None, "No chip-positive path"
    return None, None, "No supported legal action"
