"""Preflop range priors by position and action context."""

from __future__ import annotations

from poker_bot.range_model.hand_range import HandRange

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

THREE_BET_RANGES = {
    position: {"AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs"}
    for position in OPEN_RANGES
}

BUTTON_POSITIONS = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["BTN", "SB", "BB", "CO"],
    5: ["BTN", "SB", "BB", "MP", "CO"],
    6: ["BTN", "SB", "BB", "UTG", "MP", "CO"],
}


def normalize_position(position):
    position = str(position or "MP").strip().upper()
    aliases = {
        "BUTTON": "BTN",
        "DEALER": "BTN",
        "LATE": "BTN",
        "MIDDLE": "MP",
        "EARLY": "UTG",
        "BLIND": "BB",
        "SHORT": "BTN",
    }
    return aliases.get(position, position if position in OPEN_RANGES else "MP")


def active_seat_numbers(table):
    return sorted(
        int(seat["seatNumber"])
        for seat in table.get("seats", [])
        if seat.get("seatNumber") is not None
        and not seat.get("folded", False)
        and not seat.get("hasFolded", False)
    )


def position_label(table, seat):
    seats = active_seat_numbers(table)
    seat_number = (seat or {}).get("seatNumber")
    button = table.get("buttonSeatNumber")
    if button in seats and seat_number in seats:
        ordered = seats[seats.index(button) :] + seats[: seats.index(button)]
        labels = BUTTON_POSITIONS.get(len(ordered), BUTTON_POSITIONS[6])
        return labels[ordered.index(seat_number)]
    return normalize_position(None)


def default_preflop_range(position="MP", situation="open"):
    position = normalize_position(position)
    situation = str(situation or "open").lower()
    if situation in {"open", "unopened", "raise", "bet"}:
        return HandRange.from_classes(OPEN_RANGES[position])
    if situation in {"defend", "call", "facing_raise"}:
        return HandRange.from_classes(DEFEND_RANGES[position])
    if situation in {"three_bet", "3bet", "reraise"}:
        return HandRange.from_classes(THREE_BET_RANGES[position])
    if situation == "all":
        return HandRange.all()
    if situation == "empty":
        return HandRange.empty()
    raise ValueError(f"unknown preflop range situation: {situation!r}")
