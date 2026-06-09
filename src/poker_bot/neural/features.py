"""Feature encoding for neural policy/value models.

The encoder intentionally mirrors ``decision_telemetry`` columns. Training rows
and live table states should pass through this module so model behavior stays
consistent across offline self-play, arena telemetry, and future CFR labels.
"""

from __future__ import annotations

from dataclasses import dataclass

from poker_bot.strategies.adaptive import (
    board_texture,
    has_top_pair_or_better,
    made_hand_rank,
    preflop_score,
)

CHIP_SCALE = 50.0

ACTIONS = ("fold", "check", "call", "bet", "raise")
STREETS = ("Preflop", "Flop", "Turn", "River")
POSITIONS = ("BTN/SB", "BTN", "SB", "BB", "UTG", "HJ", "CO")
HAND_BUCKETS = ("air", "medium", "strong")
TABLE_STYLES = (
    "unknown",
    "short_handed",
    "loose_aggressive",
    "loose_passive",
    "tight",
)

NUMERIC_FEATURE_NAMES = (
    "active_players_6",
    "seated_players_6",
    "hero_position_offset_5",
    "pot_bb",
    "current_bet_bb",
    "call_amount_bb",
    "min_bet_bb",
    "min_raise_to_bb",
    "hero_stack_bb",
    "hero_current_bet_bb",
    "max_opponent_stack_bb",
    "chosen_amount_bb",
    "amount_ratio_pot",
    "amount_ratio_stack",
    "preflop_score_100",
    "made_hand_rank_8",
    "board_wet",
    "board_paired",
    "board_high",
    "top_pair_or_better",
    "facing_bet",
    "voluntary",
    "covered_by_larger_stack",
)


def _slug(value):
    return str(value).strip().lower().replace("/", "_").replace(" ", "_")


def _category_feature_names(prefix, values):
    return tuple(f"{prefix}_{_slug(value)}" for value in values)


FEATURE_NAMES = (
    NUMERIC_FEATURE_NAMES
    + _category_feature_names("street", STREETS)
    + _category_feature_names("position", POSITIONS)
    + _category_feature_names("bucket", HAND_BUCKETS)
    + _category_feature_names("style", TABLE_STYLES)
    + _category_feature_names("available", ACTIONS)
    + _category_feature_names("action", ACTIONS)
)


@dataclass(frozen=True)
class FeatureVector:
    names: tuple[str, ...]
    values: tuple[float, ...]

    def as_dict(self):
        return dict(zip(self.names, self.values, strict=True))


def _lookup(mapping, key, default=None):
    if mapping is None:
        return default
    if hasattr(mapping, "get"):
        value = mapping.get(key, default)
    else:
        try:
            value = mapping[key]
        except (IndexError, KeyError, TypeError):
            value = default
    return default if value is None else value


def _safe_float(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _scaled(value, scale):
    return _safe_float(value) / scale if scale else 0.0


def _clamped_ratio(value, scale):
    return max(0.0, min(1.0, _scaled(value, scale)))


def _flag(value):
    return 1.0 if bool(_safe_int(value)) else 0.0


def _category(value, expected):
    return 1.0 if str(value or "").casefold() == str(expected).casefold() else 0.0


def _parse_actions(value):
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip().lower() for part in value.split(",") if part.strip()}
    return {str(part).strip().lower() for part in value if str(part).strip()}


def _amount_ratio(amount, denominator):
    amount = _safe_float(amount)
    denominator = _safe_float(denominator)
    if denominator <= 0:
        return 0.0
    return amount / denominator


def _active_players(table):
    return sum(
        1
        for seat in table.get("seats", [])
        if not seat.get("folded", False) and not seat.get("hasFolded", False)
    )


def _seated_numbers(table):
    numbers = []
    for seat in table.get("seats", []):
        seat_number = _safe_int(seat.get("seatNumber"), default=None)
        if seat_number is not None:
            numbers.append(seat_number)
    return sorted(set(numbers))


def _button_seat_number(table):
    value = table.get("buttonSeatNumber")
    if value is not None:
        return _safe_int(value, default=None)
    for key in (
        "dealerButtonSeatNumber",
        "dealerSeatNumber",
        "buttonSeat",
        "dealerSeat",
        "button",
    ):
        value = table.get(key)
        if isinstance(value, dict):
            value = value.get("seatNumber")
        seat_number = _safe_int(value, default=None)
        if seat_number is not None:
            return seat_number
    return None


def _position_label(player_count, offset):
    labels_by_count = {
        2: ["BTN/SB", "BB"],
        3: ["BTN", "SB", "BB"],
        4: ["BTN", "SB", "BB", "CO"],
        5: ["BTN", "SB", "BB", "UTG", "CO"],
        6: ["BTN", "SB", "BB", "UTG", "HJ", "CO"],
    }
    labels = labels_by_count.get(player_count)
    if labels is None:
        if offset == 0:
            return "BTN"
        if offset == 1:
            return "SB"
        if offset == 2:
            return "BB"
        return f"POS{offset}"
    if offset < len(labels):
        return labels[offset]
    return None


def _hero_position(table, seat):
    seats = _seated_numbers(table)
    button = _button_seat_number(table)
    hero_seat = _safe_int((seat or {}).get("seatNumber"), default=None)
    if not seats or button is None or hero_seat is None:
        return None, None, len(seats)
    if button not in seats or hero_seat not in seats:
        return None, None, len(seats)

    button_index = seats.index(button)
    ordered = seats[button_index:] + seats[:button_index]
    offset = ordered.index(hero_seat)
    return _position_label(len(ordered), offset), offset, len(ordered)


def _table_style(table, seat):
    try:
        from poker_bot.strategies import survival_lookup

        return survival_lookup.table_style(table, seat)
    except Exception:
        return "unknown"


def _hand_bucket(hole_cards, board_cards):
    if not board_cards:
        score = preflop_score(hole_cards)
        if score >= 80:
            return "strong"
        if score >= 50:
            return "medium"
        return "air"
    rank = made_hand_rank(hole_cards, board_cards)
    if rank >= 2:
        return "strong"
    if rank == 1 or has_top_pair_or_better(hole_cards, board_cards):
        return "medium"
    return "air"


def state_action_mapping(table, seat, action, amount=None, voluntary=None):
    """Return telemetry-shaped fields for a live table state and candidate action."""
    table = table or {}
    seat = seat or {}
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    hole_cards = seat.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    pot = _safe_int(table.get("potChips"))
    stack = _safe_int(seat.get("stackChips"))
    call_amount = _safe_int(allowed.get("callAmount") or allowed.get("callChips"))
    amount_value = _safe_int(amount, default=None)
    texture = board_texture(board_cards) if board_cards else {}
    rank = made_hand_rank(hole_cards, board_cards) if board_cards else 0
    top_pair = has_top_pair_or_better(hole_cards, board_cards)
    opponent_stacks = [
        _safe_int(other.get("stackChips"))
        for other in table.get("seats", [])
        if other.get("agentId") != seat.get("agentId")
        and not other.get("folded", False)
        and not other.get("hasFolded", False)
    ]
    max_opponent_stack = max(opponent_stacks, default=0)
    hero_position, hero_position_offset, seated_players = _hero_position(table, seat)
    action_text = str(action or "").lower()
    if voluntary is None:
        voluntary = action_text in {"call", "bet", "raise"}

    return {
        "street": table.get("street"),
        "hero_position": hero_position,
        "hero_position_offset": hero_position_offset,
        "seated_players": seated_players,
        "active_players": _active_players(table),
        "table_style": _table_style(table, seat),
        "pot_chips": pot,
        "current_bet": table.get("currentBet"),
        "call_amount": call_amount,
        "min_bet": allowed.get("minBet"),
        "min_raise_to": allowed.get("minRaiseTo"),
        "hero_stack": stack,
        "hero_current_bet": seat.get("currentBetChips"),
        "max_opponent_stack": max_opponent_stack,
        "covered_by_larger_stack": int(max_opponent_stack > stack),
        "preflop_score": preflop_score(hole_cards),
        "made_hand_rank": rank,
        "hand_bucket": _hand_bucket(hole_cards, board_cards),
        "board_wet": int(texture.get("wet", False)),
        "board_paired": int(texture.get("paired", False)),
        "board_high": int(texture.get("high", False)),
        "top_pair_or_better": int(top_pair),
        "available_actions": ",".join(available),
        "chosen_action": action_text,
        "chosen_amount": amount_value,
        "amount_ratio_pot": _amount_ratio(amount_value, pot),
        "amount_ratio_stack": _amount_ratio(amount_value, stack),
        "facing_bet": int(call_amount > 0),
        "voluntary": int(voluntary),
    }


def _numeric_values(mapping):
    return {
        "active_players_6": _clamped_ratio(_lookup(mapping, "active_players"), 6),
        "seated_players_6": _clamped_ratio(_lookup(mapping, "seated_players"), 6),
        "hero_position_offset_5": _clamped_ratio(
            _lookup(mapping, "hero_position_offset"), 5
        ),
        "pot_bb": _scaled(_lookup(mapping, "pot_chips"), CHIP_SCALE),
        "current_bet_bb": _scaled(_lookup(mapping, "current_bet"), CHIP_SCALE),
        "call_amount_bb": _scaled(_lookup(mapping, "call_amount"), CHIP_SCALE),
        "min_bet_bb": _scaled(_lookup(mapping, "min_bet"), CHIP_SCALE),
        "min_raise_to_bb": _scaled(_lookup(mapping, "min_raise_to"), CHIP_SCALE),
        "hero_stack_bb": _scaled(_lookup(mapping, "hero_stack"), CHIP_SCALE),
        "hero_current_bet_bb": _scaled(
            _lookup(mapping, "hero_current_bet"), CHIP_SCALE
        ),
        "max_opponent_stack_bb": _scaled(
            _lookup(mapping, "max_opponent_stack"), CHIP_SCALE
        ),
        "chosen_amount_bb": _scaled(_lookup(mapping, "chosen_amount"), CHIP_SCALE),
        "amount_ratio_pot": _safe_float(_lookup(mapping, "amount_ratio_pot")),
        "amount_ratio_stack": _safe_float(_lookup(mapping, "amount_ratio_stack")),
        "preflop_score_100": _scaled(_lookup(mapping, "preflop_score"), 100),
        "made_hand_rank_8": _scaled(_lookup(mapping, "made_hand_rank"), 8),
        "board_wet": _flag(_lookup(mapping, "board_wet")),
        "board_paired": _flag(_lookup(mapping, "board_paired")),
        "board_high": _flag(_lookup(mapping, "board_high")),
        "top_pair_or_better": _flag(_lookup(mapping, "top_pair_or_better")),
        "facing_bet": _flag(_lookup(mapping, "facing_bet")),
        "voluntary": _flag(_lookup(mapping, "voluntary")),
        "covered_by_larger_stack": _flag(
            _lookup(mapping, "covered_by_larger_stack")
        ),
    }


def encode_mapping(mapping):
    """Encode a telemetry-shaped mapping into a stable numeric feature vector."""
    numeric = _numeric_values(mapping)
    values = [numeric[name] for name in NUMERIC_FEATURE_NAMES]

    street = _lookup(mapping, "street")
    values.extend(_category(street, expected) for expected in STREETS)

    position = _lookup(mapping, "hero_position")
    values.extend(_category(position, expected) for expected in POSITIONS)

    bucket = _lookup(mapping, "hand_bucket")
    values.extend(_category(bucket, expected) for expected in HAND_BUCKETS)

    style = _lookup(mapping, "table_style", "unknown")
    values.extend(_category(style, expected) for expected in TABLE_STYLES)

    available = _parse_actions(_lookup(mapping, "available_actions"))
    values.extend(1.0 if action in available else 0.0 for action in ACTIONS)

    chosen = str(_lookup(mapping, "chosen_action", "")).lower()
    values.extend(1.0 if chosen == action else 0.0 for action in ACTIONS)

    return FeatureVector(FEATURE_NAMES, tuple(float(value) for value in values))


def encode_state_action(table, seat, action, amount=None, voluntary=None):
    mapping = state_action_mapping(
        table,
        seat,
        action,
        amount=amount,
        voluntary=voluntary,
    )
    return encode_mapping(mapping)
