from poker_bot.range_model import (
    HandRange,
    apply_action_update,
    class_strength,
    combo_class,
    combos_for_class,
    default_preflop_range,
    estimate_action_range,
    position_label,
    remove_blockers,
)
from poker_bot.range_model.hand_range import all_starting_combos, normalize_combo


def make_seat(agent_id="hero", seat_number=6, folded=False):
    seat = {"agentId": agent_id, "seatNumber": seat_number}
    if folded:
        seat["folded"] = True
    return seat


def make_table():
    return {
        "buttonSeatNumber": 1,
        "seats": [
            make_seat("button", 1),
            make_seat("sb", 2),
            make_seat("bb", 3),
            make_seat("utg", 4),
            make_seat("mp", 5),
            make_seat("hero", 6),
        ],
    }


def test_combo_generation_counts_holdem_starting_combos():
    assert len(all_starting_combos()) == 1326
    assert len(combos_for_class("AA")) == 6
    assert len(combos_for_class("AKs")) == 4
    assert len(combos_for_class("AKo")) == 12


def test_combo_class_canonicalizes_order_and_suitedness():
    assert combo_class(["KS", "AS"]) == "AKs"
    assert combo_class(["AD", "KC"]) == "AKo"
    assert combo_class(["2D", "2C"]) == "22"
    assert normalize_combo(["ks", "as"]) == ("AS", "KS")


def test_hand_range_removes_blocked_cards():
    hand_range = HandRange.from_classes(["AA", "AKs"])
    blocked = remove_blockers(hand_range, ["AS"])

    assert all("AS" not in combo for combo in blocked.weights)
    assert blocked.total_weight() < hand_range.total_weight()
    assert blocked.probability_of_class("AKs") > 0


def test_position_label_uses_button_order():
    table = make_table()

    assert position_label(table, table["seats"][0]) == "BTN"
    assert position_label(table, table["seats"][2]) == "BB"
    assert position_label(table, table["seats"][5]) == "CO"


def test_default_preflop_range_changes_by_position_and_situation():
    button_open = default_preflop_range("BTN", "open")
    utg_open = default_preflop_range("UTG", "open")
    bb_defend = default_preflop_range("BB", "defend")

    assert button_open.total_weight() > utg_open.total_weight()
    assert bb_defend.total_weight() > utg_open.total_weight()
    assert button_open.probability_of_class("76s") > 0
    assert utg_open.probability_of_class("76s") == 0


def test_raise_update_increases_premium_probability():
    hand_range = default_preflop_range("BTN", "open")
    before = hand_range.probability_of_class("AA")

    updated = apply_action_update(hand_range, "raise", amount=300, pot=200)

    assert updated.probability_of_class("AA") > before
    assert updated.probability_of_class("AKs") > hand_range.probability_of_class(
        "AKs"
    )


def test_call_update_keeps_middle_range_more_than_bottom_range():
    hand_range = default_preflop_range("BTN", "open")
    updated = apply_action_update(hand_range, "call", amount=50, pot=250)

    assert updated.probability_of_class("KQs") > updated.probability_of_class("76s")


def test_estimate_action_range_applies_blockers_and_action_update():
    estimated = estimate_action_range(
        position="BTN",
        situation="open",
        action="raise",
        known_cards=["AS", "KH", "7D"],
        amount=300,
        pot=200,
    )

    assert estimated.total_weight() > 0
    assert all("AS" not in combo and "KH" not in combo for combo in estimated.weights)
    assert estimated.probability_of_class("AA") > estimated.probability_of_class("76s")


def test_class_strength_orders_premiums_above_trash():
    assert class_strength("AA") > class_strength("AKs")
    assert class_strength("AKs") > class_strength("72o")
