from poker_bot.neural.features import (
    ACTIONS,
    FEATURE_NAMES,
    encode_mapping,
    encode_state_action,
)


def make_seat():
    return {
        "agentId": "hero",
        "seatNumber": 1,
        "holeCards": ["AS", "KS"],
        "stackChips": 1800,
        "currentBetChips": 50,
    }


def make_table(seat):
    return {
        "street": "Flop",
        "boardCards": ["QS", "JS", "2C"],
        "potChips": 300,
        "currentBet": 100,
        "buttonSeatNumber": 1,
        "allowedActions": {
            "availableActions": ["fold", "call", "raise"],
            "callAmount": 100,
            "callChips": 100,
            "minBet": 50,
            "minRaiseTo": 250,
            "maxCommit": 1800,
        },
        "seats": [
            seat,
            {
                "agentId": "villain",
                "seatNumber": 2,
                "holeCards": [],
                "stackChips": 2200,
                "currentBetChips": 100,
            },
        ],
    }


def test_encode_mapping_has_stable_schema_and_action_flags():
    row = {
        "street": "Preflop",
        "hero_position": "BTN/SB",
        "hero_position_offset": 0,
        "seated_players": 2,
        "active_players": 2,
        "table_style": "short_handed",
        "pot_chips": 75,
        "current_bet": 50,
        "call_amount": 50,
        "min_bet": 50,
        "min_raise_to": 150,
        "hero_stack": 1800,
        "hero_current_bet": 25,
        "max_opponent_stack": 2000,
        "chosen_amount": 150,
        "amount_ratio_pot": 2.0,
        "amount_ratio_stack": 150 / 1800,
        "preflop_score": 78,
        "made_hand_rank": 0,
        "hand_bucket": "medium",
        "board_wet": 0,
        "board_paired": 0,
        "board_high": 0,
        "top_pair_or_better": 0,
        "facing_bet": 1,
        "voluntary": 1,
        "covered_by_larger_stack": 1,
        "available_actions": "fold,call,raise",
        "chosen_action": "raise",
    }

    vector = encode_mapping(row)
    data = vector.as_dict()

    assert vector.names == FEATURE_NAMES
    assert len(vector.values) == len(FEATURE_NAMES)
    assert data["street_preflop"] == 1.0
    assert data["position_btn_sb"] == 1.0
    assert data["available_raise"] == 1.0
    assert data["available_bet"] == 0.0
    assert data["action_raise"] == 1.0
    assert data["action_call"] == 0.0


def test_encode_state_action_matches_live_table_shape():
    seat = make_seat()
    table = make_table(seat)

    vector = encode_state_action(table, seat, "raise", amount=250)
    data = vector.as_dict()

    assert len(vector.values) == len(FEATURE_NAMES)
    assert data["street_flop"] == 1.0
    assert data["action_raise"] == 1.0
    assert data["chosen_amount_bb"] == 5.0
    assert data["facing_bet"] == 1.0
    assert data["covered_by_larger_stack"] == 1.0
    for action in ACTIONS:
        assert data[f"available_{action}"] in {0.0, 1.0}
