import simulator


def make_seat(agent_id, stack=1000, current_bet=0, seat_number=1):
    return {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": ["AS", "KD"],
        "stackChips": stack,
        "currentBetChips": current_bet,
    }


def test_short_stack_cannot_raise_without_enough_for_minimum_raise():
    seat = make_seat(
        simulator.PLAYER_AGENT_ID,
        stack=74,
        current_bet=simulator.SMALL_BLIND,
    )

    allowed = simulator.build_allowed_actions(seat, current_bet=simulator.BIG_BLIND)

    assert allowed["callAmount"] == simulator.SMALL_BLIND
    assert "call" in allowed["availableActions"]
    assert "raise" not in allowed["availableActions"]
    assert allowed["minRaiseTo"] is None
    assert allowed["maxCommit"] == 99


def test_raise_is_allowed_when_stack_can_reach_minimum_raise_to_amount():
    seat = make_seat(
        simulator.PLAYER_AGENT_ID,
        stack=75,
        current_bet=simulator.SMALL_BLIND,
    )

    allowed = simulator.build_allowed_actions(seat, current_bet=simulator.BIG_BLIND)

    assert "raise" in allowed["availableActions"]
    assert allowed["minRaiseTo"] == 100
    assert allowed["maxCommit"] == 100


def test_custom_min_raise_controls_next_reraise_size():
    seat = make_seat(simulator.BOT_AGENT_ID, stack=200, current_bet=50)

    allowed = simulator.build_allowed_actions(seat, current_bet=150, min_raise=100)

    assert allowed["callAmount"] == 100
    assert allowed["minRaiseTo"] == 250
    assert "raise" in allowed["availableActions"]


def test_resolve_raise_coerces_below_minimum_target_to_min_raise_to():
    seat = make_seat(simulator.PLAYER_AGENT_ID, stack=500, current_bet=50)

    current_bet = simulator.resolve_action(
        seat,
        action="raise",
        amount=200,
        current_bet=150,
        min_raise_to=250,
    )

    assert current_bet == 250
    assert seat["currentBetChips"] == 250
    assert seat["stackChips"] == 300


def test_preflop_call_then_check_moves_only_live_bets_to_pot(monkeypatch):
    player = make_seat(
        simulator.PLAYER_AGENT_ID,
        stack=1975,
        current_bet=simulator.SMALL_BLIND,
        seat_number=1,
    )
    bot = make_seat(
        simulator.BOT_AGENT_ID,
        stack=1950,
        current_bet=simulator.BIG_BLIND,
        seat_number=2,
    )

    monkeypatch.setattr(simulator, "print_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator, "prompt_user_action", lambda allowed, call: "call")
    monkeypatch.setattr(
        simulator,
        "choose_action",
        lambda table, agent_id: ("check", None, "test check"),
    )

    fold_winner, pot = simulator.run_betting_round(
        [player, bot],
        board=[],
        pot=0,
        current_bet=simulator.BIG_BLIND,
        street="Preflop",
        first_actor_idx=0,
    )

    assert fold_winner is None
    assert pot == 100
    assert player["stackChips"] == 1950
    assert bot["stackChips"] == 1950
    assert player["currentBetChips"] == 0
    assert bot["currentBetChips"] == 0


def test_raise_to_then_call_settles_correct_pot(monkeypatch):
    player = make_seat(
        simulator.PLAYER_AGENT_ID,
        stack=1975,
        current_bet=simulator.SMALL_BLIND,
        seat_number=1,
    )
    bot = make_seat(
        simulator.BOT_AGENT_ID,
        stack=1950,
        current_bet=simulator.BIG_BLIND,
        seat_number=2,
    )
    bot_tables = []

    monkeypatch.setattr(simulator, "print_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator, "prompt_user_action", lambda allowed, call: "raise")
    monkeypatch.setattr("builtins.input", lambda prompt: "150")

    def bot_action(table, agent_id):
        bot_tables.append(table)
        return "call", table["allowedActions"]["callAmount"], "test call"

    monkeypatch.setattr(simulator, "choose_action", bot_action)

    fold_winner, pot = simulator.run_betting_round(
        [player, bot],
        board=[],
        pot=0,
        current_bet=simulator.BIG_BLIND,
        street="Preflop",
        first_actor_idx=0,
    )

    assert fold_winner is None
    assert pot == 300
    assert player["stackChips"] == 1850
    assert bot["stackChips"] == 1850
    assert bot_tables[0]["allowedActions"]["callAmount"] == 100
    assert bot_tables[0]["allowedActions"]["minRaiseTo"] == 250


def test_zero_raise_amount_returns_to_available_actions(monkeypatch):
    allowed = {
        "availableActions": ["fold", "call", "raise"],
        "callAmount": 25,
        "minRaiseTo": 100,
        "minBet": 50,
    }
    action_choices = iter(["raise", "call"])
    amount_entries = iter(["0"])

    monkeypatch.setattr(
        simulator,
        "prompt_user_action",
        lambda available, call_amount: next(action_choices),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(amount_entries))

    action, amount = simulator.prompt_player_action(allowed)

    assert action == "call"
    assert amount is None


def test_raise_cancel_then_call_resolves_call_in_betting_round(monkeypatch):
    player = make_seat(
        simulator.PLAYER_AGENT_ID,
        stack=1975,
        current_bet=simulator.SMALL_BLIND,
        seat_number=1,
    )
    bot = make_seat(
        simulator.BOT_AGENT_ID,
        stack=1950,
        current_bet=simulator.BIG_BLIND,
        seat_number=2,
    )
    action_choices = iter(["raise", "call"])
    amount_entries = iter(["0"])

    monkeypatch.setattr(simulator, "print_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        simulator,
        "prompt_user_action",
        lambda available, call_amount: next(action_choices),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(amount_entries))
    monkeypatch.setattr(
        simulator,
        "choose_action",
        lambda table, agent_id: ("check", None, "test check"),
    )

    fold_winner, pot = simulator.run_betting_round(
        [player, bot],
        board=[],
        pot=0,
        current_bet=simulator.BIG_BLIND,
        street="Preflop",
        first_actor_idx=0,
    )

    assert fold_winner is None
    assert pot == 100
    assert player["stackChips"] == 1950
    assert bot["stackChips"] == 1950


def test_zero_stack_player_facing_bet_has_no_call_action():
    seat = make_seat(simulator.PLAYER_AGENT_ID, stack=0, current_bet=250)

    allowed = simulator.build_allowed_actions(seat, current_bet=950)

    assert allowed["availableActions"] == []
    assert allowed["callAmount"] == 0
    assert allowed["minRaiseTo"] is None


def test_short_all_in_call_amount_is_capped_to_remaining_stack():
    seat = make_seat(simulator.PLAYER_AGENT_ID, stack=300, current_bet=250)

    allowed = simulator.build_allowed_actions(seat, current_bet=950)

    assert allowed["availableActions"] == ["fold", "call"]
    assert allowed["callAmount"] == 300


def test_all_in_unequal_bets_end_round_and_return_uncalled_excess(monkeypatch):
    player = make_seat(
        simulator.PLAYER_AGENT_ID,
        stack=0,
        current_bet=250,
        seat_number=1,
    )
    bot = make_seat(
        simulator.BOT_AGENT_ID,
        stack=0,
        current_bet=950,
        seat_number=2,
    )

    monkeypatch.setattr(
        simulator,
        "prompt_user_action",
        lambda allowed, call: (_ for _ in ()).throw(AssertionError("prompted")),
    )
    monkeypatch.setattr(
        simulator,
        "choose_action",
        lambda table, agent_id: (_ for _ in ()).throw(AssertionError("bot acted")),
    )

    fold_winner, pot = simulator.run_betting_round(
        [player, bot],
        board=["4S", "6D", "9C", "KC", "2D"],
        pot=2800,
        current_bet=950,
        street="River",
        first_actor_idx=0,
    )

    assert fold_winner is None
    assert pot == 3300
    assert player["stackChips"] == 0
    assert bot["stackChips"] == 700
    assert player["currentBetChips"] == 0
    assert bot["currentBetChips"] == 0


def test_short_all_in_call_closes_betting_round(monkeypatch):
    player = make_seat(
        simulator.PLAYER_AGENT_ID,
        stack=300,
        current_bet=250,
        seat_number=1,
    )
    bot = make_seat(
        simulator.BOT_AGENT_ID,
        stack=0,
        current_bet=950,
        seat_number=2,
    )

    monkeypatch.setattr(simulator, "print_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator, "prompt_user_action", lambda allowed, call: "call")
    monkeypatch.setattr(
        simulator,
        "choose_action",
        lambda table, agent_id: (_ for _ in ()).throw(AssertionError("bot acted")),
    )

    fold_winner, pot = simulator.run_betting_round(
        [player, bot],
        board=["4S", "6D", "9C", "KC", "2D"],
        pot=2800,
        current_bet=950,
        street="River",
        first_actor_idx=0,
    )

    assert fold_winner is None
    assert pot == 3900
    assert player["stackChips"] == 0
    assert bot["stackChips"] == 400
    assert player["currentBetChips"] == 0
    assert bot["currentBetChips"] == 0


def test_play_hand_posts_player_small_blind_by_default(monkeypatch):
    rounds = []

    def capture_round(seats, board, pot, current_bet, street, first_actor_idx):
        rounds.append(
            {
                "seats": [seat.copy() for seat in seats],
                "current_bet": current_bet,
                "street": street,
                "first_actor_idx": first_actor_idx,
            }
        )
        return simulator.PLAYER_AGENT_ID, pot + sum(
            seat["currentBetChips"] for seat in seats
        )

    monkeypatch.setattr(simulator, "run_betting_round", capture_round)

    player_stack, bot_stack = simulator.play_hand(2000, 2000)

    preflop = rounds[0]
    player, bot = preflop["seats"]
    assert player["currentBetChips"] == simulator.SMALL_BLIND
    assert bot["currentBetChips"] == simulator.BIG_BLIND
    assert player["stackChips"] == 1975
    assert bot["stackChips"] == 1950
    assert preflop["current_bet"] == simulator.BIG_BLIND
    assert preflop["first_actor_idx"] == 0
    assert player_stack == 2050
    assert bot_stack == 1950


def test_play_hand_can_post_bot_small_blind(monkeypatch):
    rounds = []

    def capture_round(seats, board, pot, current_bet, street, first_actor_idx):
        rounds.append(
            {
                "seats": [seat.copy() for seat in seats],
                "current_bet": current_bet,
                "street": street,
                "first_actor_idx": first_actor_idx,
            }
        )
        return simulator.BOT_AGENT_ID, pot + sum(
            seat["currentBetChips"] for seat in seats
        )

    monkeypatch.setattr(simulator, "run_betting_round", capture_round)

    player_stack, bot_stack = simulator.play_hand(
        2000, 2000, player_is_small_blind=False
    )

    preflop = rounds[0]
    player, bot = preflop["seats"]
    assert player["currentBetChips"] == simulator.BIG_BLIND
    assert bot["currentBetChips"] == simulator.SMALL_BLIND
    assert player["stackChips"] == 1950
    assert bot["stackChips"] == 1975
    assert preflop["current_bet"] == simulator.BIG_BLIND
    assert preflop["first_actor_idx"] == 1
    assert player_stack == 1950
    assert bot_stack == 2050


def test_player_stack_increases_when_bot_folds_preflop(monkeypatch):
    monkeypatch.setattr(simulator, "print_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator, "prompt_user_action", lambda allowed, call: "raise")
    monkeypatch.setattr("builtins.input", lambda prompt: "150")
    monkeypatch.setattr(
        simulator,
        "choose_action",
        lambda table, agent_id: ("fold", None, "test fold"),
    )

    player_stack, bot_stack = simulator.play_hand(2000, 2000)

    assert player_stack == 2050
    assert bot_stack == 1950


def test_bot_stack_increases_when_player_folds_preflop(monkeypatch):
    monkeypatch.setattr(simulator, "print_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator, "prompt_user_action", lambda allowed, call: "fold")

    player_stack, bot_stack = simulator.play_hand(2000, 2000)

    assert player_stack == 1975
    assert bot_stack == 2025


def test_main_alternates_blinds_between_hands(monkeypatch):
    blind_roles = []

    def fake_play_hand(player_stack, bot_stack, player_is_small_blind=True):
        blind_roles.append(player_is_small_blind)
        if len(blind_roles) == 1:
            return player_stack, bot_stack
        return 0, bot_stack

    monkeypatch.setattr(simulator, "play_hand", fake_play_hand)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    simulator.main()

    assert blind_roles == [True, False]
