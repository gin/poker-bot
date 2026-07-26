import io
import re
from contextlib import redirect_stdout

import simulator


def make_seat(agent_id, stack=1000, current_bet=0, seat_number=1):
    return {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": ["AS", "KD"],
        "stackChips": stack,
        "currentBetChips": current_bet,
    }


def _visible_line(line):
    return re.sub(r"\033\[[0-9;]*m", "", line)


def test_print_table_uses_consistent_borders_and_dynamic_width():
    player = make_seat(
        simulator.PLAYER_AGENT_ID,
        stack=990,
        current_bet=0,
        seat_number=1,
    )
    player["holeCards"] = ["3C", "AH"]
    bot = make_seat(
        simulator.BOT_AGENT_ID,
        stack=1900,
        current_bet=50,
        seat_number=2,
    )

    output = io.StringIO()
    with redirect_stdout(output):
        simulator.print_table(
            [player, bot],
            board=["AS", "3D", "6S"],
            pot=15,
            street="Flop",
            current_bet=10,
            active_id=simulator.PLAYER_AGENT_ID,
        )

    table_lines = [
        _visible_line(line).rstrip() for line in output.getvalue().splitlines()[4:9]
    ]
    assert all(
        line.startswith("  │") and line.endswith("│") for line in table_lines[:4]
    )
    assert table_lines[4].startswith("  └") and table_lines[4].endswith("┘")
    assert "║" not in "\n".join(table_lines)
    assert (
        len(table_lines[0])
        == len(table_lines[1])
        == len(table_lines[2])
        == len(table_lines[3])
        == len(table_lines[4])
    )
    assert "Pot: $15   Bet: $10" in table_lines[0]
    assert "cards=3♣ A♥  ◀" in table_lines[2]


def test_simulator_parser_defaults_to_simple_strategy():
    args = simulator.build_parser().parse_args([])

    assert args.strat == "simple"


def test_simulator_parser_accepts_strategy_flag():
    args = simulator.build_parser().parse_args(["--strat", "all_in_everytime"])

    assert args.strat == "all_in_everytime"


def test_simulator_main_uses_selected_bot_strategy(monkeypatch):
    selected_strategy = object()
    captured = {}

    def fake_load_strategy(name):
        captured["strategy_name"] = name
        return selected_strategy

    def fake_play_hand(
        player_stack,
        bot_stack,
        player_is_small_blind=True,
        player_strategy=None,
        bot_strategy=None,
        rng=None,
        verbose=True,
    ):
        captured["bot_strategy"] = bot_strategy
        return 0, bot_stack

    monkeypatch.setattr(simulator, "load_strategy", fake_load_strategy)
    monkeypatch.setattr(simulator, "play_hand", fake_play_hand)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    simulator.main(["--strat", "all_in_everytime"])

    assert captured["strategy_name"] == "all_in_everytime"
    assert captured["bot_strategy"] is selected_strategy


def test_short_stack_cannot_raise_without_enough_for_minimum_raise():
    seat = make_seat(
        simulator.PLAYER_AGENT_ID,
        stack=5,
        current_bet=simulator.SMALL_BLIND,
    )

    allowed = simulator.build_allowed_actions(seat, current_bet=simulator.BIG_BLIND)

    assert allowed["callAmount"] == simulator.SMALL_BLIND
    assert "call" in allowed["availableActions"]
    assert "raise" not in allowed["availableActions"]
    assert allowed["minRaiseTo"] is None
    assert allowed["maxCommit"] == 10


def test_raise_is_allowed_when_stack_can_reach_minimum_raise_to_amount():
    seat = make_seat(
        simulator.PLAYER_AGENT_ID,
        stack=15,
        current_bet=simulator.SMALL_BLIND,
    )

    allowed = simulator.build_allowed_actions(seat, current_bet=simulator.BIG_BLIND)

    assert allowed["availableActions"] == ["fold", "call", "raise", "all-in"]
    assert allowed["minRaiseTo"] == 20
    assert allowed["maxCommit"] == 20


def test_custom_min_raise_controls_next_reraise_size():
    seat = make_seat(simulator.BOT_AGENT_ID, stack=200, current_bet=50)

    allowed = simulator.build_allowed_actions(seat, current_bet=150, min_raise=100)

    assert allowed["callAmount"] == 100
    assert allowed["minRaiseTo"] == 250
    assert "raise" in allowed["availableActions"]


def test_short_all_in_does_not_lower_min_raise():
    """NLHE rule: a short all-in (< min raise increment) does NOT reduce
    the minimum raise for subsequent actions."""
    # Player A bet BB ($10). Player B goes all-in for $15 (only $5 more,
    # less than the BB raise of $10). The min_raise should still be $10,
    # so the next raise minimum = $10 + $10 = $20, not $10 + $5 = $15.
    seat = make_seat(simulator.PLAYER_AGENT_ID, stack=2000, current_bet=5)

    allowed = simulator.build_allowed_actions(seat, current_bet=10, min_raise=10)

    assert allowed["minRaiseTo"] == 20, (
        f"Expected min raise to 20 from bet=10 + min_raise=10, "
        f"got {allowed['minRaiseTo']}"
    )
    assert "raise" in allowed["availableActions"]


def test_full_raise_updates_min_raise():
    """A proper full raise (≥ min raise increment) becomes the new min raise
    for the next player."""
    # Seat committed 50, then opponent raised from 100 to 200 (inc 100).
    # The original raiser now faces a re-raise opportunity with min_raise=100.
    # Min re-raise to = 200 + 100 = 300.
    seat = make_seat(simulator.BOT_AGENT_ID, stack=2000, current_bet=50)

    allowed = simulator.build_allowed_actions(seat, current_bet=200, min_raise=100)

    assert allowed["minRaiseTo"] == 300
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
        stack=995,
        current_bet=simulator.SMALL_BLIND,
        seat_number=1,
    )
    bot = make_seat(
        simulator.BOT_AGENT_ID,
        stack=990,
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
    assert pot == 20
    assert player["stackChips"] == 990
    assert bot["stackChips"] == 990
    assert player["currentBetChips"] == 0
    assert bot["currentBetChips"] == 0


def test_raise_to_then_call_settles_correct_pot(monkeypatch):
    player = make_seat(
        simulator.PLAYER_AGENT_ID,
        stack=995,
        current_bet=simulator.SMALL_BLIND,
        seat_number=1,
    )
    bot = make_seat(
        simulator.BOT_AGENT_ID,
        stack=990,
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
    assert player["stackChips"] == 850
    assert bot["stackChips"] == 850
    assert bot_tables[0]["allowedActions"]["callAmount"] == 140
    assert bot_tables[0]["allowedActions"]["minRaiseTo"] == 290


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
        stack=995,
        current_bet=simulator.SMALL_BLIND,
        seat_number=1,
    )
    bot = make_seat(
        simulator.BOT_AGENT_ID,
        stack=990,
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
    assert pot == 20
    assert player["stackChips"] == 990
    assert bot["stackChips"] == 990


def test_zero_stack_player_facing_bet_has_no_call_action():
    seat = make_seat(simulator.PLAYER_AGENT_ID, stack=0, current_bet=250)

    allowed = simulator.build_allowed_actions(seat, current_bet=950)

    assert allowed["availableActions"] == []
    assert allowed["callAmount"] == 0
    assert allowed["minRaiseTo"] is None


def test_short_all_in_call_amount_is_capped_to_remaining_stack():
    seat = make_seat(simulator.PLAYER_AGENT_ID, stack=300, current_bet=250)

    allowed = simulator.build_allowed_actions(seat, current_bet=950)

    assert allowed["availableActions"] == ["fold", "call", "all-in"]
    assert allowed["callAmount"] == 300


def test_all_in_option_is_available_and_commits_stack():
    seat = make_seat(simulator.PLAYER_AGENT_ID, stack=100, current_bet=0)

    allowed = simulator.build_allowed_actions(seat, current_bet=0)

    assert allowed["availableActions"] == ["fold", "check", "bet", "all-in"]
    assert allowed["allInToAmount"] == 100

    current_bet = simulator.resolve_action(
        seat, "all-in", None, 0, allowed["minRaiseTo"]
    )

    assert current_bet == 100
    assert seat["currentBetChips"] == 100
    assert seat["stackChips"] == 0


def test_all_in_facing_bet_commits_remaining_stack():
    seat = make_seat(simulator.PLAYER_AGENT_ID, stack=30, current_bet=0)

    allowed = simulator.build_allowed_actions(seat, current_bet=100)

    assert allowed["availableActions"] == ["fold", "call", "all-in"]
    assert allowed["callAmount"] == 30

    current_bet = simulator.resolve_action(
        seat, "all-in", None, 100, allowed["minRaiseTo"]
    )

    assert current_bet == 100
    assert seat["currentBetChips"] == 30
    assert seat["stackChips"] == 0


def test_prompt_player_action_accepts_all_in(monkeypatch):
    monkeypatch.setattr(simulator, "prompt_user_action", lambda allowed, call: "all-in")

    action, amount = simulator.prompt_player_action(
        {
            "availableActions": ["fold", "call", "all-in"],
            "callAmount": 30,
            "allInToAmount": 30,
        }
    )

    assert action == "all-in"
    assert amount is None


def test_player_all_in_requires_bot_to_call_or_fold(monkeypatch):
    player = make_seat(
        simulator.PLAYER_AGENT_ID,
        stack=995,
        current_bet=simulator.SMALL_BLIND,
        seat_number=1,
    )
    bot = make_seat(
        simulator.BOT_AGENT_ID,
        stack=990,
        current_bet=simulator.BIG_BLIND,
        seat_number=2,
    )

    monkeypatch.setattr(simulator, "print_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator, "prompt_user_action", lambda allowed, call: "all-in")
    monkeypatch.setattr(
        simulator,
        "choose_action",
        lambda table, agent_id: (
            "call",
            table["allowedActions"]["callAmount"],
            "test call",
        ),
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
    assert pot == 2000
    assert player["stackChips"] == 0
    assert bot["stackChips"] == 0
    assert player["currentBetChips"] == 0
    assert bot["currentBetChips"] == 0


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


def test_player_cannot_check_when_bot_bets_all_in(monkeypatch):
    player = make_seat(
        simulator.PLAYER_AGENT_ID,
        stack=1000,
        current_bet=0,
        seat_number=1,
    )
    bot = make_seat(
        simulator.BOT_AGENT_ID,
        stack=300,
        current_bet=0,
        seat_number=2,
    )
    player_actions = []

    def bot_all_in(table, agent_id):
        return "bet", table["allowedActions"]["maxCommit"], "test all-in"

    def player_fold(allowed, call_amount):
        player_actions.append((allowed.copy(), call_amount))
        return "fold"

    monkeypatch.setattr(simulator, "print_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator, "prompt_user_action", player_fold)

    fold_winner, pot = simulator.run_betting_round(
        [player, bot],
        board=["4S", "6D", "9C"],
        pot=100,
        current_bet=0,
        street="Flop",
        first_actor_idx=1,
        action_providers={simulator.BOT_AGENT_ID: bot_all_in},
    )

    assert player_actions == [(["fold", "call", "all-in"], 300)]
    assert fold_winner == simulator.BOT_AGENT_ID
    assert pot == 400


def test_play_hand_posts_player_small_blind_by_default(monkeypatch):
    rounds = []

    def capture_round(
        seats,
        board,
        pot,
        current_bet,
        street,
        first_actor_idx,
        action_providers=None,
        verbose=True,
        **kwargs,
    ):
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

    player_stack, bot_stack = simulator.play_hand(1000, 1000)

    preflop = rounds[0]
    player, bot = preflop["seats"]
    assert player["currentBetChips"] == simulator.SMALL_BLIND
    assert bot["currentBetChips"] == simulator.BIG_BLIND
    assert player["stackChips"] == 995
    assert bot["stackChips"] == 990
    assert preflop["current_bet"] == simulator.BIG_BLIND
    assert preflop["first_actor_idx"] == 0
    assert player_stack == 1010
    assert bot_stack == 990


def test_play_hand_can_post_bot_small_blind(monkeypatch):
    rounds = []

    def capture_round(
        seats,
        board,
        pot,
        current_bet,
        street,
        first_actor_idx,
        action_providers=None,
        verbose=True,
        **kwargs,
    ):
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
        1000, 1000, player_is_small_blind=False
    )

    preflop = rounds[0]
    player, bot = preflop["seats"]
    assert player["currentBetChips"] == simulator.BIG_BLIND
    assert bot["currentBetChips"] == simulator.SMALL_BLIND
    assert player["stackChips"] == 990
    assert bot["stackChips"] == 995
    assert preflop["current_bet"] == simulator.BIG_BLIND
    assert preflop["first_actor_idx"] == 1
    assert player_stack == 990
    assert bot_stack == 1010


def test_player_stack_increases_when_bot_folds_preflop(monkeypatch):
    monkeypatch.setattr(simulator, "print_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator, "prompt_user_action", lambda allowed, call: "raise")
    monkeypatch.setattr("builtins.input", lambda prompt: "150")
    monkeypatch.setattr(
        simulator,
        "choose_action",
        lambda table, agent_id: ("fold", None, "test fold"),
    )

    player_stack, bot_stack = simulator.play_hand(1000, 1000)

    assert player_stack == 1010
    assert bot_stack == 990


def test_bot_stack_increases_when_player_folds_preflop(monkeypatch):
    monkeypatch.setattr(simulator, "print_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator, "prompt_user_action", lambda allowed, call: "fold")

    player_stack, bot_stack = simulator.play_hand(1000, 1000)

    assert player_stack == 995
    assert bot_stack == 1005


def test_main_alternates_blinds_between_hands(monkeypatch):
    blind_roles = []

    def fake_play_hand(
        player_stack,
        bot_stack,
        player_is_small_blind=True,
        player_strategy=None,
        bot_strategy=None,
        rng=None,
        verbose=True,
    ):
        blind_roles.append(player_is_small_blind)
        if len(blind_roles) == 1:
            return player_stack, bot_stack
        return 0, bot_stack

    monkeypatch.setattr(simulator, "play_hand", fake_play_hand)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    simulator.main([])

    assert blind_roles == [True, False]


# ── Pot settlement primitive (build_pots / settle_pots) ────────────────────


def test_build_pots_single_layer_when_all_committed_equally():
    seats = [
        {"agentId": "a", "committedChips": 100, "folded": False},
        {"agentId": "b", "committedChips": 100, "folded": False},
        {"agentId": "c", "committedChips": 100, "folded": False},
    ]

    pots = simulator.build_pots(seats)

    assert pots == ((300, ("a", "b", "c")),)


def test_build_pots_splits_main_and_side_pot_for_short_all_in():
    seats = [
        {"agentId": "short", "committedChips": 40, "folded": False},
        {"agentId": "b", "committedChips": 100, "folded": False},
        {"agentId": "c", "committedChips": 100, "folded": False},
    ]

    pots = simulator.build_pots(seats)

    assert pots == (
        (120, ("short", "b", "c")),
        (120, ("b", "c")),
    )
    assert sum(amount for amount, _ in pots) == 240


def test_build_pots_excludes_folded_seats_from_eligibility_but_keeps_their_chips():
    seats = [
        {"agentId": "folded_short", "committedChips": 40, "folded": True},
        {"agentId": "b", "committedChips": 100, "folded": False},
        {"agentId": "c", "committedChips": 100, "folded": False},
    ]

    pots = simulator.build_pots(seats)

    assert pots == (
        (120, ("b", "c")),
        (120, ("b", "c")),
    )
    assert sum(amount for amount, _ in pots) == 240


def test_build_pots_returns_unmatched_single_contributor_layer():
    seats = [
        {"agentId": "a", "committedChips": 100, "folded": False},
        {"agentId": "b", "committedChips": 50, "folded": False},
        {"agentId": "c", "committedChips": 50, "folded": False},
    ]

    pots = simulator.build_pots(seats)

    assert pots == (
        (150, ("a", "b", "c")),
        (50, ("a",)),
    )


def test_settle_pots_awards_short_stack_only_the_main_pot():
    """Regression: a short-stacked showdown winner must not receive chips
    from a side pot they were never eligible for."""
    short = make_seat("short", stack=0, current_bet=0)
    short["committedChips"] = 40
    short["folded"] = False
    short["holeCards"] = ["AS", "AH"]

    mid = make_seat("mid", stack=0, current_bet=0, seat_number=2)
    mid["committedChips"] = 100
    mid["folded"] = False
    mid["holeCards"] = ["7C", "7D"]

    big = make_seat("big", stack=0, current_bet=0, seat_number=3)
    big["committedChips"] = 100
    big["folded"] = False
    big["holeCards"] = ["2C", "2D"]

    board = ["9S", "4H", "3D", "KC", "6S"]

    winnings = simulator.settle_pots([short, mid, big], board)

    # `short` has pocket aces (best hand) but only committed 40, so they can
    # only win the 120-chip main pot, never the 120-chip side pot between
    # mid and big.
    assert winnings["short"] == 120
    assert winnings["mid"] == 120
    assert winnings["big"] == 0
    assert sum(winnings.values()) == 240


def test_settle_pots_splits_ties_and_gives_odd_chip_to_seat_after_button():
    # Three seats commit equally (51 each -> single 153-chip layer). "a"
    # and "b" tie for the best hand; "c" loses outright but still pays
    # into (and stays eligible for) the layer they contributed to. The
    # 51-chip tied share (153 - c's non-winning stake is irrelevant here;
    # only a/b split the pot) is odd, so the remainder chip must land on
    # whichever tied seat sits closest to acting first next hand.
    def make_tied_seat(agent_id, seat_number, hole_cards):
        seat = make_seat(agent_id, stack=0, current_bet=0, seat_number=seat_number)
        seat["committedChips"] = 51
        seat["folded"] = False
        seat["holeCards"] = hole_cards
        return seat

    board = ["9S", "4H", "3D", "KC", "6S"]

    def build_seats():
        return [
            make_tied_seat("a", 1, ["AS", "AH"]),
            make_tied_seat("b", 2, ["AD", "AC"]),
            make_tied_seat("c", 3, ["2C", "2D"]),
        ]

    # Button is seat 3 ("c"): order after the button is a, b, c -- "a" is
    # the first tied seat in that order and gets the odd chip.
    winnings = simulator.settle_pots(build_seats(), board, button_seat_number=3)
    assert winnings["a"] == 77
    assert winnings["b"] == 76
    assert winnings["c"] == 0
    assert sum(winnings.values()) == 153

    # Button is seat 1 ("a"): order after the button is b, c, a -- "b" is
    # now the first tied seat in that order and gets the odd chip instead.
    winnings_flipped = simulator.settle_pots(build_seats(), board, button_seat_number=1)
    assert winnings_flipped["b"] == 77
    assert winnings_flipped["a"] == 76
    assert winnings_flipped["c"] == 0


def test_settle_pots_raises_on_impossible_pot_with_no_eligible_winner():
    only_folded = make_seat("a", stack=0, current_bet=0)
    only_folded["committedChips"] = 20
    only_folded["folded"] = True

    import pytest

    with pytest.raises(ValueError):
        simulator.settle_pots([only_folded], ["9S", "4H", "3D", "KC", "6S"])


# ── Uncalled-bet refund generalized to N players ────────────────────────────


def test_refund_uncalled_bet_returns_excess_to_sole_top_bettor():
    a = make_seat("a", stack=0, current_bet=300, seat_number=1)
    b = make_seat("b", stack=0, current_bet=100, seat_number=2)
    c = make_seat("c", stack=0, current_bet=100, seat_number=3)

    simulator._refund_uncalled_bet([a, b, c])

    assert a["currentBetChips"] == 100
    assert a["stackChips"] == 200
    assert b["currentBetChips"] == 100
    assert c["currentBetChips"] == 100


def test_refund_uncalled_bet_does_nothing_when_two_seats_match_the_top():
    a = make_seat("a", stack=0, current_bet=300, seat_number=1)
    b = make_seat("b", stack=0, current_bet=300, seat_number=2)
    c = make_seat("c", stack=0, current_bet=100, seat_number=3)

    simulator._refund_uncalled_bet([a, b, c])

    assert a["currentBetChips"] == 300
    assert b["currentBetChips"] == 300
    assert a["stackChips"] == 0


def test_collect_live_bets_multiway_tracks_cumulative_committed_chips():
    a = make_seat("a", stack=0, current_bet=100, seat_number=1)
    b = make_seat("b", stack=0, current_bet=100, seat_number=2)

    pot = simulator.collect_live_bets_multiway(0, [a, b])

    assert pot == 200
    assert a["committedChips"] == 100
    assert b["committedChips"] == 100
    assert a["currentBetChips"] == 0

    # A second street accumulates on top of the first.
    a["currentBetChips"] = 50
    b["currentBetChips"] = 50
    pot = simulator.collect_live_bets_multiway(pot, [a, b])
    assert a["committedChips"] == 150
    assert b["committedChips"] == 150
    assert pot == 300


# ── Blind levels threaded through table state ───────────────────────────────


def test_build_allowed_actions_min_bet_follows_custom_big_blind():
    seat = make_seat(simulator.PLAYER_AGENT_ID, stack=1000, current_bet=0)

    allowed = simulator.build_allowed_actions(seat, current_bet=0, big_blind=200)

    assert allowed["minBet"] == 200
    assert allowed["betRange"]["min"] == 200


def test_build_table_exposes_current_blind_level():
    table = simulator.build_table(
        [], [], 0, 0, "Preflop", 1, small_blind=25, big_blind=50
    )

    assert table["smallBlindChips"] == 25
    assert table["bigBlindChips"] == 50


def test_resolve_action_bet_respects_custom_big_blind():
    seat = make_seat(simulator.PLAYER_AGENT_ID, stack=1000, current_bet=0)

    simulator.resolve_action(seat, "bet", 5, current_bet=0, big_blind=200)

    # A bet below the level's big blind is coerced up to that big blind,
    # not the module-level default.
    assert seat["currentBetChips"] == 200


class _NoShuffleRng:
    """Deterministic stand-in for random.Random: leaves the deck order
    untouched so tests can predict exactly which hole cards are dealt."""

    def shuffle(self, deck):
        return None


def _force_all_in(table, seat):
    return "all-in", None, "forced all-in for test"


def test_play_hand_multiway_blind_level_affects_posted_blinds_and_table_state():
    seen_tables = []

    def observer(**event):
        table = event.get("table")
        if table is not None:
            seen_tables.append(table)

    def fold_immediately(table, seat):
        return "fold", None, "fold for test"

    stacks = simulator.play_hand_multiway(
        [1000, 1000],
        [fold_immediately, None],
        button_index=0,
        rng=_NoShuffleRng(),
        action_observer=observer,
        small_blind=25,
        big_blind=50,
    )

    # Hero (button/small blind) folds to the big blind's forced win.
    assert stacks[0] == 975
    assert stacks[1] == 1025
    assert seen_tables, "expected at least one observed table"
    assert seen_tables[0]["smallBlindChips"] == 25
    assert seen_tables[0]["bigBlindChips"] == 50



def test_play_hand_multiway_heads_up_button_posts_small_blind_and_acts_first():
    """Heads-up is the one case where the button posts the small blind
    and acts first preflop -- the opposite of the 3+ player convention
    where the button itself never posts a blind and acts last preflop."""
    first_actor_ids = []

    def record_first_actor_and_fold(table, seat):
        if not first_actor_ids:
            first_actor_ids.append(seat["agentId"])
        return "fold", None, "fold for test"

    # button_index=0 -> seat 0 (hero) is the button/small blind.
    stacks_a = simulator.play_hand_multiway(
        [1000, 1000],
        [record_first_actor_and_fold, record_first_actor_and_fold],
        button_index=0,
        rng=_NoShuffleRng(),
    )
    assert first_actor_ids == [simulator.PLAYER_AGENT_ID]
    # The button (hero) posted only the small blind (5) and folded to the
    # big blind (10) without ever facing a raise.
    assert stacks_a[0] == 995
    assert stacks_a[1] == 1005

    first_actor_ids.clear()
    # button_index=1 -> seat 1 (villain) is now the button/small blind.
    stacks_b = simulator.play_hand_multiway(
        [1000, 1000],
        [record_first_actor_and_fold, record_first_actor_and_fold],
        button_index=1,
        rng=_NoShuffleRng(),
    )
    assert first_actor_ids == ["bot-agent-1"]
    assert stacks_b[1] == 995
    assert stacks_b[0] == 1005

def test_play_hand_multiway_accepts_custom_agent_ids():
    stacks = simulator.play_hand_multiway(
        [500, 500, 500],
        [_force_all_in, _force_all_in, _force_all_in],
        button_index=0,
        rng=_NoShuffleRng(),
        agent_ids=["hero", "villain-a", "villain-b"],
    )

    assert len(stacks) == 3
    assert sum(stacks) == 1500


# ── End-to-end multiway side-pot correctness through play_hand_multiway ────


def test_play_hand_multiway_side_pot_never_awards_short_stack_the_whole_pot(
    monkeypatch,
):
    """A short all-in winner must only collect the main pot; the side pot
    between the two deep-stacked losers/other player must not leak to
    them. This is a full play_hand_multiway integration test covering the
    exact failure mode described in the audit: 'never award an entire pot
    to a short-stack winner'."""

    hole_card_scores = {
        ("AS", "AH"): 100,  # hero: short stack, best hand
        ("AD", "AC"): 50,  # botA: deep stack, second best
        ("KS", "KH"): 10,  # botB: deep stack, worst
    }

    def fake_evaluate_hand(cards):
        return hole_card_scores[(cards[0], cards[1])]

    monkeypatch.setattr(simulator, "evaluate_hand", fake_evaluate_hand)

    stacks = simulator.play_hand_multiway(
        [40, 200, 200],
        [_force_all_in, _force_all_in, _force_all_in],
        button_index=2,
        rng=_NoShuffleRng(),
    )

    hero_stack, bot_a_stack, bot_b_stack = stacks

    # Total chips are conserved.
    assert sum(stacks) == 440
    # Hero (short stack, best hand) wins only the 120-chip main pot they
    # were eligible for -- never the full 440-chip pot.
    assert hero_stack == 120
    # botA (best hand among the two deep stacks) wins the 320-chip side
    # pot hero was never eligible for.
    assert bot_a_stack == 320
    assert bot_b_stack == 0


def test_play_hand_multiway_chip_conservation_across_seeds_and_sizes(monkeypatch):
    """Property test: for many seeds/player-counts/unequal starting stacks
    with an all-in-every-time lineup, total chips before must equal total
    chips after -- no leaks, no fabricated chips, regardless of how many
    side-pot layers are created."""
    import random as random_module

    for seed in range(15):
        for player_count in (2, 3, 4, 5, 6):
            rng = random_module.Random(seed * 100 + player_count)
            stacks_in = [rng.randint(10, 500) for _ in range(player_count)]
            total_in = sum(stacks_in)
            strategies = [_force_all_in] * player_count

            stacks_out = simulator.play_hand_multiway(
                stacks_in,
                strategies,
                button_index=seed % player_count,
                rng=random_module.Random(seed),
            )

            assert sum(stacks_out) == total_in, (
                f"chip leak: seed={seed} players={player_count} "
                f"in={stacks_in} out={stacks_out}"
            )
            assert all(s >= 0 for s in stacks_out)


# ── NLHE reopening semantics: short all-in must not reopen raise rights ────


def test_short_all_in_does_not_reopen_raise_for_players_who_already_acted():
    """A already called 10, B already called 10, then C goes all-in for
    only 15 (a SHORT all-in: the 5-chip increase is below the 10-chip
    min-raise). Per NLHE rules, a short all-in obliges A and B to respond
    (call the extra 5 or fold) but must NOT reopen their right to raise --
    and since going all-in for their full stack would itself function as
    a raise here, "all-in" must be withheld from them too. A player who
    had not yet acted this round would keep normal raise rights, but here
    everyone has already acted once."""
    seat_a = make_seat("a", stack=1000, current_bet=0, seat_number=1)
    seat_b = make_seat("b", stack=1000, current_bet=0, seat_number=2)
    seat_c = make_seat("c", stack=15, current_bet=0, seat_number=3)

    seen_allowed = {"a": [], "b": [], "c": []}

    def make_caller(agent_id):
        def provider(table, seat):
            allowed = table["allowedActions"]
            seen_allowed[agent_id].append(allowed)
            return "call", None, f"{agent_id} calls"

        return provider

    def c_provider(table, seat):
        allowed = table["allowedActions"]
        seen_allowed["c"].append(allowed)
        return "all-in", None, "c short all-in"

    fold_winner, pot = simulator.run_betting_round_multiway(
        [seat_a, seat_b, seat_c],
        board=[],
        pot=0,
        current_bet=10,
        street="Flop",
        first_actor_idx=0,
        button_seat_number=3,
        action_providers={
            "a": make_caller("a"),
            "b": make_caller("b"),
            "c": c_provider,
        },
        small_blind=5,
        big_blind=10,
    )

    assert fold_winner is None
    # A and B each acted twice: once before C's short all-in, once after.
    assert len(seen_allowed["a"]) == 2
    assert len(seen_allowed["b"]) == 2
    assert len(seen_allowed["c"]) == 1

    # First round of action (before C's short all-in): raise was a normal,
    # available option for A and B.
    assert "raise" in seen_allowed["a"][0]["availableActions"]
    assert "raise" in seen_allowed["b"][0]["availableActions"]

    # After C's short all-in (10 -> 15, a 5-chip increase below the
    # 10-chip min-raise), A and B must call again but may not raise --
    # and since an all-in for their full stack would itself be a raise
    # here, "all-in" is withheld too.
    for agent_id in ("a", "b"):
        second_allowed = seen_allowed[agent_id][1]
        assert second_allowed["availableActions"] == ["fold", "call"], (
            f"{agent_id} must only be able to call the short all-in's extra "
            f"amount or fold, not raise; got {second_allowed['availableActions']}"
        )
        assert second_allowed["callAmount"] == 5

    # Final state: everyone matched at 15, pot conserved.
    assert seat_a["currentBetChips"] == 0  # swept into pot at street end
    assert seat_a["stackChips"] == 985
    assert seat_b["stackChips"] == 985
    assert seat_c["stackChips"] == 0
    assert pot == 45


def test_short_all_in_preserves_raise_rights_for_players_who_have_not_yet_acted():
    """X already called; Y (short stack) then goes all-in short (10 -> 15,
    below the 10-chip min-raise) -- X loses raise rights (already acted),
    but Z, who has NOT yet acted this round, must keep normal raise
    rights when their turn comes. A third deep-stacked player (X) is
    needed so raising remains meaningful (i.e. `can_raise` stays true)
    after Y busts to zero."""
    seat_x = make_seat("x", stack=1000, current_bet=0, seat_number=1)
    seat_y = make_seat("y", stack=15, current_bet=0, seat_number=2)
    seat_z = make_seat("z", stack=1000, current_bet=0, seat_number=3)

    seen_allowed_z = []

    def x_provider(table, seat):
        return "call", None, "x calls"

    def y_provider(table, seat):
        return "all-in", None, "y short all-in"

    def z_provider(table, seat):
        seen_allowed_z.append(table["allowedActions"])
        return "call", None, "z calls"

    # current_bet starts at 10 (as if a big blind was already posted). X
    # acts first and calls; Y then goes all-in short for 15; Z -- who has
    # not acted at all yet this round -- acts last.
    fold_winner, _pot = simulator.run_betting_round_multiway(
        [seat_x, seat_y, seat_z],
        board=[],
        pot=0,
        current_bet=10,
        street="Flop",
        first_actor_idx=0,
        button_seat_number=3,
        action_providers={"x": x_provider, "y": y_provider, "z": z_provider},
        small_blind=5,
        big_blind=10,
    )

    assert fold_winner is None
    assert len(seen_allowed_z) == 1
    # Z had not acted before Y's short all-in, so Z's normal raise rights
    # (and all-in) must still be available.
    assert "raise" in seen_allowed_z[0]["availableActions"]
    assert "all-in" in seen_allowed_z[0]["availableActions"]


# ── resolve_action normalizes strategy-supplied float amounts to int ───────


def test_resolve_action_call_float_amount_normalizes_to_int_and_matches_int():
    """A strategy that computes a call amount via float arithmetic (e.g.
    forgetting to cast) must produce the exact same integer chip result as
    the equivalent int amount -- only the input type differs, never the
    resulting value or the resulting state's type."""
    seat_float = make_seat("a", stack=1000, current_bet=0)
    seat_int = make_seat("a", stack=1000, current_bet=0)

    simulator.resolve_action(seat_float, "call", 45.0, current_bet=45)
    simulator.resolve_action(seat_int, "call", 45, current_bet=45)

    assert seat_float == seat_int
    assert isinstance(seat_float["stackChips"], int)
    assert isinstance(seat_float["currentBetChips"], int)
    assert seat_float["stackChips"] == 955
    assert seat_float["currentBetChips"] == 45


def test_resolve_action_call_fractional_float_amount_truncates_toward_zero():
    """A genuinely fractional amount (not just a whole number typed as
    float) is truncated toward zero -- never rounded up past what the
    strategy asked for."""
    seat = make_seat("a", stack=1000, current_bet=0)

    simulator.resolve_action(seat, "call", 45.9, current_bet=100)

    assert isinstance(seat["stackChips"], int)
    assert isinstance(seat["currentBetChips"], int)
    assert seat["currentBetChips"] == 45
    assert seat["stackChips"] == 955


def test_resolve_action_bet_float_amount_normalizes_to_int():
    seat_float = make_seat("a", stack=1000, current_bet=0)
    seat_int = make_seat("a", stack=1000, current_bet=0)

    # A strategy computing BIG_BLIND * 2.5 without casting.
    simulator.resolve_action(seat_float, "bet", 25.0, current_bet=0)
    simulator.resolve_action(seat_int, "bet", 25, current_bet=0)

    assert seat_float == seat_int
    assert isinstance(seat_float["stackChips"], int)
    assert isinstance(seat_float["currentBetChips"], int)
    assert seat_float["currentBetChips"] == 25
    assert seat_float["stackChips"] == 975


def test_resolve_action_raise_float_amount_normalizes_to_int():
    seat_float = make_seat("a", stack=1000, current_bet=50)
    seat_int = make_seat("a", stack=1000, current_bet=50)

    simulator.resolve_action(
        seat_float, "raise", 150.0, current_bet=100, min_raise_to=110
    )
    simulator.resolve_action(
        seat_int, "raise", 150, current_bet=100, min_raise_to=110
    )

    assert seat_float == seat_int
    assert isinstance(seat_float["stackChips"], int)
    assert isinstance(seat_float["currentBetChips"], int)
    assert seat_float["currentBetChips"] == 150


def test_resolve_action_raise_float_min_raise_to_normalizes_to_int():
    """``min_raise_to`` is engine-computed, but resolve_action still
    normalizes it defensively -- a float min_raise_to must not leak a
    float target into stackChips/currentBetChips via ``max(target,
    min_raise_to)``."""
    seat = make_seat("a", stack=1000, current_bet=50)

    current_bet = simulator.resolve_action(
        seat, "raise", 90, current_bet=100, min_raise_to=150.9
    )

    assert isinstance(seat["stackChips"], int)
    assert isinstance(seat["currentBetChips"], int)
    assert isinstance(current_bet, int)
    # target = max(90, int(150.9)) = max(90, 150) = 150.
    assert seat["currentBetChips"] == 150


def test_resolve_action_all_in_target_stays_int_with_normal_int_stack():
    seat = make_seat("a", stack=777, current_bet=23)

    current_bet = simulator.resolve_action(seat, "all-in", None, current_bet=100)

    assert isinstance(seat["stackChips"], int)
    assert isinstance(seat["currentBetChips"], int)
    assert isinstance(current_bet, int)
    assert seat["stackChips"] == 0
    assert seat["currentBetChips"] == 800


def test_resolve_action_bet_float_below_big_blind_coerces_to_int_big_blind():
    """A float amount below the big blind is still coerced up to the big
    blind for a 'bet' action, and the coerced result stays int."""
    seat = make_seat("a", stack=1000, current_bet=0)

    simulator.resolve_action(seat, "bet", 3.0, current_bet=0, big_blind=10)

    assert isinstance(seat["currentBetChips"], int)
    assert seat["currentBetChips"] == 10


def test_resolve_action_call_float_amount_preserves_all_in_capping():
    """A float call amount larger than the seat's stack still caps
    correctly at the (integer) stack, preserving all-in legality."""
    seat = make_seat("a", stack=40, current_bet=0)

    simulator.resolve_action(seat, "call", 500.0, current_bet=500)

    assert isinstance(seat["stackChips"], int)
    assert isinstance(seat["currentBetChips"], int)
    assert seat["stackChips"] == 0
    assert seat["currentBetChips"] == 40


def test_play_hand_multiway_chip_conservation_with_float_valued_strategy():
    """A strategy that always returns float bet/raise/call amounts must
    still produce fully-integer, chip-conserving final stacks -- proving
    the normalization holds across a complete, multi-street hand, not
    just a single resolve_action call."""
    import random

    def float_amount_strategy(table, seat):
        allowed = table["allowedActions"]
        available = allowed["availableActions"]
        if "raise" in available:
            # Deliberately float, like an uncast BIG_BLIND * multiplier.
            return "raise", float(allowed["minRaiseTo"]) + 0.0, "float raise"
        if "bet" in available:
            return "bet", float(allowed["minBet"]) + 0.0, "float bet"
        if "call" in available:
            return "call", float(allowed["callAmount"]) + 0.0, "float call"
        if "check" in available:
            return "check", None, "check"
        return "fold", None, "fold"

    for seed in range(5):
        for player_count in (2, 3, 4, 6):
            stacks_in = [1000] * player_count
            stacks_out = simulator.play_hand_multiway(
                list(stacks_in),
                [float_amount_strategy] * player_count,
                button_index=seed % player_count,
                rng=random.Random(seed),
            )

            assert all(isinstance(stack, int) for stack in stacks_out), (
                f"non-integer final stack(s) with float-valued strategy: "
                f"seed={seed} players={player_count} stacks_out={stacks_out}"
            )
            assert sum(stacks_out) == sum(stacks_in)
            assert all(stack >= 0 for stack in stacks_out)
