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
