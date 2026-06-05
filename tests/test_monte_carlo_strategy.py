from poker_bot.strategies import monte_carlo


def make_table(available_actions, **allowed_overrides):
    allowed = {
        "availableActions": available_actions,
        "callAmount": 100,
        "minBet": 50,
        "minRaiseTo": 300,
        "maxCommit": 1000,
    }
    allowed.update(allowed_overrides)
    return {
        "street": "Flop",
        "boardCards": ["AH", "7D", "2C"],
        "potChips": 400,
        "currentBet": 200,
        "allowedActions": allowed,
    }


def make_seat(cards=None, stack=900, current_bet=100):
    return {
        "agentId": "my-agent",
        "holeCards": cards or ["AS", "AD"],
        "stackChips": stack,
        "currentBetChips": current_bet,
    }


def test_high_equity_raises_when_raise_is_available(monkeypatch):
    monkeypatch.setattr(monte_carlo, "estimate_equity", lambda hole, board: 0.75)

    action, amount, message = monte_carlo.choose_action(
        make_table(["fold", "call", "raise"]), make_seat()
    )

    assert action == "raise"
    assert amount >= 300
    assert "raising for value" in message


def test_calls_when_equity_beats_pot_odds(monkeypatch):
    monkeypatch.setattr(monte_carlo, "estimate_equity", lambda hole, board: 0.35)

    action, amount, message = monte_carlo.choose_action(
        make_table(["fold", "call"], callAmount=100), make_seat()
    )

    assert action == "call"
    assert amount == 100
    assert "beats pot odds" in message


def test_folds_when_equity_is_below_pot_odds_and_check_unavailable(monkeypatch):
    monkeypatch.setattr(monte_carlo, "estimate_equity", lambda hole, board: 0.10)

    action, amount, message = monte_carlo.choose_action(
        make_table(["fold", "call"], callAmount=300), make_seat()
    )

    assert action == "fold"
    assert amount is None
    assert "below pot odds" in message


def test_checks_instead_of_folding_when_check_is_available(monkeypatch):
    monkeypatch.setattr(monte_carlo, "estimate_equity", lambda hole, board: 0.05)

    action, amount, message = monte_carlo.choose_action(
        make_table(["fold", "check", "bet"]), make_seat(cards=["3S", "8D"])
    )

    assert action == "check"
    assert amount is None
    assert "checking" in message


def test_high_equity_bets_when_no_call_is_pending(monkeypatch):
    monkeypatch.setattr(monte_carlo, "estimate_equity", lambda hole, board: 0.70)

    action, amount, message = monte_carlo.choose_action(
        make_table(["fold", "check", "bet"]), make_seat()
    )

    assert action == "bet"
    assert amount >= 50
    assert "betting" in message


def test_estimate_equity_is_deterministic_for_same_cards():
    first = monte_carlo.estimate_equity(
        ["AS", "AD"], ["2C", "7H", "TD"], max_simulations=25
    )
    second = monte_carlo.estimate_equity(
        ["AS", "AD"], ["2C", "7H", "TD"], max_simulations=25
    )

    assert first == second


def test_estimate_equity_uses_fallback_when_time_budget_is_exhausted(monkeypatch):
    times = iter([0.0, monte_carlo.MAX_EQUITY_SECONDS + 1.0])
    monkeypatch.setattr(monte_carlo.time, "perf_counter", lambda: next(times))

    equity = monte_carlo.estimate_equity(["AS", "AD"], [], max_simulations=25)

    assert equity == monte_carlo.quick_preflop_estimate(["AS", "AD"])
