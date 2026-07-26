from poker_bot.strategies import royal_adaptive as strategy
from poker_bot.strategies.loader import load_strategy


def make_seat(agent_id="hero", seat_number=1, cards=None, stack=1800):
    return {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": cards or ["AS", "KS"],
        "stackChips": stack,
        "currentBetChips": 0,
    }


def make_profile(
    agent_id,
    label="patient_methodical",
    hands_seen=40,
    vpip=6,
    pfr=1,
    calls=1,
    bets=0,
    raises=0,
    folds=20,
):
    return {
        agent_id: {
            "name": agent_id,
            "hands_seen": hands_seen,
            "preflop_hands_seen": hands_seen,
            "profile_stats_schema_version": 2,
            "profile_stats_provenance": "canonical",
            "vpip": vpip,
            "pfr": pfr,
            "calls": calls,
            "bets": bets,
            "raises": raises,
            "folds": folds,
            "recent_actions": [{"action": "fold"}] * 8,
            "_label": label,
        }
    }


def make_table(street="Preflop", board=None, actions=None, hero=None, profiles=None):
    hero = hero or make_seat()
    villain = {"agentId": "villain", "seatNumber": 2, "stackChips": 1800}
    return {
        "street": street,
        "boardCards": board or [],
        "potChips": 120,
        "currentBet": 0,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": actions or ["fold", "check", "raise"],
            "callAmount": 0,
            "callChips": 0,
            "minBet": 50,
            "minRaiseTo": 150,
            "maxCommit": hero["stackChips"],
            "raiseRange": {"min": 150, "max": hero["stackChips"]},
            "betRange": {"min": 50, "max": hero["stackChips"]},
        },
        "seats": [hero, villain],
        "opponentProfiles": profiles or {},
    }


def force_base(monkeypatch, action, amount=None, message="forced base"):
    monkeypatch.setattr(
        strategy.royal_flush,
        "choose_action",
        lambda _table, _seat: (action, amount, message),
    )


def test_royal_adaptive_loads_as_strategy():
    assert load_strategy("royal_adaptive") is strategy.choose_action


def test_royal_adaptive_preserves_royal_flush_override(monkeypatch):
    force_base(monkeypatch, "fold")
    hero = make_seat(cards=["AS", "KS"])
    table = make_table(
        street="Flop",
        board=["QS", "JS", "2D"],
        actions=["fold", "call", "raise"],
        hero=hero,
        profiles=make_profile("villain"),
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "raise"
    assert amount >= table["allowedActions"]["minRaiseTo"]
    assert "royal flush possible" in message


def test_royal_adaptive_steals_against_patient_table(monkeypatch):
    force_base(monkeypatch, "fold")
    hero = make_seat(cards=["AD", "QC"])
    table = make_table(hero=hero, profiles=make_profile("villain"))

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "raise"
    assert amount >= table["allowedActions"]["minRaiseTo"]
    assert "patient table" in message


def test_royal_adaptive_bluff_catches_against_bluffer(monkeypatch):
    force_base(monkeypatch, "fold")
    hero = make_seat(cards=["KD", "9H"])
    table = make_table(
        street="Flop",
        board=["KC", "7S", "2D"],
        actions=["fold", "call"],
        hero=hero,
        profiles=make_profile(
            "villain",
            hands_seen=40,
            vpip=24,
            pfr=12,
            calls=2,
            bets=16,
            raises=8,
            folds=2,
        ),
    )
    table["potChips"] = 500
    table["currentBet"] = 100
    table["allowedActions"]["callAmount"] = 100
    table["allowedActions"]["callChips"] = 100

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "call"
    assert amount == 100
    assert "bluff-catch" in message


def test_royal_adaptive_avoids_bluffing_calling_station(monkeypatch):
    force_base(monkeypatch, "bet", 80)
    hero = make_seat(cards=["8C", "3D"])
    table = make_table(
        street="Flop",
        board=["KS", "9H", "2C"],
        actions=["fold", "check", "bet"],
        hero=hero,
        profiles=make_profile(
            "villain",
            hands_seen=40,
            vpip=24,
            pfr=1,
            calls=24,
            bets=1,
            raises=0,
            folds=2,
        ),
    )

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "check"
    assert amount is None
    assert "calling station" in message


def test_royal_adaptive_low_confidence_uses_base(monkeypatch):
    force_base(monkeypatch, "check", None, "base check")
    hero = make_seat(cards=["8C", "3D"])
    table = make_table(street="Flop", board=["KS", "9H", "2C"], hero=hero)

    action, amount, message = strategy.choose_action(table, hero)

    assert action == "check"
    assert amount is None
    assert "low-confidence base" in message
