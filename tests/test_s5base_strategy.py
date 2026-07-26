from poker_bot.mixing import choose_weighted
from poker_bot.strategies import s5base


def make_profile(*, calls=120, folds_to_bet=180, opportunities=240):
    return {
        "hands_seen": 100,
        "calls": calls,
        "bets": 20,
        "raises": 20,
        "folds": 160,
        "fold_to_bet": folds_to_bet,
        "opportunities_to_fold_to_bet": opportunities,
    }


def make_seat(agent_id="hero", seat_number=1, cards=None, folded=False):
    seat = {
        "agentId": agent_id,
        "seatNumber": seat_number,
        "holeCards": cards or ["AS", "KS"],
        "stackChips": 1800,
        "currentBetChips": 0,
    }
    if folded:
        seat["folded"] = True
    return seat


def make_table(*, hero=None):
    hero = hero or make_seat()
    seats = [
        hero,
        make_seat("villain-2", 2, cards=[]),
        make_seat("villain-3", 3, cards=[]),
        make_seat("villain-4", 4, cards=[]),
        make_seat("villain-5", 5, cards=[], folded=True),
        make_seat("villain-6", 6, cards=[], folded=True),
    ]
    return {
        "street": "Flop",
        "boardCards": ["7C", "4D", "2S"],
        "potChips": 300,
        "currentBet": 0,
        "buttonSeatNumber": 1,
        "actingSeatNumber": hero["seatNumber"],
        "selfSeatNumber": hero["seatNumber"],
        "allowedActions": {
            "availableActions": ["fold", "check", "bet"],
            "callAmount": 0,
            "callChips": 0,
            "minBet": 50,
            "maxCommit": hero["stackChips"],
            "betRange": {"min": 50, "max": hero["stackChips"]},
        },
        "seats": seats,
        "opponentProfiles": {
            f"villain-{index}": make_profile() for index in range(2, 7)
        },
    }


def _expected_range_mixed_dry_probe_result(table, hero):
    base = ("check", None, "forced check")
    action, _amount, _message = base
    allowed = table.get("allowedActions", {})
    available = allowed.get("availableActions", [])
    if action != "check" or "bet" not in available:
        return None
    if not s5base.no_one_has_bet(allowed, table):
        return None
    if not s5base.high_fold_to_bet_table(table):
        return None

    hole_cards = hero.get("holeCards", [])
    board_cards = table.get("boardCards", [])
    texture = s5base.board_texture(board_cards) if board_cards else {"wet": False}
    if texture.get("wet", False) or texture.get("paired", False):
        return None

    rank = s5base.made_hand_rank(hole_cards, board_cards) if board_cards else 0
    top_pair = s5base.has_top_pair_or_better(hole_cards, board_cards)
    if rank > 0 or top_pair:
        return None

    opponents = s5base.active_opponents(table, hero)
    if opponents > 3:
        return None

    hero_strength = s5base.hero_preflop_range_strength(hero)
    opponent_strength = s5base.average_opponent_range_strength(table, hero)
    edge = hero_strength - opponent_strength
    if edge < 0.03 or s5base.preflop_score(hole_cards) < 70:
        return None

    return choose_weighted(
        (("bet", 0.24), ("check", 0.76)),
        "v003-range-dry-probe",
        table,
        hero,
        strategy="auto_research_v003",
        extra=(round(edge, 2), opponents),
    )


def test_range_mixed_dry_probe_follows_mixing_path_without_name_error():
    hero = make_seat(cards=["AS", "KS"])
    table = make_table(hero=hero)
    base = ("check", None, "forced check")

    decision = s5base.range_mixed_dry_probe(table, hero, base)
    expected = _expected_range_mixed_dry_probe_result(table, hero)

    if expected == "bet":
        assert decision is not None
        action, amount, message = decision
        assert action == "bet"
        assert amount == s5base.bet_amount_frac(table, table["allowedActions"], 0.26)
        assert "mixed range probe" in message
    else:
        assert decision is None




