"""Unit tests for poker_bot.hand_utils — shared utility module extracted
from hubase.py. Verifies extracted functions work standalone."""

from poker_bot.hand_utils import (
    evaluate_hand,
    made_hand_rank,
    board_texture,
    card_values,
    pot_odds,
    seated_players,
    active_opponents,
    active_seat_numbers,
    active_players,
    live_opponent_seats,
    has_top_pair_or_better,
    fragile_rank_two,
    board_dominated_two_pair,
    paired_board_ranks,
    board_has_pair,
    board_has_two_pair,
    has_flush_draw,
    has_open_ended_straight_draw,
    has_good_draw,
    OpponentProfile,
    profile_value,
    opponent_is_bluffy,
    is_tight_opponent,
    single_opponent_profile,
    count_dealt_in_players,
    player_regime,
    dealt_in_regime,
    REGIME_HEADS_UP,
    REGIME_THREE_HANDED,
    REGIME_FULL_TABLE,
)


class TestHandEvaluation:
    def test_royal_flush(self):
        assert evaluate_hand(["AH", "KH", "QH", "JH", "TH"]) == (8, 14)

    def test_quads(self):
        assert evaluate_hand(["AS", "AD", "AC", "AH", "2D"])[0] == 7

    def test_full_house(self):
        assert evaluate_hand(["AS", "AD", "AC", "KH", "KD"])[0] == 6

    def test_flush(self):
        assert evaluate_hand(["AH", "2H", "3H", "4H", "7H"])[0] == 5

    def test_straight(self):
        assert evaluate_hand(["6S", "7D", "8C", "9H", "TH"])[0] == 4

    def test_trips(self):
        assert evaluate_hand(["AS", "AD", "AC", "2H", "3D"])[0] == 3

    def test_two_pair(self):
        assert evaluate_hand(["AS", "AD", "KC", "KH", "2D"])[0] == 2

    def test_one_pair(self):
        assert evaluate_hand(["AS", "AD", "2C", "3H", "5D"])[0] == 1

    def test_high_card(self):
        assert evaluate_hand(["AS", "2D", "3C", "5H", "7D"])[0] == 0

    def test_seven_card_eval(self):
        assert evaluate_hand(["3D", "3S", "AS", "KS", "TC", "KH", "QS"])[0] == 2


class TestMadeHandRank:
    def test_pocket_pair_on_paired_board(self):
        assert made_hand_rank(["3D", "3S"], ["AS", "KS", "TC", "KH", "QS"]) == 2

    def test_top_pair(self):
        assert made_hand_rank(["KH", "3C"], ["7S", "9D", "KH", "2C"]) == 1

    def test_high_card(self):
        assert made_hand_rank(["JC", "5D"], ["7S", "9D", "KH", "2C"]) == 0

    def test_board_made_returns_zero(self):
        assert made_hand_rank(["2C", "3D"], ["AH", "2H", "3H", "4H", "5H"]) == 0


class TestBoardTexture:
    def test_dry_board(self):
        t = board_texture(["7S", "9D", "KH", "2C"])
        assert t["wet"] is False and t["paired"] is False and t["high"] is True

    def test_wet_board(self):
        assert board_texture(["7S", "8S", "9H", "2C"])["wet"] is True

    def test_paired_board(self):
        assert board_texture(["7S", "7D", "KH", "2C"])["paired"] is True


class TestFragileTwoPair:
    def test_fragile_pocket_pair_on_high_board(self):
        assert fragile_rank_two(["3D", "3S"], ["AS", "KS", "TC", "KH", "QS"], 2) is True

    def test_board_dominated_two_pair(self):
        assert (
            board_dominated_two_pair(["5S", "5H"], ["KS", "KH", "QD", "QS", "3C"], 2)
            is True
        )

    def test_not_board_dominated_single_pair(self):
        assert (
            board_dominated_two_pair(["3D", "3S"], ["AS", "KS", "TC", "KH", "QS"], 2)
            is False
        )


class TestDraws:
    def test_flush_draw(self):
        assert has_flush_draw(["AH", "2H"], ["7H", "9D", "KH"]) is True

    def test_no_flush_draw(self):
        assert has_flush_draw(["AH", "2D"], ["7H", "9D", "KH"]) is False

    def test_oesd(self):
        assert has_open_ended_straight_draw(["6S", "7D"], ["8C", "9H", "2D"]) is True

    def test_good_draw(self):
        assert has_good_draw(["AH", "2H"], ["7H", "9D", "KH"]) is True


class TestPotOdds:
    def test_25_percent(self):
        assert abs(pot_odds(100, 300) - 0.25) < 1e-6

    def test_zero_call(self):
        assert pot_odds(0, 100) == 0.0


class TestTableUtils:
    def test_seated_players(self):
        table = {"seats": [{"agentId": "a"}, {"agentId": "b"}, {}]}
        assert seated_players(table) == 2

    def test_active_opponents(self):
        table = {
            "seats": [
                {"agentId": "hero", "folded": False, "hasFolded": False},
                {"agentId": "v1", "folded": False, "hasFolded": False},
                {"agentId": "v2", "folded": True, "hasFolded": False},
            ]
        }
        assert active_opponents(table, {"agentId": "hero"}) == 1

    def test_active_seat_numbers_excludes_empty_numbered_slot(self):
        # An empty/unfilled seat (agentId falsy) carries no dead status or
        # folded flag, so it looked "live" to seat_is_live alone. A truthy
        # agentId is required before liveness counts, matching
        # count_dealt_in_players/multi_core's dealt-in semantics.
        table = {
            "seats": [
                {"agentId": "hero", "seatNumber": 1},
                {"agentId": "v1", "seatNumber": 2},
                {"agentId": None, "seatNumber": 3},
            ]
        }
        assert active_seat_numbers(table) == [1, 2]
        assert active_players(table) == 2

    def test_active_opponents_excludes_empty_numbered_slot(self):
        table = {
            "seats": [
                {"agentId": "hero", "seatNumber": 1},
                {"agentId": "v1", "seatNumber": 2},
                {"agentId": None, "seatNumber": 3},
            ]
        }
        assert active_opponents(table, {"agentId": "hero"}) == 1

    def test_live_opponent_seats_excludes_empty_numbered_slot(self):
        # Same invariant as active_seat_numbers/active_opponents: an empty
        # numbered seat (agentId falsy) must never be returned as a live
        # opponent. guard_pre's spr_commitment_lock (and other guards) sum
        # stacks over this list, so an empty slot silently pulled in a
        # phantom opponent with stackChips defaulting via `or 0`.
        table = {
            "seats": [
                {"agentId": "hero", "seatNumber": 1, "stackChips": 2000},
                {"agentId": "v1", "seatNumber": 2, "stackChips": 1500},
                {"agentId": None, "seatNumber": 3, "stackChips": 0},
            ]
        }
        hero = table["seats"][0]
        opponents = live_opponent_seats(table, hero)
        assert [s["agentId"] for s in opponents] == ["v1"]


class TestPlayerRegime:
    """Contract tests for the shared dealt-in count / canonical regime,
    shared verbatim by multi_core routing and GuardContext/GuardRail."""

    def test_dealt_in_counts_live_and_folded_this_hand_simulator_style(self):
        # Simulator schema: 'folded'/'hasFolded' flags, no 'status'. All
        # seats are dealt in regardless of fold state.
        table = {
            "seats": [
                {"agentId": "hero", "folded": False, "hasFolded": False},
                {"agentId": "v1", "folded": True, "hasFolded": False},
                {"agentId": "v2", "folded": False, "hasFolded": True},
                {},  # no agentId: never dealt in
            ]
        }
        assert count_dealt_in_players(table) == 3

    def test_dealt_in_excludes_never_dealt_in_arena_statuses(self):
        # Arena schema: 'status' field. Busted/SittingOut/Waiting never took
        # part in this hand; Folded/Active/AllIn did.
        table = {
            "seats": [
                {"agentId": "hero", "status": "Active"},
                {"agentId": "v1", "status": "Folded"},
                {"agentId": "v2", "status": "AllIn"},
                {"agentId": "v3", "status": "Busted"},
                {"agentId": "v4", "status": "SittingOut"},
                {"agentId": "v5", "status": "Waiting"},
            ]
        }
        assert count_dealt_in_players(table) == 3

    def test_dealt_in_zero_when_no_seats(self):
        assert count_dealt_in_players({"seats": []}) == 0
        assert count_dealt_in_players({}) == 0

    def test_player_regime_boundaries(self):
        assert player_regime(0) == REGIME_HEADS_UP
        assert player_regime(1) == REGIME_HEADS_UP
        assert player_regime(2) == REGIME_HEADS_UP
        assert player_regime(3) == REGIME_THREE_HANDED
        assert player_regime(4) == REGIME_FULL_TABLE
        assert player_regime(6) == REGIME_FULL_TABLE
        assert player_regime(9) == REGIME_FULL_TABLE

    def test_dealt_in_regime_busted_down_six_seat_table_is_heads_up(self):
        # 6 seats, 4 Busted + 2 live: routing and guards must both see HU,
        # not full_table (the seated-count bug this replaces).
        table = {
            "seats": [
                {"agentId": "hero", "status": "Active"},
                {"agentId": "v1", "status": "Active"},
                {"agentId": "v2", "status": "Busted"},
                {"agentId": "v3", "status": "Busted"},
                {"agentId": "v4", "status": "Busted"},
                {"agentId": "v5", "status": "Busted"},
            ]
        }
        assert count_dealt_in_players(table) == 2
        assert dealt_in_regime(table) == REGIME_HEADS_UP

    def test_dealt_in_regime_three_dealt_in_is_three_handed(self):
        table = {
            "seats": [
                {"agentId": "hero", "folded": False},
                {"agentId": "v1", "folded": False},
                {"agentId": "v2", "folded": True},
            ]
        }
        assert count_dealt_in_players(table) == 3
        assert dealt_in_regime(table) == REGIME_THREE_HANDED

    def test_dealt_in_regime_six_seats_four_folded_stays_full_table(self):
        # Dealt-in stays stable across the hand: folds don't shrink the
        # regime mid-hand the way a live active count would.
        table = {
            "seats": [
                {"agentId": "hero", "folded": False},
                {"agentId": "v1", "folded": True},
                {"agentId": "v2", "folded": True},
                {"agentId": "v3", "folded": True},
                {"agentId": "v4", "folded": True},
                {"agentId": "v5", "folded": False},
            ]
        }
        assert count_dealt_in_players(table) == 6
        assert dealt_in_regime(table) == REGIME_FULL_TABLE


class TestOpponentProfile:
    def _bluffy(self):
        return OpponentProfile(
            agent_id="vill",
            hands_seen=20,
            preflop_hands_seen=20,
            vpip=10,
            calls=3,
            bets=4,
            raises=2,
            folds=6,
            fold_to_bet=4,
            opportunities_to_fold_to_bet=8,
            showdowns=5,
            weak_aggressive_showdowns=2,
        )

    def _tight(self):
        return OpponentProfile(
            agent_id="vill",
            hands_seen=20,
            preflop_hands_seen=20,
            vpip=3,
            calls=2,
            bets=1,
            raises=0,
            folds=14,
            fold_to_bet=10,
            opportunities_to_fold_to_bet=12,
            showdowns=3,
            weak_aggressive_showdowns=0,
        )

    def test_vpip_frequency(self):
        assert abs(self._bluffy().vpip_frequency - 0.5) < 1e-6

    def test_aggression_frequency(self):
        assert abs(self._bluffy().aggression_frequency - 0.4) < 1e-6

    def test_wasd(self):
        assert abs(self._bluffy().weak_aggressive_showdown_frequency - 0.4) < 1e-6

    def test_is_bluffy(self):
        assert opponent_is_bluffy(self._bluffy()) is True

    def test_is_not_bluffy(self):
        assert opponent_is_bluffy(self._tight()) is False

    def test_is_tight(self):
        table = {
            "opponentProfiles": {
                "vill": {
                    "hands_seen": 20,
                    "preflop_hands_seen": 20,
                    "profile_stats_schema_version": 2,
                    "profile_stats_provenance": "canonical",
                    "vpip": 3,
                }
            }
        }
        assert is_tight_opponent(table) is True

    def test_is_not_tight(self):
        table = {
            "opponentProfiles": {
                "vill": {
                    "hands_seen": 20,
                    "preflop_hands_seen": 20,
                    "profile_stats_schema_version": 2,
                    "profile_stats_provenance": "canonical",
                    "vpip": 10,
                }
            }
        }
        assert is_tight_opponent(table) is False

    def test_single_opponent_profile(self):
        p = self._bluffy()
        assert (
            single_opponent_profile({"opponentProfiles": {"vill": p}}, min_hands=15)
            is p
        )

    def test_single_opponent_profile_not_enough_hands(self):
        p = OpponentProfile(agent_id="vill", hands_seen=5)
        assert (
            single_opponent_profile({"opponentProfiles": {"vill": p}}, min_hands=15)
            is None
        )

    def test_profile_value_dict(self):
        d = {"hands_seen": 42, "vpip": 10}
        assert profile_value(d, "hands_seen") == 42
        assert profile_value(d, "calls") is None

    def test_profile_value_object(self):
        assert profile_value(self._bluffy(), "hands_seen") == 20
