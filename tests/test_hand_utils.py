"""Unit tests for poker_bot.hand_utils — shared utility module extracted
from hubase.py. Verifies extracted functions work standalone."""

from poker_bot.hand_utils import (
    evaluate_hand, made_hand_rank, board_texture, card_values, pot_odds,
    seated_players, active_opponents, has_top_pair_or_better,
    fragile_rank_two, board_dominated_two_pair, paired_board_ranks,
    board_has_pair, board_has_two_pair, has_flush_draw,
    has_open_ended_straight_draw, has_good_draw, OpponentProfile,
    profile_value, opponent_is_bluffy, is_tight_opponent,
    single_opponent_profile,
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
        assert board_dominated_two_pair(["5S", "5H"], ["KS", "KH", "QD", "QS", "3C"], 2) is True
    def test_not_board_dominated_single_pair(self):
        assert board_dominated_two_pair(["3D", "3S"], ["AS", "KS", "TC", "KH", "QS"], 2) is False


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
        table = {"seats": [
            {"agentId": "hero", "folded": False, "hasFolded": False},
            {"agentId": "v1", "folded": False, "hasFolded": False},
            {"agentId": "v2", "folded": True, "hasFolded": False},
        ]}
        assert active_opponents(table, {"agentId": "hero"}) == 1


class TestOpponentProfile:
    def _bluffy(self):
        return OpponentProfile(
            agent_id="vill", hands_seen=20, vpip=10, calls=3, bets=4, raises=2,
            folds=6, fold_to_bet=4, opportunities_to_fold_to_bet=8,
            showdowns=5, weak_aggressive_showdowns=2,
        )
    def _tight(self):
        return OpponentProfile(
            agent_id="vill", hands_seen=20, vpip=3, calls=2, bets=1, raises=0,
            folds=14, fold_to_bet=10, opportunities_to_fold_to_bet=12,
            showdowns=3, weak_aggressive_showdowns=0,
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
        table = {"opponentProfiles": {"vill": {"hands_seen": 20, "vpip": 3}}}
        assert is_tight_opponent(table) is True
    def test_is_not_tight(self):
        table = {"opponentProfiles": {"vill": {"hands_seen": 20, "vpip": 10}}}
        assert is_tight_opponent(table) is False
    def test_single_opponent_profile(self):
        p = self._bluffy()
        assert single_opponent_profile({"opponentProfiles": {"vill": p}}, min_hands=15) is p
    def test_single_opponent_profile_not_enough_hands(self):
        p = OpponentProfile(agent_id="vill", hands_seen=5)
        assert single_opponent_profile({"opponentProfiles": {"vill": p}}, min_hands=15) is None
    def test_profile_value_dict(self):
        d = {"hands_seen": 42, "vpip": 10}
        assert profile_value(d, "hands_seen") == 42
        assert profile_value(d, "calls") is None
    def test_profile_value_object(self):
        assert profile_value(self._bluffy(), "hands_seen") == 20