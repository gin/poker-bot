from poker_bot.hand_eval import (
    best_hand_rank_without,
    best_hand_with_combo,
    choose_dummy_card,
    evaluate_hand,
)
from poker_bot.strategies.s2base import (
    private_made_hand,
    private_made_hand_rank,
    relative_hand_drop,
)


def test_choose_dummy_card_avoids_board():
    board = ["2c", "2d", "2h", "3c", "4d"]
    assert choose_dummy_card(board) not in board


def test_best_hand_with_combo_matches_evaluate_hand():
    cards = ["KS", "KD", "AS", "QD", "JS"]
    assert best_hand_with_combo(cards)[0] == evaluate_hand(cards)


def test_best_hand_rank_without_drops_category():
    # Pair of kings: dropping one K drops the category from pair (1) to high card (0).
    pool = ["KS", "KD", "AS", "QD", "JS"]
    assert best_hand_rank_without(pool, ["KS"])[0] == 0


def test_private_made_hand_kq_on_trips_river():
    # KQ on a trips board: best hand is trips with K/Q kickers. Category is
    # board-made; the hole cards only contribute as kickers, so used == 0.
    hole = ["KS", "QD"]
    board = ["AS", "AH", "AD", "5C", "4H"]
    category, used = private_made_hand(hole, board)
    assert used == 0
    assert category == 3


def test_private_made_hand_22_on_full_house_board_is_board_made():
    # 22 on A A A 5 5: the board alone is already a full house (A full 5), and
    # the pocket pair is lower than the board's pair, so the best hand with
    # the hero is still the board's hand. private_made_hand treats this as
    # board-made.
    hole = ["2S", "2D"]
    board = ["AS", "AH", "AD", "5C", "5H"]
    category, used = private_made_hand(hole, board)
    assert used == 0
    assert category == 0


def test_private_made_hand_22_on_trips_no_pair_makes_private_full_house():
    # 22 on A A A 5 4: board has trips but no pair. The pocket pair becomes
    # the pair component of the full house, so the category is private.
    hole = ["2S", "2D"]
    board = ["AS", "AH", "AD", "5C", "4H"]
    category, used = private_made_hand(hole, board)
    assert used == 2
    assert category == 6


def test_private_made_hand_rank_zero_for_board_made():
    hole = ["KS", "QD"]
    board = ["AS", "AH", "AD", "5C", "4H"]
    assert private_made_hand_rank(hole, board) == 0


def test_private_made_hand_rank_nonzero_for_private_hand():
    # Pocket aces on a board that includes two more aces: dropping a hole card
    # drops the category from quads (7) to trips (3), so used == 2 and the
    # private rank is the full category.
    hole = ["AS", "AC"]
    board = ["AH", "AD", "5C", "2D", "3H"]
    assert private_made_hand_rank(hole, board) == 7


def test_relative_hand_drop_detects_weaker_than_preflop():
    # KQ preflop is strong (score ~68) but trips on board are board-made.
    assert relative_hand_drop(["KS", "QD"], ["AS", "AH", "AD", "5C", "4H"]) is True
    # 22 on a full-house board where the board already has the higher pair is
    # also "weaker than preflop" from a category standpoint: the pocket pair
    # didn't contribute.
    assert relative_hand_drop(["2S", "2D"], ["AS", "AH", "AD", "5C", "5H"]) is True
