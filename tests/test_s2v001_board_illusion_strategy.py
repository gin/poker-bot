"""Loader smoke tests for the board-illusion strategy variant."""

from poker_bot.strategies.loader import load_strategy


def test_board_illusion_variant_loads():
    assert load_strategy("s2v001_board_illusion").__name__ == "choose_action"
