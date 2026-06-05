import pytest

from poker_bot.strategies.all_in_everytime import choose_action as all_in
from poker_bot.strategies.loader import load_strategy


def test_load_strategy_loads_strategy_choose_action():
    assert load_strategy("all_in_everytime") is all_in


def test_load_strategy_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        load_strategy("does_not_exist")
