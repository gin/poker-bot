"""Season 2 v001 wrapped with the board-illusion river exploit."""

from __future__ import annotations

from poker_bot.strategies import s2v001 as base_strategy
from poker_bot.strategies.exploits.board_illusion import apply_board_illusion

ActionDecision = tuple[str | None, int | None, str]


def choose_action(table, my_seat) -> ActionDecision:
    base = base_strategy.choose_action(table, my_seat)
    override = apply_board_illusion(table, my_seat, base)
    if override is not None:
        return override
    return base
