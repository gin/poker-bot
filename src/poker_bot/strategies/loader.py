"""Strategy loading helpers."""

import importlib
from collections.abc import Callable

type Strategy = Callable[[dict, str], tuple[str | None, int | None, str]]


def load_strategy(name) -> Strategy:
    module_name = f"poker_bot.strategies.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ValueError(f"Unknown strategy: {name}") from exc
        raise

    strategy = getattr(module, "choose_action", None)
    if strategy is None:
        raise ValueError(f"Strategy {name} does not define choose_action()")
    return strategy
