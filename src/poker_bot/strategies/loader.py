"""Strategy loading helpers."""

import importlib
from collections.abc import Callable

Strategy = Callable[[dict, dict | None], tuple[str | None, int | None, str]]


def load_strategy(name) -> Strategy:
    if "." in name:
        module_name = name
        fallback = None
    else:
        module_name = f"poker_bot.strategies.{name}"
        fallback = name

    module = None
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        if fallback is None:
            raise
        try:
            module = importlib.import_module(fallback)
        except ModuleNotFoundError:
            pass

    if module is None:
        raise ValueError(f"Unknown strategy: {name}")

    strategy = getattr(module, "choose_action", None)
    if strategy is None:
        raise ValueError(f"Strategy {name} does not define choose_action()")
    return strategy
