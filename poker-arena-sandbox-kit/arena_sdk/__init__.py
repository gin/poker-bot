"""dev.fun Arena SDK — build, test, and submit Arena agents.

Platform layer (submit/pack/comps) is environment-agnostic; per-game logic lives
under an environment package (the first is `arena_sdk.poker`). Test the SAME
strategy.py you submit, run self-play offline, get bb/100.
"""
from .poker.contract import load_strategy, normalize_action, clamp_to_range
from .poker.read import (hero, hole_cards, button_seat, is_button, to_call,
                         pot_odds, can)
from .pack import build_bundle, BundleError
from .submit import submit, poll

# The self-play engine pulls in pokerkit (the optional `[selfplay]` extra). Load it
# lazily so `import arena_sdk` and the whole pack/submit flow work with zero deps;
# only touching run_match/build_table/etc. needs pokerkit installed.
_ENGINE = {"run_match", "play_one_hand", "build_table", "OPPONENTS"}


def __getattr__(name):
    if name in _ENGINE:
        try:
            from .poker import engine
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"arena_sdk.{name} needs the self-play engine — "
                "install it with: pip install 'arena-sdk[selfplay]'") from e
        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["run_match", "play_one_hand", "build_table", "OPPONENTS",
           "load_strategy", "normalize_action", "clamp_to_range",
           "hero", "hole_cards", "button_seat", "is_button", "to_call",
           "pot_odds", "can", "build_bundle", "BundleError", "submit", "poll"]
__version__ = "0.8.1"
