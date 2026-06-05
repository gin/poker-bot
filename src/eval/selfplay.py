"""Run quiet heads-up self-play evaluations between poker strategies."""

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from poker_bot.strategies.loader import load_strategy  # noqa: E402
from simulator import BIG_BLIND, INITIAL_STACK, play_hand  # noqa: E402

DEFAULT_HANDS = 200
DEFAULT_OPPONENT = "simple"


@dataclass(frozen=True)
class SelfPlayResult:
    hands: int
    strat: str
    opponent: str
    wins: int
    losses: int
    pushes: int
    net_chips: int
    elapsed: float

    @property
    def bb_per_100(self):
        if self.hands == 0:
            return 0.0
        return self.net_chips / BIG_BLIND / self.hands * 100

    @property
    def hands_per_second(self):
        if self.elapsed <= 0:
            return 0
        return int(self.hands / self.elapsed)


def run_selfplay(
    strat_name,
    hands=DEFAULT_HANDS,
    seed=None,
    opponent_name=DEFAULT_OPPONENT,
):
    if hands < 0:
        raise ValueError("--hands must be non-negative")

    strat = load_strategy(strat_name)
    opponent = load_strategy(opponent_name)
    rng = random.Random(seed)

    wins = 0
    losses = 0
    pushes = 0
    net_chips = 0
    started = time.perf_counter()

    for hand_index in range(hands):
        player_is_small_blind = hand_index % 2 == 0
        player_stack, opponent_stack = play_hand(
            INITIAL_STACK,
            INITIAL_STACK,
            player_is_small_blind=player_is_small_blind,
            player_strategy=strat,
            bot_strategy=opponent,
            rng=rng,
            verbose=False,
        )
        net_chips += player_stack - INITIAL_STACK
        if player_stack > opponent_stack:
            wins += 1
        elif player_stack < opponent_stack:
            losses += 1
        else:
            pushes += 1

    elapsed = time.perf_counter() - started
    return SelfPlayResult(
        hands=hands,
        strat=strat_name,
        opponent=opponent_name,
        wins=wins,
        losses=losses,
        pushes=pushes,
        net_chips=net_chips,
        elapsed=elapsed,
    )


def format_signed_number(value):
    return f"{value:+d}"


def format_signed_float(value):
    return f"{value:+.1f}"


def format_result(result):
    return "\n".join(
        [
            f"  hands       : {result.hands}",
            f"  opponent    : {result.opponent} x1",
            (
                f"  wins/losses : {result.wins}/{result.losses}  "
                f"(push: {result.pushes})"
            ),
            f"  net chips   : {format_signed_number(result.net_chips)}",
            f"  bb/100      : {format_signed_float(result.bb_per_100)}",
            (
                f"  elapsed     : {result.elapsed:.1f}s  "
                f"({result.hands_per_second} hands/s)"
            ),
        ]
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run heads-up self-play against the simple strategy."
    )
    parser.add_argument(
        "--strat",
        required=True,
        help="Strategy module under poker_bot.strategies, e.g. all_in_everytime.",
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=DEFAULT_HANDS,
        help=f"Number of hands to play. Defaults to {DEFAULT_HANDS}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible self-play.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_selfplay(args.strat, hands=args.hands, seed=args.seed)
    print(format_result(result))


if __name__ == "__main__":
    main()
