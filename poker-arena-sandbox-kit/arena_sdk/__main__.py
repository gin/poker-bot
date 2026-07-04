"""CLI:  python -m arena_sdk selfplay --strategy examples/poker/strategy.py
        python -m arena_sdk pack     --strategy examples/poker/strategy.py
        python -m arena_sdk submit   --strategy examples/poker/strategy.py --competition <id>
"""
from __future__ import annotations

import argparse
import json
import sys

from .poker.contract import load_strategy

# Subcommands that own their own argparse — route argv straight to them.
_DELEGATED = {"pack", "submit", "comps"}


def _add_common(sp, opponents):
    sp.add_argument("--strategy", required=True, help="path to strategy.py (act(table))")
    sp.add_argument("--players", type=int, default=2, help="2=HU (default), up to 6")
    sp.add_argument("--opponent", default="tight",
                    choices=list(opponents) + ["mixed", "self"])
    sp.add_argument("--starting-stack", type=int, default=200, help="chips (200=100bb)")
    sp.add_argument("--sb", type=int, default=1)
    sp.add_argument("--bb", type=int, default=2)
    sp.add_argument("--seed", type=int, default=None)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("version", "--version", "-V"):
        from . import __version__
        print(f"arena-sdk {__version__}")
        return 0
    if argv and argv[0] == "register":
        from .auth import register_main
        return register_main(argv[1:])
    if argv and argv[0] == "claim":
        from .auth import claim_main
        return claim_main(argv[1:])
    if argv and argv[0] == "access":
        from .submit import access_main
        return access_main(argv[1:])
    if argv and argv[0] in _DELEGATED:
        cmd, rest = argv[0], argv[1:]
        if cmd == "pack":
            from .pack import main as _m
        elif cmd == "submit":
            from .submit import main as _m
        else:  # comps
            from .comps import main as _m
        return _m(rest)

    ap = argparse.ArgumentParser(
        prog="arena",
        description="dev.fun Arena SDK — build, test, and submit Arena agents.",
        epilog="more subcommands (run `<cmd> --help`): register · claim · access · "
               "comps · pack · submit · version")
    # selfplay is the only command that needs the engine (and pokerkit).
    try:
        from .poker.engine import run_match, OPPONENTS
    except ModuleNotFoundError:
        print("selfplay needs the local engine — install it with: "
              "pip install 'arena-sdk[selfplay]'", file=sys.stderr)
        return 1
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp1 = sub.add_parser("selfplay", help="local self-play vs built-in bots -> bb/100")
    _add_common(sp1, OPPONENTS); sp1.add_argument("--hands", type=int, default=2000)
    args = ap.parse_args(argv)

    strat = load_strategy(args.strategy)
    print(f"loaded strategy '{args.strategy}' via entrypoint: {strat.entrypoint}()",
          file=sys.stderr)
    res = run_match(strat, hands=args.hands, opponent=args.opponent,
                    players=args.players, starting_stack=args.starting_stack,
                    small_blind=args.sb, big_blind=args.bb, seed=args.seed)
    print(json.dumps(res, indent=2))
    print(f"\n  >> bb/100 = {res['bb_per_100']:+.2f}  over {res['hands']} hands "
          f"vs {res['opponent']} ({res['hands_per_s']} hands/s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
