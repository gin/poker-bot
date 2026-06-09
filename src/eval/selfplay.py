"""Run quiet self-play evaluations between poker strategies."""

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from poker_bot.opponent_store import (  # noqa: E402
    connect,
    create_telemetry_run,
    increment_hand_seen,
    load_profiles_for_agents,
    record_decision_telemetry,
    record_observed_action,
    update_hand_telemetry_outcome,
)
from poker_bot.strategies.loader import load_strategy  # noqa: E402
from simulator import (  # noqa: E402
    BIG_BLIND,
    INITIAL_STACK,
    PLAYER_AGENT_ID,
    play_hand,
    play_hand_multiway,
)

DEFAULT_HANDS = 200
DEFAULT_OPPONENT = "simple"
DEFAULT_PLAYERS = 2
OPPONENT_LINEUP_SEPARATOR = "+"


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
    players: int = DEFAULT_PLAYERS

    @property
    def bb_per_100(self):
        if self.hands == 0:
            return 0.0
        return self.net_chips / BIG_BLIND / self.hands * 100

    @property
    def chips_per_hand(self):
        if self.hands == 0:
            return 0.0
        return self.net_chips / self.hands

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
    players=DEFAULT_PLAYERS,
    track_opponents=False,
    opponent_db=None,
    telemetry=False,
    telemetry_run_id=None,
    platform="selfplay",
):
    if hands < 0:
        raise ValueError("--hands must be non-negative")
    if players < 2 or players > 6:
        raise ValueError("--players must be between 2 and 6")

    opponent_names = resolve_opponent_lineup(opponent_name, players)
    opponent_label = format_opponent_label(opponent_names)
    strat = load_strategy(strat_name)
    opponent_strategies = tuple(load_strategy(name) for name in opponent_names)
    rng = random.Random(seed)
    db_conn = connect(opponent_db) if opponent_db is not None else None
    should_track = track_opponents or db_conn is not None
    should_record_telemetry = telemetry or telemetry_run_id is not None
    if should_record_telemetry and db_conn is None:
        db_conn = connect()
    run_id = None
    if should_record_telemetry:
        run_id = create_telemetry_run(
            db_conn,
            strategy=strat_name,
            opponent=opponent_label,
            players=players,
            seed=seed,
            platform=platform,
            run_id=telemetry_run_id,
        )
    agent_ids = [PLAYER_AGENT_ID] + [
        f"bot-agent-{index}" for index in range(1, players)
    ]
    opponent_profiles = {}
    if should_track and db_conn is not None:
        opponent_profiles.update(load_profiles_for_agents(db_conn, platform, agent_ids))
    decision_indexes = {}

    def observe_action(**event):
        if db_conn is None:
            return
        seat = event["seat"]
        table = event["table"]
        hero = next(
            (
                table_seat
                for table_seat in table.get("seats", [])
                if table_seat.get("agentId") == PLAYER_AGENT_ID
            ),
            {},
        )
        record_observed_action(
            db_conn,
            platform=platform,
            agent_id=seat["agentId"],
            handle=seat.get("name"),
            hand_id=event.get("hand_id"),
            street=table.get("street"),
            action=event["action"],
            amount=event.get("amount"),
            pot=table.get("potChips"),
            message=event.get("message"),
            facing_bet=event.get("facing_bet", False),
            stack_chips=seat.get("stackChips"),
            hero_stack_chips=hero.get("stackChips"),
            voluntary=event.get("voluntary", False),
        )
        if should_record_telemetry and seat.get("agentId") == PLAYER_AGENT_ID:
            hand_id = event.get("hand_id")
            decision_index = decision_indexes.get(hand_id, 0)
            decision_indexes[hand_id] = decision_index + 1
            record_decision_telemetry(
                db_conn,
                run_id=run_id,
                hand_id=hand_id,
                decision_index=decision_index,
                strategy=strat_name,
                table=table,
                seat=seat,
                action=event["action"],
                amount=event.get("amount"),
                message=event.get("message"),
                facing_bet=event.get("facing_bet", False),
                voluntary=event.get("voluntary", False),
            )

    wins = 0
    losses = 0
    pushes = 0
    net_chips = 0
    started = time.perf_counter()

    for hand_index in range(hands):
        if players == 2:
            player_is_small_blind = hand_index % 2 == 0
            player_stack, opponent_stack = play_hand(
                INITIAL_STACK,
                INITIAL_STACK,
                player_is_small_blind=player_is_small_blind,
                player_strategy=strat,
                bot_strategy=opponent_strategies[0],
                rng=rng,
                verbose=False,
            )
            hero_delta = player_stack - INITIAL_STACK
            hero_won = player_stack > opponent_stack
            hero_lost = player_stack < opponent_stack
        else:
            if db_conn is not None:
                for agent_id in agent_ids:
                    increment_hand_seen(db_conn, platform, agent_id)
            hand_id = f"{seed or 'run'}-{hand_index}"
            stacks = play_hand_multiway(
                [INITIAL_STACK] * players,
                [strat, *opponent_strategies],
                button_index=hand_index % players,
                rng=rng,
                opponent_profiles=opponent_profiles if should_track else None,
                action_observer=observe_action if db_conn is not None else None,
                hand_id=hand_id,
                verbose=False,
            )
            hero_delta = stacks[0] - INITIAL_STACK
            hero_won = hero_delta > 0
            hero_lost = hero_delta < 0
            if should_record_telemetry:
                update_hand_telemetry_outcome(
                    db_conn,
                    run_id=run_id,
                    hand_id=hand_id,
                    hero_net_chips=hero_delta,
                    won_hand=hero_won,
                    final_pot=None,
                )

        net_chips += hero_delta
        if hero_won:
            wins += 1
        elif hero_lost:
            losses += 1
        else:
            pushes += 1

    elapsed = time.perf_counter() - started
    return SelfPlayResult(
        hands=hands,
        strat=strat_name,
        opponent=opponent_label,
        wins=wins,
        losses=losses,
        pushes=pushes,
        net_chips=net_chips,
        elapsed=elapsed,
        players=players,
    )


def parse_opponent_lineup(value):
    if isinstance(value, str):
        return tuple(
            part.strip()
            for part in value.split(OPPONENT_LINEUP_SEPARATOR)
            if part.strip()
        )
    return tuple(value)


def resolve_opponent_lineup(opponent_name, players):
    names = parse_opponent_lineup(opponent_name)
    if not names:
        raise ValueError("opponent lineup must include at least one strategy")
    if len(names) == 1:
        return names * (players - 1)
    if len(names) != players - 1:
        raise ValueError(
            "mixed opponent lineups must provide exactly one strategy per "
            f"opponent seat: got {len(names)} for {players} players"
        )
    return names


def format_opponent_label(opponent_names):
    if len(set(opponent_names)) == 1:
        return opponent_names[0]
    return OPPONENT_LINEUP_SEPARATOR.join(opponent_names)


def format_signed_number(value):
    return f"{value:+d}"


def format_signed_float(value):
    return f"{value:+.1f}"


def format_result(result):
    opponent = result.opponent
    if OPPONENT_LINEUP_SEPARATOR not in opponent:
        opponent = f"{opponent} x{result.players - 1}"
    return "\n".join(
        [
            f"  hands       : {result.hands}",
            f"  opponent    : {opponent}",
            (
                f"  wins/losses : {result.wins}/{result.losses}  "
                f"(push: {result.pushes})"
            ),
            f"  net chips   : {format_signed_number(result.net_chips)}",
            f"  chips/hand  : {format_signed_float(result.chips_per_hand)}",
            f"  bb/100      : {format_signed_float(result.bb_per_100)}",
            (
                f"  elapsed     : {result.elapsed:.1f}s  "
                f"({result.hands_per_second} hands/s)"
            ),
        ]
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run self-play between poker strategies."
    )
    parser.add_argument(
        "--strat",
        required=True,
        help="Strategy module under poker_bot.strategies, e.g. all_in_everytime.",
    )
    parser.add_argument(
        "--opponent",
        default=DEFAULT_OPPONENT,
        help=(
            "Opponent strategy module. For multiway mixed lineups, separate "
            f"one strategy per opponent seat with '{OPPONENT_LINEUP_SEPARATOR}'. "
            f"Defaults to {DEFAULT_OPPONENT}."
        ),
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=DEFAULT_HANDS,
        help=f"Number of hands to play. Defaults to {DEFAULT_HANDS}.",
    )
    parser.add_argument(
        "--players",
        type=int,
        default=DEFAULT_PLAYERS,
        help=f"Total players including hero, 2-6. Defaults to {DEFAULT_PLAYERS}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible self-play.",
    )
    parser.add_argument(
        "--track-opponents",
        action="store_true",
        help="Maintain in-memory opponent profiles during multiway self-play.",
    )
    parser.add_argument(
        "--opponent-db",
        default=None,
        help="Optional SQLite database path for persistent opponent profiles.",
    )
    parser.add_argument(
        "--telemetry",
        action="store_true",
        help="Record hero decision telemetry to the opponent SQLite database.",
    )
    parser.add_argument(
        "--telemetry-run-id",
        default=None,
        help="Optional run id for decision telemetry.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_selfplay(
        args.strat,
        hands=args.hands,
        seed=args.seed,
        opponent_name=args.opponent,
        players=args.players,
        track_opponents=args.track_opponents,
        opponent_db=args.opponent_db,
        telemetry=args.telemetry,
        telemetry_run_id=args.telemetry_run_id,
    )
    print(format_result(result))


if __name__ == "__main__":
    main()
