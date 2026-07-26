"""Run quiet self-play evaluations between poker strategies."""

import argparse
import concurrent.futures
import multiprocessing
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from uuid import uuid4

# Use 'spawn' to avoid fork() deadlock issues with threaded parents
multiprocessing.set_start_method("spawn", force=True)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval.profiler import StrategyProfile, format_profile  # noqa: E402
from poker_bot.guards.telemetry import drain_events as drain_guard_events  # noqa: E402
from poker_bot.opponent_store import (  # noqa: E402
    connect,
    create_telemetry_run,
    default_telemetry_db_path,
    increment_hand_seen,
    load_profiles_for_agents,
    merge_worker_db,
    record_decision_telemetry,
    record_guard_event,
    record_observed_action,
    update_hand_telemetry_outcome,
)
from poker_bot.strategies.loader import load_strategy  # noqa: E402
from simulator import (  # noqa: E402
    BIG_BLIND,
    BOT_AGENT_ID,
    INITIAL_STACK,
    PLAYER_AGENT_ID,
    play_hand,
    play_hand_multiway,
)

DEFAULT_HANDS = 200
DEFAULT_OPPONENT = "simple"
DEFAULT_PLAYERS = 2
DEFAULT_DB_COMMIT_INTERVAL = 1000
DEFAULT_WORKER_CAP = 16
OPPONENT_LINEUP_SEPARATOR = "+"
OPPONENT_LINEUP_INPUT_SEPARATORS = (OPPONENT_LINEUP_SEPARATOR, ",")
PROFILE_STATE_UNTRACKED = "untracked"
PROFILE_STATE_PERSISTENT = "persistent"
PROFILE_STATE_SHARDED_RESEARCH = "sharded_research"
PROFILE_STATE_MODES = frozenset(
    {
        PROFILE_STATE_UNTRACKED,
        PROFILE_STATE_PERSISTENT,
        PROFILE_STATE_SHARDED_RESEARCH,
    }
)


# This schema describes evaluator-observed route messages only. Profile-stat
# producers may later attach their own versioned provenance without this
# evaluator reading or interpreting profile internals.
PROFILE_ROUTE_DIAGNOSTICS_SCHEMA_VERSION = 1
ALTERNATE_V007_ROUTE_TAG = "[short_handed profile-gated s3v013]"
CANONICAL_V007_ROUTE_TAG = "[v007 canonical low-VPIP"
BASELINE_FALLBACK_ROUTE_TAG = "[short_handed]"


@dataclass(frozen=True)
class RouteDiagnostics:
    """Behavior-neutral route observations for one self-play result.

    Route state is inferred solely from stable strategy messages emitted at a
    hero decision. Unknown covers guards and strategies that do not expose a
    recognized message tag. It must never be used to influence play or gates.
    """

    schema_version: int = PROFILE_ROUTE_DIAGNOSTICS_SCHEMA_VERSION
    observed_hands: int = 0
    hero_decisions: int = 0
    alternate_decisions: int = 0
    fallback_decisions: int = 0
    unknown_decisions: int = 0
    alternate_hands: int = 0
    fallback_hands: int = 0
    unknown_hands: int = 0
    # Reserved integration point for the profile-stat owner. This evaluator
    # intentionally leaves it unset rather than inferring profile semantics.
    profile_stats_schema_version: int | None = None
    profile_stats_provenance: str | None = None
    profile_state_mode: str = PROFILE_STATE_UNTRACKED

    @property
    def activation_fraction(self):
        if self.observed_hands == 0:
            return 0.0
        return self.alternate_hands / self.observed_hands


def _route_state_from_message(message):
    if not isinstance(message, str):
        return "unknown"
    if (
        CANONICAL_V007_ROUTE_TAG in message
        or ALTERNATE_V007_ROUTE_TAG in message
    ):
        return "alternate"
    if BASELINE_FALLBACK_ROUTE_TAG in message:
        return "fallback"
    return "unknown"


class _RouteDiagnosticCollector:
    """Collect one decision state per hero action and one canonical state/hand."""

    _STATE_PRIORITY = {"unknown": 0, "fallback": 1, "alternate": 2}

    def __init__(self):
        self._decision_counts = {"alternate": 0, "fallback": 0, "unknown": 0}
        self._hand_states = {}

    def observe(self, *, hand_id, message):
        state = _route_state_from_message(message)
        self._decision_counts[state] += 1
        if hand_id is None:
            return
        previous = self._hand_states.get(hand_id)
        if (
            previous is None
            or self._STATE_PRIORITY[state] > self._STATE_PRIORITY[previous]
        ):
            self._hand_states[hand_id] = state

    def result(self):
        hand_counts = {"alternate": 0, "fallback": 0, "unknown": 0}
        for state in self._hand_states.values():
            hand_counts[state] += 1
        return RouteDiagnostics(
            observed_hands=len(self._hand_states),
            hero_decisions=sum(self._decision_counts.values()),
            alternate_decisions=self._decision_counts["alternate"],
            fallback_decisions=self._decision_counts["fallback"],
            unknown_decisions=self._decision_counts["unknown"],
            alternate_hands=hand_counts["alternate"],
            fallback_hands=hand_counts["fallback"],
            unknown_hands=hand_counts["unknown"],
        )


def merge_route_diagnostics(diagnostics):
    """Sum independent case or worker observations without changing behavior."""

    diagnostics = tuple(diagnostics)
    if not diagnostics:
        return RouteDiagnostics()

    def common_optional(name):
        observed = {
            getattr(item, name)
            for item in diagnostics
            if getattr(item, name) is not None
        }
        return next(iter(observed)) if len(observed) == 1 else None
    return RouteDiagnostics(
        schema_version=max(item.schema_version for item in diagnostics),
        observed_hands=sum(item.observed_hands for item in diagnostics),
        hero_decisions=sum(item.hero_decisions for item in diagnostics),
        alternate_decisions=sum(item.alternate_decisions for item in diagnostics),
        fallback_decisions=sum(item.fallback_decisions for item in diagnostics),
        unknown_decisions=sum(item.unknown_decisions for item in diagnostics),
        alternate_hands=sum(item.alternate_hands for item in diagnostics),
        fallback_hands=sum(item.fallback_hands for item in diagnostics),
        unknown_hands=sum(item.unknown_hands for item in diagnostics),
        profile_stats_schema_version=common_optional(
            "profile_stats_schema_version"
        ),
        profile_stats_provenance=common_optional("profile_stats_provenance"),
        profile_state_mode=common_optional("profile_state_mode")
        or PROFILE_STATE_UNTRACKED,
    )


def _exported_profile_metadata(profiles):
    """Return stable profile metadata without interpreting profile statistics."""

    values = tuple((profiles or {}).values())
    if not values:
        return None, None

    def exported_value(profile, name):
        if isinstance(profile, dict):
            return profile.get(name)
        return getattr(profile, name, None)

    def common(name):
        observed = {
            exported_value(profile, name)
            for profile in values
            if exported_value(profile, name) is not None
        }
        return next(iter(observed)) if len(observed) == 1 else None

    return common("profile_stats_schema_version"), common(
        "profile_stats_provenance"
    )


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
    initial_stack: int = INITIAL_STACK
    profile: StrategyProfile | None = None
    route_diagnostics: RouteDiagnostics = field(default_factory=RouteDiagnostics)

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
    players=None,
    track_opponents=False,
    opponent_db=None,
    telemetry=False,
    telemetry_run_id=None,
    platform="selfplay",
    profiler=None,
    db_commit_interval=DEFAULT_DB_COMMIT_INTERVAL,
    sqlite_fast=False,
    hand_id_prefix=None,
    initial_stack=INITIAL_STACK,
    profile_state_mode=PROFILE_STATE_PERSISTENT,
):
    if hands < 0:
        raise ValueError("--hands must be non-negative")
    if initial_stack <= 0:
        raise ValueError("--initial-stack must be positive")
    players = resolve_player_count(opponent_name, players)
    if players < 2 or players > 6:
        raise ValueError("--players must be between 2 and 6")
    if db_commit_interval is not None and db_commit_interval < 0:
        raise ValueError("--commit-every must be non-negative")
    if profile_state_mode not in PROFILE_STATE_MODES - {PROFILE_STATE_UNTRACKED}:
        raise ValueError(
            "profile_state_mode must be 'persistent' or 'sharded_research'"
        )
    if (
        profile_state_mode == PROFILE_STATE_SHARDED_RESEARCH
        and hand_id_prefix is None
    ):
        raise ValueError(
            "sharded_research profile mode is only available via parallel self-play"
        )


    opponent_names = resolve_opponent_lineup(opponent_name, players)
    opponent_label = format_opponent_label(opponent_names)
    strat = load_strategy(strat_name)
    opponent_strategies = tuple(load_strategy(name) for name in opponent_names)
    rng = random.Random(seed)
    db_conn = (
        connect(opponent_db, optimize_writes=sqlite_fast)
        if opponent_db is not None
        else None
    )
    should_track = track_opponents or db_conn is not None
    should_record_telemetry = telemetry or telemetry_run_id is not None
    if should_record_telemetry and db_conn is None:
        db_conn = connect(telemetry=True, optimize_writes=sqlite_fast)
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
            commit=False,
        )
    if players == 2:
        agent_ids = [PLAYER_AGENT_ID, BOT_AGENT_ID]
    else:
        agent_ids = [PLAYER_AGENT_ID] + [
            f"bot-agent-{index}" for index in range(1, players)
        ]
    opponent_profiles = {}
    if should_track and db_conn is not None:
        opponent_profiles.update(load_profiles_for_agents(db_conn, platform, agent_ids))
    decision_indexes = {}
    route_diagnostics = _RouteDiagnosticCollector()

    # Build the profiler observer if a profiler was given
    use_profiler = profiler is not None

    def combined_observer(**event):
        """Action observer used in ALL modes (heads-up and multiway).
        Forwards to the profiler and the DB-backed observer_action."""
        # Drain the guard event buffer on EVERY action so opponents' guard
        # fires (mirror matches) are discarded rather than misattributed to
        # the hero's next decision. Hero events are persisted below.
        guard_events = drain_guard_events()
        if use_profiler:
            profiler._observe(**event)
        seat = event.get("seat") or {}
        if seat.get("agentId") == PLAYER_AGENT_ID:
            route_diagnostics.observe(
                hand_id=event.get("hand_id"),
                message=event.get("message"),
            )
        if db_conn is not None and "seat" in event and "table" in event:
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
                commit=False,
            )
            if should_record_telemetry and seat.get("agentId") == PLAYER_AGENT_ID:
                h_id = event.get("hand_id")
                decision_index = decision_indexes.get(h_id, 0)
                decision_indexes[h_id] = decision_index + 1
                record_decision_telemetry(
                    db_conn,
                    run_id=run_id,
                    hand_id=h_id,
                    decision_index=decision_index,
                    strategy=strat_name,
                    table=table,
                    seat=seat,
                    action=event["action"],
                    amount=event.get("amount"),
                    message=event.get("message"),
                    facing_bet=event.get("facing_bet", False),
                    voluntary=event.get("voluntary", False),
                    commit=False,
                )
                for guard_event in guard_events:
                    record_guard_event(
                        db_conn,
                        run_id=run_id,
                        hand_id=h_id,
                        decision_index=decision_index,
                        event=guard_event,
                        commit=False,
                    )

    wins = 0
    losses = 0
    pushes = 0
    net_chips = 0
    started = time.perf_counter()

    for hand_index in range(hands):
        hand_id_base = f"{seed or 'run'}-{hand_index}"
        hand_id = f"{hand_id_prefix}-{hand_id_base}" if hand_id_prefix else hand_id_base
        if use_profiler:
            profiler.start_hand(hand_id)

        if players == 2:
            if db_conn is not None:
                for agent_id in agent_ids:
                    increment_hand_seen(
                        db_conn,
                        platform,
                        agent_id,
                        hand_id=hand_id,
                        commit=False,
                    )
            player_is_small_blind = hand_index % 2 == 0
            player_stack, opponent_stack = play_hand(
                initial_stack,
                initial_stack,
                player_is_small_blind=player_is_small_blind,
                player_strategy=strat,
                bot_strategy=opponent_strategies[0],
                rng=rng,
                verbose=False,
                action_observer=combined_observer,
                hand_id=hand_id,
                opponent_profiles=opponent_profiles if should_track else None,
            )
            hero_delta = player_stack - initial_stack
            hero_won = player_stack > opponent_stack
            hero_lost = player_stack < opponent_stack
        else:
            if db_conn is not None:
                for agent_id in agent_ids:
                    increment_hand_seen(
                        db_conn,
                        platform,
                        agent_id,
                        hand_id=hand_id,
                        commit=False,
                    )
            stacks = play_hand_multiway(
                [initial_stack] * players,
                [strat, *opponent_strategies],
                button_index=hand_index % players,
                rng=rng,
                opponent_profiles=opponent_profiles if should_track else None,
                action_observer=combined_observer,
                hand_id=hand_id,
                verbose=False,
            )
            hero_delta = stacks[0] - initial_stack
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
                commit=False,
            )

        net_chips += hero_delta
        if hero_won:
            wins += 1
        elif hero_lost:
            losses += 1
        else:
            pushes += 1

        if use_profiler:
            cur = profiler._current
            # Use the showdown event flag if set by the simulator,
            # otherwise fall back to river-action heuristics
            showdown = (cur is not None and cur.saw_showdown) or (
                cur is not None
                and cur.saw_flop
                and not cur.folded_hand
                and cur.river_acted
            )
            profiler.end_hand(won=hero_won, showdown=showdown)

        if (
            db_conn is not None
            and db_commit_interval
            and db_commit_interval > 0
            and (hand_index + 1) % db_commit_interval == 0
        ):
            db_conn.commit()

    elapsed = time.perf_counter() - started
    if db_conn is not None:
        db_conn.commit()
    profile_schema_version, profile_provenance = _exported_profile_metadata(
        opponent_profiles
    )
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
        initial_stack=initial_stack,
        route_diagnostics=replace(
            route_diagnostics.result(),
            profile_stats_schema_version=profile_schema_version,
            profile_stats_provenance=profile_provenance,
            profile_state_mode=(
                profile_state_mode if should_track else PROFILE_STATE_UNTRACKED
            ),
        ),
    )


def _resolve_workers(workers: int) -> int:
    if workers <= 0:
        cpu = os.cpu_count() or 1
        workers = max(1, cpu // 2)
    if (
        workers > DEFAULT_WORKER_CAP
        and os.environ.get("POKER_SELFPLAY_ALLOW_HIGH_WORKERS") != "1"
    ):
        raise ValueError(
            f"workers={workers} exceeds safety cap {DEFAULT_WORKER_CAP}; "
            "set POKER_SELFPLAY_ALLOW_HIGH_WORKERS=1 to override"
        )
    return workers


def _split_hands(hands, workers):
    base, remainder = divmod(hands, workers)
    return tuple(base + (1 if index < remainder else 0) for index in range(workers))


def _run_selfplay_worker(args):
    (
        worker_index,
        worker_hands,
        strat_name,
        seed,
        opponent_name,
        players,
        track_opponents,
        opponent_db,
        telemetry,
        telemetry_run_id,
        platform,
        db_commit_interval,
        sqlite_fast,
        initial_stack,
        profile_state_mode,
    ) = args
    worker_seed = None if seed is None else seed + worker_index
    return run_selfplay(
        strat_name,
        hands=worker_hands,
        seed=worker_seed,
        opponent_name=opponent_name,
        players=players,
        track_opponents=track_opponents,
        opponent_db=opponent_db,
        telemetry=telemetry,
        telemetry_run_id=telemetry_run_id,
        platform=platform,
        db_commit_interval=db_commit_interval,
        sqlite_fast=sqlite_fast,
        hand_id_prefix=f"w{worker_index}",
        initial_stack=initial_stack,
        profile_state_mode=profile_state_mode,
    )


def _aggregate_worker_results(results, *, elapsed):
    if not results:
        raise ValueError("no worker results to aggregate")
    first = results[0]
    return SelfPlayResult(
        hands=sum(result.hands for result in results),
        strat=first.strat,
        opponent=first.opponent,
        wins=sum(result.wins for result in results),
        losses=sum(result.losses for result in results),
        pushes=sum(result.pushes for result in results),
        net_chips=sum(result.net_chips for result in results),
        elapsed=elapsed,
        players=first.players,
        initial_stack=first.initial_stack,
        route_diagnostics=merge_route_diagnostics(
            result.route_diagnostics for result in results
        ),
    )


def run_selfplay_parallel(
    strat_name,
    *,
    hands=DEFAULT_HANDS,
    seed=None,
    opponent_name=DEFAULT_OPPONENT,
    players=None,
    track_opponents=False,
    opponent_db=None,
    telemetry=False,
    telemetry_run_id=None,
    platform="selfplay",
    workers=0,
    db_commit_interval=DEFAULT_DB_COMMIT_INTERVAL,
    sqlite_fast=False,
    initial_stack=INITIAL_STACK,
    profile_state_mode=PROFILE_STATE_PERSISTENT,
):
    if hands < 0:
        raise ValueError("--hands must be non-negative")
    if initial_stack <= 0:
        raise ValueError("--initial-stack must be positive")
    if db_commit_interval is not None and db_commit_interval < 0:
        raise ValueError("--commit-every must be non-negative")
    players = resolve_player_count(opponent_name, players)
    actual_workers = _resolve_workers(workers)
    if profile_state_mode not in PROFILE_STATE_MODES - {PROFILE_STATE_UNTRACKED}:
        raise ValueError(
            "profile_state_mode must be 'persistent' or 'sharded_research'"
        )
    tracks_profiles = track_opponents or opponent_db is not None
    if (
        profile_state_mode == PROFILE_STATE_SHARDED_RESEARCH
        and not tracks_profiles
    ):
        raise ValueError(
            "sharded_research profile mode requires tracked opponent profiles"
        )
    if (
        profile_state_mode == PROFILE_STATE_SHARDED_RESEARCH
        and actual_workers <= 1
    ):
        raise ValueError("sharded_research profile mode requires more than one worker")

    if (
        tracks_profiles
        and actual_workers > 1
        and profile_state_mode != PROFILE_STATE_SHARDED_RESEARCH
    ):
        raise ValueError(
            "tracked parallel self-play requires one worker for persistent "
            "profiles; set profile_state_mode='sharded_research' only for "
            "explicit research"
        )
    if actual_workers <= 1 or hands <= 1:
        return run_selfplay(
            strat_name,
            hands=hands,
            seed=seed,
            opponent_name=opponent_name,
            players=players,
            track_opponents=track_opponents,
            opponent_db=opponent_db,
            telemetry=telemetry,
            telemetry_run_id=telemetry_run_id,
            platform=platform,
            db_commit_interval=db_commit_interval,
            sqlite_fast=sqlite_fast,
            initial_stack=initial_stack,
            profile_state_mode=profile_state_mode,
        )

    hand_splits = tuple(split for split in _split_hands(hands, actual_workers) if split)
    shared_run_id = telemetry_run_id or (uuid4().hex if telemetry else None)
    merge_db = opponent_db
    if merge_db is None and telemetry:
        merge_db = default_telemetry_db_path()

    started = time.perf_counter()
    if merge_db is None:
        worker_db_paths = [None] * len(hand_splits)
        temp_dir_ctx = None
    else:
        temp_dir_ctx = tempfile.TemporaryDirectory()
        worker_db_paths = [
            os.path.join(temp_dir_ctx.name, f"worker_{index}.sqlite")
            for index in range(len(hand_splits))
        ]
        for path in worker_db_paths:
            connect(path, optimize_writes=sqlite_fast).close()

    try:
        worker_args = [
            (
                index,
                worker_hands,
                strat_name,
                seed,
                opponent_name,
                players,
                track_opponents,
                worker_db_paths[index],
                telemetry,
                shared_run_id,
                platform,
                db_commit_interval,
                sqlite_fast,
                initial_stack,
                profile_state_mode,
            )
            for index, worker_hands in enumerate(hand_splits)
        ]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(hand_splits)
        ) as pool:
            results = list(pool.map(_run_selfplay_worker, worker_args))
        if merge_db is not None:
            for path in worker_db_paths:
                merge_worker_db(merge_db, path)
    finally:
        if temp_dir_ctx is not None:
            temp_dir_ctx.cleanup()

    return _aggregate_worker_results(results, elapsed=time.perf_counter() - started)


def parse_opponent_lineup(value):
    if isinstance(value, str):
        normalized = value
        for separator in OPPONENT_LINEUP_INPUT_SEPARATORS:
            normalized = normalized.replace(separator, OPPONENT_LINEUP_SEPARATOR)
        return tuple(
            part.strip()
            for part in normalized.split(OPPONENT_LINEUP_SEPARATOR)
            if part.strip()
        )
    return tuple(value)


def infer_player_count(opponent_name):
    return len(parse_opponent_lineup(opponent_name)) + 1


def resolve_player_count(opponent_name, players):
    if players is None:
        return infer_player_count(opponent_name)
    return players


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
    return f"{int(value):+d}"


def format_signed_float(value):
    return f"{value:+.1f}"


def format_result(result):
    opponent = result.opponent
    if OPPONENT_LINEUP_SEPARATOR not in opponent:
        opponent = f"{opponent} x{result.players - 1}"
    lines = [
        f"  hands       : {result.hands}",
        f"  opponent    : {opponent}",
        (f"  wins/losses : {result.wins}/{result.losses}  (push: {result.pushes})"),
        f"  net chips   : {format_signed_number(result.net_chips)}",
        f"  chips/hand  : {format_signed_float(result.chips_per_hand)}",
        f"  bb/100      : {format_signed_float(result.bb_per_100)}",
        (f"  elapsed     : {result.elapsed:.1f}s  ({result.hands_per_second} hands/s)"),
        (
            "  route diag  : "
            f"v{result.route_diagnostics.schema_version}, "
            f"{result.route_diagnostics.alternate_hands}/"
            f"{result.route_diagnostics.observed_hands} alternate hands "
            f"({result.route_diagnostics.activation_fraction:.1%}), "
            f"{result.route_diagnostics.alternate_decisions}/"
            f"{result.route_diagnostics.fallback_decisions}/"
            f"{result.route_diagnostics.unknown_decisions} decisions "
            "(alternate/fallback/unknown)"
        ),
        f"  profile state: {result.route_diagnostics.profile_state_mode}",
    ]
    if result.profile is not None:
        profile_text = format_profile(result.profile)
        lines.append("  profile     :")
        for line in profile_text.split("\n"):
            lines.append(f"    {line}")
    else:
        lines.append("  profile     : (not profiled)")
    return "\n".join(lines)


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
            "one strategy per opponent seat with ',' or "
            f"'{OPPONENT_LINEUP_SEPARATOR}'. "
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
        default=None,
        help=("Total players including hero, 2-6. Defaults to opponent count + 1."),
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
        help="Maintain in-memory opponent profiles during self-play.",
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
    parser.add_argument(
        "--commit-every",
        type=int,
        default=DEFAULT_DB_COMMIT_INTERVAL,
        help=(
            "SQLite commit interval in hands when tracking opponents or telemetry. "
            "Use 0 to commit once at the end."
        ),
    )
    parser.add_argument(
        "--sqlite-fast",
        action="store_true",
        help=(
            "Use faster SQLite pragmas for bulk self-play writes "
            "(WAL + synchronous=NORMAL + larger cache)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Run hands across multiple worker processes. Use 0 for half the CPU "
            f"count. Safety cap: {DEFAULT_WORKER_CAP} workers."
        ),
    )
    parser.add_argument(
        "--initial-stack",
        type=int,
        default=INITIAL_STACK,
        dest="initial_stack",
        help=(
            "Starting stack for hero and every opponent, in chips "
            f"(default: {INITIAL_STACK}, i.e. {INITIAL_STACK // BIG_BLIND} big "
            "blinds). Hero's per-hand delta is measured against this stack."
        ),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    players = resolve_player_count(args.opponent, args.players)
    result = run_selfplay_parallel(
        args.strat,
        hands=args.hands,
        seed=args.seed,
        opponent_name=args.opponent,
        players=players,
        track_opponents=args.track_opponents,
        opponent_db=args.opponent_db,
        telemetry=args.telemetry,
        telemetry_run_id=args.telemetry_run_id,
        workers=args.workers,
        db_commit_interval=args.commit_every,
        sqlite_fast=args.sqlite_fast,
        initial_stack=args.initial_stack,
    )
    print(format_result(result))


if __name__ == "__main__":
    main()
