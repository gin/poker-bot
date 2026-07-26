"""Persistent multi-hand tournament evaluator.

Unlike ``eval.selfplay``/``eval.benchmark`` (which reset every player back
to a fixed starting stack each hand and report hand-level bb/100 EV), this
module plays real freezeout tournaments: stacks persist across hands, the
button rotates among surviving seats, busted players are removed before the
next deal, and blinds escalate on a configurable schedule. It reports
tournament-level outcomes (wins, finish position, hands survived) -- never
bb/100 -- because hand-EV and tournament placement are different metrics
that must not be conflated.

Hands are played through :func:`simulator.play_hand_multiway`, the exact
same table/action path ``eval.selfplay`` uses for multiway hands, so
strategies see the same table contract they see in every other evaluator.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval.selfplay import (  # noqa: E402
    format_opponent_label,
    resolve_opponent_lineup,
    resolve_player_count,
)
from poker_bot.strategies.loader import load_strategy  # noqa: E402
from simulator import (  # noqa: E402
    INITIAL_STACK,
    PLAYER_AGENT_ID,
    play_hand_multiway,
)

DEFAULT_INITIAL_STACK = INITIAL_STACK
DEFAULT_TOURNAMENT_COUNT = 100
DEFAULT_SEED = 1
DEFAULT_MAX_HANDS = 5000
DEFAULT_OPPONENT = "simple"


@dataclass(frozen=True)
class BlindLevel:
    small_blind: int
    big_blind: int
    # Number of hands this level lasts before advancing. ``None`` (or 0)
    # means "lasts for the rest of the tournament" -- only valid on the
    # final level of a schedule.
    hands: int | None = None


DEFAULT_BLIND_LEVELS: tuple[BlindLevel, ...] = (
    BlindLevel(5, 10, 20),
    BlindLevel(10, 20, 20),
    BlindLevel(15, 30, 20),
    BlindLevel(25, 50, 20),
    BlindLevel(50, 100, 20),
    BlindLevel(75, 150, 20),
    BlindLevel(100, 200, None),
)


def parse_blind_schedule(value):
    """Parse ``"sb/bb[:hands],sb/bb[:hands],..."`` into a tuple of
    :class:`BlindLevel`. Only the final level may omit ``:hands`` (or use
    ``:0``), meaning it lasts for the rest of the tournament."""
    if value is None:
        return DEFAULT_BLIND_LEVELS
    if isinstance(value, tuple) and all(isinstance(v, BlindLevel) for v in value):
        levels = value
    else:
        levels = []
        for raw_level in str(value).split(","):
            raw_level = raw_level.strip()
            if not raw_level:
                continue
            blind_part, _, hands_part = raw_level.partition(":")
            sb_str, _, bb_str = blind_part.partition("/")
            if not sb_str or not bb_str:
                raise ValueError(
                    f"invalid blind schedule entry {raw_level!r}; "
                    "expected 'sb/bb' or 'sb/bb:hands'"
                )
            hands = int(hands_part) if hands_part else None
            levels.append(BlindLevel(int(sb_str), int(bb_str), hands))
        levels = tuple(levels)
    validate_blind_schedule(levels)
    return levels


def validate_blind_schedule(levels):
    if not levels:
        raise ValueError("blind schedule must have at least one level")
    for level in levels[:-1]:
        if not level.hands or level.hands <= 0:
            raise ValueError(
                "only the final blind level may have an unbounded (None/0) "
                f"duration; got {level} before the end of the schedule"
            )
    for level in levels:
        if level.small_blind <= 0 or level.big_blind <= 0:
            raise ValueError(f"blind levels must be positive chip amounts: {level}")
        if level.small_blind > level.big_blind:
            raise ValueError(f"small blind must not exceed big blind: {level}")


def format_blind_schedule(levels):
    parts = []
    for level in levels:
        part = f"{level.small_blind}/{level.big_blind}"
        if level.hands:
            part += f":{level.hands}"
        parts.append(part)
    return ",".join(parts)


def blind_level_for_hand(blind_levels, hand_index):
    """``hand_index`` is 0-based across the whole tournament (not reset per
    level). Returns the :class:`BlindLevel` active for that hand; the final
    level persists indefinitely once earlier levels' hand budgets run out.
    """
    remaining = hand_index
    for level in blind_levels[:-1]:
        if remaining < level.hands:
            return level
        remaining -= level.hands
    return blind_levels[-1]


@dataclass(frozen=True)
class TournamentConfig:
    hero_strategy: str
    opponent_lineup: tuple[str, ...]
    initial_stack: int = DEFAULT_INITIAL_STACK
    blind_levels: tuple[BlindLevel, ...] = DEFAULT_BLIND_LEVELS
    tournament_count: int = DEFAULT_TOURNAMENT_COUNT
    seed: int = DEFAULT_SEED
    max_hands: int = DEFAULT_MAX_HANDS

    def __post_init__(self):
        if self.initial_stack <= 0:
            raise ValueError("initial_stack must be positive")
        if self.tournament_count <= 0:
            raise ValueError("tournament_count must be positive")
        if self.max_hands <= 0:
            raise ValueError("max_hands must be positive")
        if not (2 <= self.players <= 6):
            raise ValueError("tournament supports 2 to 6 starting players")
        validate_blind_schedule(self.blind_levels)

    @property
    def players(self):
        return len(self.opponent_lineup) + 1


@dataclass
class _TournamentSeat:
    agent_id: str
    strategy_name: str
    strategy: object
    stack: int
    busted_hand: int | None = None
    finish_position: int | None = None


@dataclass(frozen=True)
class SeatResult:
    agent_id: str
    strategy: str
    finish_position: int
    busted_hand: int | None
    final_stack: int


@dataclass(frozen=True)
class HeroStageDelta:
    """One hand's hero chip delta, recorded only while the hero was still
    dealt into that hand, tagged with the table stage (players actually
    dealt in this hand, and the blind level) it occurred at.

    This is a diagnostic breakdown separate from tournament placement
    metrics (finish position / win rate) -- it never feeds back into
    them, and its bb/100 figure is a per-stage hand-EV diagnostic, not a
    substitute for the tournament-level outcome.
    """

    players_dealt_in: int
    small_blind: int
    big_blind: int
    hero_chip_delta: int


@dataclass(frozen=True)
class TournamentResult:
    index: int
    seed: int
    hands_played: int
    seats: tuple[SeatResult, ...]
    hero_finish_position: int
    hero_won: bool
    hero_stage_deltas: tuple[HeroStageDelta, ...] = ()


def _build_seats(config):
    strategies = [load_strategy(config.hero_strategy)] + [
        load_strategy(name) for name in config.opponent_lineup
    ]
    agent_ids = [PLAYER_AGENT_ID] + [
        f"bot-agent-{index}" for index in range(1, config.players)
    ]
    strategy_names = [config.hero_strategy, *config.opponent_lineup]
    return [
        _TournamentSeat(
            agent_id=agent_ids[index],
            strategy_name=strategy_names[index],
            strategy=strategies[index],
            stack=config.initial_stack,
        )
        for index in range(config.players)
    ]


def run_tournament(config, tournament_index, *, action_observer=None):
    """Play one freezeout tournament to completion (or until
    ``config.max_hands`` is hit as a safety cap) and return a
    :class:`TournamentResult`.

    Deterministic: the same ``(config.seed, tournament_index)`` pair always
    produces the same sequence of deals and therefore the same outcome,
    since neither button rotation nor blind escalation depends on chance --
    only the deck shuffle does, and it is seeded solely from this pair.
    """
    rng = random.Random(f"{config.seed}:{tournament_index}")
    seats = _build_seats(config)
    n = len(seats)
    opponent_profiles: dict = {}
    button_slot = 0
    hands_played = 0
    finish_rank = n
    hero_stage_deltas: list[HeroStageDelta] = []

    while hands_played < config.max_hands:
        alive_seats = [seat for seat in seats if seat.busted_hand is None]
        if len(alive_seats) <= 1:
            break

        level = blind_level_for_hand(config.blind_levels, hands_played)
        # Table order starting at the button; core routing (and the deck
        # deal) only ever sees seats still dealt into this hand.
        order = [seats[(button_slot + offset) % n] for offset in range(n)]
        ordered_alive = [seat for seat in order if seat.busted_hand is None]
        starting_stacks = [seat.stack for seat in ordered_alive]
        # Hero (always seats[0]) may or may not be dealt into this hand;
        # once busted, `_build_seats` never removes them from `seats`, but
        # `ordered_alive` (and therefore the hand itself) stops including
        # them from that point on.
        hero_seat = seats[0]
        hero_dealt_in = hero_seat.busted_hand is None
        hero_stack_before = hero_seat.stack if hero_dealt_in else None

        hand_id = f"tournament{tournament_index}-hand{hands_played}"
        final_stacks = play_hand_multiway(
            list(starting_stacks),
            [seat.strategy for seat in ordered_alive],
            button_index=0,
            rng=rng,
            opponent_profiles=opponent_profiles,
            action_observer=action_observer,
            hand_id=hand_id,
            verbose=False,
            agent_ids=[seat.agent_id for seat in ordered_alive],
            small_blind=level.small_blind,
            big_blind=level.big_blind,
        )
        for seat, stack in zip(ordered_alive, final_stacks, strict=True):
            seat.stack = stack
        hands_played += 1

        if hero_dealt_in:
            hero_stage_deltas.append(
                HeroStageDelta(
                    players_dealt_in=len(ordered_alive),
                    small_blind=level.small_blind,
                    big_blind=level.big_blind,
                    hero_chip_delta=hero_seat.stack - hero_stack_before,
                )
            )

        newly_busted = [seat for seat in ordered_alive if seat.stack == 0]
        if newly_busted:
            # Simultaneous bust-outs are ranked by the chip stack they
            # brought into the hand: `finish_rank` here is the WORST
            # remaining position and decreases with each assignment below,
            # so the smallest stack must be assigned first (worst) and the
            # largest stack last (best of the batch) -- ascending order.
            newly_busted.sort(
                key=lambda seat: starting_stacks[ordered_alive.index(seat)],
            )
            for seat in newly_busted:
                seat.busted_hand = hands_played
                seat.finish_position = finish_rank
                finish_rank -= 1

        remaining_alive = [seat for seat in seats if seat.busted_hand is None]
        if len(remaining_alive) <= 1:
            break
        next_slot = (button_slot + 1) % n
        while seats[next_slot].busted_hand is not None:
            next_slot = (next_slot + 1) % n
        button_slot = next_slot

    survivors = [seat for seat in seats if seat.busted_hand is None]
    if len(survivors) == 1:
        survivors[0].finish_position = 1
    elif len(survivors) > 1:
        # Safety cap reached before a heads-up winner emerged: rank the
        # remaining survivors by stack. `finish_rank` is the worst
        # remaining position and decreases with each assignment, so the
        # smallest stack must be assigned first (worst) and the largest
        # stack last, ending at position 1 (best remaining finish).
        for seat in sorted(survivors, key=lambda s: s.stack):
            seat.finish_position = finish_rank
            finish_rank -= 1

    seat_results = tuple(
        SeatResult(
            agent_id=seat.agent_id,
            strategy=seat.strategy_name,
            finish_position=seat.finish_position,
            busted_hand=seat.busted_hand,
            final_stack=seat.stack,
        )
        for seat in seats
    )
    hero = seats[0]
    return TournamentResult(
        index=tournament_index,
        seed=config.seed,
        hands_played=hands_played,
        seats=seat_results,
        hero_finish_position=hero.finish_position,
        hero_won=hero.finish_position == 1,
        hero_stage_deltas=tuple(hero_stage_deltas),
    )


@dataclass(frozen=True)
class ChipEVStage:
    """Aggregated hero chip-EV diagnostic for one (players dealt in, small
    blind, big blind) stage, pooled across an entire tournament batch.

    This is a hand-EV diagnostic breakdown, separate from (and never fed
    back into) the tournament-level placement metrics on
    :class:`TournamentBatchReport` -- it exists to show which table
    stages the hero is actually winning or losing chips at, not to
    replace win-rate/finish-position as the tournament outcome.
    """

    players_dealt_in: int
    small_blind: int
    big_blind: int
    hands: int
    net_chips: int

    @property
    def chips_per_hand(self):
        if self.hands == 0:
            return 0.0
        return self.net_chips / self.hands

    @property
    def bb_per_100(self):
        if self.hands == 0 or self.big_blind == 0:
            return 0.0
        return self.net_chips / self.big_blind / self.hands * 100


@dataclass(frozen=True)
class TournamentBatchReport:
    hero_strategy: str
    opponent_lineup: tuple[str, ...]
    players: int
    initial_stack: int
    blind_levels: tuple[BlindLevel, ...]
    tournament_count: int
    seed: int
    results: tuple[TournamentResult, ...]
    elapsed: float

    @property
    def hero_wins(self):
        return sum(1 for result in self.results if result.hero_won)

    @property
    def hero_win_rate(self):
        if not self.results:
            return 0.0
        return self.hero_wins / len(self.results)

    @property
    def mean_hands_per_tournament(self):
        if not self.results:
            return 0.0
        return sum(result.hands_played for result in self.results) / len(self.results)

    @property
    def mean_hero_finish_position(self):
        if not self.results:
            return 0.0
        return sum(result.hero_finish_position for result in self.results) / len(
            self.results
        )

    @property
    def finish_position_distribution(self):
        counts: dict[int, int] = {}
        for result in self.results:
            counts[result.hero_finish_position] = (
                counts.get(result.hero_finish_position, 0) + 1
            )
        return dict(sorted(counts.items()))

    @property
    def chip_ev_by_stage(self):
        """Hero chip-EV diagnostic per (players dealt in, small blind, big
        blind) stage, pooled across every tournament in this batch.
        Ordered by blind level (ascending), then by player count
        (descending, full table first)."""
        grouped: dict[tuple[int, int, int], list[int]] = {}
        for result in self.results:
            for delta in result.hero_stage_deltas:
                key = (delta.players_dealt_in, delta.small_blind, delta.big_blind)
                grouped.setdefault(key, []).append(delta.hero_chip_delta)
        rows = [
            ChipEVStage(
                players_dealt_in=players_dealt_in,
                small_blind=small_blind,
                big_blind=big_blind,
                hands=len(deltas),
                net_chips=sum(deltas),
            )
            for (players_dealt_in, small_blind, big_blind), deltas in grouped.items()
        ]
        rows.sort(
            key=lambda row: (row.small_blind, row.big_blind, -row.players_dealt_in)
        )
        return tuple(rows)


def run_tournament_batch(config):
    started = time.perf_counter()
    results = tuple(
        run_tournament(config, index) for index in range(config.tournament_count)
    )
    return TournamentBatchReport(
        hero_strategy=config.hero_strategy,
        opponent_lineup=config.opponent_lineup,
        players=config.players,
        initial_stack=config.initial_stack,
        blind_levels=config.blind_levels,
        tournament_count=config.tournament_count,
        seed=config.seed,
        results=results,
        elapsed=time.perf_counter() - started,
    )


def _signed_int(value):
    """Format a chip total with an explicit sign. Chip totals are always
    conceptually integral, but upstream arithmetic can leave one typed as
    a float (e.g. a strategy computing a bet size via a float multiplier
    such as ``BIG_BLIND * 2.5``) -- Python's ``:+d`` format spec rejects
    floats outright even when whole-valued, so round-trip through ``int``
    first rather than assuming the type."""
    return f"{int(value):+d}"


def format_report(report):
    opponent_label = format_opponent_label(report.opponent_lineup)
    lines = [
        f"hero           : {report.hero_strategy}",
        f"opponents      : {opponent_label}",
        f"players        : {report.players}",
        f"initial stack  : {report.initial_stack}",
        f"blind schedule : {format_blind_schedule(report.blind_levels)}",
        f"tournaments    : {report.tournament_count}",
        f"hero wins      : {report.hero_wins}/{report.tournament_count} "
        f"({report.hero_win_rate:.1%})",
        f"mean finish    : {report.mean_hero_finish_position:.2f} "
        f"(1 = winner, {report.players} = first out)",
        f"mean hands/t   : {report.mean_hands_per_tournament:.1f}",
        f"elapsed        : {report.elapsed:.1f}s",
        "finish distribution (hero):",
    ]
    for position, count in report.finish_position_distribution.items():
        share = count / report.tournament_count if report.tournament_count else 0.0
        lines.append(f"  #{position}: {count} ({share:.1%})")
    stage_rows = report.chip_ev_by_stage
    if stage_rows:
        lines.append(
            "hero chip-EV by stage (diagnostic; placement metrics above are "
            "the tournament outcome, this is not bb/100 tournament EV):"
        )
        for row in stage_rows:
            lines.append(
                f"  {row.players_dealt_in}p @ {row.small_blind}/{row.big_blind}: "
                f"{row.hands} hands, net {_signed_int(row.net_chips)}, "
                f"{row.chips_per_hand:+.2f}/hand, {row.bb_per_100:+.1f} bb/100"
            )
    return "\n".join(lines)


def report_to_jsonable(report):
    return {
        "hero_strategy": report.hero_strategy,
        "opponent_lineup": list(report.opponent_lineup),
        "players": report.players,
        "initial_stack": report.initial_stack,
        "blind_schedule": format_blind_schedule(report.blind_levels),
        "tournament_count": report.tournament_count,
        "seed": report.seed,
        "elapsed": report.elapsed,
        "hero_wins": report.hero_wins,
        "hero_win_rate": report.hero_win_rate,
        "mean_hero_finish_position": report.mean_hero_finish_position,
        "mean_hands_per_tournament": report.mean_hands_per_tournament,
        "finish_position_distribution": {
            str(position): count
            for position, count in report.finish_position_distribution.items()
        },
        "chip_ev_by_stage": [
            {
                "players_dealt_in": row.players_dealt_in,
                "small_blind": row.small_blind,
                "big_blind": row.big_blind,
                "hands": row.hands,
                "net_chips": row.net_chips,
                "chips_per_hand": row.chips_per_hand,
                "bb_per_100": row.bb_per_100,
            }
            for row in report.chip_ev_by_stage
        ],
        "tournaments": [
            {
                "index": result.index,
                "hands_played": result.hands_played,
                "hero_finish_position": result.hero_finish_position,
                "hero_won": result.hero_won,
                "seats": [asdict(seat) for seat in result.seats],
            }
            for result in report.results
        ],
    }


def write_json_report(report, path):
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report_to_jsonable(report), f, indent=2)
        f.write("\n")


def write_text_report(report, path):
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_report(report) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Play persistent freezeout tournaments (stacks carry across "
            "hands, blinds escalate, busted players are removed) and "
            "report tournament-level outcomes -- not bb/100."
        )
    )
    parser.add_argument("--strat", required=True, help="Hero strategy to evaluate.")
    parser.add_argument(
        "--opponent",
        default=DEFAULT_OPPONENT,
        help=(
            "Opponent strategy name, or a '+'/','-separated lineup for a "
            "heterogeneous table (e.g. 'simple+adaptive+royal_adaptive')."
        ),
    )
    parser.add_argument(
        "--players",
        type=int,
        default=None,
        help="Starting player count (2-6). Inferred from --opponent if omitted.",
    )
    parser.add_argument(
        "--initial-stack",
        type=int,
        default=DEFAULT_INITIAL_STACK,
        help="Starting stack for every seat (default: %(default)s).",
    )
    parser.add_argument(
        "--blind-schedule",
        default=None,
        help=(
            "'sb/bb[:hands],...' blind levels; only the final level may "
            "omit ':hands' (unbounded). "
            f"Default: {format_blind_schedule(DEFAULT_BLIND_LEVELS)}"
        ),
    )
    parser.add_argument(
        "--tournaments",
        type=int,
        default=DEFAULT_TOURNAMENT_COUNT,
        help="Number of independent tournaments to play (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Base RNG seed."
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=DEFAULT_MAX_HANDS,
        help="Safety cap on hands per tournament (default: %(default)s).",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-text", default=None)
    return parser


def config_from_args(args):
    players = resolve_player_count(args.opponent, args.players)
    opponent_lineup = resolve_opponent_lineup(args.opponent, players)
    blind_levels = parse_blind_schedule(args.blind_schedule)
    return TournamentConfig(
        hero_strategy=args.strat,
        opponent_lineup=opponent_lineup,
        initial_stack=args.initial_stack,
        blind_levels=blind_levels,
        tournament_count=args.tournaments,
        seed=args.seed,
        max_hands=args.max_hands,
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    report = run_tournament_batch(config)
    write_json_report(report, args.output_json)
    write_text_report(report, args.output_text)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
