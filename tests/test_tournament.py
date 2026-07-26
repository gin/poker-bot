import json

import pytest

import eval.tournament as tournament_mod
from eval.tournament import (
    BlindLevel,
    TournamentConfig,
    blind_level_for_hand,
    format_blind_schedule,
    format_report,
    parse_blind_schedule,
    report_to_jsonable,
    run_tournament,
    run_tournament_batch,
    validate_blind_schedule,
    write_json_report,
    write_text_report,
)

# ── Blind schedule parsing / lookup ─────────────────────────────────────────


def test_parse_blind_schedule_default_matches_module_default():
    assert parse_blind_schedule(None) == tournament_mod.DEFAULT_BLIND_LEVELS


def test_parse_blind_schedule_parses_custom_string():
    levels = parse_blind_schedule("5/10:20,25/50:15,100/200")

    assert levels == (
        BlindLevel(5, 10, 20),
        BlindLevel(25, 50, 15),
        BlindLevel(100, 200, None),
    )


def test_parse_blind_schedule_rejects_malformed_entry():
    with pytest.raises(ValueError, match="invalid blind schedule entry"):
        parse_blind_schedule("bogus")


def test_validate_blind_schedule_rejects_unbounded_non_final_level():
    with pytest.raises(ValueError, match="final blind level"):
        validate_blind_schedule((BlindLevel(5, 10, None), BlindLevel(10, 20, 20)))


def test_validate_blind_schedule_rejects_small_blind_above_big_blind():
    with pytest.raises(ValueError, match="small blind must not exceed"):
        validate_blind_schedule((BlindLevel(20, 10, None),))


def test_format_blind_schedule_round_trips():
    levels = (BlindLevel(5, 10, 20), BlindLevel(25, 50, None))
    assert parse_blind_schedule(format_blind_schedule(levels)) == levels


def test_blind_level_for_hand_progresses_then_holds_final_level():
    levels = (BlindLevel(5, 10, 2), BlindLevel(25, 50, 3), BlindLevel(100, 200, None))

    assert blind_level_for_hand(levels, 0) == levels[0]
    assert blind_level_for_hand(levels, 1) == levels[0]
    assert blind_level_for_hand(levels, 2) == levels[1]
    assert blind_level_for_hand(levels, 4) == levels[1]
    assert blind_level_for_hand(levels, 5) == levels[2]
    assert blind_level_for_hand(levels, 500) == levels[2]  # holds forever


# ── Config validation ────────────────────────────────────────────────────────


def test_tournament_config_rejects_out_of_range_player_count():
    with pytest.raises(ValueError, match="2 to 6"):
        TournamentConfig(hero_strategy="simple", opponent_lineup=())


def test_tournament_config_rejects_too_many_players():
    with pytest.raises(ValueError, match="2 to 6"):
        TournamentConfig(
            hero_strategy="simple",
            opponent_lineup=("simple",) * 6,
        )


def test_tournament_config_rejects_non_positive_initial_stack():
    with pytest.raises(ValueError, match="initial_stack"):
        TournamentConfig(
            hero_strategy="simple", opponent_lineup=("simple",), initial_stack=0
        )


# ── Deterministic replay ────────────────────────────────────────────────────


def test_run_tournament_is_deterministic_for_same_seed_and_index():
    config = TournamentConfig(
        hero_strategy="royal_adaptive",
        opponent_lineup=("simple", "adaptive", "all_in_everytime"),
        initial_stack=300,
        seed=123,
        max_hands=200,
    )

    first = run_tournament(config, 0)
    second = run_tournament(config, 0)

    assert first == second


def test_run_tournament_batch_different_indices_are_independent_seeds():
    config = TournamentConfig(
        hero_strategy="simple",
        opponent_lineup=("all_in_everytime",),
        initial_stack=200,
        seed=5,
        tournament_count=8,
        max_hands=200,
    )

    report = run_tournament_batch(config)

    # Not every tournament should have an identical hand count/outcome --
    # otherwise the per-tournament seeding isn't actually varying.
    hand_counts = {result.hands_played for result in report.results}
    assert len(hand_counts) > 1


# ── Heads-up and multiway bust-outs, chip conservation ──────────────────────


def test_run_tournament_heads_up_produces_single_winner_and_conserves_chips():
    config = TournamentConfig(
        hero_strategy="all_in_everytime",
        opponent_lineup=("all_in_everytime",),
        initial_stack=150,
        seed=11,
        max_hands=200,
    )

    result = run_tournament(config, 0)

    assert len(result.seats) == 2
    assert result.hero_finish_position in (1, 2)
    finishers = sorted(seat.finish_position for seat in result.seats)
    assert finishers == [1, 2]
    assert sum(seat.final_stack for seat in result.seats) == 150 * 2
    winner = next(seat for seat in result.seats if seat.finish_position == 1)
    assert winner.final_stack == 150 * 2
    loser = next(seat for seat in result.seats if seat.finish_position == 2)
    assert loser.final_stack == 0
    assert loser.busted_hand is not None


@pytest.mark.parametrize("players", [2, 3, 4, 5, 6])
def test_run_tournament_chip_conservation_across_player_counts_and_seeds(players):
    opponent_lineup = tuple(
        [
            "all_in_everytime",
            "simple",
            "adaptive",
            "counter_adaptive",
            "royal_adaptive",
        ][: players - 1]
    )
    for seed in range(4):
        config = TournamentConfig(
            hero_strategy="simple",
            opponent_lineup=opponent_lineup,
            initial_stack=250,
            seed=seed,
            max_hands=300,
        )

        result = run_tournament(config, 0)

        total = sum(seat.final_stack for seat in result.seats)
        assert total == 250 * players, (
            f"chip leak: players={players} seed={seed} total={total}"
        )
        positions = sorted(seat.finish_position for seat in result.seats)
        assert positions == list(range(1, players + 1))
        # Every seat must have a final finish position -- no stragglers.
        assert all(seat.finish_position is not None for seat in result.seats)


def test_run_tournament_multiway_side_pots_still_conserve_chips_over_many_hands():
    """All-in-happy lineup forces frequent multiway all-ins with unequal
    remaining stacks (i.e. genuine multi-level side pots) over the course
    of a tournament; chip totals must still balance exactly every hand."""
    config = TournamentConfig(
        hero_strategy="all_in_everytime",
        opponent_lineup=("all_in_everytime",) * 4,
        initial_stack=180,
        seed=77,
        max_hands=500,
    )

    for tournament_index in range(6):
        result = run_tournament(config, tournament_index)
        total = sum(seat.final_stack for seat in result.seats)
        assert total == 180 * 5
        assert sorted(seat.finish_position for seat in result.seats) == [1, 2, 3, 4, 5]


# ── Button rotation, busted-player removal, blind escalation visibility ────
# Driven with a fake play_hand_multiway so the outcome of each hand is
# fully controlled -- this isolates the tournament orchestration logic
# (who acts, who busts, how the button moves, what blinds are visible)
# from poker hand-strength mechanics, which are already covered by
# tests/test_simulator.py.


def test_run_tournament_button_rotation_bust_removal_and_blind_visibility(monkeypatch):
    calls = []

    def fake_play_hand_multiway(
        stacks,
        strategies,
        *,
        button_index,
        rng,
        opponent_profiles,
        action_observer,
        hand_id,
        verbose,
        agent_ids,
        small_blind,
        big_blind,
    ):
        calls.append(
            {
                "agent_ids": tuple(agent_ids),
                "small_blind": small_blind,
                "big_blind": big_blind,
            }
        )
        # Deterministic script: the seat immediately after the button (table
        # position 1) always busts entirely to the button (table position 0).
        result = list(stacks)
        if len(result) > 1:
            result[0] += result[1]
            result[1] = 0
        return result

    monkeypatch.setattr(tournament_mod, "play_hand_multiway", fake_play_hand_multiway)

    config = TournamentConfig(
        hero_strategy="simple",
        opponent_lineup=("simple", "simple", "simple"),
        initial_stack=100,
        blind_levels=(BlindLevel(5, 10, 1), BlindLevel(50, 100, None)),
        seed=1,
        max_hands=10,
    )

    result = run_tournament(config, 0)

    # 4 players -> exactly 3 bust-outs before a sole winner remains.
    assert result.hands_played == 3
    assert len(calls) == 3

    # Hand 0: initial seat order, no one busted yet.
    assert calls[0]["agent_ids"] == (
        "player-agent",
        "bot-agent-1",
        "bot-agent-2",
        "bot-agent-3",
    )
    assert (calls[0]["small_blind"], calls[0]["big_blind"]) == (5, 10)

    # Hand 1: bot-agent-1 busted, button moved past it to bot-agent-2;
    # only surviving seats are dealt in.
    assert calls[1]["agent_ids"] == ("bot-agent-2", "bot-agent-3", "player-agent")
    assert "bot-agent-1" not in calls[1]["agent_ids"]
    # Blind schedule's first level only lasts 1 hand -> hand index 1 is on
    # the second (escalated, unbounded) level.
    assert (calls[1]["small_blind"], calls[1]["big_blind"]) == (50, 100)

    # Hand 2: bot-agent-3 busted in hand 1 (to bot-agent-2, not the hero),
    # button moves on; only the two remaining seats are dealt in.
    assert calls[2]["agent_ids"] == ("player-agent", "bot-agent-2")
    assert set(calls[2]["agent_ids"]) == {"player-agent", "bot-agent-2"}

    # Hero wins the whole tournament in this script.
    assert result.hero_finish_position == 1
    assert result.hero_won is True

    seats_by_id = {seat.agent_id: seat for seat in result.seats}
    assert seats_by_id["bot-agent-1"].finish_position == 4
    assert seats_by_id["bot-agent-1"].busted_hand == 1
    assert seats_by_id["bot-agent-3"].finish_position == 3
    assert seats_by_id["bot-agent-3"].busted_hand == 2
    assert seats_by_id["bot-agent-2"].finish_position == 2
    assert seats_by_id["bot-agent-2"].busted_hand == 3
    assert seats_by_id["player-agent"].finish_position == 1
    assert seats_by_id["player-agent"].busted_hand is None
    assert seats_by_id["player-agent"].final_stack == 100 * 4


# ── Opponent identity / profile persistence across busts ───────────────────


def test_run_tournament_preserves_stable_agent_identity_as_seats_bust():
    """Regardless of how seating order shifts as players bust, each
    surviving seat's agentId (and therefore its strategy identity and any
    accumulated opponent profile) must stay the same for the whole
    tournament."""
    seen_ids_per_hand = []

    def observer(**event):
        table = event.get("table")
        if table is not None:
            seen_ids_per_hand.append(
                tuple(seat["agentId"] for seat in table["seats"])
            )

    config = TournamentConfig(
        hero_strategy="all_in_everytime",
        opponent_lineup=("all_in_everytime", "all_in_everytime", "all_in_everytime"),
        initial_stack=120,
        seed=13,
        max_hands=200,
    )

    result = run_tournament(config, 0, action_observer=observer)

    assert seen_ids_per_hand, "expected at least one observed hand"
    # Every id ever observed must be one of the four canonical agent ids
    # assigned once at tournament start.
    canonical_ids = {"player-agent", "bot-agent-1", "bot-agent-2", "bot-agent-3"}
    for ids in seen_ids_per_hand:
        assert set(ids) <= canonical_ids

    # A seat's strategy name (recorded in the final result) must match the
    # strategy it was assigned at tournament start, keyed by its stable id.
    strategy_by_id = {seat.agent_id: seat.strategy for seat in result.seats}
    assert strategy_by_id["player-agent"] == "all_in_everytime"
    assert strategy_by_id["bot-agent-1"] == "all_in_everytime"


# ── Reporting / JSON schema ──────────────────────────────────────────────────


def test_run_tournament_batch_report_aggregates_and_json_schema(tmp_path):
    config = TournamentConfig(
        hero_strategy="simple",
        opponent_lineup=("all_in_everytime",),
        initial_stack=100,
        seed=9,
        tournament_count=6,
        max_hands=200,
    )

    report = run_tournament_batch(config)

    assert report.hero_wins == sum(1 for r in report.results if r.hero_won)
    assert report.hero_win_rate == pytest.approx(report.hero_wins / 6)
    assert report.mean_hands_per_tournament > 0
    assert sum(report.finish_position_distribution.values()) == 6
    assert set(report.finish_position_distribution) <= {1, 2}

    payload = report_to_jsonable(report)
    for key in (
        "hero_strategy",
        "opponent_lineup",
        "players",
        "initial_stack",
        "blind_schedule",
        "tournament_count",
        "seed",
        "elapsed",
        "hero_wins",
        "hero_win_rate",
        "mean_hero_finish_position",
        "mean_hands_per_tournament",
        "finish_position_distribution",
        "tournaments",
    ):
        assert key in payload

    assert len(payload["tournaments"]) == 6
    for tournament_payload in payload["tournaments"]:
        for key in (
            "index",
            "hands_played",
            "hero_finish_position",
            "hero_won",
            "seats",
        ):
            assert key in tournament_payload
        assert len(tournament_payload["seats"]) == 2
        for seat_payload in tournament_payload["seats"]:
            for key in (
                "agent_id",
                "strategy",
                "finish_position",
                "busted_hand",
                "final_stack",
            ):
                assert key in seat_payload

    output_json = tmp_path / "report.json"
    write_json_report(report, output_json)
    reloaded = json.loads(output_json.read_text())
    assert reloaded == payload

    output_text = tmp_path / "report.txt"
    write_text_report(report, output_text)
    text = output_text.read_text()
    assert "hero wins" in text
    assert "finish distribution" in text
    assert text.rstrip("\n") == format_report(report)


def test_format_report_placement_metrics_never_use_bb_per_100():
    """bb/100 is a hand-EV concept; the tournament's PLACEMENT metrics
    (wins, win rate, finish position, hands/tournament) must never be
    expressed in bb/100. The chip-EV-by-stage diagnostic is a deliberate,
    clearly-labeled exception: it reports a per-stage hand-EV breakdown
    alongside (never instead of) placement, so bb/100 is expected there
    and there only."""
    config = TournamentConfig(
        hero_strategy="simple",
        opponent_lineup=("all_in_everytime",),
        initial_stack=100,
        seed=1,
        tournament_count=2,
        max_hands=200,
    )

    report = run_tournament_batch(config)
    text = format_report(report)

    placement_section, _, diagnostic_section = text.partition(
        "finish distribution (hero):"
    )
    assert "bb/100" not in placement_section

    payload = report_to_jsonable(report)
    placement_keys = (
        "hero_wins",
        "hero_win_rate",
        "mean_hero_finish_position",
        "mean_hands_per_tournament",
        "finish_position_distribution",
        "tournaments",
    )
    for key in placement_keys:
        assert "bb_per_100" not in json.dumps(payload[key])

    # The diagnostic section is present, clearly labeled, and does use
    # bb/100 -- proving it wasn't simply omitted.
    assert "chip-EV by stage" in diagnostic_section
    assert "diagnostic" in diagnostic_section
    assert "bb/100" in diagnostic_section
    assert "bb_per_100" in json.dumps(payload["chip_ev_by_stage"])


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_build_parser_and_config_from_args_infer_players_from_opponent():
    args = tournament_mod.build_parser().parse_args(
        [
            "--strat",
            "simple",
            "--opponent",
            "simple+adaptive+royal_adaptive",
            "--tournaments",
            "3",
            "--seed",
            "4",
            "--max-hands",
            "50",
        ]
    )
    config = tournament_mod.config_from_args(args)

    assert config.players == 4
    assert config.opponent_lineup == ("simple", "adaptive", "royal_adaptive")
    assert config.tournament_count == 3
    assert config.seed == 4
    assert config.max_hands == 50


# ── Bust-order and cap-ranking must favor larger stacks ─────────────────────


def test_run_tournament_simultaneous_unequal_bust_stacks_ranked_correctly(
    monkeypatch,
):
    """When two seats bust in the SAME hand with unequal starting stacks,
    the one that brought MORE chips into the hand must receive the
    better (lower) finish position -- not the other way around."""
    calls = []

    def fake_play_hand_multiway(stacks, strategies, *, agent_ids, **kwargs):
        calls.append(tuple(agent_ids))
        by_id = dict(zip(agent_ids, stacks, strict=True))
        if len(calls) == 1:
            # Redistribute unevenly without busting anyone: bot-agent-1
            # (150) ends up with much more than bot-agent-2 (50).
            by_id["bot-agent-1"] += 50
            by_id["bot-agent-2"] -= 50
        elif len(calls) == 2:
            # bot-agent-1 (150) and bot-agent-2 (50) bust simultaneously,
            # both losing everything to the hero.
            pool = by_id["bot-agent-1"] + by_id["bot-agent-2"]
            by_id["player-agent"] += pool
            by_id["bot-agent-1"] = 0
            by_id["bot-agent-2"] = 0
        else:
            # Final hand: bot-agent-3 busts too, hero wins outright.
            pool = by_id["player-agent"] + by_id["bot-agent-3"]
            by_id["player-agent"] = pool
            by_id["bot-agent-3"] = 0
        return [by_id[agent_id] for agent_id in agent_ids]

    monkeypatch.setattr(tournament_mod, "play_hand_multiway", fake_play_hand_multiway)

    config = TournamentConfig(
        hero_strategy="simple",
        opponent_lineup=("simple", "simple", "simple"),
        initial_stack=100,
        blind_levels=(BlindLevel(5, 10, None),),
        seed=1,
        max_hands=10,
    )

    result = run_tournament(config, 0)

    seats_by_id = {seat.agent_id: seat for seat in result.seats}
    assert result.hands_played == 3
    assert sum(seat.final_stack for seat in result.seats) == 100 * 4

    # bot-agent-2 brought only 50 chips into the simultaneous-bust hand
    # (the smaller stack) and must get the WORSE finish position.
    assert seats_by_id["bot-agent-2"].finish_position == 4
    # bot-agent-1 brought 150 chips (the larger stack) and must get the
    # BETTER finish position among the two simultaneous bust-outs.
    assert seats_by_id["bot-agent-1"].finish_position == 3
    assert seats_by_id["bot-agent-1"].busted_hand == 2
    assert seats_by_id["bot-agent-2"].busted_hand == 2
    assert seats_by_id["bot-agent-3"].finish_position == 2
    assert seats_by_id["player-agent"].finish_position == 1
    assert result.hero_won is True


def test_run_tournament_max_hands_cap_ranks_survivors_by_stack_size(monkeypatch):
    """If the safety cap is hit before a winner emerges, remaining
    survivors must be ranked by final stack -- larger stack gets the
    better (lower) finish position, largest ending at position 1."""

    def fake_play_hand_multiway(stacks, strategies, *, agent_ids, **kwargs):
        by_id = dict(zip(agent_ids, stacks, strict=True))
        # Non-busting redistribution every hand: bot-agent-1 always nets
        # +10 from bot-agent-2.
        take = min(10, by_id["bot-agent-2"])
        by_id["bot-agent-1"] += take
        by_id["bot-agent-2"] -= take
        return [by_id[agent_id] for agent_id in agent_ids]

    monkeypatch.setattr(tournament_mod, "play_hand_multiway", fake_play_hand_multiway)

    config = TournamentConfig(
        hero_strategy="simple",
        opponent_lineup=("simple", "simple"),
        initial_stack=100,
        blind_levels=(BlindLevel(5, 10, None),),
        seed=1,
        max_hands=5,  # safety cap hit well before anyone actually busts
    )

    result = run_tournament(config, 0)

    assert result.hands_played == 5
    seats_by_id = {seat.agent_id: seat for seat in result.seats}
    assert all(seat.busted_hand is None for seat in result.seats)
    assert seats_by_id["bot-agent-1"].final_stack == 150
    assert seats_by_id["player-agent"].final_stack == 100
    assert seats_by_id["bot-agent-2"].final_stack == 50

    # Larger final stack -> better (lower) finish position.
    assert seats_by_id["bot-agent-1"].finish_position == 1
    assert seats_by_id["player-agent"].finish_position == 2
    assert seats_by_id["bot-agent-2"].finish_position == 3
    assert result.hero_finish_position == 2
    assert result.hero_won is False


# ── Chip-EV-by-stage diagnostics: stage assignment and arithmetic ──────────


def test_run_tournament_chip_ev_by_stage_tracks_transitions(monkeypatch):
    """Controlled fake play_hand_multiway drives hero through a blind
    escalation (5/10 -> 10/20) and a player-count contraction (4p -> 3p
    after an opponent busts), proving each hand's hero delta lands in the
    correct (players_dealt_in, small_blind, big_blind) stage bucket with
    correct hands/net_chips/chips_per_hand/bb_per_100 arithmetic."""
    call_count = 0

    def fake_play_hand_multiway(stacks, strategies, *, agent_ids, **kwargs):
        nonlocal call_count
        call_count += 1
        by_id = dict(zip(agent_ids, stacks, strict=True))
        if call_count == 1:
            # 4p dealt in @ 5/10 (level 0, lasts exactly 1 hand): hero +50.
            by_id["player-agent"] += 50
            by_id["bot-agent-1"] -= 50
        elif call_count == 2:
            # 4p dealt in @ 10/20 (level 0 exhausted): hero +100; C busts.
            by_id["player-agent"] += 100
            by_id["bot-agent-1"] += by_id["bot-agent-3"] - 100
            by_id["bot-agent-3"] = 0
        elif call_count == 3:
            # 3p dealt in @ 10/20 (C removed before this deal): hero -70.
            by_id["player-agent"] -= 70
            by_id["bot-agent-1"] += 70
        elif call_count == 4:
            # 3p dealt in @ 10/20: hero +20.
            by_id["player-agent"] += 20
            by_id["bot-agent-2"] -= 20
        else:
            raise AssertionError(f"unexpected extra hand call #{call_count}")
        return [by_id[agent_id] for agent_id in agent_ids]

    monkeypatch.setattr(tournament_mod, "play_hand_multiway", fake_play_hand_multiway)

    config = TournamentConfig(
        hero_strategy="simple",
        opponent_lineup=("simple", "simple", "simple"),
        initial_stack=1000,
        blind_levels=(BlindLevel(5, 10, 1), BlindLevel(10, 20, None)),
        seed=1,
        max_hands=4,
    )

    result = run_tournament(config, 0)

    assert call_count == 4
    assert result.hands_played == 4

    deltas = result.hero_stage_deltas
    assert len(deltas) == 4
    assert deltas[0] == tournament_mod.HeroStageDelta(
        players_dealt_in=4, small_blind=5, big_blind=10, hero_chip_delta=50
    )
    assert deltas[1] == tournament_mod.HeroStageDelta(
        players_dealt_in=4, small_blind=10, big_blind=20, hero_chip_delta=100
    )
    assert deltas[2] == tournament_mod.HeroStageDelta(
        players_dealt_in=3, small_blind=10, big_blind=20, hero_chip_delta=-70
    )
    assert deltas[3] == tournament_mod.HeroStageDelta(
        players_dealt_in=3, small_blind=10, big_blind=20, hero_chip_delta=20
    )

    report = tournament_mod.TournamentBatchReport(
        hero_strategy=config.hero_strategy,
        opponent_lineup=config.opponent_lineup,
        players=config.players,
        initial_stack=config.initial_stack,
        blind_levels=config.blind_levels,
        tournament_count=1,
        seed=config.seed,
        results=(result,),
        elapsed=0.0,
    )
    stages = report.chip_ev_by_stage

    # Ordered ascending by blind level, then descending by player count.
    assert [(s.players_dealt_in, s.small_blind, s.big_blind) for s in stages] == [
        (4, 5, 10),
        (4, 10, 20),
        (3, 10, 20),
    ]

    stage_5_10 = stages[0]
    assert stage_5_10.hands == 1
    assert stage_5_10.net_chips == 50
    assert stage_5_10.chips_per_hand == pytest.approx(50.0)
    assert stage_5_10.bb_per_100 == pytest.approx(50 / 10 / 1 * 100)

    stage_4p_10_20 = stages[1]
    assert stage_4p_10_20.hands == 1
    assert stage_4p_10_20.net_chips == 100
    assert stage_4p_10_20.bb_per_100 == pytest.approx(100 / 20 / 1 * 100)

    # The two 3-handed @ 10/20 hands are pooled together: net = -70 + 20.
    stage_3p_10_20 = stages[2]
    assert stage_3p_10_20.hands == 2
    assert stage_3p_10_20.net_chips == -50
    assert stage_3p_10_20.chips_per_hand == pytest.approx(-25.0)
    assert stage_3p_10_20.bb_per_100 == pytest.approx(-50 / 20 / 2 * 100)


def test_run_tournament_chip_ev_by_stage_stops_once_hero_busts(monkeypatch):
    """Once the hero busts, later hands (which no longer deal the hero in)
    must not contribute any further stage deltas."""
    call_count = 0

    def fake_play_hand_multiway(stacks, strategies, *, agent_ids, **kwargs):
        nonlocal call_count
        call_count += 1
        by_id = dict(zip(agent_ids, stacks, strict=True))
        if call_count == 1:
            # Hero busts entirely to bot-agent-1 on the very first hand.
            by_id["bot-agent-1"] += by_id["player-agent"]
            by_id["player-agent"] = 0
        elif call_count == 2:
            # Hero is no longer dealt in (already busted). bot-agent-2
            # busts to bot-agent-1, ending the tournament -- hero must
            # not gain a second stage delta from this hand.
            by_id["bot-agent-1"] += by_id["bot-agent-2"]
            by_id["bot-agent-2"] = 0
        else:
            raise AssertionError(f"unexpected extra hand call #{call_count}")
        return [by_id[agent_id] for agent_id in agent_ids]

    monkeypatch.setattr(tournament_mod, "play_hand_multiway", fake_play_hand_multiway)

    config = TournamentConfig(
        hero_strategy="simple",
        opponent_lineup=("simple", "simple"),
        initial_stack=500,
        blind_levels=(BlindLevel(5, 10, None),),
        seed=1,
        max_hands=10,
    )

    result = run_tournament(config, 0)

    # Hand 1 busts the hero (1 stage delta recorded); hand 2 is played
    # between the two remaining opponents only (no hero delta) and ends
    # the tournament by busting one of them.
    assert result.hands_played == 2
    assert len(result.hero_stage_deltas) == 1
    assert result.hero_stage_deltas[0].hero_chip_delta == -500
    assert result.hero_finish_position == 3


# ── Regression: float-typed chip totals must not crash formatting ──────────


def test_signed_int_formats_whole_valued_floats_and_ints():
    assert tournament_mod._signed_int(-2280.0) == "-2280"
    assert tournament_mod._signed_int(2280.0) == "+2280"
    assert tournament_mod._signed_int(-2280) == "-2280"
    assert tournament_mod._signed_int(0) == "+0"


def test_format_report_handles_float_contaminated_net_chips_without_crashing():
    """Regression: a strategy that computes a bet size via a float
    multiplier (e.g. ``BIG_BLIND * 2.5``) can leave a seat's ``stackChips``
    -- and therefore ``hero_chip_delta`` / the aggregated stage
    ``net_chips`` -- typed as a whole-valued float rather than int.
    ``format_report`` previously used Python's ``:+d`` format spec on
    ``net_chips``, which rejects floats outright even when whole-valued,
    crashing with ``ValueError: Unknown format code 'd' for object of
    type 'float'`` -- reproduced here with the exact shape/value observed
    in a real multi_core_v007 tournament smoke report (a -2280.0 net at
    3p @ 10/20)."""
    stage_delta = tournament_mod.HeroStageDelta(
        players_dealt_in=3,
        small_blind=10,
        big_blind=20,
        hero_chip_delta=-2280.0,
    )
    result = tournament_mod.TournamentResult(
        index=0,
        seed=1,
        hands_played=1,
        seats=(),
        hero_finish_position=2,
        hero_won=False,
        hero_stage_deltas=(stage_delta,),
    )
    report = tournament_mod.TournamentBatchReport(
        hero_strategy="multi_core_v007",
        opponent_lineup=("adaptive",),
        players=2,
        initial_stack=1000,
        blind_levels=tournament_mod.DEFAULT_BLIND_LEVELS,
        tournament_count=1,
        seed=1,
        results=(result,),
        elapsed=0.0,
    )

    # sum() of a deltas list containing one float promotes the whole
    # aggregated stage total to float -- exactly the observed shape.
    stage_row = report.chip_ev_by_stage[0]
    assert isinstance(stage_row.net_chips, float)

    text = format_report(report)  # must not raise ValueError

    assert "net -2280" in text
    assert "3p @ 10/20" in text

    payload = report_to_jsonable(report)
    assert payload["chip_ev_by_stage"][0]["net_chips"] == -2280.0
