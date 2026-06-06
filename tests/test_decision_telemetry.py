from eval.selfplay import run_selfplay
from poker_bot.opponent_store import (
    connect,
    create_telemetry_run,
    record_decision_telemetry,
    summarize_losing_buckets,
    update_hand_telemetry_outcome,
)


def make_seat():
    return {
        "agentId": "player-agent",
        "seatNumber": 1,
        "holeCards": ["AS", "KS"],
        "stackChips": 1800,
        "currentBetChips": 50,
    }


def make_table(seat):
    return {
        "street": "Preflop",
        "boardCards": [],
        "potChips": 75,
        "currentBet": 50,
        "buttonSeatNumber": 1,
        "allowedActions": {
            "availableActions": ["fold", "call", "raise"],
            "callAmount": 50,
            "callChips": 50,
            "minBet": 50,
            "minRaiseTo": 150,
        },
        "seats": [
            seat,
            {
                "agentId": "bot-agent-1",
                "seatNumber": 2,
                "stackChips": 2200,
                "currentBetChips": 25,
                "holeCards": [],
            },
        ],
    }


def test_decision_telemetry_records_and_updates_outcome(tmp_path):
    conn = connect(tmp_path / "telemetry.sqlite")
    run_id = create_telemetry_run(
        conn,
        strategy="survival_balanced",
        opponent="adaptive",
        players=6,
        seed=1,
    )
    seat = make_seat()

    record_decision_telemetry(
        conn,
        run_id=run_id,
        hand_id="h1",
        decision_index=0,
        strategy="survival_balanced",
        table=make_table(seat),
        seat=seat,
        action="raise",
        amount=150,
        message="test raise",
        facing_bet=True,
        voluntary=True,
    )
    update_hand_telemetry_outcome(
        conn,
        run_id=run_id,
        hand_id="h1",
        hero_net_chips=125,
        won_hand=True,
        final_pot=300,
    )

    row = conn.execute("select * from decision_telemetry").fetchone()
    assert row["run_id"] == run_id
    assert row["chosen_action"] == "raise"
    assert row["preflop_score"] > 0
    assert row["button_seat_number"] == 1
    assert row["hero_position"] == "BTN/SB"
    assert row["hero_position_offset"] == 0
    assert row["seated_players"] == 2
    assert row["hero_net_chips"] == 125
    assert row["won_hand"] == 1


def test_decision_telemetry_records_sixmax_position(tmp_path):
    conn = connect(tmp_path / "telemetry.sqlite")
    run_id = create_telemetry_run(conn, strategy="s", opponent="o")
    seats = [
        {
            "agentId": f"agent-{seat_number}",
            "seatNumber": seat_number,
            "holeCards": ["AS", "KS"] if seat_number == 6 else [],
            "stackChips": 2000,
            "currentBetChips": 0,
        }
        for seat_number in range(1, 7)
    ]
    hero = seats[-1]
    table = {
        "street": "Preflop",
        "boardCards": [],
        "potChips": 75,
        "currentBet": 50,
        "buttonSeatNumber": 1,
        "allowedActions": {
            "availableActions": ["fold", "call", "raise"],
            "callAmount": 50,
            "minRaiseTo": 150,
        },
        "seats": seats,
    }

    record_decision_telemetry(
        conn,
        run_id=run_id,
        hand_id="h1",
        decision_index=0,
        strategy="s",
        table=table,
        seat=hero,
        action="raise",
        amount=150,
    )

    row = conn.execute("select * from decision_telemetry").fetchone()
    assert row["button_seat_number"] == 1
    assert row["hero_position"] == "CO"
    assert row["hero_position_offset"] == 5
    assert row["seated_players"] == 6


def test_decision_telemetry_uses_button_alias_for_position(tmp_path):
    conn = connect(tmp_path / "telemetry.sqlite")
    run_id = create_telemetry_run(conn, strategy="s", opponent="o")
    seat = make_seat()
    table = make_table(seat)
    table.pop("buttonSeatNumber")
    table["dealerSeatNumber"] = "2"

    record_decision_telemetry(
        conn,
        run_id=run_id,
        hand_id="h1",
        decision_index=0,
        strategy="s",
        table=table,
        seat=seat,
        action="call",
        amount=50,
    )

    row = conn.execute("select * from decision_telemetry").fetchone()
    assert row["button_seat_number"] == 2
    assert row["hero_position"] == "BB"
    assert row["hero_position_offset"] == 1


def test_selfplay_can_record_decision_telemetry(tmp_path):
    db_path = tmp_path / "telemetry.sqlite"

    result = run_selfplay(
        "survival_balanced",
        opponent_name="adaptive",
        hands=8,
        seed=3,
        players=6,
        opponent_db=db_path,
        telemetry=True,
        telemetry_run_id="test-run",
    )
    conn = connect(db_path)
    decision_count = conn.execute(
        "select count(*) as count from decision_telemetry where run_id = ?",
        ("test-run",),
    ).fetchone()["count"]
    missing_outcomes = conn.execute(
        """
        select count(*) as count
        from decision_telemetry
        where run_id = ? and hero_net_chips is null
        """,
        ("test-run",),
    ).fetchone()["count"]

    assert result.hands == 8
    assert decision_count > 0
    assert missing_outcomes == 0


def test_summarize_losing_buckets_returns_rows(tmp_path):
    conn = connect(tmp_path / "telemetry.sqlite")
    run_id = create_telemetry_run(conn, strategy="s", opponent="o")
    seat = make_seat()
    table = make_table(seat)

    for index in range(3):
        record_decision_telemetry(
            conn,
            run_id=run_id,
            hand_id=f"h{index}",
            decision_index=0,
            strategy="s",
            table=table,
            seat=seat,
            action="fold",
            message="test fold",
        )
        update_hand_telemetry_outcome(
            conn,
            run_id=run_id,
            hand_id=f"h{index}",
            hero_net_chips=-50,
            won_hand=False,
        )

    rows = summarize_losing_buckets(conn, run_id, min_spots=1)

    assert rows
    assert rows[0]["chosen_action"] == "fold"
