import json

from eval import neural_train
from poker_bot.neural.value_model import (
    LinearValueModel,
    evaluate_model,
    load_labeled_telemetry,
    split_examples,
    train_linear_value_model,
)
from poker_bot.opponent_store import (
    connect,
    create_telemetry_run,
    record_decision_telemetry,
    update_hand_telemetry_outcome,
)


def make_seat(cards=None):
    return {
        "agentId": "player-agent",
        "seatNumber": 1,
        "holeCards": cards or ["AS", "KS"],
        "stackChips": 1800,
        "currentBetChips": 50,
    }


def make_table(seat, *, action_set=None, pot=120, street="Preflop"):
    return {
        "street": street,
        "boardCards": [] if street == "Preflop" else ["QS", "JS", "2C"],
        "potChips": pot,
        "currentBet": 50,
        "buttonSeatNumber": 1,
        "allowedActions": {
            "availableActions": action_set or ["fold", "call", "raise"],
            "callAmount": 50,
            "callChips": 50,
            "minBet": 50,
            "minRaiseTo": 150,
            "maxCommit": 1800,
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


def seed_labeled_telemetry(db_path):
    conn = connect(db_path)
    run_id = create_telemetry_run(
        conn,
        strategy="neural_seed",
        opponent="simple",
        players=2,
        seed=1,
    )
    rows = [
        (["AS", "KS"], "raise", 150, 220),
        (["AH", "AD"], "raise", 200, 300),
        (["9C", "8C"], "call", 50, -50),
        (["7D", "2C"], "fold", None, -25),
        (["QS", "JS"], "raise", 175, 180),
        (["4D", "3C"], "fold", None, -50),
    ]
    for index, (cards, action, amount, outcome) in enumerate(rows):
        seat = make_seat(cards)
        record_decision_telemetry(
            conn,
            run_id=run_id,
            hand_id=f"h{index}",
            decision_index=0,
            strategy="neural_seed",
            table=make_table(seat),
            seat=seat,
            action=action,
            amount=amount,
            message="seed row",
            facing_bet=True,
            voluntary=action in {"call", "raise"},
        )
        update_hand_telemetry_outcome(
            conn,
            run_id=run_id,
            hand_id=f"h{index}",
            hero_net_chips=outcome,
            won_hand=outcome > 0,
        )
    return conn, run_id


def test_load_labeled_telemetry_excludes_missing_outcomes(tmp_path):
    conn, run_id = seed_labeled_telemetry(tmp_path / "telemetry.sqlite")
    seat = make_seat(["2D", "7C"])
    record_decision_telemetry(
        conn,
        run_id=run_id,
        hand_id="missing",
        decision_index=0,
        strategy="neural_seed",
        table=make_table(seat),
        seat=seat,
        action="fold",
    )

    examples = load_labeled_telemetry(conn, run_id=run_id)

    assert len(examples) == 6
    assert {example.hand_id for example in examples} == {
        "h0",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
    }
    assert examples[0].target_bb == 220 / 50


def test_train_linear_value_model_round_trips_json(tmp_path):
    conn, run_id = seed_labeled_telemetry(tmp_path / "telemetry.sqlite")
    examples = load_labeled_telemetry(conn, run_id=run_id)

    model = train_linear_value_model(examples, epochs=5, learning_rate=0.01)
    metrics = evaluate_model(model, examples)
    model_path = tmp_path / "model.json"
    model.write_json(model_path)
    reloaded = LinearValueModel.read_json(model_path)

    assert metrics.count == len(examples)
    assert metrics.mae_bb >= 0
    assert reloaded.feature_names == model.feature_names
    assert reloaded.predict_bb(examples[0].features) == model.predict_bb(
        examples[0].features
    )


def test_split_examples_is_deterministic(tmp_path):
    conn, run_id = seed_labeled_telemetry(tmp_path / "telemetry.sqlite")
    examples = load_labeled_telemetry(conn, run_id=run_id)

    first = split_examples(examples, validation_fraction=0.33, seed=7)
    second = split_examples(examples, validation_fraction=0.33, seed=7)

    assert first == second
    assert len(first[0]) + len(first[1]) == len(examples)
    assert first[1]


def test_neural_train_cli_writes_model_and_report(tmp_path):
    db_path = tmp_path / "telemetry.sqlite"
    seed_labeled_telemetry(db_path)
    output_dir = tmp_path / "neural"

    exit_code = neural_train.main(
        [
            "--db",
            str(db_path),
            "--strategy",
            "neural_seed",
            "--output-dir",
            str(output_dir),
            "--epochs",
            "3",
        ]
    )

    model_path = output_dir / "linear-value-neural_seed.json"
    report_path = output_dir / "linear-value-neural_seed-report.json"
    payload = json.loads(report_path.read_text())

    assert exit_code == 0
    assert model_path.exists()
    assert payload["examples"] == 6
    assert payload["train_metrics"]["count"] > 0
