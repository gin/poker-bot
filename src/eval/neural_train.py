"""Train lightweight neural/value-model baselines from decision telemetry."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from poker_bot.neural.value_model import (
    evaluate_model,
    load_labeled_telemetry,
    metrics_to_jsonable,
    split_examples,
    train_linear_value_model,
)
from poker_bot.opponent_store import default_db_path

DEFAULT_OUTPUT_DIR = Path("benchmark-runs/neural")


@dataclass(frozen=True)
class NeuralTrainingReport:
    examples: int
    train_examples: int
    validation_examples: int
    strategy: str | None
    run_id: str | None
    epochs: int
    learning_rate: float
    train_metrics: dict
    validation_metrics: dict
    model_path: str


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a lightweight value model from decision telemetry."
    )
    parser.add_argument(
        "--db",
        default=str(default_db_path()),
        help="SQLite telemetry database path.",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Optional decision_telemetry strategy filter.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional telemetry run id filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum labeled rows to load.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="SGD training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.02,
        help="SGD learning rate.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of examples held out for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Deterministic split/training seed.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for default model/report outputs.",
    )
    parser.add_argument("--model-out", default=None)
    parser.add_argument("--report-out", default=None)
    return parser


def _write_json(path, payload):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _default_stem(args):
    parts = ["linear-value"]
    if args.strategy:
        parts.append(args.strategy)
    if args.run_id:
        parts.append(args.run_id)
    return "-".join(part.replace("/", "_") for part in parts)


def train_from_args(args):
    if args.epochs < 0:
        raise ValueError("--epochs must be non-negative")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if not 0 <= args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be in [0, 1)")

    conn = sqlite3.connect(Path(args.db).expanduser())
    conn.row_factory = sqlite3.Row
    examples = load_labeled_telemetry(
        conn,
        run_id=args.run_id,
        strategy=args.strategy,
        limit=args.limit,
    )
    if not examples:
        raise ValueError(
            "no labeled telemetry found; run selfplay with --telemetry first"
        )

    train_examples, validation_examples = split_examples(
        examples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    if not train_examples:
        raise ValueError("training split is empty")

    model = train_linear_value_model(
        train_examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    train_metrics = evaluate_model(model, train_examples)
    validation_metrics = evaluate_model(
        model,
        validation_examples,
        baseline_prediction_bb=train_metrics.target_mean_bb,
    )

    output_dir = Path(args.output_dir)
    stem = _default_stem(args)
    model_path = Path(args.model_out) if args.model_out else output_dir / f"{stem}.json"
    report_path = (
        Path(args.report_out) if args.report_out else output_dir / f"{stem}-report.json"
    )
    model.write_json(model_path)
    report = NeuralTrainingReport(
        examples=len(examples),
        train_examples=len(train_examples),
        validation_examples=len(validation_examples),
        strategy=args.strategy,
        run_id=args.run_id,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_metrics=metrics_to_jsonable(train_metrics),
        validation_metrics=metrics_to_jsonable(validation_metrics),
        model_path=str(model_path),
    )
    _write_json(report_path, asdict(report))
    return report, report_path


def format_report(report, report_path):
    validation = report.validation_metrics
    return "\n".join(
        [
            "# Neural Value Model",
            "",
            f"examples: {report.examples}",
            f"train_examples: {report.train_examples}",
            f"validation_examples: {report.validation_examples}",
            f"validation_mae_bb: {validation['mae_bb']:.3f}",
            f"validation_baseline_mae_bb: {validation['baseline_mae_bb']:.3f}",
            f"model_path: {report.model_path}",
            f"report_path: {report_path}",
        ]
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        report, report_path = train_from_args(args)
    except (ValueError, sqlite3.Error) as exc:
        print(f"neural-train: {exc}", file=sys.stderr)
        return 2
    print(format_report(report, report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
