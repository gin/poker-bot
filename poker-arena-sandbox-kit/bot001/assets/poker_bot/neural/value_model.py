"""Dependency-free linear value model for poker decision telemetry."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from poker_bot.neural.features import (
    CHIP_SCALE,
    FEATURE_NAMES,
    FeatureVector,
    encode_mapping,
)


@dataclass(frozen=True)
class LabeledDecision:
    features: tuple[float, ...]
    target_bb: float
    run_id: str | None = None
    hand_id: str | None = None
    decision_index: int | None = None


@dataclass(frozen=True)
class ValueModelMetrics:
    count: int
    mae_bb: float
    rmse_bb: float
    baseline_mae_bb: float
    target_mean_bb: float
    prediction_mean_bb: float

    @property
    def mae_chips(self):
        return self.mae_bb * CHIP_SCALE

    @property
    def rmse_chips(self):
        return self.rmse_bb * CHIP_SCALE


@dataclass(frozen=True)
class LinearValueModel:
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    bias: float
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    target_scale: float = CHIP_SCALE

    def _coerce_features(self, features):
        if isinstance(features, FeatureVector):
            if tuple(features.names) != tuple(self.feature_names):
                raise ValueError("feature names do not match model schema")
            return tuple(features.values)
        return tuple(float(value) for value in features)

    def _normalized(self, features):
        raw = self._coerce_features(features)
        if len(raw) != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features, got {len(raw)}"
            )
        return tuple(
            (value - mean) / scale
            for value, mean, scale in zip(
                raw, self.feature_means, self.feature_scales, strict=True
            )
        )

    def predict_bb(self, features):
        normalized = self._normalized(features)
        return self.bias + sum(
            weight * value
            for weight, value in zip(self.weights, normalized, strict=True)
        )

    def predict_chips(self, features):
        return self.predict_bb(features) * self.target_scale

    def to_jsonable(self):
        return {
            "model_type": "linear_value_v1",
            "feature_names": list(self.feature_names),
            "weights": list(self.weights),
            "bias": self.bias,
            "feature_means": list(self.feature_means),
            "feature_scales": list(self.feature_scales),
            "target_scale": self.target_scale,
        }

    @classmethod
    def from_jsonable(cls, data):
        if data.get("model_type") != "linear_value_v1":
            raise ValueError("unsupported value model type")
        return cls(
            feature_names=tuple(data["feature_names"]),
            weights=tuple(float(value) for value in data["weights"]),
            bias=float(data["bias"]),
            feature_means=tuple(float(value) for value in data["feature_means"]),
            feature_scales=tuple(float(value) for value in data["feature_scales"]),
            target_scale=float(data.get("target_scale", CHIP_SCALE)),
        )

    def write_json(self, path):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(self.to_jsonable(), f, indent=2)
            f.write("\n")

    @classmethod
    def read_json(cls, path):
        with Path(path).open() as f:
            return cls.from_jsonable(json.load(f))


def _row_mapping(row, columns):
    if hasattr(row, "keys"):
        return row
    return dict(zip(columns, row, strict=True))


def load_labeled_telemetry(
    conn,
    *,
    run_id=None,
    strategy=None,
    limit=None,
):
    """Load value-labeled decisions from ``decision_telemetry``.

    Targets are normalized to big blinds. Arena telemetry without
    ``hero_net_chips`` remains useful for future behavior cloning, but it is
    intentionally excluded from this value dataset.
    """
    clauses = [
        "hero_net_chips is not null",
        "chosen_action is not null",
    ]
    params = []
    if run_id is not None:
        clauses.append("run_id = ?")
        params.append(run_id)
    if strategy is not None:
        clauses.append("strategy = ?")
        params.append(strategy)

    query = (
        "select * from decision_telemetry "
        f"where {' and '.join(clauses)} "
        "order by run_id, hand_id, decision_index, id"
    )
    if limit is not None:
        query += " limit ?"
        params.append(int(limit))

    cursor = conn.execute(query, params)
    columns = tuple(description[0] for description in cursor.description)
    examples = []
    for row in cursor.fetchall():
        mapping = _row_mapping(row, columns)
        vector = encode_mapping(mapping)
        target_bb = float(mapping["hero_net_chips"]) / CHIP_SCALE
        examples.append(
            LabeledDecision(
                features=vector.values,
                target_bb=target_bb,
                run_id=mapping["run_id"],
                hand_id=mapping["hand_id"],
                decision_index=mapping["decision_index"],
            )
        )
    return tuple(examples)


def split_examples(examples, *, validation_fraction=0.2, seed=1):
    examples = tuple(examples)
    if not examples:
        return (), ()
    if validation_fraction <= 0:
        return examples, ()
    if validation_fraction >= 1:
        return (), examples

    indexes = list(range(len(examples)))
    random.Random(seed).shuffle(indexes)
    validation_count = max(1, int(round(len(indexes) * validation_fraction)))
    if validation_count >= len(indexes):
        validation_count = len(indexes) - 1
    validation_indexes = set(indexes[:validation_count])
    train = tuple(
        example
        for index, example in enumerate(examples)
        if index not in validation_indexes
    )
    validation = tuple(
        example for index, example in enumerate(examples) if index in validation_indexes
    )
    return train, validation


def _feature_stats(examples):
    width = len(examples[0].features)
    means = []
    scales = []
    for index in range(width):
        values = [example.features[index] for example in examples]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scale = math.sqrt(variance)
        means.append(mean)
        scales.append(scale if scale > 1e-9 else 1.0)
    return tuple(means), tuple(scales)


def _normalize(features, means, scales):
    return tuple(
        (value - mean) / scale
        for value, mean, scale in zip(features, means, scales, strict=True)
    )


def train_linear_value_model(
    examples,
    *,
    epochs=60,
    learning_rate=0.02,
    l2=0.0005,
    seed=1,
):
    examples = tuple(examples)
    if not examples:
        raise ValueError("at least one labeled decision is required")
    width = len(examples[0].features)
    if width != len(FEATURE_NAMES):
        raise ValueError(f"expected {len(FEATURE_NAMES)} features, got {width}")
    if any(len(example.features) != width for example in examples):
        raise ValueError("all examples must share the same feature width")
    if epochs < 0:
        raise ValueError("epochs must be non-negative")

    means, scales = _feature_stats(examples)
    weights = [0.0] * width
    bias = sum(example.target_bb for example in examples) / len(examples)
    rng = random.Random(seed)
    indexes = list(range(len(examples)))

    for _epoch in range(epochs):
        rng.shuffle(indexes)
        for index in indexes:
            example = examples[index]
            features = _normalize(example.features, means, scales)
            prediction = bias + sum(
                weight * value for weight, value in zip(weights, features, strict=True)
            )
            error = max(-20.0, min(20.0, prediction - example.target_bb))
            for feature_index, value in enumerate(features):
                gradient = error * value + l2 * weights[feature_index]
                weights[feature_index] -= learning_rate * gradient
            bias -= learning_rate * error

    return LinearValueModel(
        feature_names=FEATURE_NAMES,
        weights=tuple(weights),
        bias=bias,
        feature_means=means,
        feature_scales=scales,
    )


def evaluate_model(model, examples, *, baseline_prediction_bb=None):
    examples = tuple(examples)
    if not examples:
        return ValueModelMetrics(
            count=0,
            mae_bb=0.0,
            rmse_bb=0.0,
            baseline_mae_bb=0.0,
            target_mean_bb=0.0,
            prediction_mean_bb=0.0,
        )

    targets = [example.target_bb for example in examples]
    predictions = [model.predict_bb(example.features) for example in examples]
    baseline = (
        sum(targets) / len(targets)
        if baseline_prediction_bb is None
        else baseline_prediction_bb
    )
    errors = [
        prediction - target
        for prediction, target in zip(predictions, targets, strict=True)
    ]
    baseline_errors = [baseline - target for target in targets]
    return ValueModelMetrics(
        count=len(examples),
        mae_bb=sum(abs(error) for error in errors) / len(errors),
        rmse_bb=math.sqrt(sum(error**2 for error in errors) / len(errors)),
        baseline_mae_bb=sum(abs(error) for error in baseline_errors)
        / len(baseline_errors),
        target_mean_bb=sum(targets) / len(targets),
        prediction_mean_bb=sum(predictions) / len(predictions),
    )


def metrics_to_jsonable(metrics):
    return asdict(metrics)
