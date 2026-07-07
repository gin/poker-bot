"""Persistent Bayesian range tracker for opponent modeling."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poker_bot.range_model.hand_range import (
    HandRange,
    class_strength,
    combo_class,
    normalize_combo,
)
from poker_bot.range_model.preflop import default_preflop_range
from poker_bot.range_model.update import (
    apply_action_update,
    remove_blockers,
)

RANGE_STATE_DIR_ENV = "POKER_RANGE_STATE_DIR"
_MAX_TOP_CLASSES = 8
_CONFIDENCE_INCREMENT = 0.15
_MAX_CONFIDENCE = 0.95


def default_state_dir() -> Path:
    configured = os.environ.get(RANGE_STATE_DIR_ENV)
    if configured and configured.strip():
        return Path(configured).expanduser()
    return Path.home() / ".poker-range-state"


DEFAULT_STATE_DIR = default_state_dir()


@dataclass
class RangeTrackerState:
    """Persistable posterior state for one opponent."""

    agent_id: str
    prior_range: HandRange = field(default_factory=HandRange.all)
    posterior_range: HandRange = field(default_factory=HandRange.all)
    action_history: list[dict[str, Any]] = field(default_factory=list)
    showdown_hands: list[tuple[str, str]] = field(default_factory=list)
    confidence: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "prior_range": sorted(self.prior_range.weights.items()),
            "posterior_range": sorted(self.posterior_range.weights.items()),
            "action_history": self.action_history,
            "showdown_hands": [list(hand) for hand in self.showdown_hands],
            "confidence": self.confidence,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RangeTrackerState:
        return cls(
            agent_id=data["agent_id"],
            prior_range=HandRange(_normalize_weights(data.get("prior_range", []))),
            posterior_range=HandRange(
                _normalize_weights(data.get("posterior_range", [])),
            ),
            action_history=list(data.get("action_history", [])),
            showdown_hands=[tuple(hand) for hand in data.get("showdown_hands", [])],
            confidence=float(data.get("confidence", 0.0)),
        )


class BayesianRangeTracker:
    """Maintain and persist Bayesian posterior ranges per opponent.

    The first implementation stays intentionally small: each action scales the
    opponent's range by the existing action likelihood model, then normalizes.
    Blockers and showdown observations remove impossible combos.
    """

    def __init__(
        self,
        state_dir: str | Path | None = None,
        *,
        auto_save: bool = True,
    ) -> None:
        if state_dir is None:
            self.state_dir = default_state_dir()
        else:
            self.state_dir = Path(state_dir)
        self.auto_save = auto_save
        self._states: dict[str, RangeTrackerState] = {}

    def state_for(self, agent_id: str) -> RangeTrackerState:
        agent_id = _agent_id(agent_id)
        if agent_id not in self._states:
            self._states[agent_id] = self._load_state(agent_id)
        return self._states[agent_id]

    def reset_agent(
        self,
        agent_id: str,
        position: str = "MP",
        situation: str = "open",
    ) -> RangeTrackerState:
        state = RangeTrackerState(agent_id=_agent_id(agent_id))
        state.prior_range = default_preflop_range(position, situation)
        state.posterior_range = state.prior_range
        self._states[state.agent_id] = state
        if self.auto_save:
            self.save(agent_id)
        return state

    def update(
        self,
        agent_id: str,
        *,
        position: str = "MP",
        situation: str = "open",
        action: str | None = None,
        known_cards: list[str] | None = None,
        amount: int | float | None = None,
        pot: int | float | None = None,
    ) -> RangeTrackerState:
        state = self.state_for(agent_id)
        if not state.action_history:
            state.prior_range = default_preflop_range(position, situation)
            state.posterior_range = state.prior_range
        if state.posterior_range.total_weight() <= 0:
            self.reset_agent(state.agent_id, position, situation)
            state = self._states[state.agent_id]

        prior = remove_blockers(state.posterior_range, known_cards)
        if prior.total_weight() <= 0:
            prior = remove_blockers(
                default_preflop_range(position, situation),
                known_cards,
            )

        updated = prior
        if action is not None:
            updated = apply_action_update(
                prior,
                action,
                amount=amount,
                pot=pot,
                normalize=True,
            )
        if updated.total_weight() <= 0:
            updated = prior.normalized()

        state.prior_range = default_preflop_range(position, situation)
        state.posterior_range = updated
        state.action_history.append(
            {
                "position": position,
                "situation": situation,
                "action": action,
                "amount": amount,
                "pot": pot,
            }
        )
        state.confidence = _confidence_after_samples(len(state.action_history))
        if self.auto_save:
            self.save(state.agent_id)
        return state

    def record_showdown(
        self,
        agent_id: str,
        hole_cards: list[str] | tuple[str, str],
    ) -> RangeTrackerState:
        state = self.state_for(agent_id)
        shown = normalize_combo(hole_cards)
        state.showdown_hands.append(shown)

        if state.prior_range.total_weight() > 0:
            state.prior_range = state.prior_range.scale(
                lambda combo: 1.0 if combo == shown else 0.0
            )
            if state.prior_range.total_weight() <= 0:
                state.prior_range = HandRange.from_classes([combo_class(shown)])
        else:
            state.prior_range = HandRange.from_classes([combo_class(shown)])

        if state.posterior_range.total_weight() > 0:
            state.posterior_range = state.posterior_range.scale(
                lambda combo: 1.0 if combo == shown else 0.0
            )
            if state.posterior_range.total_weight() <= 0:
                state.posterior_range = HandRange.from_classes([combo_class(shown)])
        else:
            state.posterior_range = HandRange.from_classes([combo_class(shown)])

        state.confidence = min(
            _MAX_CONFIDENCE,
            state.confidence + _CONFIDENCE_INCREMENT,
        )
        if self.auto_save:
            self.save(state.agent_id)
        return state

    def summary(self, agent_id: str) -> dict[str, Any]:
        state = self.state_for(agent_id)
        posterior_strength = _weighted_strength(state.posterior_range)
        prior_strength = _weighted_strength(state.prior_range)
        top_classes = state.posterior_range.top_classes(_MAX_TOP_CLASSES)
        action_history = state.action_history
        last_action = action_history[-1]["action"] if action_history else None
        strong_weight = _weight_above_strength(state.posterior_range, 0.75)
        weak_weight = _weight_below_strength(state.posterior_range, 0.45)
        total_weight = state.posterior_range.total_weight()

        return {
            "agent_id": state.agent_id,
            "posterior_strength": posterior_strength,
            "prior_strength": prior_strength,
            "range_advantage": posterior_strength - prior_strength,
            "bluff_frequency": _bounded(
                weak_weight / max(total_weight, 1e-9)
                if _is_bluff_action(last_action)
                else 0.0
            ),
            "value_frequency": _bounded(
                strong_weight / max(total_weight, 1e-9)
                if _is_value_action(last_action)
                else 0.0
            ),
            "capped_probability": _capped_probability(
                state.posterior_range,
                last_action,
            ),
            "top_classes": top_classes,
            "confidence": state.confidence,
            "samples": len(action_history),
            "showdowns": len(state.showdown_hands),
        }

    def save(self, agent_id: str | None = None) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        agent_ids = [agent_id] if agent_id is not None else sorted(self._states)
        for current_agent_id in agent_ids:
            if current_agent_id is None:
                continue
            state = self._states.get(current_agent_id)
            if state is None:
                continue
            path = self._path_for(current_agent_id)
            path.write_text(
                json.dumps(state.to_json(), sort_keys=True),
                encoding="utf-8",
            )

    def _load_state(self, agent_id: str) -> RangeTrackerState:
        path = self._path_for(agent_id)
        if not path.exists():
            return RangeTrackerState(agent_id=agent_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return RangeTrackerState(agent_id=agent_id)
        state = RangeTrackerState.from_json(data)
        if state.agent_id != agent_id:
            state.agent_id = agent_id
        return state

    def _path_for(self, agent_id: str) -> Path:
        return self.state_dir / f"{_safe_filename(agent_id)}.json"


def average_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Average tracker summaries into one compact feature dict."""
    if not summaries:
        return {
            "tracker_strength": 0.0,
            "tracker_range_advantage": 0.0,
            "tracker_bluff_frequency": 0.0,
            "tracker_value_frequency": 0.0,
            "tracker_capped_probability": 0.0,
            "tracker_confidence": 0.0,
            "tracker_samples": 0,
            "tracker_top_classes": [],
        }

    averaged = {
        "tracker_strength": _average(
            summary["posterior_strength"] for summary in summaries
        ),
        "tracker_range_advantage": _average(
            summary["range_advantage"] for summary in summaries
        ),
        "tracker_bluff_frequency": _average(
            summary["bluff_frequency"] for summary in summaries
        ),
        "tracker_value_frequency": _average(
            summary["value_frequency"] for summary in summaries
        ),
        "tracker_capped_probability": _average(
            summary["capped_probability"] for summary in summaries
        ),
        "tracker_confidence": _average(summary["confidence"] for summary in summaries),
        "tracker_samples": sum(int(summary.get("samples", 0)) for summary in summaries),
    }
    top_classes: list[tuple[str, float]] = []
    for summary in summaries:
        top_classes.extend(summary.get("top_classes", []))
    merged: dict[str, float] = {}
    for hand_class, weight in top_classes:
        merged[hand_class] = merged.get(hand_class, 0.0) + float(weight)
    averaged["tracker_top_classes"] = sorted(
        merged.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:_MAX_TOP_CLASSES]
    return averaged


def _normalize_weights(weights: Any) -> dict[tuple[str, str], float]:
    normalized: dict[tuple[str, str], float] = {}
    for combo, weight in weights or []:
        try:
            normalized[normalize_combo(combo)] = float(weight)
        except ValueError:
            continue
    return normalized


def _agent_id(agent_id: str) -> str:
    value = str(agent_id or "unknown").strip()
    return value or "unknown"


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "unknown"


def _confidence_after_samples(samples: int) -> float:
    if samples <= 0:
        return 0.0
    return min(_MAX_CONFIDENCE, 1.0 - (0.85 / (1.0 + samples * 0.35)))


def _weighted_strength(hand_range: HandRange) -> float:
    total = hand_range.total_weight()
    if total <= 0:
        return 0.0
    weighted = 0.0
    for combo, weight in hand_range.weights.items():
        if weight <= 0:
            continue
        weighted += weight * class_strength(combo_class(combo))
    return _bounded(weighted / total)


def _weight_above_strength(hand_range: HandRange, threshold: float) -> float:
    return sum(
        weight
        for combo, weight in hand_range.weights.items()
        if weight > 0 and class_strength(combo_class(combo)) >= threshold
    )


def _weight_below_strength(hand_range: HandRange, threshold: float) -> float:
    return sum(
        weight
        for combo, weight in hand_range.weights.items()
        if weight > 0 and class_strength(combo_class(combo)) <= threshold
    )


def _is_bluff_action(action: str | None) -> bool:
    return str(action or "").lower() in {"bet", "raise", "3bet", "all-in", "allin"}


def _is_value_action(action: str | None) -> bool:
    return _is_bluff_action(action)


def _capped_probability(hand_range: HandRange, action: str | None) -> float:
    if not _is_bluff_action(action):
        return 0.0
    total = hand_range.total_weight()
    if total <= 0:
        return 0.0
    strong_weight = _weight_above_strength(hand_range, 0.75)
    return _bounded(1.0 - (strong_weight / max(total, 1e-9)) * 1.35)


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _average(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)
