"""Convert opponent profiles and API stats into table-level reads."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TableTendencies:
    labels: tuple[str, ...] = ()
    avg_vpip: float = 0.0
    avg_pfr: float = 0.0
    avg_aggression: float = 0.0
    avg_fold_to_bet: float = 0.0
    confidence: float = 0.0
    samples: int = 0
    has_bluffer: bool = False
    has_calling_station: bool = False
    has_patient: bool = False
    has_aggressive: bool = False
    all_patient: bool = False
    raw_labels: tuple[str, ...] = field(default_factory=tuple)


def _clean_label(label):
    return str(label or "unknown").strip().lower().replace("-", "_")


def _safe_float(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _profile_row(profile):
    label = _clean_label(profile.label())
    return {
        "label": label,
        "sample": int(getattr(profile, "hands_seen", 0) or 0),
        "vpip": profile.vpip_frequency,
        "pfr": profile.pfr_frequency,
        "aggression": profile.aggression_frequency,
        "fold_to_bet": profile.fold_to_bet_frequency,
    }


def _agent_stats_label(stats):
    style = stats.get("playingStyle") or {}
    label = _clean_label(style.get("label"))
    archetype = _clean_label(style.get("archetype"))
    tightness = _clean_label(style.get("tightness"))
    aggression = _clean_label(style.get("aggression"))

    if archetype in {"maniac", "bluffer"}:
        return "bluffer"
    if archetype in {"fish", "calling_station"}:
        return "calling_station"
    if tightness == "tight" and aggression in {"passive", "measured"}:
        return "patient_methodical"
    if aggression == "aggressive" and tightness == "loose":
        return "loose_aggressive"
    if aggression == "aggressive":
        return "tight_aggressive"
    if "loose_passive" in label:
        return "calling_station"
    if "tight" in label:
        return "patient_methodical"
    return label or "unknown"


def _agent_stats_row(stats):
    style = stats.get("playingStyle") or {}
    aggression_label = _clean_label(style.get("aggression"))
    af = stats.get("af")
    if af is None:
        aggression = {"passive": 0.10, "measured": 0.28, "aggressive": 0.46}.get(
            aggression_label,
            0.0,
        )
    else:
        aggression = min(0.70, _safe_float(af) / 5)

    return {
        "label": _agent_stats_label(stats),
        "sample": int(stats.get("sampleSize") or 0),
        "vpip": _safe_float(stats.get("vpip")),
        "pfr": _safe_float(stats.get("pfr")),
        "aggression": aggression,
        "fold_to_bet": 0.0,
    }


def _weighted_average(rows, key):
    total = sum(max(1, row["sample"]) for row in rows)
    if total <= 0:
        return 0.0
    return sum(row[key] * max(1, row["sample"]) for row in rows) / total


def summarize_tendencies(profiles=(), agent_stats=()) -> TableTendencies:
    rows = [_profile_row(profile) for profile in profiles]
    rows.extend(
        _agent_stats_row(stats) for stats in agent_stats if isinstance(stats, dict)
    )
    if not rows:
        return TableTendencies()

    labels = tuple(row["label"] for row in rows)
    samples = sum(row["sample"] for row in rows)
    confidence = min(1.0, samples / max(20, 20 * len(rows)))

    has_bluffer = any(label in {"bluffer", "loose_aggressive"} for label in labels)
    has_calling_station = any(label == "calling_station" for label in labels)
    has_patient = any(label == "patient_methodical" for label in labels)
    has_aggressive = any(
        label in {"bluffer", "loose_aggressive", "tight_aggressive"} for label in labels
    )
    all_patient = bool(labels) and all(
        label in {"patient_methodical", "unknown"} for label in labels
    )

    return TableTendencies(
        labels=labels,
        avg_vpip=_weighted_average(rows, "vpip"),
        avg_pfr=_weighted_average(rows, "pfr"),
        avg_aggression=_weighted_average(rows, "aggression"),
        avg_fold_to_bet=_weighted_average(rows, "fold_to_bet"),
        confidence=confidence,
        samples=samples,
        has_bluffer=has_bluffer,
        has_calling_station=has_calling_station,
        has_patient=has_patient,
        has_aggressive=has_aggressive,
        all_patient=all_patient,
        raw_labels=labels,
    )
