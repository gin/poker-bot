"""Offline calibration of one configured strategy's three-player profiles.

This module only observes the configured self-play strategy. Its labels and
candidate limits are report data; they are never passed to a strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from eval.selfplay import (
    PROFILE_STATE_MODES,
    PROFILE_STATE_PERSISTENT,
    run_selfplay_parallel,
)
from poker_bot.opponent_store import connect, load_profile, profile_to_mapping

DEFAULT_CONFIG = Path("benchmarks/profile_calibration.json")
DEFAULT_SMOKE_CONFIG = Path("benchmarks/profile_calibration.smoke.json")
PROFILE_SCHEMA_VERSION = 2
WILSON_95_Z = 1.96
WILSON_99_Z = 2.5758293035489004


def wilson_interval(
    successes: int, trials: int, *, z: float = WILSON_95_Z
) -> tuple[float, float] | None:
    """Return a two-sided Wilson interval for a proportion."""

    if trials <= 0 or successes < 0 or successes > trials:
        return None
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    radius = z * (
        (proportion * (1 - proportion) + z * z / (4 * trials)) / trials
    ) ** 0.5 / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    interval = wilson_interval(numerator, denominator)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "estimate": numerator / denominator if denominator else None,
        "interval95": list(interval) if interval is not None else None,
    }


def _profile_record(profile: object, agent_id: str) -> dict[str, Any]:
    values = profile_to_mapping(profile)
    schema_version = int(values["profile_stats_schema_version"])
    provenance = values["profile_stats_provenance"]
    preflop_hands = int(values["preflop_hands_seen"])
    vpip = int(values["vpip"])
    pfr = int(values["pfr"])
    calls = int(values["calls"])
    bets = int(values["bets"])
    raises = int(values["raises"])
    fold_to_bet = int(values["fold_to_bet"])
    fold_opportunities = int(values["opportunities_to_fold_to_bet"])

    if schema_version != PROFILE_SCHEMA_VERSION or provenance != "canonical":
        raise ValueError(
            f"untrusted profile for {agent_id}: schema={schema_version}, "
            f"provenance={provenance!r}"
        )
    if not 0 <= pfr <= vpip <= preflop_hands:
        raise ValueError(
            f"canonical preflop invariant failed for {agent_id}: "
            f"pfr={pfr}, vpip={vpip}, preflop_hands_seen={preflop_hands}"
        )
    if not 0 <= fold_to_bet <= fold_opportunities:
        raise ValueError(
            f"fold-to-bet invariant failed for {agent_id}: "
            f"fold_to_bet={fold_to_bet}, opportunities={fold_opportunities}"
        )

    return {
        "agent_id": agent_id,
        "canonical": {
            "profile_stats_schema_version": schema_version,
            "profile_stats_provenance": provenance,
            "hands_seen": int(values["hands_seen"]),
            "preflop_hands_seen": preflop_hands,
            "vpip": vpip,
            "pfr": pfr,
            "calls": calls,
            "bets": bets,
            "raises": raises,
            "fold_to_bet": fold_to_bet,
            "opportunities_to_fold_to_bet": fold_opportunities,
        },
        "metrics": {
            "vpip": _metric(vpip, preflop_hands),
            "pfr": _metric(pfr, preflop_hands),
            "action_aggression": _metric(bets + raises, calls + bets + raises),
            "fold_to_bet": _metric(fold_to_bet, fold_opportunities),
        },
    }


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "strategy",
        "opponents",
        "stacks",
        "calibration_seeds",
        "holdout_seeds",
        "hands",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"calibration config missing: {', '.join(missing)}")
    if not config["opponents"] or not config["stacks"]:
        raise ValueError("opponents and stacks must be non-empty")
    if not config["calibration_seeds"] or not config["holdout_seeds"]:
        raise ValueError("calibration and holdout seeds must be non-empty")
    if set(config["calibration_seeds"]) & set(config["holdout_seeds"]):
        raise ValueError("calibration_seeds and holdout_seeds must be disjoint")
    if not isinstance(config["strategy"], str) or not config["strategy"].strip():
        raise ValueError("strategy must be a non-empty string")
    if int(config["hands"]) <= 0:
        raise ValueError("hands must be positive")
    profile_state_mode = config.get(
        "profile_state_mode", PROFILE_STATE_PERSISTENT
    )
    if profile_state_mode not in PROFILE_STATE_MODES - {"untracked"}:
        raise ValueError(
            "profile_state_mode must be 'persistent' or 'sharded_research'"
        )


    labels = config.get("offline_labels", {})
    allowed = {"gain", "fallback", "unlabeled_validation"}
    unknown = sorted(set(labels.values()) - allowed)
    if unknown:
        raise ValueError(f"unknown offline labels: {', '.join(unknown)}")
    for opponent in labels:
        if opponent not in config["opponents"]:
            raise ValueError(f"offline label references unknown opponent {opponent!r}")
    unclassified = sorted(set(config["opponents"]) - set(labels))
    if unclassified:
        raise ValueError(
            "offline_labels must classify every configured opponent: "
            + ", ".join(unclassified)
        )
    gains = {name for name, label in labels.items() if label == "gain"}
    fallbacks = {name for name, label in labels.items() if label == "fallback"}
    if not gains or not fallbacks:
        raise ValueError("offline_labels must identify at least one gain and fallback")


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    _validate_config(config)
    return config


def _run_case(
    *,
    strategy: str,
    opponent: str,
    stack: int,
    seed: int,
    hands: int,
    workers: int,
    profile_state_mode: str,
) -> dict[str, Any]:
    """Run one isolated three-player case and return its two opponent seats."""

    with tempfile.TemporaryDirectory(prefix="profile-calibration-") as directory:
        db_path = Path(directory) / "opponents.sqlite"
        result = run_selfplay_parallel(
            strategy,
            hands=hands,
            seed=seed,
            opponent_name=opponent,
            players=3,
            track_opponents=True,
            opponent_db=db_path,
            workers=workers,
            db_commit_interval=0,
            initial_stack=stack,
            profile_state_mode=profile_state_mode,
        )
        if result.strat != strategy:
            raise ValueError(
                f"calibration strategy mismatch: requested {strategy!r}, "
                f"runner returned {result.strat!r}"
            )
        conn = connect(db_path)
        try:
            seats = []
            for seat_index in (1, 2):
                agent_id = f"bot-agent-{seat_index}"
                profile = load_profile(conn, "selfplay", agent_id)
                if profile is None:
                    raise ValueError(f"missing profile for {agent_id}")
                seats.append(_profile_record(profile, agent_id))
        finally:
            conn.close()
    return {
        "strategy": strategy,
        "opponent": opponent,
        "players": 3,
        "initial_stack": stack,
        "seed": seed,
        "profile_state_mode": profile_state_mode,
        "seats": seats,
    }


def _samples(
    cases: list[dict[str, Any]], labels: dict[str, str]
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        label = labels.get(case["opponent"], "unlabeled_validation")
        for seat in case["seats"]:
            rows.append({"opponent": case["opponent"], "label": label, **seat})
    return rows


def derive_candidate_limits(
    calibration_cases: list[dict[str, Any]], labels: dict[str, str]
) -> dict[str, Any]:
    """Derive a conservative 99% VPIP ceiling from labeled gain profiles only."""

    gain_samples = [
        row for row in _samples(calibration_cases, labels) if row["label"] == "gain"
    ]
    if not gain_samples:
        raise ValueError("no labeled gain calibration profiles")

    vpip_uppers = [
        wilson_interval(
            sample["metrics"]["vpip"]["numerator"],
            sample["metrics"]["vpip"]["denominator"],
            z=WILSON_99_Z,
        )[1]
        for sample in gain_samples
        if sample["metrics"]["vpip"]["denominator"]
    ]
    if not vpip_uppers:
        raise ValueError("gain calibration profiles had no VPIP support")
    return {
        "source": "labeled_gain_calibration_vpip_upper_99_only",
        "gain_samples": len(gain_samples),
        "confidence_level": 0.99,
        "both_seats_required": True,
        "limits": {"vpip": {"max": max(vpip_uppers)}},
        "unavailable_metrics": [],
    }


def _activates(sample: dict[str, Any], limits: dict[str, Any]) -> bool:
    vpip = sample["metrics"]["vpip"]["estimate"]
    maximum = limits["vpip"]["max"]
    return vpip is not None and vpip <= maximum


def _activation_summary(
    cases: list[dict[str, Any]], labels: dict[str, str], limits: dict[str, Any]
) -> dict[str, Any]:
    summary = {}
    for case in cases:
        opponent = case["opponent"]
        entry = summary.setdefault(
            opponent,
            {
                "label": labels.get(opponent, "unlabeled_validation"),
                "profiles": 0,
                "activated": 0,
                "cases": 0,
                "both_seats_activated": 0,
            },
        )
        seat_activations = []
        for seat in case["seats"]:
            active = _activates(
                {"opponent": opponent, "label": entry["label"], **seat},
                limits,
            )
            entry["profiles"] += 1
            entry["activated"] += int(active)
            seat_activations.append(active)
        entry["cases"] += 1
        entry["both_seats_activated"] += int(all(seat_activations))
    for entry in summary.values():
        entry["activation_rate"] = entry["activated"] / entry["profiles"]
        entry["both_seats_activation_rate"] = (
            entry["both_seats_activated"] / entry["cases"]
        )
    return dict(sorted(summary.items()))


def _candidate_assessment(
    calibration_cases: list[dict[str, Any]],
    holdout_cases: list[dict[str, Any]],
    labels: dict[str, str],
) -> dict[str, Any]:
    candidate = derive_candidate_limits(calibration_cases, labels)
    limits = candidate["limits"]
    calibration = _activation_summary(calibration_cases, labels, limits)
    holdout = _activation_summary(holdout_cases, labels, limits)
    fallback_activations = [
        opponent
        for opponent, summary in holdout.items()
        if summary["label"] == "fallback" and summary["activated"]
    ]
    gain_support_failures = [
        (
            f"{opponent} (profiles {summary['activated']}/"
            f"{summary['profiles']}; both seats {summary['both_seats_activated']}/"
            f"{summary['cases']})"
        )
        for opponent, summary in holdout.items()
        if summary["label"] == "gain"
        and (
            summary["activated"] != summary["profiles"]
            or summary["both_seats_activated"] != summary["cases"]
        )
    ]
    rejection_reasons = []
    if fallback_activations:
        rejection_reasons.append(
            "labeled fallback activated on holdout: "
            + ", ".join(fallback_activations)
        )
    if gain_support_failures:
        rejection_reasons.append(
            "labeled gain did not activate every holdout profile and both seats: "
            + ", ".join(gain_support_failures)
        )
    candidate.update(
        {
            "calibration_separation": calibration,
            "holdout_activation": holdout,
            "gain_support_failures": gain_support_failures,
            "status": (
                "rejected"
                if fallback_activations or gain_support_failures
                else "screening_candidate"
            ),
            "rejection_reason": "; ".join(rejection_reasons) or None,
            "unlabeled_validation_activation": {
                opponent: summary
                for opponent, summary in holdout.items()
                if summary["label"] == "unlabeled_validation"
            },
            "unlabeled_validation_requirement": (
                "Any activation for an unlabeled_validation opponent requires "
                "a separate exact-wrapper screen before production."
                if any(
                    summary["label"] == "unlabeled_validation"
                    for summary in holdout.values()
                )
                else None
            ),
        }
    )
    return candidate


def run_profile_calibration(
    config: dict[str, Any], *, workers: int | None = None
) -> dict[str, Any]:
    """Collect isolated calibration/holdout profiles and assess offline limits."""

    _validate_config(config)
    actual_workers = int(config.get("workers", 1) if workers is None else workers)
    if actual_workers < 0:
        raise ValueError("workers must be non-negative")
    profile_state_mode = config.get(
        "profile_state_mode", PROFILE_STATE_PERSISTENT
    )
    labels = config.get("offline_labels", {})

    def cases_for(seeds: list[int]) -> list[dict[str, Any]]:
        return [
            _run_case(
                strategy=config["strategy"],
                opponent=opponent,
                stack=int(stack),
                seed=int(seed),
                hands=int(config["hands"]),
                workers=actual_workers,
                profile_state_mode=profile_state_mode,
            )
            for opponent in config["opponents"]
            for stack in config["stacks"]
            for seed in seeds
        ]

    calibration_cases = cases_for(config["calibration_seeds"])
    holdout_cases = cases_for(config["holdout_seeds"])
    return {
        "schema_version": 1,
        "strategy": config["strategy"],
        "config": {
            "strategy": config["strategy"],
            "opponents": config["opponents"],
            "stacks": config["stacks"],
            "players": 3,
            "hands": int(config["hands"]),
            "workers": actual_workers,
            "calibration_seeds": config["calibration_seeds"],
            "holdout_seeds": config["holdout_seeds"],
            "offline_labels": labels,
            "profile_state_mode": profile_state_mode,
        },
        "calibration_cases": calibration_cases,
        "holdout_cases": holdout_cases,
        "candidate": _candidate_assessment(calibration_cases, holdout_cases, labels),
    }


def _format_metric(metric: dict[str, Any]) -> str:
    interval = metric["interval95"]
    if interval is None:
        return f"{metric['numerator']}/{metric['denominator']} (unsupported)"
    return (
        f"{metric['numerator']}/{metric['denominator']} "
        f"[{interval[0]:.4f}, {interval[1]:.4f}]"
    )


def format_markdown_report(report: dict[str, Any]) -> str:
    candidate = report["candidate"]
    lines = [
        "# Offline profile calibration",
        "",
        (
            "This report is evaluator-only. Its labels and limits are never "
            "runtime strategy inputs."
        ),
        "",
        f"- Strategy: `{report['strategy']}`",
        f"- Candidate status: **{candidate['status']}**",
        f"- Calibration cases: {len(report['calibration_cases'])}",
        f"- Holdout cases: {len(report['holdout_cases'])}",
        f"- Profile state: `{report['config']['profile_state_mode']}`",
        "",
        "## Candidate VPIP threshold (simple calibration only)",
        "",
        (
            f"- Two-sided Wilson confidence: {candidate['confidence_level']:.0%}; "
            "the maximum upper bound across calibration simple profiles is used."
        ),
        "- Both opponent seats must satisfy VPIP at or below this threshold.",
        "",
        "| Metric | Maximum |",
        "| --- | ---: |",
        f"| VPIP | {candidate['limits']['vpip']['max']:.4f} |",
    ]
    lines.extend(
        [
            "",
            "## Holdout activation",
            "",
            (
                "| Opponent | Label | Activated | Profiles | Rate | "
                "Both seats | Cases |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {opponent} | {summary['label']} | {summary['activated']} | "
        f"{summary['profiles']} | {summary['activation_rate']:.1%} | "
        f"{summary['both_seats_activated']} | {summary['cases']} |"
        for opponent, summary in candidate["holdout_activation"].items()
    )
    if candidate["unlabeled_validation_requirement"]:
        lines.extend(["", candidate["unlabeled_validation_requirement"]])
    lines.extend(
        [
            "",
            "## Per-seat canonical observations",
            "",
            (
                "| Phase | Opponent | Stack | Seed | Seat | Support | Provenance | "
                "VPIP (n/d [95% CI]) | PFR (n/d [95% CI]) | "
                "Action aggression (n/d [95% CI]) | Fold-to-bet (n/d [95% CI]) |"
            ),
            (
                "| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- | "
                "--- | --- |"
            ),
        ]
    )
    for phase, cases in (
        ("calibration", report["calibration_cases"]),
        ("holdout", report["holdout_cases"]),
    ):
        for case in cases:
            for seat in case["seats"]:
                metrics = seat["metrics"]
                lines.append(
                    f"| {phase} | {case['opponent']} | {case['initial_stack']} | "
                    f"{case['seed']} | {seat['agent_id']} | "
                    f"{seat['canonical']['preflop_hands_seen']} | "
                    f"{seat['canonical']['profile_stats_provenance']} "
                    f"v{seat['canonical']['profile_stats_schema_version']} | "
                    f"{_format_metric(metrics['vpip'])} | "
                    f"{_format_metric(metrics['pfr'])} | "
                    f"{_format_metric(metrics['action_aggression'])} | "
                    f"{_format_metric(metrics['fold_to_bet'])} |"
                )
    if candidate["rejection_reason"]:
        lines.extend(["", f"Rejection: {candidate['rejection_reason']}"])
    return "\n".join(lines) + "\n"


def write_json_report(report: dict[str, Any], path: str | Path | None) -> None:
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def write_markdown_report(report: dict[str, Any], path: str | Path | None) -> None:
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_markdown_report(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate offline profile limits.")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--smoke", action="store_true", help="Use the cheap smoke config."
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--profile-state-mode",
        choices=sorted(PROFILE_STATE_MODES - {"untracked"}),
        default=None,
        help="Profile accumulation mode; sharded_research is never production-safe.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = (
        Path(args.config)
        if args.config
        else (DEFAULT_SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG)
    )
    try:
        config = load_config(config_path)
        if args.profile_state_mode is not None:
            config = {**config, "profile_state_mode": args.profile_state_mode}
        report = run_profile_calibration(config, workers=args.workers)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"profile-calibration: {exc}", file=sys.stderr)
        return 2
    write_json_report(report, args.output_json)
    write_markdown_report(report, args.output_markdown)
    print(format_markdown_report(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
