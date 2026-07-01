"""Persistent self-improvement loop for the NN poker bot.

Orchestrates: train → version → stage → gate → deploy → repeat.

Designed to run as a long-lived process (systemd, screen, or nohup).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval import checkpoint_registry  # noqa: E402
from eval.promotion_gate import run_promotion_gate  # noqa: E402

# Training defaults
DEFAULT_TRAIN_ITERATIONS = 10
DEFAULT_HANDS_PER_ITERATION = 5000
DEFAULT_LR = 5e-6
DEFAULT_SLEEP_SECONDS = 1800  # 30 minutes between cycles
DEFAULT_OPPONENTS = [
    "simple",
    "survival_balanced",
    "survival_aggressive",
    "auto_research_v008",
    "flattened_v2",
    "s2v015",
    "s3v017",
    "s4v002",
    "hu008",
    "nnprev",
    "nnnext",
]

POLICY_PPO_PATH = ROOT / "src" / "poker_bot" / "neural" / "policy_ppo.pt"
VALUE_PPO_PATH = ROOT / "src" / "poker_bot" / "neural" / "value_ppo.pt"
DEPLOYED_POLICY = ROOT / "src" / "poker_bot" / "neural" / "policy_v1.pt"
DEPLOYED_VALUE = ROOT / "src" / "poker_bot" / "neural" / "value_v1.pt"
CHAMPION_JSON = ROOT / "benchmarks" / "champion.json"
LOG_DIR = ROOT / "benchmark-runs" / "self_improve"
HISTORY_LOG = LOG_DIR / "history.jsonl"


def _log_iteration(record: dict) -> None:
    """Append a JSONL summary row for this cycle."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    with HISTORY_LOG.open("a") as f:
        json.dump(record, f)
        f.write("\n")


def _train(
    iterations: int,
    hands_per_iteration: int,
    lr: float,
    opponents: list[str],
    device: str = "cpu",
    workers: int = 0,
) -> bool:
    """Run PPO training. Returns True on success."""
    from poker_bot.neural.ppo_trainer import train_ppo

    try:
        train_ppo(
            opponent_strategies=opponents,
            iterations=iterations,
            hands_per_iteration=hands_per_iteration,
            lr=lr,
            device=device,
            workers=workers,
        )
        return True
    except Exception as exc:
        print(f"[train] Training failed: {exc}")
        return False


def _stage_checkpoint(version: str) -> bool:
    """Copy policy_ppo.pt to versioned path and deploy to policy_v1.pt."""
    if not POLICY_PPO_PATH.exists():
        print(f"[stage] No checkpoint at {POLICY_PPO_PATH}")
        return False

    # Save versioned copy
    checkpoint_registry.save_checkpoint(
        source_policy=POLICY_PPO_PATH,
        version=version,
        source_value=VALUE_PPO_PATH if VALUE_PPO_PATH.exists() else None,
    )
    print(f"[stage] Saved checkpoint as {version}")

    # Deploy to policy_v1.pt (staged for gate evaluation)
    shutil.copy2(str(POLICY_PPO_PATH), str(DEPLOYED_POLICY))
    if VALUE_PPO_PATH.exists():
        shutil.copy2(str(VALUE_PPO_PATH), str(DEPLOYED_VALUE))
    print(f"[stage] Deployed {version} to policy_v1.pt")
    return True


def _run_gate(candidate_checkpoint: str) -> bool:
    """Run promotion gate. Returns True if promoted."""
    champion_ckpt = None
    if CHAMPION_JSON.exists():
        with CHAMPION_JSON.open() as f:
            champ = json.load(f)
        champion_ckpt = champ.get("model_checkpoint")

    try:
        report = run_promotion_gate(
            "nnbase",
            nn_mode="6max_active",
            candidate_checkpoint=candidate_checkpoint,
            champion_checkpoint=champion_ckpt,
        )
        print(f"[gate] Passed: {report.passed}, Promoted: {report.promoted}")
        if report.passed:
            print("[gate] Gate checks:")
            for check in report.checks:
                status = "PASS" if check.passed else "FAIL"
                print(f"  {status}: {check.name} — {check.detail}")
        return report.promoted
    except Exception as exc:
        print(f"[gate] Gate failed with error: {exc}")
        return False


def run_cycle(
    *,
    train_iterations: int,
    hands_per_iteration: int,
    lr: float,
    opponents: list[str],
    device: str = "cpu",
    workers: int = 0,
    dry_run: bool = False,
) -> dict:
    """Execute one full self-improvement cycle."""
    cycle_start = time.perf_counter()
    result: dict = {
        "cycle_started": datetime.now(UTC).isoformat(timespec="seconds"),
        "version": None,
        "trained": False,
        "staged": False,
        "promoted": False,
        "error": None,
    }

    try:
        # 1. Determine next version
        version = checkpoint_registry.next_version()
        result["version"] = version
        print(f"\n{'=' * 60}")
        print(f"  Self-Improvement Cycle - Target version: {version}")
        print(f"{'=' * 60}")

        # 2. Train
        print(f"\n[1/4] Training PPO ({train_iterations} iterations)...")
        trained = _train(
            iterations=train_iterations,
            hands_per_iteration=hands_per_iteration,
            lr=lr,
            opponents=opponents,
            device=device,
            workers=workers,
        )
        result["trained"] = trained
        if not trained:
            result["error"] = "training failed"
            return result

        # 3. Stage checkpoint
        print(f"\n[2/4] Staging checkpoint as {version}...")
        staged = _stage_checkpoint(version)
        result["staged"] = staged
        if not staged:
            result["error"] = "no checkpoint to stage"
            return result

        # 4. Run promotion gate
        print("\n[3/4] Running promotion gate...")
        if dry_run:
            print("[gate] DRY RUN - skipping actual gate execution")
            result["promoted"] = False
            result["dry_run"] = True
        else:
            ckpt_path = str(checkpoint_registry.checkpoint_path(version))
            promoted = _run_gate(candidate_checkpoint=ckpt_path)
            result["promoted"] = promoted

        # 5. Report
        elapsed = time.perf_counter() - cycle_start
        result["elapsed_seconds"] = round(elapsed, 1)
        if result["promoted"]:
            print(f"\n[4/4] Cycle complete in {elapsed:.0f}s")
            print(f"  {version} PROMOTED to champion!")
        else:
            print(f"\n[4/4] Cycle complete in {elapsed:.0f}s")
            print("  Did not pass gate. Champion unchanged.")

    except Exception as exc:
        result["error"] = str(exc)
        print(f"[cycle] Error: {exc}")

    return result


def run_loop(
    *,
    train_iterations: int,
    hands_per_iteration: int,
    lr: float,
    opponents: list[str],
    sleep_seconds: int,
    device: str = "cpu",
    workers: int = 0,
    max_cycles: int | None = None,
    dry_run: bool = False,
) -> None:
    """Run the self-improvement loop continuously."""
    cycle = 0
    print("=" * 60)
    print("  NN Poker Bot - Self-Improvement Loop")
    print(f"  Training {train_iterations} iters x {hands_per_iteration} hands")
    print(f"  LR: {lr}")
    print(f"  Opponents: {', '.join(opponents)}")
    print(f"  Sleep between cycles: {sleep_seconds}s")
    print(f"  Max cycles: {max_cycles or 'unlimited'}")
    print(f"  Dry run: {dry_run}")
    print("=" * 60)

    while True:
        cycle += 1
        if max_cycles and cycle > max_cycles:
            print(f"\nReached max cycles ({max_cycles}). Stopping.")
            break

        print(f"\n\n>>> CYCLE {cycle} <<<")
        result = run_cycle(
            train_iterations=train_iterations,
            hands_per_iteration=hands_per_iteration,
            lr=lr,
            opponents=opponents,
            device=device,
            workers=workers,
            dry_run=dry_run,
        )
        _log_iteration(result)

        if max_cycles and cycle >= max_cycles:
            break

        print(f"\nSleeping {sleep_seconds}s until next cycle...")
        time.sleep(sleep_seconds)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Persistent self-improvement loop for NN poker bot."
    )
    parser.add_argument(
        "--train-iterations",
        type=int,
        default=DEFAULT_TRAIN_ITERATIONS,
        help="PPO training iterations per cycle (default: %(default)s)",
    )
    parser.add_argument(
        "--hands-per-iteration",
        type=int,
        default=DEFAULT_HANDS_PER_ITERATION,
        help="Hands per training iteration (default: %(default)s)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=DEFAULT_OPPONENTS,
        help="Opponent strategies for self-play",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=DEFAULT_SLEEP_SECONDS,
        help="Seconds to sleep between cycles (default: %(default)s)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Stop after N cycles (default: run forever)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of workers for self-play (0 = auto, default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and stage but skip the promotion gate",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single cycle and exit (no loop)",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.once:
        result = run_cycle(
            train_iterations=args.train_iterations,
            hands_per_iteration=args.hands_per_iteration,
            lr=args.lr,
            opponents=args.opponents,
            device=args.device,
            dry_run=args.dry_run,
        )
        _log_iteration(result)
        return 0 if result.get("promoted") or result.get("dry_run") else 1

    run_loop(
        train_iterations=args.train_iterations,
        hands_per_iteration=args.hands_per_iteration,
        lr=args.lr,
        opponents=args.opponents,
        sleep_seconds=args.sleep,
        device=args.device,
        max_cycles=args.max_cycles,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
