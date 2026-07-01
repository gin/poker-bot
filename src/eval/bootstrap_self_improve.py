"""Bootstrap the self-improvement loop from the current deployed model.

Seeds nn001 from the existing policy_v1.pt and value_v1.pt, then updates
champion.json to reflect the initial state.

Run once before starting the self-improvement loop for the first time:
    uv run python src/eval/bootstrap_self_improve.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval import checkpoint_registry  # noqa: E402


def main():
    policy_src = ROOT / "src" / "poker_bot" / "neural" / "policy_v1.pt"
    value_src = ROOT / "src" / "poker_bot" / "neural" / "value_v1.pt"
    champion_path = ROOT / "benchmarks" / "champion.json"

    if not policy_src.exists():
        print(f"ERROR: {policy_src} not found. Train a model first.")
        return 1

    # Check if already bootstrapped
    versions = checkpoint_registry.list_versions()
    if versions:
        print(f"Already bootstrapped. Existing versions: {versions}")
        print(f"Current version: {checkpoint_registry.current_version()}")
        return 0

    # Seed nn001
    version = checkpoint_registry.save_checkpoint(
        source_policy=policy_src,
        version="nn001",
        source_value=value_src if value_src.exists() else None,
    )
    print(f"Seeded {version} from {policy_src}")

    # Compute SHA256
    ckpt_path = checkpoint_registry.checkpoint_path("nn001")
    sha = checkpoint_registry.file_sha256(ckpt_path)

    # Update champion.json
    if champion_path.exists():
        with champion_path.open() as f:
            data = json.load(f)
    else:
        data = {}

    data.update(
        {
            "strategy": "nnbase",
            "model_version": "nn001",
            "model_checkpoint": str(ckpt_path.relative_to(ROOT)),
            "model_sha256": sha,
            "previous_strategy": data.get("previous_strategy", "auto_research_v008"),
            "previous_version": None,
            "config": "benchmarks/promotion_gate.json",
            "report": None,
            "promoted_at": None,
            "summary": {
                "status": "bootstrapped",
                "reason": (
                    "Initial NN champion seeded from existing PPO checkpoint. "
                    "Promotion gate passes required for future automated promotion."
                ),
            },
        }
    )

    with champion_path.open("w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")

    print(f"Updated {champion_path}")
    print("  Strategy: nnbase vnn001")
    print("Ready to run: uv run self-improve")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  SHA256: {sha[:16]}...")
    print(f"\nNext version will be: {checkpoint_registry.next_version()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
