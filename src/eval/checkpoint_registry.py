"""Checkpoint registry for versioning NN model weights.

Manages nn001, nn002, ... model checkpoints and the deployed alias.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "src" / "poker_bot" / "neural" / "models"
DEPLOYED_POLICY = ROOT / "src" / "poker_bot" / "neural" / "policy_v1.pt"
CHAMPION_JSON = ROOT / "benchmarks" / "champion.json"

VERSION_PREFIX = "nn"
VERSION_DIGITS = 3


def _format_version(number: int) -> str:
    return f"{VERSION_PREFIX}{number:0{VERSION_DIGITS}d}"


def list_versions() -> list[str]:
    """Return all version names sorted ascending (['nn001', 'nn002', ...])."""
    if not MODELS_DIR.exists():
        return []
    versions = []
    for p in MODELS_DIR.glob(f"{VERSION_PREFIX}*.pt"):
        name = p.stem  # e.g. 'nn001'
        if name.endswith("_value"):
            continue
        if name.startswith(VERSION_PREFIX) and name[len(VERSION_PREFIX) :].isdigit():
            versions.append(name)
    return sorted(versions)


def parse_version(name: str) -> int:
    """Extract numeric version from name like 'nn001' -> 1."""
    if name.startswith(VERSION_PREFIX) and name[len(VERSION_PREFIX) :].isdigit():
        return int(name[len(VERSION_PREFIX) :])
    raise ValueError(f"Invalid version name: {name!r}")


def next_version() -> str:
    """Determine the next version name based on existing checkpoints."""
    versions = list_versions()
    if not versions:
        return _format_version(1)
    highest = max(parse_version(v) for v in versions)
    return _format_version(highest + 1)


def checkpoint_path(version: str, *, value: bool = False) -> Path:
    """Path to a versioned checkpoint file."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_value" if value else ""
    return MODELS_DIR / f"{version}{suffix}.pt"


def current_version() -> str | None:
    """Read current model version from champion.json, or None if not an NN champion."""
    import json

    if not CHAMPION_JSON.exists():
        return None
    with CHAMPION_JSON.open() as f:
        data = json.load(f)
    return data.get("model_version")


def current_checkpoint() -> Path | None:
    """Path to the currently deployed checkpoint (per champion.json)."""
    import json

    if not CHAMPION_JSON.exists():
        return None
    with CHAMPION_JSON.open() as f:
        data = json.load(f)
    ckpt = data.get("model_checkpoint")
    if ckpt:
        return Path(ckpt)
    return None


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deploy_version(version: str) -> None:
    """Copy versioned checkpoint to the deployed policy_v1.pt path."""
    src = checkpoint_path(version)
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint {src} does not exist")
    shutil.copy2(str(src), str(DEPLOYED_POLICY))
    # Also copy value net if present
    value_src = checkpoint_path(version, value=True)
    value_dst = DEPLOYED_POLICY.parent / "value_v1.pt"
    if value_src.exists():
        shutil.copy2(str(value_src), str(value_dst))


def save_checkpoint(
    source_policy: str | Path,
    version: str | None = None,
    *,
    source_value: str | Path | None = None,
) -> str:
    """Save a checkpoint under a versioned name.

    Returns the version string used.
    """
    if version is None:
        version = next_version()
    dst = checkpoint_path(version)
    shutil.copy2(str(source_policy), str(dst))
    if source_value is not None and source_value.exists():
        dst_value = checkpoint_path(version, value=True)
        shutil.copy2(str(source_value), str(dst_value))
    return version
