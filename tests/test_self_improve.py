"""Smoke tests for the self-improvement loop components."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval import checkpoint_registry  # noqa: E402


@pytest.fixture
def tmp_checkpoint_env(tmp_path):
    """Set up a temporary checkpoint registry environment."""
    tmp_models = tmp_path / "models"
    tmp_models.mkdir()

    orig_models = checkpoint_registry.MODELS_DIR
    orig_deployed = checkpoint_registry.DEPLOYED_POLICY
    orig_champion = checkpoint_registry.CHAMPION_JSON

    checkpoint_registry.MODELS_DIR = tmp_models
    checkpoint_registry.DEPLOYED_POLICY = tmp_path / "policy_v1.pt"

    yield tmp_path, tmp_models

    checkpoint_registry.MODELS_DIR = orig_models
    checkpoint_registry.DEPLOYED_POLICY = orig_deployed
    checkpoint_registry.CHAMPION_JSON = orig_champion


def _make_fake_checkpoint(path, content="fake"):
    with open(path, "wb") as f:
        f.write(content.encode())


def test_next_version_starts_at_1(tmp_checkpoint_env):
    assert checkpoint_registry.next_version() == "nn001"


def test_save_and_list_versions(tmp_checkpoint_env):
    tmp_path, _ = tmp_checkpoint_env
    fake_ppo = tmp_path / "policy_ppo.pt"
    _make_fake_checkpoint(fake_ppo)
    checkpoint_registry.save_checkpoint(fake_ppo, version="nn001")
    assert checkpoint_registry.list_versions() == ["nn001"]


def test_next_version_increments(tmp_checkpoint_env):
    tmp_path, _ = tmp_checkpoint_env
    fake_ppo = tmp_path / "policy_ppo.pt"
    _make_fake_checkpoint(fake_ppo, "v1")
    checkpoint_registry.save_checkpoint(fake_ppo, version="nn001")
    _make_fake_checkpoint(fake_ppo, "v2")
    checkpoint_registry.save_checkpoint(fake_ppo, version="nn002")
    assert checkpoint_registry.next_version() == "nn003"


def test_checkpoint_path():
    path = checkpoint_registry.checkpoint_path("nn001")
    assert path.name == "nn001.pt"
    path_value = checkpoint_registry.checkpoint_path("nn001", value=True)
    assert path_value.name == "nn001_value.pt"


def test_file_sha256(tmp_checkpoint_env):
    tmp_path, _ = tmp_checkpoint_env
    f = tmp_path / "test.pt"
    _make_fake_checkpoint(f, "test-content")
    sha = checkpoint_registry.file_sha256(f)
    assert sha is not None
    assert len(sha) == 64  # hex digest of sha256


def test_deploy_version(tmp_checkpoint_env):
    tmp_path, _ = tmp_checkpoint_env
    fake_ppo = tmp_path / "policy_ppo.pt"
    _make_fake_checkpoint(fake_ppo)
    checkpoint_registry.save_checkpoint(fake_ppo, version="nn001")
    checkpoint_registry.deploy_version("nn001")
    assert checkpoint_registry.DEPLOYED_POLICY.exists()


def test_current_version(tmp_checkpoint_env):
    tmp_path, _ = tmp_checkpoint_env
    champion_json = tmp_path / "champion.json"
    with open(champion_json, "w") as f:
        json.dump({"strategy": "nnbase", "model_version": "nn001"}, f)
    checkpoint_registry.CHAMPION_JSON = champion_json
    assert checkpoint_registry.current_version() == "nn001"


def test_save_with_value_net(tmp_checkpoint_env):
    tmp_path, _ = tmp_checkpoint_env
    fake_policy = tmp_path / "policy_ppo.pt"
    fake_value = tmp_path / "value_ppo.pt"
    _make_fake_checkpoint(fake_policy, "policy-weights")
    _make_fake_checkpoint(fake_value, "value-weights")
    checkpoint_registry.save_checkpoint(
        fake_policy, version="nn003", source_value=fake_value
    )
    assert checkpoint_registry.checkpoint_path("nn003").exists()
    assert checkpoint_registry.checkpoint_path("nn003", value=True).exists()
