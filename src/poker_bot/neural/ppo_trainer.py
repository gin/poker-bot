"""Proximal Policy Optimization (PPO) Self-Play Trainer (NN-IMPROVE-008)

Self-play reinforcement learning with experience replay:
  1. Run N hands of self-play with current policy (nnbase active)
  2. Collect per-decision telemetry from SQLite
  3. Add to replay buffer (rolling window of past experiences)
  4. Compute chip-EV advantage over full buffer
  5. PPO update on the policy network
  6. Repeat
"""

import contextlib
import json
import os
import random
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical

from poker_bot.neural.encoder import encode_state
from poker_bot.neural.models import PolicyNetwork, ACTION_MAP

INPUT_DIM = 31
NUM_ACTIONS = 5
BLIND_SIZE = 10


# Default diverse opponent pool for training
DEFAULT_TRAINING_OPPONENTS = [
    # Baseline heuristics
    "simple",
    "adaptive",
    "auto_research_v008",
    # Counter strategies
    "counter_adaptive",
    "threshold_pressure",
    "anti_threshold",
    # S2-S4 evolution (latest from each season)
    "s2v015",
    "s3v017",
    "s4v002",
    # Survival variants
    "survival_balanced",
    "survival_aggressive",
    "survival_sixmax",
    # Advanced flattened
    "flattened_v5",
    # Previous NN champions (self-play)
    "nnbase",
    "nnnext",
]


def _build_opponent_pool(
    opponent_strategies: list[str] | None,
    num_players: int,
    pool_size: int | None = None,
    rotate: bool = True,
) -> list[str]:
    """Build a diverse opponent pool for training.

    Args:
        opponent_strategies: User-provided list or None for default
        num_players: Number of players at table (need num_players-1 opponents)
        pool_size: Optional target size for rotating pool (None = use all)
        rotate: If True, shuffle pool each iteration for variety

    Returns:
        List of opponent strategy names
    """
    if opponent_strategies is None:
        opponent_pool = DEFAULT_TRAINING_OPPONENTS.copy()
    else:
        opponent_pool = opponent_strategies.copy()

    # If pool_size specified, sample from pool each iteration for diversity
    if pool_size is not None and len(opponent_pool) > pool_size:
        if rotate:
            random.shuffle(opponent_pool)
        opponent_pool = opponent_pool[:pool_size]

    # Ensure we have at least num_players-1 opponents
    if len(opponent_pool) < (num_players - 1):
        repeats = (num_players - 1) // len(opponent_pool) + 1
        opponent_pool = opponent_pool * repeats

    return opponent_pool


class ReplayBuffer:
    """Rolling buffer of (state, action, log_prob, reward, done) tuples."""

    def __init__(self, max_size=50000):
        self.max_size = max_size
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def add(self, states, actions, log_probs, rewards, dones):
        """Add new trajectory data, evicting oldest if over capacity."""
        self.states.extend(states)
        self.actions.extend(actions)
        # log_probs may be None placeholder — replaced by _recompute_log_probs
        n = len(states)
        self.log_probs.extend(log_probs if log_probs else [None] * n)
        self.rewards.extend(rewards)
        self.dones.extend(dones)
        overflow = len(self.states) - self.max_size
        if overflow > 0:
            self.states = self.states[overflow:]
            self.actions = self.actions[overflow:]
            self.log_probs = self.log_probs[overflow:]
            self.rewards = self.rewards[overflow:]
            self.dones = self.dones[overflow:]

    def __len__(self):
        return len(self.states)

    def sample(self, batch_size):
        """Random sample of indices."""
        idx = torch.randint(len(self), (batch_size,))
        return self._index_list(self.states, idx)

    def _index_list(self, items, idx):
        return [items[i] for i in idx.tolist()]


def _recompute_log_probs(model, states, actions, device):
    """Re-evaluate log-probs under current policy (on-policy correction)."""
    if not states:
        return []
    with torch.no_grad():
        state_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        logits = model(state_tensor)
        log_probs = torch.log_softmax(logits, dim=-1)
        action_tensor = torch.tensor(actions, dtype=torch.long, device=device)
        return log_probs.gather(1, action_tensor.unsqueeze(1)).squeeze(1).tolist()


def compute_gae(rewards, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    if not rewards:
        return []
    n = len(rewards)
    advantages = [0.0] * n
    gae = 0.0
    for t in reversed(range(n)):
        next_val = 0.0 if t == n - 1 else 0.0
        delta = rewards[t] + gamma * next_val - 0.0
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages


def collect_from_telemetry(model, db_path, device):
    """Read telemetry SQLite and extract (state, action, log_prob, reward, done)."""
    if not Path(db_path).exists():
        return [], [], [], [], []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT hand_id, decision_index, hero_seat_number, chosen_action,
               hero_net_chips, won_hand
        FROM decision_telemetry ORDER BY hand_id, decision_index
    """).fetchall()
    conn.close()

    if not rows:
        return [], [], [], [], []

    states = []
    actions = []
    rewards = []
    dones = []

    for r in rows:
        vec = torch.zeros(INPUT_DIM, dtype=torch.float32)
        # Simple feature extraction from telemetry
        vec[0] = r["hero_seat_number"] / 6.0
        vec[1] = 0.5  # placeholder
        states.append(vec.numpy())
        # Map action string to index
        action_str = r["chosen_action"]
        action_idx = ACTION_MAP.get(action_str, 0)
        actions.append(action_idx)
        # Compute reward from hero_net_chips (only at terminal)
        net_chips = r["hero_net_chips"] if r["hero_net_chips"] else 0
        is_terminal = r["won_hand"] is not None
        reward = (net_chips / BLIND_SIZE) if is_terminal else 0.0
        rewards.append(reward)
        dones.append(1.0 if is_terminal else 0.0)

    # Log probs from model (will be overwritten by _recompute_log_probs)
    with torch.no_grad():
        state_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        logits = model(state_tensor)
        log_probs = torch.log_softmax(logits, dim=-1)
        action_tensor = torch.tensor(actions, dtype=torch.long, device=device)
        lp = log_probs.gather(1, action_tensor.unsqueeze(1)).squeeze(1).tolist()

    return states, actions, lp, rewards, dones


def train_ppo(
    opponent_strategies=None,
    iterations=50,
    hands_per_iteration=5000,
    num_players=6,
    lr=5e-6,
    device="cpu",
    checkpoint_dir="src/poker_bot/neural",
    workers=0,
    pool_size=None,
    rotate_opponents=True,
):
    """
    PPO training loop with experience replay buffer.

    Each iteration:
      1. Self-play hands in parallel (workers=min(cpu_count,12))
      2. Collect telemetry data
      3. Add to replay buffer (max 50K samples)
      4. PPO update on full buffer (not just fresh data)
      5. Save best checkpoint
    """
    # Use relative path instead of hardcoded absolute path
    project_root = Path(__file__).resolve().parents[3]

    # Build opponent pool with optional rotation for diversity
    opponent_pool = _build_opponent_pool(
        opponent_strategies=opponent_strategies,
        num_players=num_players,
        pool_size=pool_size,
        rotate=rotate_opponents,
    )

    best_bb100 = -float("inf")

    # Resolve workers: 0 = auto (cpu_count // 2, capped at 12)
    if workers <= 0:
        num_workers = min((os.cpu_count() or 4) // 2, 12)
    else:
        num_workers = min(workers, 12)
    hands_per_worker = max(hands_per_iteration // num_workers, 100)

    # Model and optimizer setup
    checkpoint_dir = Path(checkpoint_dir)
    model_path = checkpoint_dir / "policy_ppo.pt"
    model = PolicyNetwork(input_dim=INPUT_DIM, num_actions=NUM_ACTIONS).to(device)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Resumed from {model_path}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = ReplayBuffer(max_size=50000)

    print(
        f"PPO Training Start (experience replay)\n"
        f"  iterations: {iterations}\n"
        f"  hands/iter: {hands_per_iteration}\n"
        f"  lr: {lr}\n"
        f"  buffer: {buffer.max_size}\n"
        f"  workers: {num_workers}\n"
        f"  opponents: {opponent_pool}\n"
    )

    for it in range(iterations):
        telemetry_db = Path(tempfile.mktemp(suffix=".sqlite", prefix="ppo_telemetry_"))

        # Build opponent string: need exactly num_players-1 opponents
        # Rebuild pool each iteration if rotating for diversity
        cycle_pool = (
            _build_opponent_pool(
                opponent_strategies=opponent_strategies,
                num_players=num_players,
                pool_size=pool_size,
                rotate=rotate_opponents,
            )
            if rotate_opponents
            else opponent_pool
        )
        opponent_sequences = []
        for i in range(num_players - 1):
            opponent_sequences.append(cycle_pool[i % len(cycle_pool)])
        opponent_str = ",".join(opponent_sequences)

        cmd = [
            "uv",
            "run",
            "selfplay",
            "--strat",
            "nnbase",
            "--opponent",
            opponent_str,
            "--players",
            str(num_players),
            "--hands",
            str(hands_per_worker),
            "--seed",
            str(1000 + it),
            "--workers",
            str(num_workers),
            "--telemetry",
            "--commit-every",
            "0",
            "--sqlite-fast",
            "--opponent-db",
            str(telemetry_db),
        ]
        env = {**os.environ, "NN_MODE": "6max_active"}

        try:
            result = subprocess.run(
                cmd,
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            print(f"Iter {it + 1}: self-play timed out")
            continue

        bb100 = 0.0
        if result.returncode != 0:
            print(f"Iter {it + 1}: self-play failed: {result.stderr[-300:]}")
            continue

        for line_text in result.stdout.split("\n"):
            if "bb/100" in line_text:
                with contextlib.suppress(ValueError, IndexError):
                    bb100 = float(line_text.strip().split()[-1])

        # Collect new data (log_probs recomputed below to match
        # current policy — eliminates off-policy drift)
        new_states, new_actions, _, new_rewards, new_dones = collect_from_telemetry(
            model, str(telemetry_db), device
        )
        buffer.add(new_states, new_actions, _, new_rewards, new_dones)

        if len(buffer) < 100:
            print(f"Iter {it + 1}: buffer too small ({len(buffer)}, need 100+)")
            continue

        # Recompute log_probs with CURRENT policy (on-policy)
        buffer.log_probs = _recompute_log_probs(
            model, buffer.states, buffer.actions, device
        )

        # PPO update
        states = torch.tensor(buffer.states, dtype=torch.float32, device=device)
        actions = torch.tensor(buffer.actions, dtype=torch.long, device=device)
        old_log_probs = torch.tensor(
            buffer.log_probs, dtype=torch.float32, device=device
        )
        rewards = torch.tensor(buffer.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(buffer.dones, dtype=torch.float32, device=device)

        advantages = compute_gae(buffer.rewards, buffer.dones)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = advantages + 0.0  # value_net not used yet

        batch_size = min(2048, len(buffer))
        idx = torch.randperm(len(buffer))[:batch_size]

        b_states = states[idx]
        b_actions = actions[idx]
        b_old_log_probs = old_log_probs[idx]
        b_advantages = advantages[idx]
        b_returns = returns[idx]

        # PPO clip
        clip_eps = 0.2
        logits = model(b_states)
        log_probs = torch.log_softmax(logits, dim=-1)
        new_log_probs = log_probs.gather(1, b_actions.unsqueeze(1)).squeeze(1)
        ratio = (new_log_probs - b_old_log_probs).exp()
        surr1 = ratio * b_advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy = -(log_probs.exp() * log_probs).sum(-1).mean()

        loss = policy_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(
            f"Iter [{it + 1:3d}/{iterations}] "
            f"bb/100={bb100:+7.1f} buffer={len(buffer):5d} "
            f"loss={loss.item():+.4f} entropy={entropy.item():.3f}"
        )

        if bb100 > best_bb100:
            best_bb100 = bb100
            torch.save(model.state_dict(), model_path)
            print(f"  → New best model saved!")

    print(f"\nTraining complete. Best bb/100: {best_bb100:.1f}")
    return model
