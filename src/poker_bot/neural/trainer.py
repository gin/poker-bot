"""
Trainer: extracts state-action pairs from all telemetry databases,
encodes them via encoder.py, and trains the policy / value networks.

v3 (NN-IMPROVE-004):
  - Chip-outcome sample weighting: decisions that won/lost more chips
    get higher weight in CrossEntropyLoss, so the NN prioritizes
    learning high-stakes situations correctly.
  - WeightedRandomSampler so each batch sees proportional representation
    of high-impact decisions.
"""

import sqlite3
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from poker_bot.neural.encoder import encode_state
from poker_bot.neural.models import PolicyNetwork, ValueNetwork

# ── Strategy tiers ──────────────────────────────────────────────────────
# Tier 1: Proven strong strategies — always imitate
TIER_1_STRATEGIES = [
    "hubase",
    "hu005",
    "hu006",
    "hu007",
    "s3v016",
    "s3v017",
    "s4base",
    "s4v001",
]

# Tier 2: Acceptable but older/volatile — include if volume needed
TIER_2_STRATEGIES = [
    "s3v015",
    "s4v002",
]

# Excluded (known leaks or weak): s2base, s2v009, s2v010, auto_research_*

# ── Sample weighting ────────────────────────────────────────────────────
# Weight = sqrt(|net_chips| / BB + 1) — gives ~1.0 for small pots,
# ~3.0 for 100BB swings, diminishing returns beyond.
# This focuses the NN on high-EV decisions so large pots don't get
# swamped by the many trivial folds in CrossEntropyLoss.
BLIND_SIZE = 10


def discover_sqlite_dbs(root: str = ".") -> list[str]:
    """Find all .sqlite files in project root (not in .venv or __pycache__)."""
    dbs = []
    for p in Path(root).glob("*.sqlite"):
        if ".venv" not in str(p) and "__pycache__" not in str(p):
            dbs.append(str(p))
    return sorted(dbs)


def _parse_card_list(raw) -> list[str]:
    """Parse hole_cards/board_cards from various storage formats."""
    if not raw:
        return []
    if isinstance(raw, str):
        # JSON array '["Ah","Kd"]', CSV 'Ah,Kd', or space-separated
        raw = raw.strip()
        if raw.startswith("["):
            import json

            try:
                return json.loads(raw)
            except Exception:
                pass
        # Try comma or space separated
        if "," in raw:
            return [c.strip().strip("'\"") for c in raw.split(",")]
        return raw.split()
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return []


def _parse_available_actions(raw) -> list[str]:
    """Parse available_actions from storage."""
    if not raw:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            import json

            try:
                return json.loads(raw)
            except Exception:
                pass
        if "," in raw:
            return [a.strip().strip("'\"") for a in raw.split(",")]
        return raw.split()
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return []


def _chip_weight(net_chips: float) -> float:
    """Sample weight from chip outcome — sqrt(|chips|/BB + 1)."""
    import math

    return math.sqrt(abs(net_chips) / BLIND_SIZE + 1.0)


class PokerTelemetryDataset(Dataset):
    """
    Dataset extracting (state, label, weight) from decision_telemetry.
    Auto-discovers .sqlite files and filters by strategy tier.

    Weight comes from ``hero_net_chips`` when available (gameplay.sqlite),
    falling back to uniform weighting for DBs without outcome data.
    """

    def __init__(
        self,
        db_paths: list[str] | None = None,
        target_strategies: list[str] | None = None,
        project_root: str = ".",
    ):
        self.target_strategies = target_strategies or TIER_1_STRATEGIES
        if db_paths is None:
            db_paths = discover_sqlite_dbs(project_root)
        self.db_paths = [Path(p) for p in db_paths if Path(p).exists()]
        self.data = []  # list of (state_vec, label_int)
        self.weights = []  # parallel list of float weights

        print(
            f"Loading {len(self.target_strategies)} strategies "
            f"from {len(self.db_paths)} databases..."
        )
        for db_path in self.db_paths:
            before = len(self.data)
            self._load_db(db_path)
            delta = len(self.data) - before
            if delta > 0:
                print(f"  {db_path.name}: +{delta} samples (total: {len(self.data)})")

        if self.weights:
            w_avg = sum(self.weights) / len(self.weights)
            print(f"Weight stats: mean={w_avg:.2f} max={max(self.weights):.1f}")
        print(f"Total training samples collected: {len(self.data)}")

    def _load_db(self, db_path: Path):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        strategy_filter = ",".join([f"'{s}'" for s in self.target_strategies])
        query = (
            f"SELECT * FROM decision_telemetry WHERE strategy IN ({strategy_filter})"
        )

        try:
            rows = cursor.execute(query).fetchall()
            for row in rows:
                row_dict = dict(row)

                hole_cards = _parse_card_list(row_dict.get("hole_cards"))
                board_cards = _parse_card_list(row_dict.get("board_cards"))
                available = _parse_available_actions(row_dict.get("available_actions"))

                call_amt = (
                    row_dict.get("call_amount") or row_dict.get("callAmount", 0) or 0
                )
                table_state = {
                    "potChips": row_dict.get("pot_chips")
                    or row_dict.get("potChips", 0)
                    or 0,
                    "allowedActions": {
                        "callAmount": call_amt,
                        "availableActions": available,
                    },
                    "buttonSeatNumber": row_dict.get("button_seat_number")
                    or row_dict.get("buttonSeatNumber", 0),
                    "boardCards": board_cards,
                    "street": row_dict.get("street", "Flop"),
                    "currentBet": row_dict.get("current_bet")
                    or row_dict.get("currentBet", 0)
                    or 0,
                    "actionHistory": [],
                    "seats": _build_seats(row_dict),
                }

                hero_seat = {
                    "seatNumber": row_dict.get("hero_seat_number")
                    or row_dict.get("heroSeatNumber", 1),
                    "stackChips": row_dict.get("hero_stack")
                    or row_dict.get("heroStack", 2000),
                    "holeCards": hole_cards,
                }

                action = row_dict.get("chosen_action") or row_dict.get("action", "fold")
                action_map = {
                    "fold": 0,
                    "call": 1,
                    "check": 2,
                    "raise": 3,
                    "all-in": 4,
                    "allin": 4,
                    "bet": 3,
                }
                label = action_map.get(action, 0)

                # Weight: chip outcome if available, else 1.0
                net_chips = row_dict.get("hero_net_chips")
                weight = (
                    _chip_weight(float(net_chips)) if net_chips is not None else 1.0
                )

                try:
                    state_vec = encode_state(table_state, hero_seat)
                    self.data.append((state_vec, label))
                    self.weights.append(weight)
                except Exception:
                    continue

        except sqlite3.OperationalError as e:
            print(f"  Skipping {db_path.name}: {e}")
        finally:
            conn.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, label = self.data[idx]
        weight = self.weights[idx]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(weight, dtype=torch.float32),
        )


def _build_seats(row_dict: dict) -> list[dict]:
    """Rebuild minimal seats list for encoder's num_active_players."""
    active = row_dict.get("active_players") or row_dict.get("activePlayers", 2)
    hero_id = row_dict.get("hero_agent_id") or "hero"
    seats = [{"agentId": hero_id, "folded": False}]
    for _ in range(max(0, int(active) - 1)):
        seats.append({"agentId": "opp", "folded": False})
    return seats


def train_policy_network(
    db_files: list[str] | None = None,
    epochs: int = 25,
    lr: float = 0.0005,
    batch_size: int = 256,
    project_root: str = ".",
):
    """
    Supervised pre-training to imitate strong strategies.
    Uses sqrt(|net_chips| / BB + 1) sample weighting so high-stakes
    decisions dominate the gradient.
    """
    dataset = PokerTelemetryDataset(db_paths=db_files, project_root=project_root)
    if len(dataset) == 0:
        print("No data found for training.")
        return None

    # Train/val split 90/10
    n = len(dataset)
    n_train = int(n * 0.9)
    n_val = n - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    # Weighted sampler: high-|chips| decisions appear more often
    train_weights = [dataset.weights[i] for i in train_set.indices]
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_set),
        replacement=True,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    sample_state, _, _ = dataset[0]
    input_dim = sample_state.shape[0]
    model = PolicyNetwork(input_dim=input_dim, dropout=0.15)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    print(
        f"Weighted training on {n_train} samples, validating on {n_val} "
        f"(input_dim={input_dim}, batch={batch_size})"
    )

    best_val_acc = 0.0
    for epoch in range(epochs):
        # ── Train ──
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for states, labels, weights in train_loader:
            optimizer.zero_grad()
            outputs = model(states)
            # Per-sample weighted loss
            loss_per_sample = nn.functional.cross_entropy(
                outputs, labels, reduction="none"
            )
            loss = (loss_per_sample * weights).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for states, labels, _ in val_loader:
                outputs = model(states)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * (val_correct / val_total) if val_total > 0 else 0
        print(
            f"Epoch [{epoch + 1:2d}/{epochs}] "
            f"loss={avg_loss:.4f} train_acc={train_acc:.1f}% "
            f"val_acc={val_acc:.1f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "src/poker_bot/neural/policy_v1.pt")

    print(f"Best val accuracy: {best_val_acc:.1f}%")
    print("Model saved to src/poker_bot/neural/policy_v1.pt")
    return model


class ValueTelemetryDataset(Dataset):
    """
    Dataset that extracts state→EV pairs for Value Network training.
    Only uses rows where hero_net_chips is available (gameplay.sqlite).
    Target: net_chips / hero_stack, clipped to [-1, 1].
    """

    def __init__(
        self,
        db_paths: list[str] | None = None,
        target_strategies: list[str] | None = None,
        project_root: str = ".",
    ):
        self.target_strategies = target_strategies or TIER_1_STRATEGIES
        if db_paths is None:
            db_paths = discover_sqlite_dbs(project_root)
        self.db_paths = [Path(p) for p in db_paths if Path(p).exists()]
        self.data = []

        print(f"Loading value targets from {len(self.db_paths)} databases...")
        for db_path in self.db_paths:
            before = len(self.data)
            self._load_db(db_path)
            delta = len(self.data) - before
            if delta > 0:
                print(f"  {db_path.name}: +{delta} samples (total: {len(self.data)})")

        print(f"Total value samples collected: {len(self.data)}")

    def _load_db(self, db_path: Path):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        strategy_filter = ",".join([f"'{s}'" for s in self.target_strategies])
        # Only rows with non-null net_chips and positive stack
        query = (
            f"SELECT * FROM decision_telemetry "
            f"WHERE strategy IN ({strategy_filter}) "
            f"AND hero_net_chips IS NOT NULL "
            f"AND hero_stack > 0"
        )

        try:
            rows = cursor.execute(query).fetchall()
            for row in rows:
                row_dict = dict(row)

                hole_cards = _parse_card_list(row_dict.get("hole_cards"))
                board_cards = _parse_card_list(row_dict.get("board_cards"))
                available = _parse_available_actions(row_dict.get("available_actions"))

                call_amt = (
                    row_dict.get("call_amount") or row_dict.get("callAmount", 0) or 0
                )
                table_state = {
                    "potChips": row_dict.get("pot_chips")
                    or row_dict.get("potChips", 0)
                    or 0,
                    "allowedActions": {
                        "callAmount": call_amt,
                        "availableActions": available,
                    },
                    "buttonSeatNumber": row_dict.get("button_seat_number")
                    or row_dict.get("buttonSeatNumber", 0),
                    "boardCards": board_cards,
                    "street": row_dict.get("street", "Flop"),
                    "currentBet": row_dict.get("current_bet")
                    or row_dict.get("currentBet", 0)
                    or 0,
                    "actionHistory": [],
                    "seats": _build_seats(row_dict),
                }

                hero_seat = {
                    "seatNumber": row_dict.get("hero_seat_number")
                    or row_dict.get("heroSeatNumber", 1),
                    "stackChips": row_dict.get("hero_stack")
                    or row_dict.get("heroStack", 2000),
                    "holeCards": hole_cards,
                }

                # EV target: net_chips / stack, clipped to [-1, 1]
                net_chips = row_dict.get("hero_net_chips", 0)
                stack = row_dict.get("hero_stack", 1)
                ev_target = max(-1.0, min(1.0, net_chips / (stack + 1e-6)))

                try:
                    state_vec = encode_state(table_state, hero_seat)
                    self.data.append((state_vec, ev_target))
                except Exception:
                    continue

        except sqlite3.OperationalError as e:
            print(f"  Skipping {db_path.name}: {e}")
        finally:
            conn.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, ev = self.data[idx]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(ev, dtype=torch.float32),
        )


def train_value_network(
    db_files: list[str] | None = None,
    epochs: int = 20,
    lr: float = 0.001,
    batch_size: int = 256,
    project_root: str = ".",
):
    """
    Train the Value Network to estimate EV from game state.
    """
    dataset = ValueTelemetryDataset(db_paths=db_files, project_root=project_root)
    if len(dataset) == 0:
        print("No data found for value training.")
        return None

    n = len(dataset)
    n_train = int(n * 0.9)
    n_val = n - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    sample_state, _ = dataset[0]
    input_dim = sample_state.shape[0]
    model = ValueNetwork(input_dim=input_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    print(
        f"Value Network: training on {n_train} samples, "
        f"validating on {n_val} (input_dim={input_dim})"
    )

    best_val_loss = float("inf")
    for epoch in range(epochs):
        # ── Train ──
        model.train()
        total_loss = 0
        for states, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(states).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0
        val_total = 0
        with torch.no_grad():
            for states, targets in val_loader:
                outputs = model(states).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * states.size(0)
                val_total += states.size(0)

        avg_val_loss = val_loss / val_total if val_total > 0 else 0
        print(
            f"Epoch [{epoch + 1:2d}/{epochs}] "
            f"train_mse={avg_loss:.4f} val_mse={avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "src/poker_bot/neural/value_v1.pt")

    print(f"Best val MSE: {best_val_loss:.4f}")
    print("Model saved to src/poker_bot/neural/value_v1.pt")
    return model


if __name__ == "__main__":
    # Use relative path from this file's location
    project_root = str(Path(__file__).resolve().parents[3])
    train_policy_network(project_root=project_root)
    train_value_network(project_root=project_root)
