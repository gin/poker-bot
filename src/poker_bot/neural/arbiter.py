"""Neural arbiter for NN vs heuristic action selection."""
import os
from pathlib import Path
from typing import Any, Optional

import torch

from poker_bot.neural.encoder import encode_state
from poker_bot.neural.guardrails import GuardRail, GuardResult, get_guardrail, _action_from_guard
from poker_bot.neural.models import ACTION_MAP, PolicyNetwork

INPUT_DIM = 31


class NeuralArbiter:
    """Meta-Arbiter that selects between NN proposal and heuristic baseline."""

    def __init__(
        self,
        policy_path=None,
        value_path=None,
        trust_threshold: float = 0.85,
        mode: str = "shadow",
    ):
        root = Path(__file__).resolve().parents[3]
        self.policy_path = policy_path or os.environ.get(
            "NN_POLICY_PATH", f"{root}/assets/policy_v1.pt"
        )
        self.value_path = value_path or os.environ.get(
            "NN_VALUE_PATH", f"{root}/assets/value_v1.pt"
        )
        self.policy_net = None
        self.value_net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trust_threshold = trust_threshold
        self.mode = mode
        self.guards: GuardRail = get_guardrail()
        self._telemetry_conn: Optional[Any] = None
        self._run_id: Optional[str] = None
        self._decision_index: int = 0

    def configure_telemetry(self, conn: Any, *, run_id: str) -> None:
        """Attach a telemetry connection so guard overrides are logged."""
        self._telemetry_conn = conn
        self._run_id = run_id
        self._decision_index = 0

    def _load_policy(self):
        if self.policy_net is None:
            self.policy_net = PolicyNetwork(input_dim=INPUT_DIM)
            if Path(self.policy_path).exists():
                self.policy_net.load_state_dict(
                    torch.load(self.policy_path, map_location=self.device, weights_only=True)
                )
            self.policy_net.to(self.device)
            self.policy_net.eval()

    def _encode(self, table, hero_seat, opponent_profile=None):
        state_vec = encode_state(table, hero_seat, opponent_profile=opponent_profile)
        return torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _apply_and_log(
        self,
        guard_result: GuardResult,
        table: dict[str, Any],
        hero_seat: dict[str, Any],
        heuristic_action: str,
    ) -> tuple[str, str]:
        """Apply guard override and log to telemetry DB."""
        action, amount, reason = self.guards._action_from_guard(guard_result, table)

        if self._telemetry_conn and self._run_id is not None:
            try:
                self.guards.log_override(
                    self._telemetry_conn,
                    run_id=self._run_id,
                    hand_id=f"{table.get('tableId', 'unknown')}:{table.get('gameId', 'unknown')}",
                    decision_index=self._decision_index,
                    guard_result=guard_result,
                    table=table,
                    seat=hero_seat,
                )
            except Exception:
                pass  # Never let telemetry logging break play

        self._decision_index += 1
        return action, reason

    def decide(self, table, hero_seat, heuristic_action, opponent_profile=None):
        """NN proposes -> Pre-guard -> Guard validates -> Selection."""
        self._load_policy()

        # 1. Pre-decision guards (before NN even proposes)
        pre_guard = self.guards.run_pre(table, hero_seat)
        if pre_guard and pre_guard.fired:
            if self.mode == "shadow":
                return (
                    heuristic_action,
                    f"shadow pre_guard={pre_guard.guard_id} reason={pre_guard.reason}",
                )
            action, reason = self._apply_and_log(pre_guard, table, hero_seat, heuristic_action)
            return action, reason

        # 2. NN Proposal
        state_tensor = self._encode(table, hero_seat, opponent_profile)
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            nn_action = ACTION_MAP.get(pred_idx.item(), "fold")
            confidence = conf.item()

        # 3. Post-decision guard validation
        post_guard = self.guards.run_post(table, hero_seat, nn_action)

        # 4. Selection
        if self.mode == "shadow":
            return (
                heuristic_action,
                f"shadow nn={nn_action} conf={confidence:.2f} guard={post_guard.guard_id}",
            )

        table_street = table.get("street", "")
        street_threshold = {
            "Preflop": 0.85,
            "Flop": 0.95,
            "Turn": 0.92,
            "River": 0.80,
        }
        threshold = street_threshold.get(table_street, self.trust_threshold)

        if confidence < threshold:
            return (
                heuristic_action,
                f"heuristic_fallback street={table_street} conf={confidence:.2f}<{threshold:.2f}",
            )

        allowed = table.get("allowedActions") or {}
        call_amt = allowed.get("callAmount", 0) or 0
        pot = table.get("potChips", 0) or 0
        if call_amt > 0 and call_amt > pot * 0.5:
            return (
                heuristic_action,
                f"heuristic_shove_defense call={call_amt} pot={pot}",
            )

        # 5. Apply post-decision guard if it fired
        if post_guard.fired:
            action, reason = self._apply_and_log(post_guard, table, hero_seat, heuristic_action)
            return action, reason

        return nn_action, f"nn_active conf={confidence:.2f}"

    def shadow_decide(self, table, hero_seat, heuristic_action, opponent_profile=None):
        """Shadow-Mode decision for A/B logging."""
        orig_mode = self.mode
        self.mode = "shadow"
        action, reason = self.decide(table, hero_seat, heuristic_action, opponent_profile=opponent_profile)
        self.mode = orig_mode
        return {
            "heuristic_action": heuristic_action,
            "nn_proposal": action,
            "reason": reason,
            "match": (action == heuristic_action),
            "arbiter_mode": orig_mode,
        }
