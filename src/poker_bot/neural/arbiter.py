"""Neural arbiter for NN vs heuristic action selection."""
import os
from pathlib import Path
from typing import Any

import torch

from poker_bot.hand_eval import evaluate_hand
from poker_bot.neural.encoder import encode_state
from poker_bot.neural.guards import PokerGuard
from poker_bot.neural.models import ACTION_MAP, PolicyNetwork

RANK_VALUES = {rank: index for index, rank in enumerate("23456789TJQKA", start=2)}

def _card_values(cards):
    return [RANK_VALUES.get(card[0], 0) for card in cards]

def made_hand_rank(hole_cards, board_cards):
    if len(board_cards) < 3:
        return 0
    board_rank = evaluate_hand(board_cards) if len(board_cards) >= 5 else (0,)
    full_rank = evaluate_hand(list(hole_cards) + list(board_cards))
    category = full_rank[0]
    if len(board_cards) >= 5 and full_rank == board_rank:
        return 0
    return category

def has_top_pair_or_better(hole_cards, board_cards):
    if not board_cards:
        return False
    board_high = max(_card_values(board_cards))
    hole_values = _card_values(hole_cards)
    all_values = _card_values(list(hole_cards) + list(board_cards))
    return any(value == board_high and all_values.count(value) >= 2 for value in hole_values)

def preflop_score(hole_cards):
    if len(hole_cards) != 2:
        return 0
    first, second = _card_values(hole_cards)
    high = max(first, second)
    score = high * 3 + min(first, second)
    if first == second:
        score += 34 + high * 2
    if hole_cards[0][1] == hole_cards[1][1]:
        score += 8
    if abs(first - second) == 1:
        score += 5
    if high >= 12 and min(first, second) >= 10:
        score += 12
    return score

INPUT_DIM = 31

class NeuralArbiter:
    """Meta-Arbiter that selects between NN proposal and heuristic baseline."""

    def __init__(self, policy_path=None, value_path=None, trust_threshold=0.85, mode="shadow"):
        root = Path(__file__).resolve().parents[3]
        self.policy_path = policy_path or os.environ.get("NN_POLICY_PATH", f"{root}/assets/policy_v1.pt")
        self.value_path = value_path or os.environ.get("NN_VALUE_PATH", f"{root}/assets/value_v1.pt")
        self.policy_net = None
        self.value_net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trust_threshold = trust_threshold
        self.mode = mode

    def _load_policy(self):
        if self.policy_net is None:
            self.policy_net = PolicyNetwork(input_dim=INPUT_DIM)
            if Path(self.policy_path).exists():
                self.policy_net.load_state_dict(torch.load(self.policy_path, map_location=self.device))
            self.policy_net.to(self.device)
            self.policy_net.eval()

    def _encode(self, table, hero_seat, opponent_profile=None):
        state_vec = encode_state(table, hero_seat, opponent_profile=opponent_profile)
        return torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)

    def decide(self, table, hero_seat, heuristic_action, opponent_profile=None):
        """NN proposes -> Guard validates -> Selection."""
        self._load_policy()

        # NN Proposal
        state_tensor = self._encode(table, hero_seat, opponent_profile)
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            nn_action = ACTION_MAP.get(pred_idx.item(), "fold")
            confidence = conf.item()

        # Guard Validation
        final_nn_action, guard_reason = PokerGuard.evaluate(table, hero_seat, nn_action)

        # Selection
        if self.mode == "shadow":
            return (heuristic_action, f"shadow nn={final_nn_action} conf={confidence:.2f} guard={guard_reason}")

        table_street = table.get("street", "")
        street_threshold = {"Preflop": 0.85, "Flop": 0.95, "Turn": 0.92, "River": 0.80}
        threshold = street_threshold.get(table_street, self.trust_threshold)

        if confidence < threshold:
            return (heuristic_action, f"heuristic_fallback street={table_street} conf={confidence:.2f}<{threshold:.2f}")

        allowed = table.get("allowedActions") or {}
        call_amt = allowed.get("callAmount", 0) or 0
        pot = table.get("potChips", 0) or 0
        if call_amt > 0 and call_amt > pot * 0.5:
            return (heuristic_action, f"heuristic_shove_defense call={call_amt} pot={pot}")

        return (final_nn_action, f"nn_active conf={confidence:.2f} guard={guard_reason}")

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