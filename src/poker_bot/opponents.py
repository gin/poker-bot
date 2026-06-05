"""Opponent profiling helpers."""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class OpponentProfile:
    agent_id: str
    name: str | None = None
    hands_seen: int = 0
    vpip: int = 0
    pfr: int = 0
    calls: int = 0
    bets: int = 0
    raises: int = 0
    folds: int = 0
    fold_to_bet: int = 0
    opportunities_to_fold_to_bet: int = 0
    showdowns: int = 0
    weak_aggressive_showdowns: int = 0
    recent_actions: deque = field(default_factory=lambda: deque(maxlen=20))

    @property
    def aggression_frequency(self):
        actions = self.calls + self.bets + self.raises + self.folds
        if actions == 0:
            return 0.0
        return (self.bets + self.raises) / actions

    @property
    def call_frequency(self):
        actions = self.calls + self.bets + self.raises + self.folds
        if actions == 0:
            return 0.0
        return self.calls / actions

    @property
    def fold_to_bet_frequency(self):
        if self.opportunities_to_fold_to_bet == 0:
            return 0.0
        return self.fold_to_bet / self.opportunities_to_fold_to_bet

    @property
    def vpip_frequency(self):
        if self.hands_seen == 0:
            return 0.0
        return self.vpip / self.hands_seen

    @property
    def pfr_frequency(self):
        if self.hands_seen == 0:
            return 0.0
        return self.pfr / self.hands_seen

    @property
    def weak_aggressive_showdown_frequency(self):
        if self.showdowns == 0:
            return 0.0
        return self.weak_aggressive_showdowns / self.showdowns

    def label(self):
        if self.hands_seen < 5 and len(self.recent_actions) < 8:
            return "unknown"
        if self.weak_aggressive_showdown_frequency >= 0.35:
            return "bluffer"
        if self.vpip_frequency >= 0.45 and self.aggression_frequency >= 0.35:
            return "loose_aggressive"
        if self.vpip_frequency >= 0.45 and self.call_frequency >= 0.45:
            return "calling_station"
        if self.vpip_frequency <= 0.18 and self.pfr_frequency <= 0.08:
            return "patient_methodical"
        if self.pfr_frequency >= 0.22 and self.aggression_frequency >= 0.30:
            return "tight_aggressive"
        return "balanced"


def profile_from_mapping(agent_id, data):
    profile = OpponentProfile(agent_id=agent_id, name=data.get("name"))
    for field_name in (
        "hands_seen",
        "vpip",
        "pfr",
        "calls",
        "bets",
        "raises",
        "folds",
        "fold_to_bet",
        "opportunities_to_fold_to_bet",
        "showdowns",
        "weak_aggressive_showdowns",
    ):
        setattr(profile, field_name, int(data.get(field_name, 0)))
    for action in data.get("recent_actions", []):
        profile.recent_actions.append(action)
    return profile


def record_action(profile, action, street=None, facing_bet=False, voluntary=False):
    profile.recent_actions.append({"street": street, "action": action})
    if voluntary:
        profile.vpip += 1
    if action == "call":
        profile.calls += 1
    elif action == "bet":
        profile.bets += 1
    elif action == "raise":
        profile.raises += 1
        if street == "Preflop":
            profile.pfr += 1
    elif action == "fold":
        profile.folds += 1
        if facing_bet:
            profile.fold_to_bet += 1

    if facing_bet:
        profile.opportunities_to_fold_to_bet += 1


def summarize_profiles(profiles):
    return {
        agent_id: {
            "label": profile.label(),
            "vpip": round(profile.vpip_frequency, 3),
            "pfr": round(profile.pfr_frequency, 3),
            "aggression": round(profile.aggression_frequency, 3),
            "fold_to_bet": round(profile.fold_to_bet_frequency, 3),
        }
        for agent_id, profile in profiles.items()
    }
