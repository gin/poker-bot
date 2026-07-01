"""Poker strategy profiler — computes VPIP, PFR, AF, 3-BET%, WTSD, W$SD, and BLUFF
metrics from selfplay action streams, then classifies the strategy profile.

Metrics:
  VPIP      — Voluntarily Put money In Pot (preflop). Classification:
              <20% tight, 20-30% balanced, >30% loose.
  PFR       — Pre-Flop Raise %. Classification:
              <15% passive, 15-25% measured, >25% aggressive.
  AF        — Aggression Factor (bets + raises) / calls. Classification:
              <1.0 passive, 1.0-2.5 measured, >2.5 aggressive.
  3-BET%    — Preflop re-raise frequency when facing a raise.
              <5% tight, 5-10% balanced, >10% aggressive.
  WTSD      — Went To ShowDown (of hands that saw a flop).
              <25% tight, 25-35% balanced, >35% loose.
  W$SD      — Won money at ShowDown.
              >55% tight/value, 48-55% balanced, <48% loose/bluffy.
  BLUFF     — River aggression (bet/raise when acting on river).
              <15% tight/value, 15-30% balanced, >30% aggressive/bluffy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# ------------------------------------------------------------------
# Per-hand raw data  (collected by PlayProfiler._observe)
# ------------------------------------------------------------------


@dataclass
class HandMetrics:
    """Raw action counters for a single hand played by the hero agent."""

    hand_id: str = ""
    vpip: bool = False
    pfr: bool = False
    three_bet: bool = False
    three_bet_opportunity: bool = False
    saw_flop: bool = False
    saw_showdown: bool = False
    won_showdown: bool = False
    bets: int = 0
    raises: int = 0
    calls: int = 0
    folds: int = 0
    checks: int = 0
    river_bet_raise: bool = False
    river_acted: bool = False
    won_hand: bool = False
    folded_hand: bool = False


# ------------------------------------------------------------------
# Aggregate profile  (computed across many hands)
# ------------------------------------------------------------------


@dataclass
class StrategyProfile:
    """Aggregate poker metrics with behavioural classification labels."""

    total_hands: int
    vpip_pct: float
    pfr_pct: float
    af: float
    three_bet_pct: float
    wtsd_pct: float
    wsd_pct: float  # W$SD — Won at ShowDown
    bluff_pct: float

    # Computed labels
    vpip_label: str = ""
    pfr_label: str = ""
    af_label: str = ""
    three_bet_label: str = ""
    wtsd_label: str = ""
    wsd_label: str = ""
    bluff_label: str = ""
    profile_label: str = ""

    def __post_init__(self):
        # VPIP classification
        if self.vpip_pct < 20:
            self.vpip_label = "tight"
        elif self.vpip_pct <= 30:
            self.vpip_label = "balanced"
        else:
            self.vpip_label = "loose"

        # PFR classification
        if self.pfr_pct < 15:
            self.pfr_label = "passive"
        elif self.pfr_pct <= 25:
            self.pfr_label = "measured"
        else:
            self.pfr_label = "aggressive"

        # AF classification
        if self.af < 1.0:
            self.af_label = "passive"
        elif self.af < 2.5:
            self.af_label = "measured"
        else:
            self.af_label = "aggressive"

        # 3-BET% classification
        if self.three_bet_pct < 5:
            self.three_bet_label = "tight"
        elif self.three_bet_pct <= 10:
            self.three_bet_label = "balanced"
        else:
            self.three_bet_label = "aggressive"

        # WTSD classification
        if self.wtsd_pct < 25:
            self.wtsd_label = "tight"
        elif self.wtsd_pct <= 35:
            self.wtsd_label = "balanced"
        else:
            self.wtsd_label = "loose"

        # W$SD classification
        if self.wsd_pct > 55:
            self.wsd_label = "value"
        elif self.wsd_pct >= 48:
            self.wsd_label = "balanced"
        else:
            self.wsd_label = "bluffy"

        # BLUFF classification
        if self.bluff_pct < 15:
            self.bluff_label = "tight"
        elif self.bluff_pct <= 30:
            self.bluff_label = "balanced"
        else:
            self.bluff_label = "aggressive"

        # --- Overall profile: VPIP label + dominant aggression ---
        aggro_signals = [
            self.pfr_label == "aggressive",
            self.af_label == "aggressive",
            self.bluff_label == "aggressive",
        ]
        passive_signals = [
            self.pfr_label == "passive",
            self.af_label == "passive",
            self.bluff_label == "passive",
        ]
        aggro_count = sum(aggro_signals)
        passive_count = sum(passive_signals)

        if aggro_count >= 2:
            overall = "aggressive"
        elif passive_count >= 2:
            overall = "passive"
        else:
            overall = "measured"

        self.profile_label = f"{self.vpip_label}/{overall}"


# ------------------------------------------------------------------
# Live profiler  (hooks into simulator action_observer)
# ------------------------------------------------------------------


def _is_hero(seat: dict) -> bool:
    """Check whether *seat* belongs to the hero agent."""
    return seat.get("agentId") in ("player-agent", "agent-1", "agent-0")


_HERO_IDS = frozenset({"player-agent", "agent-1", "agent-0"})


class PlayProfiler:
    """Collects per-hand action data via the simulator action_observer hook
    and computes a StrategyProfile.

    Usage::

        profiler = PlayProfiler()

        # In the selfplay loop, for each hand:
        profiler.start_hand(hand_id)
        result = play_hand(..., action_observer=profiler.observer)
                    or play_hand_multiway(..., action_observer=profiler.observer)
        profiler.end_hand(won=hero_won, showdown=reached_showdown)

        profile = profiler.compute_profile()
    """

    def __init__(self):
        self.hands: list[HandMetrics] = []
        self._current: HandMetrics | None = None
        self._hand_id: str | None = None

    # ------------------------------------------------------------------
    # Public API used by the selfplay loop
    # ------------------------------------------------------------------

    @property
    def observer(self) -> Callable:
        """Return an action_observer callable for simulator hooks."""
        return self._observe

    def start_hand(self, hand_id: str | None = None):
        """Begin tracking a new hand. Finalises any previous incomplete hand."""
        self._finalize_hand()
        self._hand_id = hand_id
        self._current = HandMetrics(hand_id=hand_id or f"h{len(self.hands)}")

    def end_hand(self, won: bool = False, showdown: bool = False):
        """Record the outcome of the current hand.

        Args:
            won: True if hero won the hand (by fold or showdown).
            showdown: True if the hand reached showdown.
        """
        if self._current is not None:
            self._current.won_hand = won
            self._current.saw_showdown = showdown
            if showdown and won:
                self._current.won_showdown = True

    def compute_profile(self) -> StrategyProfile:
        """Finalise outstanding hand and aggregate across all hands."""
        self._finalize_hand()
        total = len(self.hands)
        if total == 0:
            return StrategyProfile(
                total_hands=0,
                vpip_pct=0.0,
                pfr_pct=0.0,
                af=0.0,
                three_bet_pct=0.0,
                wtsd_pct=0.0,
                wsd_pct=0.0,
                bluff_pct=0.0,
            )

        vpip_hands = sum(1 for h in self.hands if h.vpip)
        pfr_hands = sum(1 for h in self.hands if h.pfr)
        three_bet_hands = sum(1 for h in self.hands if h.three_bet)
        three_bet_opps = sum(1 for h in self.hands if h.three_bet_opportunity)
        showdown_hands = sum(1 for h in self.hands if h.saw_showdown)
        showdown_wins = sum(1 for h in self.hands if h.won_showdown)
        flop_or_showdown = sum(1 for h in self.hands if h.saw_flop or h.saw_showdown)
        total_bets_raises = sum(h.bets + h.raises for h in self.hands)
        total_calls = sum(h.calls for h in self.hands)
        river_aggro = sum(1 for h in self.hands if h.river_bet_raise)
        river_acted_total = sum(1 for h in self.hands if h.river_acted)

        return StrategyProfile(
            total_hands=total,
            vpip_pct=vpip_hands / total * 100,
            pfr_pct=pfr_hands / total * 100,
            af=total_bets_raises / total_calls if total_calls else float("inf"),
            three_bet_pct=(
                three_bet_hands / three_bet_opps * 100 if three_bet_opps else 0.0
            ),
            wtsd_pct=(showdown_hands / flop_or_showdown * 100)
            if flop_or_showdown
            else 0.0,
            wsd_pct=(showdown_wins / showdown_hands * 100 if showdown_hands else 0.0),
            bluff_pct=(
                river_aggro / river_acted_total * 100 if river_acted_total else 0.0
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _observe(self, **event):
        """Action-observer callback — compatible with both heads-up
        (run_betting_round) and multiway (run_betting_round_multiway) event dicts.

        Also recognises a synthetic ``action="showdown"`` event emitted by
        ``play_hand_multiway`` when the hand reaches showdown.
        """
        # Synthetic showdown event
        if event.get("action") == "showdown":
            if self._current is not None:
                self._current.saw_showdown = True
            return

        hand_id = event.get("hand_id")
        if hand_id is not None and hand_id != self._hand_id:
            self._finalize_hand()
            self._start_hand(hand_id)

        action = event.get("action", "")
        street = event.get("street", "")
        facing_bet = event.get("facing_bet", False)
        voluntary = event.get("voluntary", False)
        seat = event.get("seat", {})

        if seat.get("agentId") not in _HERO_IDS:
            return

        if self._current is None:
            self._start_hand(hand_id or f"oob-{len(self.hands)}")

        c = self._current

        # --- Preflop tracking ---
        if street == "Preflop":
            if voluntary and action in ("call", "bet", "raise"):
                c.vpip = True
            if action == "raise":
                c.pfr = True
                if facing_bet:
                    c.three_bet = True
            if facing_bet:
                c.three_bet_opportunity = True

        # --- Street tracking ---
        if street in ("Flop", "Turn", "River"):
            c.saw_flop = True

        if street == "River":
            c.river_acted = True
            if action in ("bet", "raise"):
                c.river_bet_raise = True

        # --- Action counters ---
        if action == "fold":
            c.folds += 1
            c.folded_hand = True
        elif action == "call":
            c.calls += 1
        elif action == "bet":
            c.bets += 1
        elif action == "raise":
            c.raises += 1
        elif action == "check":
            c.checks += 1

    def _start_hand(self, hand_id: str | None):
        self._hand_id = hand_id
        self._current = HandMetrics(hand_id=hand_id or f"h{len(self.hands)}")

    def _finalize_hand(self):
        if self._current is not None:
            # Folded before showdown → cannot have won at showdown
            if self._current.folded_hand and self._current.saw_showdown:
                self._current.saw_showdown = False
                self._current.won_showdown = False
            self.hands.append(self._current)
        self._current = None


# ------------------------------------------------------------------
# Aggregate multiple profiles (e.g. across seeds / opponents)
# ------------------------------------------------------------------


def aggregate_profiles(profiles: list[StrategyProfile]) -> StrategyProfile:
    """Merge several StrategyProfile instances into a hand-count-weighted
    aggregate."""
    if not profiles:
        return StrategyProfile(
            total_hands=0,
            vpip_pct=0.0,
            pfr_pct=0.0,
            af=0.0,
            three_bet_pct=0.0,
            wtsd_pct=0.0,
            wsd_pct=0.0,
            bluff_pct=0.0,
        )

    total = sum(p.total_hands for p in profiles)
    if total == 0:
        return profiles[0]

    def weighted(vals, weights):
        return sum(v * w for v, w in zip(vals, weights, strict=True)) / total

    weights = [float(p.total_hands) for p in profiles]

    return StrategyProfile(
        total_hands=total,
        vpip_pct=weighted([p.vpip_pct for p in profiles], weights),
        pfr_pct=weighted([p.pfr_pct for p in profiles], weights),
        af=weighted([p.af for p in profiles], weights),
        three_bet_pct=weighted([p.three_bet_pct for p in profiles], weights),
        wtsd_pct=weighted([p.wtsd_pct for p in profiles], weights),
        wsd_pct=weighted([p.wsd_pct for p in profiles], weights),
        bluff_pct=weighted([p.bluff_pct for p in profiles], weights),
    )


# ------------------------------------------------------------------
# Formatter
# ------------------------------------------------------------------


def format_profile(profile: StrategyProfile) -> str:
    """Render a StrategyProfile as a human-readable block."""
    if profile.total_hands == 0:
        return "  profile   : (no hands profiled)"

    profile_line = f"  profile   : {profile.profile_label}"
    lines = [
        profile_line,
        f"    VPIP    {profile.vpip_pct:>5.1f}%  ({profile.vpip_label})",
        f"    PFR     {profile.pfr_pct:>5.1f}%  ({profile.pfr_label})",
        f"    AF      {profile.af:>5.2f}    ({profile.af_label})",
        f"    3-BET%  {profile.three_bet_pct:>5.1f}%  ({profile.three_bet_label})",
        f"    WTSD    {profile.wtsd_pct:>5.1f}%  ({profile.wtsd_label})",
        f"    W$SD    {profile.wsd_pct:>5.1f}%  ({profile.wsd_label})",
        f"    BLUFF   {profile.bluff_pct:>5.1f}%  ({profile.bluff_label})",
    ]
    return "\n".join(lines)
