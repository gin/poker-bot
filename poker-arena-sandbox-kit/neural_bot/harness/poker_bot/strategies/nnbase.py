import os
from typing import Any
from poker_bot.neural.arbiter import NeuralArbiter
from poker_bot.strategies.hubase import choose_action as hubase_choose_action

_ARBITER = None


def _clear_arbiter_cache():
    global _ARBITER
    _ARBITER = None


def get_arbiter(mode: str | None = None) -> NeuralArbiter:
    global _ARBITER
    if _ARBITER is None:
        mode = mode or os.environ.get("NN_MODE", "6max_active")
        _ARBITER = NeuralArbiter(mode=mode)
    return _ARBITER


def count_active_players(table: dict[str, Any]) -> int:
    seats = table.get("seats", [])
    return sum(1 for s in seats if not s.get("folded", False))


def extract_opponent_profile(
    table: dict[str, Any], my_seat: dict[str, Any]
) -> dict[str, Any] | None:
    profiles = table.get("opponentProfiles")
    if not profiles:
        return None
    my_id = my_seat.get("agentId", "hero")
    opp_profile = None
    for seat in table.get("seats", []):
        if seat.get("agentId") == my_id:
            continue
        if seat.get("folded"):
            continue
        opp_id = seat.get("agentId")
        if opp_id in profiles:
            opp_profile = profiles[opp_id]
            break
    if opp_profile is None:
        return None
    hands_seen = max(getattr(opp_profile, "hands_seen", 0), 1)
    calls = getattr(opp_profile, "calls", 0)
    bets = getattr(opp_profile, "bets", 0)
    raises = getattr(opp_profile, "raises", 0)
    folds = getattr(opp_profile, "folds", 0)
    vpip_count = getattr(opp_profile, "vpip", 0)
    pfr_count = getattr(opp_profile, "pfr", 0)
    fold_to_bet = getattr(opp_profile, "fold_to_bet", 0)
    fold_opp = getattr(opp_profile, "opportunities_to_fold_to_bet", 0)
    showdowns = getattr(opp_profile, "showdowns", 0)
    vpip = min(vpip_count / (hands_seen * 2.5), 1.0)
    pfr = min(pfr_count / (hands_seen * 2.5), 1.0)
    fold_to_bet_rate = min(fold_to_bet / fold_opp, 1.0) if fold_opp > 0 else 0.5
    bet_raise = bets + raises
    total_actions = calls + bets + raises + folds
    showdown_rate = min(showdowns / hands_seen, 1.0)
    return {
        "hands_seen": hands_seen,
        "vpip": vpip,
        "pfr": pfr,
        "fold_to_bet": fold_to_bet_rate,
        "action_count": total_actions,
        "bet_raise_count": bet_raise,
        "showdown_rate": showdown_rate,
    }


def choose_action(
    table: dict[str, Any], my_seat: dict[str, Any]
) -> tuple[str, float, str]:
    h_action, h_amount, h_msg = hubase_choose_action(table, my_seat)
    mode = os.environ.get("NN_MODE", "6max_active")
    opp_profile = extract_opponent_profile(table, my_seat)
    if mode == "shadow":
        arbiter = get_arbiter(mode="shadow")
        shadow_result = arbiter.shadow_decide(
            table, my_seat, h_action, opponent_profile=opp_profile
        )
        nn_act = shadow_result["nn_proposal"]
        match = "MATCH" if shadow_result["match"] else "DIFF"
        return h_action, h_amount, f"[{h_msg}] | NN: {nn_act} ({match})"
    if mode == "hu_active" and count_active_players(table) > 2:
        return h_action, h_amount, f"[{h_msg}] | nn_defer multiway"
    arbiter = get_arbiter(mode="active")
    nn_action, reason = arbiter.decide(
        table, my_seat, h_action, opponent_profile=opp_profile
    )
    if nn_action == h_action:
        amount = h_amount
    elif nn_action in ("call", "check"):
        amount = (table.get("allowedActions") or {}).get("callAmount", 0) or 0
    elif nn_action == "raise":
        amount = h_amount or (table.get("allowedActions") or {}).get("minRaiseTo", 0)
    else:
        amount = None
    tag = "NN" if nn_action != h_action else "H+NN"
    msg = f"[{h_msg}] | {tag}: {nn_action} | {reason}"
    return nn_action, float(amount) if amount else 0.0, msg
