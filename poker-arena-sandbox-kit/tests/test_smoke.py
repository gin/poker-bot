"""Smoke tests — run with `pytest` or `python tests/test_smoke.py`.

Cover the contract, the local engine, the bundle validator, and the offline
submit flow. No network required.
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STRATEGY = ROOT / "examples" / "poker" / "strategy.py"

from arena_sdk import (run_match, load_strategy, normalize_action,
                              build_bundle, BundleError, submit)
from arena_sdk.poker.engine import bot_call


def test_contract_normalize():
    assert normalize_action("call") == {"action": "call"}
    assert normalize_action(("raise", 8, "3-bet")) == {
        "action": "raise", "amount": 8, "reasoning_text": "3-bet"}  # server's field name
    assert normalize_action({"action": "fold"}) == {"action": "fold"}


def test_selfplay_runs():
    res = run_match(load_strategy(str(STRATEGY)), hands=30, opponent="tight", seed=1)
    assert res["hands"] == 30 and "bb_per_100" in res


def test_pack_builds_static_bundle():
    raw = build_bundle(str(STRATEGY))
    names = zipfile.ZipFile(__import__("io").BytesIO(raw)).namelist()
    assert "harness/strategy.py" in names


def test_harness_requires_strategy_py():
    import tempfile, os
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "helper.py"), "w") as f:
        f.write("x = 1\n")
    try:
        build_bundle(harness=d)
        raise AssertionError("expected BundleError for harness without strategy.py")
    except BundleError as e:
        assert "strategy.py" in str(e)


def test_selfplay_is_position_fair():
    # A symmetric bot vs itself must net ~0 (hero seat rotates each hand);
    # guards against the positional-bias regression.
    r = run_match(bot_call, hands=1500, opponent="self", seed=11)
    assert abs(r["bb_per_100"]) < 8, r["bb_per_100"]


def test_pack_catches_missing_sibling():
    # The headline false-green guard: a strategy importing a module that won't be
    # in the bundle must FAIL locally (isolation import), not pass silently.
    import tempfile, os
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "strategy.py"), "w") as f:
        f.write("import _nope_missing_mod\ndef act(t): return {'action': 'fold'}\n")
    try:
        build_bundle(os.path.join(d, "strategy.py"))
        raise AssertionError("expected BundleError for a missing-sibling import")
    except BundleError as e:
        assert "isolation" in str(e) or "missing" in str(e), str(e)


def test_pack_harness_multifile_ok():
    # A legit multi-file bot (strategy.py + sibling helper.py) bundled via harness
    # must pass isolation and ship both files.
    import tempfile, os, io as _io
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "helper.py"), "w") as f:
        f.write("ACTION = 'fold'\n")
    with open(os.path.join(d, "strategy.py"), "w") as f:
        f.write("import helper\ndef act(t): return {'action': helper.ACTION}\n")
    raw = build_bundle(harness=d)
    names = zipfile.ZipFile(_io.BytesIO(raw)).namelist()
    assert "harness/strategy.py" in names and "harness/helper.py" in names


def test_submit_dry_run_pvp():
    res = submit(str(STRATEGY), competition_id="demo", api_key=None,
                 expect="pvp", dry_run=True)
    assert res["status"] == "Succeeded"
    assert res.get("pvp", {}).get("status") == "Active"


def test_bb_option_labeled_raise():
    # H1 regression: in a limped HU pot the BB-option spot must be labeled `raise`
    # (currentBet = posted blind > 0), NOT `bet` — matching the server. Mislabeling
    # it `bet` made self-play pass an action the server rejects.
    from pokerkit import NoLimitTexasHoldem
    from arena_sdk.poker.engine import _AUTO, build_table
    state = NoLimitTexasHoldem.create_state(
        automations=_AUTO, ante_trimming_status=True, raw_antes=0,
        raw_blinds_or_straddles=(1, 2), min_bet=2,
        raw_starting_stacks=(200, 200), player_count=2)
    state.check_or_call()                       # SB/button limp-calls → BB to act
    actor = state.actor_index
    allowed = build_table(state, actor, "t", big_blind=2)["allowedActions"]
    aa = allowed["availableActions"]
    assert "raise" in aa and "bet" not in aa, aa
    # full allowedActions surface matches the server (so a bot reading these
    # locally behaves the same online)
    for k in ("canFold", "canCall", "canCheck", "canBet", "canRaise",
              "canAllIn", "minRaiseTo", "allInToAmount"):
        assert k in allowed, k


def test_normalize_action_edges():
    assert normalize_action({"foo": 1}) == {"action": "fold"}   # dict w/o action
    assert normalize_action(("raise", 8))["action"] == "raise"
    assert normalize_action((123,))["action"] == "123"          # coerced to str
    assert normalize_action(42) == {"action": "fold"}


def test_clamp_to_range():
    from arena_sdk import clamp_to_range
    a = {"raiseRange": {"min": 40, "max": 1000}}
    assert clamp_to_range(a, "raise", 0.6, 100) == {"action": "raise", "amount": 60}
    assert clamp_to_range(a, "raise", 100, 100)["amount"] == 1000   # clamped to max
    assert clamp_to_range({"raiseRange": {"min": 0, "max": 0}}, "raise", 0.6, 100) is None
    # not in availableActions -> None even if a range is present
    assert clamp_to_range({"availableActions": ["fold", "call"],
                           "raiseRange": {"min": 40, "max": 1000}},
                          "raise", 0.6, 100) is None


def test_multipart_encoding():
    from arena_sdk.submit import _multipart
    ctype, body = _multipart({"competitionId": "cmq1", "template": "static-agent"},
                             "file", "bundle.zip", b"PK\x03\x04")
    s = body.decode("latin1")
    assert ctype.startswith("multipart/form-data; boundary=")
    assert 'name="competitionId"' in s and "cmq1" in s
    assert 'filename="bundle.zip"' in s and "application/zip" in s
    # CRLF in a field value can't inject a new header line
    _, evil = _multipart({"x": "a\r\nContent-Disposition: evil"}, "file", "f.py", b"x")
    assert b"\r\nContent-Disposition: evil" not in evil


_FIX = Path(__file__).resolve().parent / "fixtures"


def test_golden_pve_contract():
    # Real captured PvE response (Beta) — guards the SDK's parsing against the live
    # contract, not just the dry-run mock.
    import json
    from arena_sdk.submit import _print_final
    g = json.load(open(_FIX / "golden_pve.json"))
    assert g["status"] == "Succeeded" and g["pvp"] is None
    assert isinstance(g["rawBbPer100"], (int, float))
    assert isinstance(g["adjustedBbPer100"], (int, float))
    for k in ("status", "completedHands", "targetHands", "errorCode", "error",
              "traceObjectKey"):
        assert k in g, f"real PvE response missing SDK-read field {k}"
    _print_final(g)                       # must render without raising


def test_golden_pvp_contract():
    import json
    from arena_sdk.submit import _print_final, _pvp_rating
    g = json.load(open(_FIX / "golden_pvp.json"))
    assert g["status"] == "Succeeded" and isinstance(g["pvp"], dict)
    score, mu, sigma = _pvp_rating(g["pvp"])
    assert all(isinstance(x, (int, float)) for x in (score, mu, sigma)), (score, mu, sigma)
    for k in ("status", "completedHands", "targetHands"):
        assert k in g["pvp"], f"real PvP.pvp missing SDK-read field {k}"
    _print_final(g)


def test_dry_run_mock_matches_real_shape():
    # The dry-run mock must carry every field the SDK reads from a REAL response,
    # so dry-run can't go green while a real poll would print '?'.
    import json
    from arena_sdk.submit import _make_mock
    mock = _make_mock("pvp")
    url = "https://x/api/arena/submissions/sid"
    mock("GET", url, "k")                 # Running
    final = mock("GET", url, "k")         # Succeeded (+ pvp)
    real = json.load(open(_FIX / "golden_pvp.json"))
    sdk_top = {"status", "completedHands", "targetHands", "rawBbPer100",
               "adjustedBbPer100", "error", "errorCode", "traceObjectKey"}
    for k in sdk_top:
        assert k in real, f"real golden missing SDK field {k}"
    # error/errorCode are failure-only (legitimately absent on a Succeeded response)
    assert sdk_top - set(final) <= {"error", "errorCode"}, set(final)  # mock representative
    assert any(k in real["pvp"] for k in ("trueskillScore", "scaleRating", "rating"))
    assert any(k in final["pvp"] for k in ("trueskillScore", "scaleRating", "rating"))


def test_table_has_real_server_fields():
    # build_table emits the real /pending-actions fields so position/decision
    # logic is testable locally exactly as on the server.
    from pokerkit import NoLimitTexasHoldem
    from arena_sdk.poker.engine import _AUTO, build_table
    st = NoLimitTexasHoldem.create_state(
        automations=_AUTO, ante_trimming_status=True, raw_antes=0,
        raw_blinds_or_straddles=(1, 2), min_bet=2,
        raw_starting_stacks=(200, 200), player_count=2)
    t = build_table(st, st.actor_index, "t", small_blind=1, big_blind=2,
                    starting_stack=200)
    for k in ("smallBlindChips", "bigBlindChips", "currentBet", "currentSeatNumber",
              "actionDeadlineAt", "recentEvents", "minRaiseTo"):
        assert k in t, k
    for k in ("status", "currentBetChips", "totalCommittedChips"):
        assert k in t["seats"][0], k


def test_position_inference_matches_blinds():
    # is_button must equal "I posted the small blind" at every hand, across the
    # rotated hero seat (guards the position-reading the docs teach).
    from arena_sdk import is_button, to_call, run_match
    seen, res = {}, {"ok": 0, "btn": 0, "bb": 0}
    def probe(t):
        tid = t["tableId"]
        if tid not in seen:
            seen[tid] = True
            myblind = next((e["summary"]["amount"] for e in t["recentEvents"]
                            if e["type"] == "BlindPosted"
                            and e["summary"]["seatNumber"] == t["selfSeatNumber"]), None)
            truth = myblind == t["smallBlindChips"]
            if is_button(t) == truth:
                res["ok"] += 1
            res["btn" if is_button(t) else "bb"] += 1
        a = t["allowedActions"]["availableActions"]
        if "check" in a:
            return {"action": "check"}
        return {"action": "call"} if ("call" in a and to_call(t) <= 2) else {"action": "fold"}
    run_match(probe, hands=200, opponent="call", players=2, seed=3)
    assert res["ok"] == res["btn"] + res["bb"] and res["ok"] > 0, res   # all matched
    assert res["btn"] > 0 and res["bb"] > 0, res                        # both positions seen


def test_pot_odds_helper():
    from arena_sdk import pot_odds, to_call
    t = {"potChips": 100, "allowedActions": {"callChips": 50}}
    assert to_call(t) == 50 and abs(pot_odds(t) - (50 / 150)) < 1e-9
    assert pot_odds({"potChips": 100, "allowedActions": {"callChips": 0}}) == 0.0


def test_top_imports_flags_unavailable():
    # the bundle import-guard must flag a locally-installed-but-not-on-server
    # package (e.g. pokerkit), across comma / `as` / dotted forms, but not stdlib/numpy.
    from arena_sdk.pack import _top_imports, _SERVER_HAS
    got = _top_imports("import json, pokerkit as pk\nfrom os.path import join\nimport numpy.linalg\n")
    assert {"json", "pokerkit", "os", "numpy"} <= got, got
    flagged = got - _SERVER_HAS
    assert "pokerkit" in flagged                       # not on the sandbox -> flagged
    assert "json" not in flagged and "numpy" not in flagged   # stdlib/numpy -> fine


def test_button_is_heads_up_only():
    from arena_sdk.poker.read import button_seat, is_button
    hu = {"selfSeatNumber": 2, "smallBlindChips": 1,
          "seats": [{"seatNumber": 1}, {"seatNumber": 2}],
          "recentEvents": [{"type": "BlindPosted", "summary": {"amount": 1, "seatNumber": 2}}]}
    six = {**hu, "seats": [{"seatNumber": i} for i in range(1, 7)]}
    assert button_seat(hu) == 2 and is_button(hu) is True          # HU: SB poster = button
    assert button_seat(six) is None and is_button(six) is False    # >2 seats: don't mislead


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
