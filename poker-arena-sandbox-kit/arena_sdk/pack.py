"""Build + locally validate a Sandbox submission bundle.

The dev.fun Arena server accepts a `static-agent` submission as either a bare
`strategy.py` (auto-wrapped to `harness/strategy.py`) or a `bundle.zip`. Use a
zip when you ship trained weights / data under `assets/`, or multiple code files
under `harness/` (via `--harness <dir>`).

This module mirrors the SERVER's `inspectSandboxBundle` rules (size/structure)
AND — crucially — **import-validates the bundle in isolation**: it extracts the
zip to a temp dir and imports `harness/strategy.py` with ONLY the bundle on the
path, exactly as the server does. That catches the #1 silent failure — a strategy
that imports a sibling module which never made it into the bundle — locally, in
milliseconds, instead of after you've spent a metered production submission.

Bundle layout:
    bundle.zip
      harness/        # code; harness/strategy.py REQUIRED
        strategy.py
        (helpers.py)  # extra modules go here too (use --harness <dir>)
      assets/         # optional: trained weights / lookup tables (largest budget)
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path
from typing import Optional

# ── server limits (limits.ts) — keep in sync ────────────────────────────────
MiB = 1024 * 1024
KiB = 1024
TOTAL_BYTES = 100 * MiB        # zip size AND total uncompressed
HARNESS_BYTES = 256 * KiB      # harness/ uncompressed (also caps a bare strategy.py)
ASSETS_BYTES = 100 * MiB       # assets/
MAX_FILES = 512
STATIC_STRATEGY_ENTRY = "harness/strategy.py"

# A minimal but realistic preflop table (hero faces a raise) used to smoke-test
# that act() actually runs in the bundle, not just imports.
_SAMPLE_TABLE = {
    "tableId": "validate", "id": "validate", "competitionId": "validate",
    "status": "Active", "potChips": 30, "street": "Preflop", "boardCards": [],
    "currentBet": 20, "minRaiseTo": 40, "smallBlindChips": 10, "bigBlindChips": 20,
    "selfSeatNumber": 1, "currentSeatNumber": 1, "actingSeatNumber": 1,
    "actionDeadlineAt": 0, "winners": [],
    "seats": [{"seatNumber": 1, "agentHandle": "hero", "status": "Active",
               "stackChips": 990, "currentBetChips": 10, "totalCommittedChips": 10,
               "holeCards": ["Ah", "Kd"]},
              {"seatNumber": 2, "agentHandle": "villain", "status": "Active",
               "stackChips": 980, "currentBetChips": 20, "totalCommittedChips": 20,
               "holeCards": []}],
    "allowedActions": {"availableActions": ["fold", "call", "raise"],
                       "callChips": 10, "callToAmount": 20,
                       "canFold": True, "canCheck": False, "canCall": True,
                       "canBet": False, "canRaise": True, "canAllIn": False,
                       "betRange": {"min": 0, "max": 0},
                       "raiseRange": {"min": 40, "max": 990},
                       "minRaiseTo": 40, "allInToAmount": 990},
    # blind posts so position helpers resolve (seat 1 posted the small blind = button)
    "recentEvents": [
        {"type": "BlindPosted", "street": "Preflop",
         "summary": {"action": "post", "amount": 10, "toAmount": 10, "seatNumber": 1}},
        {"type": "BlindPosted", "street": "Preflop",
         "summary": {"action": "post", "amount": 20, "toAmount": 20, "seatNumber": 2}}],
}


# The sandbox image has ONLY the stdlib + numpy + torch. Anything else a bundled
# .py imports — that isn't a module you also bundled — will ImportError on the
# server, even though it may be installed in your local venv. We catch that here.
_FROM_RE = re.compile(r"^\s*from\s+([A-Za-z0-9_]+)", re.M)
_IMPORT_RE = re.compile(r"^\s*import\s+([^\n#]+)", re.M)
_SERVER_HAS = set(getattr(sys, "stdlib_module_names", ())) | {"numpy", "torch"}
_CEXT_HINT = {"eval7", "onnxruntime"}   # won't run even if vendored (C extension)


def _top_imports(src: str) -> set:
    """Top-level package names imported by `src` — handles `from x import ...`,
    `import a, b as c`, and dotted `import a.b`."""
    mods = set(_FROM_RE.findall(src))
    for line in _IMPORT_RE.findall(src):
        for part in line.split(","):
            tok = part.strip().split(" ")[0].split(".")[0]   # drop `as ...` and `.sub`
            if tok.isidentifier():
                mods.add(tok)
    return mods


def _warn_unavailable_imports(raw: bytes) -> None:
    """Warn (don't fail) if a bundled .py imports a top-level package that isn't on
    the sandbox (stdlib + numpy + torch) and isn't something you bundled yourself."""
    z = zipfile.ZipFile(io.BytesIO(raw))
    bundled, imported = set(), set()
    for n in z.namelist():
        if n.startswith("harness/"):                       # names you can import
            top = n[len("harness/"):].split("/")[0]
            bundled.add(top[:-3] if top.endswith(".py") else top)
        if n.startswith("harness/") and n.endswith(".py"):
            try:
                imported |= _top_imports(z.read(n).decode("utf-8", "ignore"))
            except Exception:
                pass
    for m in sorted(imported - _SERVER_HAS - bundled):
        extra = " (a C-extension package won't run even if vendored)" if m in _CEXT_HINT else \
                " — vendor it under your bundle (pure-Python only) or precompute"
        print(f"[pack] ⚠ '{m}' isn't on the sandbox (only stdlib + numpy + torch){extra}; "
              "it would ImportError server-side.")


class BundleError(Exception):
    """Local validation failure — message mirrors the server's 400 text."""


def _iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        if "__pycache__" in p.parts or p.suffix in (".pyc", ".pyo"):
            continue                                  # never ship bytecode caches
        if p.is_symlink():
            raise BundleError(f"bundle may not contain symlinks: {p}")
        if p.is_file():
            yield p


def _validate_isolation(raw: bytes) -> None:
    """Extract the bundle and import harness/strategy.py with ONLY the bundle on
    the path (like the server), then call act() once. Hard-fail on import error
    (the classic 'sibling module not bundled' trap); warn on a runtime throw."""
    with tempfile.TemporaryDirectory() as td:
        zipfile.ZipFile(io.BytesIO(raw)).extractall(td)
        harness = str(Path(td) / "harness")
        script = textwrap.dedent(f"""
            import sys, json, importlib.util
            sys.path.insert(0, {harness!r})
            spec = importlib.util.spec_from_file_location(
                "strategy", {harness!r} + "/strategy.py")
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)                 # import error -> exit 2
            fn = (getattr(m, "choose_action", None) or getattr(m, "act", None))
            if fn is None:
                print("NO_ENTRYPOINT"); sys.exit(3)
            try:
                r = fn({_SAMPLE_TABLE!r})
            except Exception as e:
                print("ACT_THREW " + type(e).__name__ + ": " + str(e)[:160])
            else:
                # The contract accepts a string, a dict, or a tuple. Anything else
                # (None, int, ...) is a bug — flag it, but don't block the build.
                if not isinstance(r, (str, dict, list, tuple)):
                    print("ACT_BADTYPE " + type(r).__name__)
                print("ACT_OK " + json.dumps(r, default=str)[:200])
        """)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        try:
            # -I = isolated mode: ignore PYTHONPATH + user site-packages and do
            # NOT put cwd on the path, so the check faithfully mirrors the server's
            # bare import path — a module that's only on YOUR PYTHONPATH can't mask
            # one that's missing from the bundle. We add ONLY harness/ explicitly.
            proc = subprocess.run([sys.executable, "-I", "-c", script], cwd=harness,
                                  capture_output=True, text=True, timeout=30, env=env)
        except subprocess.TimeoutExpired:
            raise BundleError(
                "importing the bundle + one act() call did not finish within 30s — "
                "likely an import-time hang or an infinite loop. (Separately, the server "
                "enforces a per-decision deadline ~10s on every act() — keep act() fast.) "
                "Fix before submitting.")
        out = (proc.stdout + proc.stderr).strip()
        # NO_ENTRYPOINT is signalled by an exact exit code (3), not a substring, so
        # source/traceback text can't misclassify it.
        if proc.returncode == 3:
            raise BundleError("strategy.py defines no entrypoint — add a "
                              "choose_action(table) (or act(table)) function.")
        if proc.returncode != 0:
            tail = out.splitlines()[-1] if out else f"exit {proc.returncode}"
            # Classify by the exception CLASS on the traceback's final line only.
            etype = tail.split(":", 1)[0].strip()
            if etype in ("ModuleNotFoundError", "ImportError"):
                raise BundleError(
                    "bundle is missing a module needed to import — a sibling .py file "
                    "→ add it with --harness <dir>, or an unvendored 3rd-party package "
                    f"→ ship it under assets/ or inline it: {tail}")
            if etype in ("SyntaxError", "IndentationError", "TabError"):
                raise BundleError(f"strategy.py has a syntax error: {tail}")
            raise BundleError(f"strategy.py failed to load in isolation: {tail}")
        if "ACT_BADTYPE" in out:
            bad = out.rsplit("ACT_BADTYPE", 1)[-1].split()[0]
            print(f"[pack] ⚠ act() returned a `{bad}` — return a dict "
                  '(e.g. {"action": "raise", "amount": 120}) or a bare action string; '
                  "a dict is the safe form.")
        last = out.splitlines()[-1] if out else ""
        if last.startswith("ACT_THREW"):
            print(f"[pack] ⚠ act() raised on a sample hand ({last[9:]}) — "
                  "may be fine (needs assets/network) but check it.")
        else:
            print(f"[pack] isolation ok: imports clean + act() ran ({last})")


def build_bundle(strategy: Optional[str] = None, *, harness: Optional[str] = None,
                 assets: Optional[str] = None, out: Optional[str] = None,
                 validate: bool = True) -> bytes:
    """Build a static-agent bundle.zip (returns its bytes; writes to `out` if given).

    Provide EITHER `strategy` (one file -> harness/strategy.py) OR `harness` (a dir
    copied into harness/, for multi-file bots; must contain strategy.py). `assets`
    is a dir copied under assets/ (trained weights / lookup tables). Validates
    against the server limits + import-isolation before returning.
    """
    if not strategy and not harness:
        raise BundleError("provide --strategy <file> or --harness <dir>")

    buf = io.BytesIO()
    nfiles = 0
    harness_b = assets_b = 0

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if harness:
            base = Path(harness).resolve()
            if not base.is_dir():
                raise BundleError(f"--harness must be a directory: {base}")
            if not (base / "strategy.py").is_file():
                raise BundleError("--harness dir must contain a strategy.py file")
            # Do NOT load_strategy() here: it imports without the harness dir on
            # sys.path, so a legitimate sibling import (`import helper`) would
            # false-fail. The isolation check below imports it correctly (with
            # only the bundled harness/ on the path, exactly like the server).
            print(f"[pack] harness: {base} (entry: strategy.py)")
            for p in _iter_files(base):
                data = p.read_bytes()
                harness_b += len(data); nfiles += 1
                z.writestr(f"harness/{p.relative_to(base).as_posix()}", data)
        else:
            strat_path = Path(strategy).resolve()
            if not strat_path.exists():
                raise BundleError(f"strategy file not found: {strat_path}")
            # one self-contained strategy.py -> harness/strategy.py.
            # Import + entrypoint validation happens in the isolation check below
            # (a faithful bare-path subprocess), NOT here in the live process —
            # so a missing import surfaces as a clean BundleError, not a traceback,
            # and can't false-pass on the caller's sys.path.
            print(f"[pack] strategy: {strat_path.name}")
            code = strat_path.read_bytes()
            harness_b += len(code); nfiles += 1
            z.writestr(STATIC_STRATEGY_ENTRY, code)

        if assets:
            base = Path(assets).resolve()
            if not base.is_dir():
                raise BundleError(f"assets path must be a directory: {base}")
            for p in _iter_files(base):
                data = p.read_bytes()
                assets_b += len(data); nfiles += 1
                z.writestr(f"assets/{p.relative_to(base).as_posix()}", data)

    raw = buf.getvalue()

    # ── mirror inspectSandboxBundle hard rules ──────────────────────────────
    if "harness/strategy.py" not in zipfile.ZipFile(io.BytesIO(raw)).namelist():
        raise BundleError("static-agent submissions must include harness/strategy.py")
    if nfiles < 1:
        raise BundleError("bundle must contain at least one file")
    if nfiles > MAX_FILES:
        raise BundleError(f"bundle has {nfiles} files; max {MAX_FILES}")
    if len(raw) > TOTAL_BYTES:
        raise BundleError(f"bundle.zip is {len(raw)//MiB}MiB; max {TOTAL_BYTES//MiB}MiB")
    if (harness_b + assets_b) > TOTAL_BYTES:
        raise BundleError(f"bundle uncompressed is {(harness_b+assets_b)//MiB}MiB; "
                          f"max {TOTAL_BYTES//MiB}MiB")
    if harness_b > HARNESS_BYTES:
        raise BundleError(f"harness/ is {harness_b//KiB}KiB; max {HARNESS_BYTES//KiB}KiB")
    if assets_b > ASSETS_BYTES:
        raise BundleError(f"assets/ is {assets_b//MiB}MiB; max {ASSETS_BYTES//MiB}MiB")

    # ── import-isolation: catch missing-sibling-import BEFORE you submit ─────
    if validate:
        _validate_isolation(raw)
        _warn_unavailable_imports(raw)

    if out:
        Path(out).write_bytes(raw)
        print(f"[pack] wrote {out}  ({len(raw)//KiB}KiB, {nfiles} files, "
              f"harness={harness_b//KiB}KiB assets={assets_b//KiB}KiB)")
    return raw


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="arena pack",
                                 description="Build + validate a Sandbox bundle.zip")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--strategy", help="path to a single self-contained strategy.py")
    src.add_argument("--harness", help="dir copied into harness/ (multi-file bot; needs strategy.py)")
    ap.add_argument("--assets", help="dir copied under assets/ (trained weights / lookup tables)")
    ap.add_argument("--out", default="bundle.zip", help="output zip path")
    a = ap.parse_args(argv)
    try:
        build_bundle(a.strategy, harness=a.harness, assets=a.assets, out=a.out)
    except BundleError as e:
        print(f"[pack] INVALID: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
